#include "src/turbomind/models/llama/GatedDeltaNetWeight.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

GatedDeltaNetWeight::GatedDeltaNetWeight(int      hidden_dim,
                                         int      num_k_heads,
                                         int      num_v_heads,
                                         int      key_head_dim,
                                         int      value_head_dim,
                                         int      d_conv,
                                         bool     bias,
                                         int      tp_size,
                                         int      tp_rank,
                                         DataType data_type,
                                         DataType weight_type,
                                         int      group_size):
    tp_rank_(tp_rank), tp_size_(tp_size)
{
    const int key_dim    = num_k_heads * key_head_dim / tp_size;
    const int value_dim  = num_v_heads * value_head_dim / tp_size;
    const int v_heads_tp = num_v_heads / tp_size;
    const int conv_dim   = key_dim * 2 + value_dim;

    // GatedDeltaNet projections are stored as plain dense weights in the checkpoint
    // (dense_wtype = data_type avoids quantization path for these projections).
    const DataType dense_wtype = data_type;
    const int      dense_gsz   = 0;

    // Individual projections registered for checkpoint loading
    in_proj_qkv.emplace(hidden_dim, conv_dim, data_type, bias, dense_wtype, dense_gsz);
    in_proj_z.emplace(hidden_dim, value_dim, data_type, bias, dense_wtype, dense_gsz);
    in_proj_b.emplace(hidden_dim, v_heads_tp, data_type, bias, dense_wtype, dense_gsz);
    in_proj_a.emplace(hidden_dim, v_heads_tp, data_type, bias, dense_wtype, dense_gsz);
    out_proj.emplace(value_dim, hidden_dim, data_type, bias, dense_wtype, dense_gsz);

    register_module("in_proj_qkv", in_proj_qkv, tp_rank_);
    register_module("in_proj_z", in_proj_z, tp_rank_);
    register_module("in_proj_b", in_proj_b, tp_rank_);
    register_module("in_proj_a", in_proj_a, tp_rank_);
    register_module("out_proj", out_proj, tp_rank_);

    // conv1d: depthwise weights, shape (conv_dim, d_conv)
    conv1d = Tensor{{conv_dim, d_conv}, data_type, kDEVICE};
    register_parameter("conv1d." + std::to_string(tp_rank_) + ".weight", conv1d);

    // A_log: log-space decay per head, shape (num_v_heads/tp,)
    A_log = Tensor{{v_heads_tp}, data_type, kDEVICE};
    register_parameter("A_log." + std::to_string(tp_rank_) + ".weight", A_log);

    // dt_bias: per head, shape (num_v_heads/tp,)
    dt_bias = Tensor{{v_heads_tp}, data_type, kDEVICE};
    register_parameter("dt_bias." + std::to_string(tp_rank_) + ".weight", dt_bias);

    // norm: RMSNormGated weight, shape (value_head_dim,)
    norm = Tensor{{value_head_dim}, data_type, kDEVICE};
    register_parameter("norm.weight", norm);
}

// ---------------------------------------------------------------------------
// Row-wise concatenation of 4 weight matrices into a single pre-allocated
// destination tensor.
//
// Each source weight has shape (input_dim, out_dim_i) in row-major storage.
// The destination has shape (input_dim, sum_i out_dim_i) and rows are filled
// by concatenating the corresponding source rows in order.
//
// Implemented with cudaMemcpy2DAsync so that no extra temporary is needed:
// each source "column block" is scattered into the correct column range of
// the destination in one pass per source.
// ---------------------------------------------------------------------------
static void
concat_weights_4(const Tensor& a, const Tensor& b, const Tensor& c, const Tensor& d, Tensor& dst, cudaStream_t st)
{
    // Tensors are (K=input_dim, M=output_dim) in row-major order.
    // Each row of `dst` is [a_row | b_row | c_row | d_row].
    const int K       = dst.shape(0);
    const int M_a     = a.shape(1);
    const int M_b     = b.shape(1);
    const int M_c     = c.shape(1);
    const int M_d     = d.shape(1);
    const int M_dst   = dst.shape(1);  // M_a + M_b + M_c + M_d
    const int elem_sz = byte_size(dst.dtype(), 1);

    // Pitch of the destination row in bytes
    const size_t dst_pitch   = (size_t)M_dst * elem_sz;
    const size_t src_pitch_a = (size_t)M_a * elem_sz;
    const size_t src_pitch_b = (size_t)M_b * elem_sz;
    const size_t src_pitch_c = (size_t)M_c * elem_sz;
    const size_t src_pitch_d = (size_t)M_d * elem_sz;

    char* dst_ptr = reinterpret_cast<char*>(dst.raw_data());

    // Columns [0, M_a)
    TM_CUDA_CHECK(
        cudaMemcpy2DAsync(dst_ptr, dst_pitch, a.raw_data(), src_pitch_a, src_pitch_a, K, cudaMemcpyDefault, st));

    // Columns [M_a, M_a+M_b)
    TM_CUDA_CHECK(cudaMemcpy2DAsync(
        dst_ptr + src_pitch_a, dst_pitch, b.raw_data(), src_pitch_b, src_pitch_b, K, cudaMemcpyDefault, st));

    // Columns [M_a+M_b, M_a+M_b+M_c)
    TM_CUDA_CHECK(cudaMemcpy2DAsync(dst_ptr + src_pitch_a + src_pitch_b,
                                    dst_pitch,
                                    c.raw_data(),
                                    src_pitch_c,
                                    src_pitch_c,
                                    K,
                                    cudaMemcpyDefault,
                                    st));

    // Columns [M_a+M_b+M_c, M_dst)
    TM_CUDA_CHECK(cudaMemcpy2DAsync(dst_ptr + src_pitch_a + src_pitch_b + src_pitch_c,
                                    dst_pitch,
                                    d.raw_data(),
                                    src_pitch_d,
                                    src_pitch_d,
                                    K,
                                    cudaMemcpyDefault,
                                    st));
}

void GatedDeltaNetWeight::prepare()
{
    auto stream = core::Context::stream().handle();

    // Preprocess individual weights (converts blockscale FP8, etc.)
    in_proj_qkv.preprocess();
    in_proj_z.preprocess();
    in_proj_b.preprocess();
    in_proj_a.preprocess();
    out_proj.preprocess();
    out_proj.prepare();

    // Build the fused input projection weight:
    //   shape (hidden_dim,  conv_dim + value_dim + 2*v_heads_tp)
    //   = [in_proj_qkv | in_proj_z | in_proj_b | in_proj_a]  (column-wise)
    const int out_all = in_proj_qkv.output_dim  //
                        + in_proj_z.output_dim  //
                        + in_proj_b.output_dim  //
                        + in_proj_a.output_dim;

    in_proj_all.emplace(in_proj_qkv.input_dim,
                        out_all,
                        in_proj_qkv.data_type,
                        /*bias=*/false,
                        in_proj_qkv.weight_type,
                        in_proj_qkv.group_size);

    concat_weights_4(
        in_proj_qkv.weight, in_proj_z.weight, in_proj_b.weight, in_proj_a.weight, in_proj_all.weight, stream);

    // Prepare (convert/repack) the fused weight for GEMM
    in_proj_all.prepare();

    // Release the now-redundant individual weight tensors to free HBM
    in_proj_qkv = {};
    in_proj_z   = {};
    in_proj_b   = {};
    in_proj_a   = {};

    // Transpose conv1d from checkpoint layout [conv_dim, d_conv] to kernel layout [d_conv, conv_dim]
    {
        const int rows = conv1d.shape(0);  // conv_dim
        const int cols = conv1d.shape(1);  // d_conv

        Tensor conv1d_t{{cols, rows}, conv1d.dtype(), kDEVICE};
        TM_CUDA_CHECK(
            invokeTransposeAxis01((uint16_t*)conv1d_t.raw_data(), (uint16_t*)conv1d.raw_data(), rows, cols, 1, stream));
        conv1d = std::move(conv1d_t);
    }
}

}  // namespace turbomind
