#include "src/turbomind/models/llama/LlamaDenseWeight.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gpt_kernels.h"

#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

void LlamaDenseWeight::emplace(
    int input_dim, int output_dim, DataType data_type, bool bias, DataType weight_type, int group_size)
{
    this->data_type   = data_type;
    this->weight_type = weight_type;
    this->input_dim   = input_dim;
    this->output_dim  = output_dim;
    this->group_size  = group_size;

    const auto wbits = byte_size(weight_type, 8);

    weight = Tensor({input_dim, output_dim}, weight_type, kDEVICE);
    register_parameter(wbits < 16 ? "qweight" : "weight", weight);

    if (bias) {
        this->bias = Tensor{{output_dim}, data_type, kDEVICE};
        register_parameter("bias", this->bias);
    }

    if (wbits < 16) {
        TM_CHECK(input_dim % group_size == 0) << input_dim << " " << group_size;
        scales = Tensor{{input_dim / group_size, output_dim}, data_type, kDEVICE};
        zeros  = Tensor{{input_dim / group_size, output_dim}, data_type, kDEVICE};
        register_parameter("scales", scales);
        register_parameter("zeros", zeros);
    }
}

static void convert_u4(LlamaDenseWeight& dense, bool is_fused_moe, bool use_simt, cudaStream_t st)
{
    TM_CHECK_EQ(dense.weight_type, data_type_v<uint4_t>);

    using namespace gemm;

    auto [order_b, pack_b, order_v, pack_v] =
        get_weight_and_scales_layout(data_type_v<uint4_t>, is_fused_moe, getSMVersion(), use_simt);

    if (order_b == kColMajor) {
        Buffer trans{dense.input_dim * dense.output_dim, data_type_v<uint4_t>, kDEVICE};
        transpose_u4(
            (uint4_t*)trans.raw_data(), (const uint4_t*)dense.weight.raw_data(), dense.input_dim, dense.output_dim, st);
        cudaMemcpyAsync(
            dense.weight.raw_data(), trans.raw_data(), dense.input_dim * dense.output_dim / 2, cudaMemcpyDefault, st);
    }

    Buffer_<uint16_t> tmp_w{dense.input_dim * dense.output_dim, kDEVICE};
    extend_to_u16(tmp_w.data(), (const uint4_t*)dense.weight.raw_data(), dense.input_dim * dense.output_dim, st);
    sync_check_cuda_error();

    MatrixLayout w_desc{
        data_type_v<half_t>,
        order_b,
        (int)dense.input_dim,   // k
        (int)dense.output_dim,  // n
        order_b == kRowMajor ? (int)dense.output_dim : (int)dense.input_dim,
    };

    MatrixLayout k_desc = w_desc;
    k_desc.type         = data_type_v<uint4_t>;
    k_desc.pack         = pack_b;

    cudaMemsetAsync(dense.weight.raw_data(), 0, dense.input_dim * dense.output_dim / 2, st);

    FT_CHECK(Convert(tmp_w.data(), w_desc, dense.weight.raw_data(), k_desc, st) == 0);
    sync_check_cuda_error();

    const int scale_count = (dense.input_dim / dense.group_size) * dense.output_dim;

    Buffer_<half> tmp_q{scale_count * 2, kDEVICE};
    fuse_scales_and_zeros(tmp_q.data(), dense.scales.data<half>(), dense.zeros.data<half>(), scale_count, st);
    sync_check_cuda_error();

    dense.scales = {};
    dense.zeros  = {};

    dense.scales_zeros = Tensor_<half>{{scale_count, 2}, kDEVICE};

    MatrixLayout s_desc{
        data_type_v<uint32_t>,
        order_v,
        (int)dense.input_dim / dense.group_size,  // k
        (int)dense.output_dim,                    // n
        (int)dense.output_dim,
    };

    MatrixLayout q_desc = s_desc;
    q_desc.pack         = pack_v;

    FT_CHECK(Convert(tmp_q.data(), s_desc, dense.scales_zeros.raw_data(), q_desc, st) == 0);
    sync_check_cuda_error();

    dense.k_desc = k_desc;
    dense.q_desc = q_desc;
}

static void convert_fp(LlamaDenseWeight& dense, bool is_fused_moe, bool use_simt, cudaStream_t st)
{
    using namespace gemm;

    if (!is_fused_moe) {
        return;
    }

    /// TODO: unify data types
    auto data_type = dense.data_type;

    const auto [order_b, pack_b, order_v, pack_v] =
        get_weight_and_scales_layout(data_type, is_fused_moe, getSMVersion(), use_simt);

    const int input_dim  = dense.input_dim;
    const int output_dim = dense.output_dim;

    TM_CHECK(dense.weight.is_contiguous());

    Buffer_<uint16_t> tmp{input_dim * output_dim, kDEVICE};

    if (order_b == kColMajor) {
        invokeTransposeAxis01(tmp.data(), (uint16_t*)dense.weight.raw_data(), input_dim, output_dim, 1, st);
        sync_check_cuda_error();
    }
    else {
        check_cuda_error(
            cudaMemcpyAsync(tmp.data(), dense.weight.raw_data(), dense.weight.byte_size(), cudaMemcpyDefault, st));
    }

    MatrixLayout src{
        data_type,
        order_b,
        input_dim,   // k
        output_dim,  // n
        order_b == kRowMajor ? output_dim : input_dim,
    };

    MatrixLayout dst = src;
    dst.pack         = pack_b;

    if (pack_b) {
        FT_CHECK(Convert(tmp.data(), src, dense.weight.raw_data(), dst, st) == 0);
        sync_check_cuda_error();
    }
    else {
        check_cuda_error(
            cudaMemcpyAsync(dense.weight.raw_data(), tmp.data(), dense.weight.byte_size(), cudaMemcpyDefault, st));
    }

    dense.k_desc = dst;
}

static void convert(LlamaDenseWeight& dense, bool is_fused_moe, DataType data_type, bool use_simt, cudaStream_t st) {}

void LlamaDenseWeight::prepare(bool fused_moe, bool use_simt)
{
    if (!weight) {
        return;
    }

    auto stream = core::Context::stream().handle();

    if (weight_type == data_type_v<uint4_t>) {
        TM_CHECK_EQ(data_type, data_type_v<half_t>);
        convert_u4(*this, fused_moe, use_simt, stream);
    }
    else {
        convert_fp(*this, fused_moe, use_simt, stream);
    }
}

LlamaAttentionWeight::LlamaAttentionWeight(int      hidden_dim,
                                           int      head_dim,
                                           int      head_num,
                                           int      kv_head_num,
                                           MLAParam mla,
                                           bool     bias,
                                           bool     qk_norm,
                                           int      tp_size,
                                           int      tp_rank,
                                           DataType data_type,
                                           DataType weight_type,
                                           int      group_size)
{
    if (mla.kv_lora_rank == 0) {
        qkv.emplace(
            hidden_dim, (head_num + 2 * kv_head_num) * head_dim / tp_size, data_type, bias, weight_type, group_size);
        register_module("w_qkv", qkv, tp_rank);
        if (qk_norm) {
            q_a_layernorm  = Tensor{{head_dim}, data_type, kDEVICE};
            kv_a_layernorm = Tensor{{head_dim}, data_type, kDEVICE};
            register_parameter("q_norm", q_a_layernorm);
            register_parameter("k_norm", kv_a_layernorm);
        }
    }
    else {
        const int qk_nope_dim = head_dim - mla.qk_rope_dim;
        if (mla.q_lora_rank) {
            q_a_proj.emplace(hidden_dim, mla.q_lora_rank, data_type, false, weight_type, group_size);
            q_b_proj.emplace(mla.q_lora_rank, head_num * head_dim / tp_size, data_type, false, weight_type, group_size);
            q_a_layernorm = Tensor{{q_b_proj.input_dim}, data_type, kDEVICE};
            register_module("q_a_proj", q_a_proj);
            register_module("q_b_proj", q_b_proj, tp_rank);
            register_parameter("q_a_layernorm", q_a_layernorm);
        }
        else {
            q_proj.emplace(hidden_dim, head_num * head_dim / tp_size, data_type, false, weight_type, group_size);
            register_module("q_proj", q_proj, tp_rank);
        }
        kv_a_proj.emplace(hidden_dim, mla.kv_lora_rank + mla.qk_rope_dim, data_type, false, weight_type, group_size);
        kv_b_proj.emplace(mla.kv_lora_rank,
                          head_num * (qk_nope_dim + mla.v_head_dim) / tp_size,
                          data_type,
                          false,
                          weight_type,
                          group_size);

        kv_a_layernorm = Tensor{{kv_b_proj.input_dim}, data_type, kDEVICE};
        register_module("kv_a_proj", kv_a_proj);
        register_module("kv_b_proj", kv_b_proj, tp_rank);
        register_parameter("kv_a_layernorm", kv_a_layernorm);
    }
    output.emplace((head_num * head_dim) / tp_size, hidden_dim, data_type, bias, weight_type, group_size);
    register_module("wo", output, tp_rank);
}

void LlamaAttentionWeight::prepare(bool use_simt)
{
    std::vector weights{
        &qkv,
        &output,
        &q_a_proj,
        &q_a_proj,
        &q_b_proj,
        &kv_a_proj,
        &kv_b_proj,
    };
    for (auto& w : weights) {
        w->prepare(false, use_simt);
    }
}

LlamaFfnWeight::LlamaFfnWeight(int      hidden_dim,
                               int      inter_size,
                               int      tp_size,
                               int      tp_rank,
                               DataType data_type,
                               DataType weight_type,
                               int      group_size,
                               bool     fuse_silu_act)
{
    TM_CHECK(inter_size % tp_size == 0) << inter_size << " " << tp_size;

    inter_size /= tp_size;

    this->inter_size = inter_size;

    gating.emplace(hidden_dim, inter_size, data_type, false, weight_type, group_size);

    intermediate.emplace(hidden_dim, inter_size, data_type, false, weight_type, group_size);

    // fused_gating_intermediate = {hidden_dim, inter_size * 2, data_type, weight_type, group_size};
    is_fused_silu = fuse_silu_act;

    output.emplace(inter_size, hidden_dim, data_type, false, weight_type, group_size);

    register_module("w1", gating, tp_rank);
    register_module("w3", intermediate, tp_rank);
    register_module("w2", output, tp_rank);
}

void interleave(LlamaDenseWeight& c, LlamaDenseWeight& a, LlamaDenseWeight& b, DataType data_type, cudaStream_t st)
{
    FT_CHECK(c.input_dim == a.input_dim);
    FT_CHECK(c.input_dim == b.input_dim);
    FT_CHECK(c.output_dim == a.output_dim * 2);
    FT_CHECK(c.output_dim == b.output_dim * 2);
    FT_CHECK(c.group_size == a.group_size);
    FT_CHECK(c.group_size == b.group_size);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        if (a.weight_type == data_type_v<uint4_t>) {
            Buffer_<uint8_t> tmp_a{a.weight.size(), kDEVICE};
            Buffer_<uint8_t> tmp_b{b.weight.size(), kDEVICE};
            Buffer_<uint8_t> tmp_c{c.weight.size(), kDEVICE};

            extend_to_u8(tmp_a.data(), (const uint4_t*)a.weight.raw_data(), a.output_dim * a.input_dim, st);
            extend_to_u8(tmp_b.data(), (const uint4_t*)b.weight.raw_data(), b.output_dim * b.input_dim, st);

            interleave_output_dims(tmp_c.data(), tmp_a.data(), tmp_b.data(), a.output_dim, a.input_dim, st);

            compact_to_u4((uint4_t*)c.weight.raw_data(), tmp_c.data(), c.output_dim * c.input_dim, st);

            interleave_output_dims(c.scales.data<T>(),
                                   a.scales.data<T>(),
                                   b.scales.data<T>(),
                                   a.output_dim,
                                   a.input_dim / a.group_size,
                                   st);
            interleave_output_dims(c.zeros.data<T>(),  //
                                   a.zeros.data<T>(),
                                   b.zeros.data<T>(),
                                   a.output_dim,
                                   a.input_dim / a.group_size,
                                   st);
        }
        else {
            interleave_output_dims(
                c.weight.data<T>(), a.weight.data<T>(), b.weight.data<T>(), a.output_dim, a.input_dim, st);
        }
        // Check at function level
        sync_check_cuda_error();
    };

    TM_DISPATCH_DTYPES(data_type, invoke, half_t, bfloat16_t);
}

void chunk(LlamaDenseWeight& c, LlamaDenseWeight& a, LlamaDenseWeight& b, DataType data_type, cudaStream_t st)
{
    FT_CHECK(c.input_dim == a.input_dim);
    FT_CHECK(c.input_dim == b.input_dim);
    FT_CHECK(c.output_dim == a.output_dim * 2);
    FT_CHECK(c.output_dim == b.output_dim * 2);
    FT_CHECK(c.group_size == a.group_size);
    FT_CHECK(c.group_size == b.group_size);

    auto _chunks = [&](auto c, auto a, auto b, int height, int width) {
        check_cuda_error(
            cudaMemcpy2DAsync((char*)c + 0x000, width * 2, a, width, width, height, cudaMemcpyDefault, st));
        check_cuda_error(
            cudaMemcpy2DAsync((char*)c + width, width * 2, b, width, width, height, cudaMemcpyDefault, st));
    };

    auto invoke = [&](auto t) {
        using T = decltype(t);
        if (c.weight_type == data_type_v<uint4_t>) {
            _chunks(c.weight.raw_data(), a.weight.raw_data(), b.weight.raw_data(), a.input_dim, 4 * a.output_dim / 8);
            _chunks(c.scales.data<T>(),
                    a.scales.data<T>(),
                    b.scales.data<T>(),
                    a.input_dim / a.group_size,
                    sizeof(T) * a.output_dim);
            _chunks(c.zeros.data<T>(),
                    a.zeros.data<T>(),
                    b.zeros.data<T>(),
                    a.input_dim / a.group_size,
                    sizeof(T) * a.output_dim);
        }
        else {
            _chunks(c.weight.data<T>(), a.weight.data<T>(), b.weight.data<T>(), a.input_dim, sizeof(T) * a.output_dim);
        }
        // Check at function level
        sync_check_cuda_error();
    };

    TM_DISPATCH_DTYPES(data_type, invoke, half_t, bfloat16_t);
}

void LlamaFfnWeight::prepare(bool fused_moe, bool use_simt)
{
    const auto data_type = gating.data_type;

    auto stream = core::Context().stream().handle();

    if (fuse_up_and_gate) {
        auto& fused_up_and_gate = fused_gating_intermediate;

        fused_up_and_gate.emplace(gating.input_dim,  //
                                  gating.output_dim * 2,
                                  gating.data_type,
                                  false,
                                  gating.weight_type,
                                  gating.group_size);

        if (is_fused_silu) {
            interleave(fused_up_and_gate, gating, intermediate, data_type, stream);
        }
        else {
            chunk(fused_up_and_gate, gating, intermediate, data_type, stream);
        }

        fused_gating_intermediate.prepare(fused_moe, use_simt);

        gating       = {};
        intermediate = {};
    }
    else {
        gating.prepare(fused_moe, use_simt);
        intermediate.prepare(fused_moe, use_simt);
    }

    output.prepare(fused_moe, use_simt);
}

MoeFfnWeight::MoeFfnWeight(int             layer_id,
                           const MoeParam& param,
                           int             hidden_dim,
                           DataType        data_type,
                           DataType        weight_type,
                           int             group_size,
                           int             tp_size,
                           int             tp_rank,
                           bool            fuse_silu_act)
{
    if ((int)param.expert_num.size() <= layer_id) {
        return;
    }

    const int expert_num = param.expert_num[layer_id];

    if (expert_num == 0) {
        return;
    }

    gate.emplace(hidden_dim, expert_num, data_type, false, data_type, 1);
    register_module("gate", gate);

    method        = param.method;
    fuse_silu_act = fuse_silu_act && method == MoeParam::kFused;

    experts.reserve(expert_num);
    for (int i = 0; i < expert_num; ++i) {
        experts.emplace_back(new LlamaFfnWeight{
            hidden_dim, param.inter_size, tp_size, tp_rank, data_type, weight_type, group_size, fuse_silu_act});
        register_module("experts", *experts.back(), i);
    }

    if (param.shared_gate) {
        shared_gate.emplace(hidden_dim, 1, data_type, false, data_type, 1);
        register_module("shared_gate", shared_gate);
    }
}

void MoeFfnWeight::prepare(bool use_simt)
{
    const auto fused_moe = method == MoeParam::kFused;

    for (auto& e : experts) {
        e->prepare(fused_moe, use_simt);
    }
    const int  n_expert = experts.size();
    const auto st       = core::Context::stream().handle();

    auto make_block_ptr = [&](const auto& ptrs) {
        return std::shared_ptr<void>{gemm::make_blocked_ptrs(ptrs, st), [](auto p) { cudaFree(p); }};
    };

    auto process = [&](auto getter) {
        std::vector<std::pair<void*, int>> weight_ptrs;
        std::vector<std::pair<void*, int>> quant_ptrs;

        for (auto& e : experts) {
            auto& m = (*e).*getter;
            weight_ptrs.push_back({m.weight.raw_data(), m.k_desc.ld});
            if (m.scales_zeros) {
                quant_ptrs.emplace_back(m.scales_zeros.raw_data(), m.q_desc.ld);
            }
        }

        LlamaDenseWeight& m = block.*getter;

        {  // Copy properties from exemplar, this assumes all experts has the same shape
            LlamaDenseWeight& e = (*experts.at(0)).*getter;
            m.input_dim         = e.input_dim;
            m.output_dim        = e.output_dim;
            m.group_size        = e.group_size;
            m.data_type         = e.data_type;
            m.weight_type       = e.weight_type;
            m.k_desc            = e.k_desc;
            m.q_desc            = e.q_desc;
        }

        // Dummy tensors to hold the blocked ptrs
        m.weight = Tensor{make_block_ptr(weight_ptrs), {n_expert}, m.weight_type, kDEVICE};
        if (!quant_ptrs.empty()) {
            TM_CHECK_EQ(quant_ptrs.size(), n_expert);
            m.scales_zeros = Tensor{make_block_ptr(quant_ptrs), {n_expert}, m.data_type, kDEVICE};
        }

        m.k_desc.num = m.q_desc.num = experts.size();
        m.k_desc.ld = m.q_desc.ld = 0;  // `ld` is meaningless in this case
    };

    process(&LlamaFfnWeight::fused_gating_intermediate);
    process(&LlamaFfnWeight::output);

    auto& e = *experts.at(0);
    // Copy MLP properties
    block.inter_size    = e.inter_size;
    block.is_fused_silu = e.is_fused_silu;
}

}  // namespace turbomind
