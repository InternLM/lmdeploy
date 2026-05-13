// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/linear_weight.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

LinearWeight::LinearWeight(const core::LinearConfig& cfg):
    input_dim(cfg.input_dim),
    output_dim(cfg.output_dim),
    data_type(cfg.data_type),
    weight_format(cfg.format),
    has_bias_(cfg.has_bias)
{
    std::tie(input_format, output_format) = DeriveActivationFormats(weight_format, data_type, getSMVersion());
}

std::pair<DataFormat, DataFormat> DeriveActivationFormats(const DataFormat& weight_format, DataType data_type, int sm)
{
    DataFormat in_fmt;
    DataFormat out_fmt;
    in_fmt.dtype        = data_type;
    in_fmt.block_sizes  = {1, 1};
    out_fmt.dtype       = data_type;
    out_fmt.block_sizes = {1, 1};

    // Empty weight_format (from LinearBuilder.set_weight path for embeddings /
    // lm_head): treat as trivial. No quantization on I/O.
    if (weight_format.dtype == DataType{}) {
        return {in_fmt, out_fmt};
    }

    if (!weight_format.is_quantized()) {
        return {in_fmt, out_fmt};
    }

    if (weight_format.dtype == kFloat8_e4m3) {
        if (sm == 90) {
            int gs              = weight_format.block_sizes[0];  // K-axis, tensor-shape order
            in_fmt.dtype        = kFloat8_e4m3;
            in_fmt.block_sizes  = {gs, 1};
            in_fmt.scales.dtype = kFloat;
        }
        return {in_fmt, out_fmt};
    }

    // FP4 / U4 / U8: input stays in model activation dtype — the GEMM
    // upcasts / dequants on the fly. output_format is also activation dtype.
    return {in_fmt, out_fmt};
}

gemm::QuantDesc MakeQuantDesc(const DataFormat& fmt)
{
    if (!fmt.is_quantized()) {
        return {gemm::QuantType::kNone, 0};
    }
    int gs = (fmt.block_sizes.size() > 0) ? fmt.block_sizes[0] : 1;

    if (fmt.dtype == kFloat8_e4m3) {
        // Weight format has bidirectional blocking {128, 128} → B-type.
        // Activation format has K-axis-only blocking {gs, 1} → K-type.
        if (fmt.block_sizes.size() > 1 && fmt.block_sizes[1] > 1) {
            return {gemm::QuantType::kB, gs};
        }
        return {gemm::QuantType::kK, gs};
    }
    // FP4 / U4 / U8: K-grouped quantization
    return {gemm::QuantType::kK, gs};
}

void LinearWeight::copy_metadata_to(LinearWeight& dst) const
{
    dst.input_dim     = input_dim;
    dst.output_dim    = output_dim;
    dst.data_type     = data_type;
    dst.weight_format = weight_format;
    dst.input_format  = input_format;
    dst.output_format = output_format;
    dst.epilogue      = epilogue;
    dst.has_bias_     = has_bias_;
    dst.is_grouped_   = is_grouped_;
    dst.k_desc        = k_desc;
    dst.q_desc        = q_desc;
}

// ======================================================================
// prepare (weight format conversion)
// ======================================================================

void LinearWeight::prepare()
{
    if (!weight) {
        return;
    }

    // Set up GEMM descriptor (was previously in do_allocate)
    k_desc.type  = weight.dtype();
    k_desc.order = gemm::kRowMajor;
    k_desc.rows  = input_dim;
    k_desc.cols  = output_dim;
    k_desc.ld    = output_dim;

    // No format conversion needed if weight_spec was never set (trivial weights
    // loaded via commit_tensor, e.g. tok_embeddings, output head).
    if (weight_format.dtype == DataType{}) {
        EnsureFloatDtype(weight, data_type);
        if (weight.dtype() == data_type) {
            k_desc.type = data_type;
        }
        return;
    }

    auto stream = core::Context::stream().handle();

    if (weight_format.dtype == kFloat8_e4m3 && input_dtype() == kFloat8_e4m3) {
        // FP8 native path: transpose weight and scales for native kernels.
        auto process = [&](Tensor& x, MatrixLayout& d, auto dtype) {
            using T = decltype(dtype);
            Tensor trans{{x.shape(1), x.shape(0)}, x.dtype(), kDEVICE};
            invokeTransposeAxis01((T*)trans.raw_data(), (T*)x.raw_data(), x.shape(0), x.shape(1), 1, stream);
            x = std::move(trans);
            d = MatrixLayout{x.dtype(), gemm::kColMajor, (int)x.shape(1), (int)x.shape(0), (int)x.stride(0)};
        };

        TM_CHECK_EQ(weight.dtype(), kFloat8_e4m3);
        process(weight, k_desc, uint8_t{});

        // FP8 native path requires f32 scales; cast if loaded as bf16/fp16.
        EnsureFloatDtype(scales, kFloat);

        TM_CHECK_EQ(scales.dtype(), kFloat);
        process(scales, q_desc, float{});
    }
    else if (weight_format.dtype == kFloat8_e4m3) {
        // FP8 non-native path (non-SM90)
    }
    else {
        // General quantization format conversion path.
        using namespace gemm;

        auto [conv_w, conv_s] =
            GetConverters(data_type, weight_format.dtype, input_dtype(), is_grouped_, getSMVersion());

        if (conv_w) {
            const auto order_w = conv_w->order;
            const bool is_A    = get_operand_tag(conv_w->pack) == OPERAND_A;
            const bool is_B    = !is_A;

            const int bits = byte_size(weight_format.dtype, 8);

            Tensor_<uint16_t> tmp{{input_dim, output_dim}, kDEVICE};

            if (bits == 4) {
                extend_to_u16(tmp.data(), (const uint4_t*)weight.raw_data(), tmp.size(), stream);
                sync_check_cuda_error();
            }
            else if (bits == 8) {
                extend_to_u16(tmp.data(), (const uint8_t*)weight.raw_data(), tmp.size(), stream);
                sync_check_cuda_error();
            }
            else if (bits == 16) {
                check_cuda_error(
                    cudaMemcpyAsync(tmp.raw_data(), weight.raw_data(), weight.byte_size(), cudaMemcpyDefault, stream));
            }

            if (order_w == kRowMajor) {
                Tensor_<uint16_t> trans{{output_dim, input_dim}, kDEVICE};
                invokeTransposeAxis01(trans.data(), tmp.data(), input_dim, output_dim, 1, stream);
                tmp = trans;
            }

            MatrixLayout w_desc{
                data_type,
                order_w,
                (int)output_dim,
                (int)input_dim,
                order_w == kRowMajor ? (int)input_dim : (int)output_dim,
            };

            if (is_B) {
                std::swap(w_desc.rows, w_desc.cols);
                w_desc.order = ~w_desc.order;
            }

            MatrixLayout kd = w_desc;
            kd.type         = weight_format.dtype;
            if (bits == 4) {
                kd.type = data_type_v<uint4_t>;
            }
            else if (bits == 8) {
                kd.type = data_type_v<uint8_t>;
            }
            kd.pack = conv_w->pack;

            check_cuda_error(cudaMemsetAsync(weight.raw_data(), 0, weight.byte_size(), stream));
            TM_CHECK(conv_w->Convert(tmp.data(), w_desc, weight.raw_data(), kd, stream) == 0);
            sync_check_cuda_error();

            kd.type = weight_format.dtype;
            if (is_A) {
                kd = transpose(kd);
            }
            k_desc = kd;
        }

        if (conv_s) {
            const auto order_s = conv_s->order;
            const auto pack_s  = conv_s->pack;
            const bool is_A    = get_operand_tag(conv_s->pack) == OPERAND_U;

            Tensor   tmp_q;
            DataType scale_type;

            if (zeros) {
                tmp_q = {{scales.size(), 2}, kHalf, kDEVICE};
                fuse_scales_and_zeros(
                    tmp_q.data<half>(), scales.data<half>(), zeros.data<half>(), scales.size(), stream);
                scale_type = kUint32;
                zeros      = {};
                scales     = empty_like(tmp_q);
            }
            else if (weight_format.dtype == kFloat8_e4m3) {
                tmp_q = empty_like(scales);
                Copy(scales, tmp_q);
                scale_type = kUint16;
            }
            else {
                tmp_q = empty_like(scales);
                Copy(scales, tmp_q);
                scale_type = kUint8;
            }

            if (data_type == kHalf && weight_format.dtype == kFloat4_e2m1) {
                AdjustUe8m0ScaleForHalf(tmp_q.data<uint8_t>(), tmp_q.size(), stream);
                sync_check_cuda_error();
            }

            int          gs = weight_format.block_sizes[0];  // K-axis, tensor-shape order
            MatrixLayout s_desc{
                scale_type,
                order_s,
                (int)output_dim,
                (int)input_dim / gs,
                (int)output_dim,
            };

            if (!is_A) {
                std::swap(s_desc.rows, s_desc.cols);
                s_desc.order = ~s_desc.order;
            }

            MatrixLayout qd = s_desc;
            qd.pack         = pack_s;

            TM_CHECK(conv_s->Convert(tmp_q.raw_data(), s_desc, scales.raw_data(), qd, stream) == 0);
            sync_check_cuda_error();

            if (is_A) {
                qd = transpose(qd);
            }
            q_desc = qd;
        }
    }
}

TM_MODULE_REGISTER(LinearWeight, core::LinearConfig);

TM_MODULE_METHODS(LinearWeight, LINEAR_WEIGHT_CHILDREN, LINEAR_WEIGHT_PARAMS)

}  // namespace turbomind
