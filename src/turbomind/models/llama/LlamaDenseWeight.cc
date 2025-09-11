// Copyright (c) OpenMMLab. All rights reserved.

#include <utility>

#include "src/turbomind/models/llama/LlamaDenseWeight.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

namespace gemm {

std::tuple<Order, Pack, Order, Pack>
get_weight_and_scales_layout(DataType dtype, bool is_fused_moe, int sm, bool force_simt)
{
    // Trivial case: floating point & dense GEMM
    if (!is_fused_moe && (dtype == kHalf || dtype == kBfloat16 || dtype == kFloat)) {
        return {kColMajor, OPERAND_A, {}, {}};
    }

    if (dtype == kFloat4_e2m1) {
        if (sm >= 80) {
            return {kColMajor, HMMA_16816 | OPERAND_A | 1, kColMajor, HMMA_16816 | OPERAND_U | 1};
        }
        else if (sm >= 75) {
            return {kColMajor, HMMA_16816 | OPERAND_A | 1, kColMajor, HMMA_16816 | OPERAND_U | 1};
        }
        else if (sm >= 70) {
            return {kColMajor, HMMA_884 | OPERAND_B | 1, kRowMajor, HMMA_884 | OPERAND_V | 1};
        }
    }

    if (is_fused_moe) {
        if (dtype == kBfloat16 && sm >= 80) {
            return {kColMajor, HMMA_16816 | OPERAND_B | 1, {}, {}};
        }

        if (dtype == kFloat16) {
            if (sm >= 80) {
                return {kColMajor, HMMA_16816 | OPERAND_B | 1, {}, {}};
            }
            else if (sm == 75) {
                return {kColMajor, HMMA_16816 | OPERAND_B | 1, {}, {}};
            }
            else if (sm == 70) {
                return {kColMajor, HMMA_884 | OPERAND_B | 1, {}, {}};
            }
        }
        else if (dtype == kUint4) {
            if (sm >= 80) {
                return {kColMajor, HMMA_16816 | OPERAND_B | 2, kRowMajor, HMMA_16816 | OPERAND_V | 1};
            }
            else if (sm == 75) {
                return {kColMajor, HMMA_16816 | OPERAND_B | 2, kRowMajor, HMMA_16816 | OPERAND_V | 1};
            }
            else if (sm == 70) {
                return {kColMajor, HMMA_884 | OPERAND_B | 1, kRowMajor, HMMA_884 | OPERAND_V | 1};
            }
        }
    }
    else {
        if (dtype == kUint4) {
            if (force_simt) {
                return {kColMajor, HMMA_SIMT | OPERAND_B | 1, kRowMajor, HMMA_SIMT | OPERAND_V | 1};
            }
            if (sm >= 80) {
                return {kRowMajor, HMMA_16816 | OPERAND_B | 2, kRowMajor, HMMA_16816 | OPERAND_V | 1};
            }
            else if (sm == 75) {
                return {kRowMajor, HMMA_16816 | OPERAND_B | 2, kRowMajor, HMMA_16816 | OPERAND_V | 1};
            }
            else if (sm == 70) {
                return {kColMajor, HMMA_884 | OPERAND_B | 1, kRowMajor, HMMA_884 | OPERAND_V | 1};
            }
        }
    }

    std::cerr << "not implemented: dtype=" << to_string(dtype) << ", is_fused_moe=" << is_fused_moe << ", sm=" << sm
              << std::endl;
    std::abort();

    return {};
}

}  // namespace gemm

void LlamaDenseWeight::emplace(
    int input_dim, int output_dim, DataType data_type, bool bias, DataType weight_type, int group_size)
{
    this->data_type   = data_type;
    this->input_type  = data_type;
    this->weight_type = weight_type;
    this->input_dim   = input_dim;
    this->output_dim  = output_dim;
    this->group_size  = group_size;

    const bool is_qweight = weight_type == kUint4 || weight_type == kUint8;

    weight = Tensor({input_dim, output_dim}, weight_type, kDEVICE);
    register_parameter(is_qweight ? "qweight" : "weight", weight);

    if (bias) {
        this->bias = Tensor{{output_dim}, data_type, kDEVICE};
        register_parameter("bias", this->bias);
    }

    if (weight_type == kFloat8_e4m3) {
        TM_CHECK_EQ(group_size, 128);
        scales       = Tensor{{cdiv(input_dim, group_size), cdiv(output_dim, group_size)}, kFloat, kDEVICE};
        input_type   = kFloat8_e4m3;
        weight_quant = QuantDesc{gemm::QuantType::kB, group_size};
        input_quant  = QuantDesc{gemm::QuantType::kK, group_size};
        register_parameter("scales", scales);
    }
    else if (weight_type == kFloat4_e2m1) {
        scales       = Tensor{{cdiv(input_dim, group_size), output_dim}, kUint8, kDEVICE};
        input_type   = data_type;
        weight_quant = QuantDesc{gemm::QuantType::kK, group_size};
        register_parameter("scales", scales);
    }
    else if (is_qweight) {
        TM_CHECK(input_dim % group_size == 0) << input_dim << " " << group_size;
        scales       = Tensor{{input_dim / group_size, output_dim}, data_type, kDEVICE};
        zeros        = Tensor{{input_dim / group_size, output_dim}, data_type, kDEVICE};
        weight_quant = QuantDesc{gemm::QuantType::kK, group_size};
        register_parameter("scales", scales);
        register_parameter("zeros", zeros);
    }

    k_desc = {};
    q_desc = {};

    // default case: floating point, N-major
    k_desc.type  = weight.dtype();
    k_desc.order = gemm::kRowMajor;
    k_desc.rows  = input_dim;
    k_desc.cols  = output_dim;
    k_desc.ld    = output_dim;
}

static void Convert(LlamaDenseWeight& dense, bool is_grouped, cudaStream_t st)
{
    using namespace gemm;

    auto [order_w, pack_w, order_s, pack_s] =
        get_weight_and_scales_layout(dense.weight_type, is_grouped, getSMVersion(), false);

    const bool is_A = get_operand_tag(pack_w) == OPERAND_A;
    const bool is_B = !is_A;

    if (get_pack_num(pack_w)) {

        const bool is_K_major = (is_A && order_w == kRowMajor) || (is_B && order_w == kColMajor);

        const int bits = byte_size(dense.weight_type, 8);

        Tensor_<uint16_t> tmp{{dense.input_dim, dense.output_dim}, kDEVICE};

        if (bits == 4) {
            extend_to_u16(tmp.data(), (const uint4_t*)dense.weight.raw_data(), tmp.size(), st);
            sync_check_cuda_error();
        }
        else if (bits == 16) {
            check_cuda_error(
                cudaMemcpyAsync(tmp.raw_data(), dense.weight.raw_data(), tmp.byte_size(), cudaMemcpyDefault, st));
        }

        if (is_K_major) {
            Tensor_<uint16_t> trans{{dense.output_dim, dense.input_dim}, kDEVICE};
            invokeTransposeAxis01(trans.data(), tmp.data(), dense.input_dim, dense.output_dim, 1, st);
            tmp = trans;
        }

        MatrixLayout w_desc{
            dense.data_type,
            order_w,
            (int)dense.output_dim,  // M
            (int)dense.input_dim,   // K
            is_K_major ? (int)dense.input_dim : (int)dense.output_dim,
        };

        if (is_B) {
            std::swap(w_desc.rows, w_desc.cols);
        }

        MatrixLayout k_desc = w_desc;
        // Converter does not recognize e2m1
        k_desc.type = bits == 4 ? data_type_v<uint4_t> : dense.weight_type;
        k_desc.pack = pack_w;

        check_cuda_error(cudaMemsetAsync(dense.weight.raw_data(), 0, dense.weight.byte_size(), st));

        FT_CHECK(Convert(tmp.data(), w_desc, dense.weight.raw_data(), k_desc, st) == 0);
        sync_check_cuda_error();

        k_desc.type = dense.weight_type;
        if (is_A) {
            k_desc = transpose(k_desc);
        }
        dense.k_desc = k_desc;
    }

    if (get_pack_num(pack_s)) {
        Tensor   tmp_q;
        DataType scale_type;

        if (dense.zeros) {  // AWQ/GPTQ fuse scales and zeros
            tmp_q = {{dense.scales.size(), 2}, kHalf, kDEVICE};
            fuse_scales_and_zeros(
                tmp_q.data<half>(), dense.scales.data<half>(), dense.zeros.data<half>(), dense.scales.size(), st);
            scale_type = kUint32;  // half2
        }
        else {
            tmp_q = empty_like(dense.scales);
            Copy(dense.scales, tmp_q);
            scale_type = kUint8;  // ue8m0
        }

        if (dense.data_type == kHalf && dense.weight_type == kFloat4_e2m1) {  // MXFP4
            AdjustUe8m0ScaleForHalf(tmp_q.data<uint8_t>(), tmp_q.size(), st);
            sync_check_cuda_error();
        }

        MatrixLayout s_desc{
            scale_type,
            order_s,
            (int)dense.output_dim,                    // M
            (int)dense.input_dim / dense.group_size,  // K
            (int)dense.output_dim,                    // always MN-major
        };

        if (is_B) {
            std::swap(s_desc.rows, s_desc.cols);
        }

        MatrixLayout q_desc = s_desc;
        q_desc.pack         = pack_s;

        FT_CHECK(Convert(tmp_q.raw_data(), s_desc, dense.scales.raw_data(), q_desc, st) == 0);
        sync_check_cuda_error();

        // weight is placed at B in `Linear`
        if (is_A) {
            q_desc = transpose(q_desc);
        }
        dense.q_desc = q_desc;
    }
}

static void ConvertBlockscaleFP8Native(LlamaDenseWeight& dense, cudaStream_t stream)
{
    using namespace gemm;

    TM_CHECK_GE(getSMVersion(), 90);
    TM_CHECK_EQ(dense.data_type, data_type_v<bfloat16_t>);

    auto process = [&](Tensor& x, MatrixLayout& d, auto dtype) {
        using T = decltype(dtype);
        Tensor trans{{x.shape(1), x.shape(0)}, x.dtype(), kDEVICE};
        invokeTransposeAxis01((T*)trans.raw_data(), (T*)x.raw_data(), x.shape(0), x.shape(1), 1, stream);
        x = std::move(trans);
        d = MatrixLayout{x.dtype(),  //
                         kColMajor,
                         (int)x.shape(1),
                         (int)x.shape(0),
                         (int)x.stride(0)};
    };

    TM_CHECK_EQ(dense.weight.dtype(), kFloat8_e4m3);
    process(dense.weight, dense.k_desc, uint8_t{});

    TM_CHECK_EQ(dense.scales.dtype(), kFloat);
    process(dense.scales, dense.q_desc, float{});
}

void LlamaDenseWeight::prepare(bool fused_moe, bool use_simt)
{
    if (!weight) {
        return;
    }

    auto stream = core::Context::stream().handle();

    if (weight_type == data_type_v<fp8_e4m3_t> && weight_quant.type == gemm::QuantType::kB) {
        ConvertBlockscaleFP8Native(*this, stream);
    }
    else {
        Convert(*this, fused_moe, stream);
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
                                           int      group_size,
                                           int      window_size,
                                           bool     sink)
{
    this->window_size = window_size;

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

    if (sink) {
        sinks = Tensor{{head_num / tp_size}, data_type, kDEVICE};
        register_parameter(std::to_string(tp_rank) + ".sinks", sinks);
    }
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

LlamaFfnWeight::LlamaFfnWeight(int            hidden_dim,
                               int            inter_size,
                               bool           bias,
                               int            tp_size,
                               int            tp_rank,
                               DataType       data_type,
                               DataType       weight_type,
                               int            group_size,
                               ActivationType act_type,
                               bool           fuse_silu_act)
{
    TM_CHECK(inter_size % tp_size == 0) << inter_size << " " << tp_size;

    inter_size /= tp_size;

    this->inter_size    = inter_size;
    this->tp_rank       = tp_rank;
    this->act_type      = act_type;
    this->is_fused_silu = fuse_silu_act && this->act_type == ActivationType::kSilu;

    gating.emplace(hidden_dim, inter_size, data_type, bias, weight_type, group_size);

    intermediate.emplace(hidden_dim, inter_size, data_type, bias, weight_type, group_size);

    output.emplace(inter_size, hidden_dim, data_type, bias, weight_type, group_size);

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

    // TODO: remove unused branches
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

    if (c.weight_type == kFloat8_e4m3) {
        _chunks(c.weight.data<fp8_e4m3_t>(),
                a.weight.data<fp8_e4m3_t>(),
                b.weight.data<fp8_e4m3_t>(),
                a.input_dim,
                a.output_dim);
        _chunks(c.scales.data<float>(),
                a.scales.data<float>(),
                b.scales.data<float>(),
                cdiv(a.input_dim, a.group_size),
                sizeof(float) * cdiv(a.output_dim, a.group_size));
    }
    else if (c.weight_type == kFloat4_e2m1) {
        _chunks(c.weight.raw_data(), a.weight.raw_data(), b.weight.raw_data(), a.input_dim, 4 * a.output_dim / 8);
        _chunks(c.scales.data<uint8_t>(),
                a.scales.data<uint8_t>(),
                b.scales.data<uint8_t>(),
                a.input_dim / a.group_size,
                sizeof(uint8_t) * a.output_dim);
    }
    else {
        TM_DISPATCH_DTYPES(data_type, invoke, half_t, bfloat16_t);
    }

    if (a.bias) {
        TM_CHECK(b.bias && c.bias);
        TM_CHECK_EQ(byte_size(a.bias.dtype()), sizeof(uint16_t));
        _chunks((uint16_t*)c.bias.raw_data(),
                (const uint16_t*)a.bias.raw_data(),
                (const uint16_t*)b.bias.raw_data(),
                1,
                sizeof(uint16_t) * a.output_dim);
    }
}

void LlamaFfnWeight::prepare(bool fused_moe, bool use_simt)
{
    const auto data_type = gating.data_type;

    auto stream = core::Context().stream().handle();

    // if (fuse_up_and_gate && gating.weight_type != DataType::kFloat8_e4m3) {
    if (fuse_up_and_gate) {
        auto& fused_up_and_gate = fused_gating_intermediate;

        fused_up_and_gate.emplace(gating.input_dim,  //
                                  gating.output_dim * 2,
                                  gating.data_type,
                                  (bool)gating.bias,
                                  gating.weight_type,
                                  gating.group_size);
        register_module("w1w3", fused_up_and_gate, this->tp_rank);

        if (is_fused_silu) {
            TM_CHECK(!gating.bias);
            interleave(fused_up_and_gate, gating, intermediate, data_type, stream);
            fused_up_and_gate.epilogue = gemm::Epilogue::kGatedSilu;
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
                           bool            mlp_bias,
                           DataType        data_type,
                           DataType        weight_type,
                           int             group_size,
                           int             tp_size,
                           int             tp_rank,
                           ActivationType  act_type,
                           bool            fuse_silu_act)
{
    if ((int)param.expert_num.size() <= layer_id) {
        return;
    }

    const int expert_num = param.expert_num[layer_id];

    if (expert_num == 0) {
        return;
    }

    gate.emplace(hidden_dim, expert_num, data_type, param.router_bias, data_type, 1);
    register_module("gate", gate);

    method        = param.method;
    fuse_silu_act = fuse_silu_act && method == MoeParam::kFused && weight_type != kFloat8_e4m3 && !mlp_bias;

    experts.reserve(expert_num);
    for (int i = 0; i < expert_num; ++i) {
        experts.emplace_back(new LlamaFfnWeight{hidden_dim,
                                                param.inter_size,
                                                mlp_bias,
                                                tp_size,
                                                tp_rank,
                                                data_type,
                                                weight_type,
                                                group_size,
                                                act_type,
                                                fuse_silu_act});
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

    gate.prepare(false, use_simt);
    shared_gate.prepare(false, use_simt);

    for (auto& e : experts) {
        e->prepare(fused_moe, use_simt);
    }

#if 0
    const int  n_expert = experts.size();
    const auto st       = core::Context::stream().handle();

    auto make_strided_ptr = [&](const auto& ptrs) {
        return std::shared_ptr<void>{gemm::make_strided_ptrs(ptrs, st), [](auto p) { cudaFree(p); }};
    };

    auto make_blocked_ptr = [&](const auto& ptrs) {
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
            else if (m.scales) {
                quant_ptrs.emplace_back(m.scales.raw_data(), m.q_desc.ld);
            }
        }

        LlamaDenseWeight& m = block.*getter;

        {  // Copy properties from exemplar, this assumes all experts has the same shape
            LlamaDenseWeight& e = (*experts.at(0)).*getter;
            m.input_dim         = e.input_dim;
            m.output_dim        = e.output_dim;
            m.group_size        = e.group_size;
            m.data_type         = e.data_type;
            m.input_type        = e.input_type;
            m.weight_type       = e.weight_type;
            m.input_quant       = e.input_quant;
            m.weight_quant      = e.weight_quant;
            m.k_desc            = e.k_desc;
            m.q_desc            = e.q_desc;
            m.epilogue          = e.epilogue;
            if (e.bias) {
                m.bias = Tensor{{(int)experts.size(), e.output_dim}, e.bias.dtype(), kDEVICE};
            }
        }

        // Dummy tensors to hold the blocked ptrs
        if (m.weight_type == kFloat8_e4m3) {
            TM_CHECK_EQ(quant_ptrs.size(), n_expert);
            m.weight = Tensor{make_blocked_ptr(weight_ptrs), {n_expert}, m.weight_type, kDEVICE};
            m.scales = Tensor{make_blocked_ptr(quant_ptrs), {n_expert}, kFloat, kDEVICE};
            // This is needed to be recognized as blocked striding mode
            m.k_desc.offsets = m.q_desc.offsets = (int*)1;
        }
        else {
            m.weight = Tensor{make_strided_ptr(weight_ptrs), {n_expert}, m.weight_type, kDEVICE};
            // pre-90 grouped gemm need `ld == 0` to resolve strided_ptr
            m.k_desc.ld = 0;
            if (!quant_ptrs.empty()) {
                TM_CHECK_EQ(quant_ptrs.size(), n_expert);
                m.scales_zeros = Tensor{make_strided_ptr(quant_ptrs), {n_expert}, m.data_type, kDEVICE};
                m.q_desc.ld    = 0;
            }
        }

        m.k_desc.num = m.q_desc.num = experts.size();

        if (m.bias) {
            for (int i = 0; i < (int)experts.size(); ++i) {
                auto& e = (*experts[i]).*getter;
                Copy(e.bias, m.bias.slice(i, 1).squeeze(0));
            }
        }
    };

    process(&LlamaFfnWeight::fused_gating_intermediate);
    process(&LlamaFfnWeight::output);
#else

    const int n = experts.size();
    LinkExperts([&](int i) { return &experts[i]->fused_gating_intermediate; }, n, block.fused_gating_intermediate);
    LinkExperts([&](int i) { return &experts[i]->output; }, n, block.output);

#endif

    auto& e = *experts.at(0);
    // Copy MLP properties
    block.inter_size    = e.inter_size;
    block.is_fused_silu = e.is_fused_silu;
    block.act_type      = e.act_type;
}

void LinkExperts(std::function<LlamaDenseWeight*(int)> experts, int n, LlamaDenseWeight& d)
{
    const auto& e = *experts(0);

    d.input_dim    = e.input_dim;
    d.output_dim   = e.output_dim;
    d.group_size   = e.group_size;
    d.data_type    = e.data_type;
    d.input_type   = e.input_type;
    d.weight_type  = e.weight_type;
    d.input_quant  = e.input_quant;
    d.weight_quant = e.weight_quant;
    d.k_desc       = e.k_desc;
    d.q_desc       = e.q_desc;
    d.epilogue     = e.epilogue;

    d.k_desc.num = d.q_desc.num = n;

    if (e.bias) {
        d.bias = Tensor{{n, e.output_dim}, e.bias.dtype(), kDEVICE};
    }

    std::vector<std::pair<void*, int>> weights;
    std::vector<std::pair<void*, int>> scales;

    for (int i = 0; i < n; ++i) {
        auto& e = *experts(i);
        weights.emplace_back(e.weight.raw_data(), e.k_desc.ld);
        if (e.scales_zeros) {
            scales.emplace_back(e.scales_zeros.raw_data(), e.q_desc.ld);
        }
        else if (e.scales) {
            scales.emplace_back(e.scales.raw_data(), e.q_desc.ld);
        }
        if (e.bias) {
            Copy(e.bias, d.bias.slice(i, 1).squeeze(0));
        }
    }

    auto stream = core::Context::stream().handle();

    if (d.weight_type == kFloat8_e4m3) {
        auto make_blocked_ptr = [&](const auto& ptrs) {
            return std::shared_ptr<void>{gemm::make_blocked_ptrs(ptrs, stream), [](auto p) { cudaFree(p); }};
        };
        d.weight = Tensor{make_blocked_ptr(weights), {n}, e.weight.dtype(), kDEVICE};
        d.scales = Tensor{make_blocked_ptr(scales), {n}, e.scales.dtype(), kDEVICE};
        // This is needed to be recognized as blocked striding mode
        d.k_desc.offsets = d.q_desc.offsets = (int*)1;
    }
    else {
        auto make_strided_ptr = [&](const auto& ptrs) {
            return std::shared_ptr<void>{gemm::make_strided_ptrs(ptrs, stream), [](auto p) { cudaFree(p); }};
        };
        d.weight = Tensor{make_strided_ptr(weights), {n}, d.weight_type, kDEVICE};
        if (e.scales_zeros) {  // u4
            d.scales_zeros = Tensor{make_strided_ptr(scales), {n}, e.scales_zeros.dtype(), kDEVICE};
        }
        else if (e.scales) {  // mxfp4
            d.scales = Tensor{make_strided_ptr(scales), {n}, e.scales.dtype(), kDEVICE};
        }
        // pre-sm90 grouped GEMM need `ld == 0 to resolve strided_ptr
        d.k_desc.ld = d.q_desc.ld = 0;
    }
}

}  // namespace turbomind
