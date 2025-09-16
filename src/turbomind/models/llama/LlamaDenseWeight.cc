// Copyright (c) OpenMMLab. All rights reserved.

#include <utility>

#include "src/turbomind/models/llama/LlamaDenseWeight.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/data_type.h"

#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

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
        weight_quant = QuantDesc{gemm::QuantType::kB, group_size};
        if (getSMVersion() == 90) {
            input_type  = kFloat8_e4m3;
            input_quant = QuantDesc{gemm::QuantType::kK, group_size};
        }
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

void LlamaDenseWeight::preprocess()
{
    if (!weight) {
        return;
    }
    if (weight_quant.type == gemm::QuantType::kB && input_quant.type == gemm::QuantType::kNone) {
        // Convert blockwise scales to groupwise scales
        weight_quant.type = gemm::QuantType::kK;
        scales            = BlockscaleToGroupscale(scales, data_type, weight_quant.group_size);
    }
}

static void Convert(LlamaDenseWeight& dense, bool is_grouped, cudaStream_t st)
{
    using namespace gemm;

    auto [conv_w, conv_s] =
        GetConverters(dense.data_type, dense.weight_type, dense.input_type, is_grouped, getSMVersion());

    if (conv_w) {
        const auto order_w = conv_w->order;
        const bool is_A    = get_operand_tag(conv_w->pack) == OPERAND_A;
        const bool is_B    = !is_A;

        const int bits = byte_size(dense.weight_type, 8);

        Tensor_<uint16_t> tmp{{dense.input_dim, dense.output_dim}, kDEVICE};

        if (bits == 4) {  // u4 -> u16
            extend_to_u16(tmp.data(), (const uint4_t*)dense.weight.raw_data(), tmp.size(), st);
            sync_check_cuda_error();
        }
        else if (bits == 8) {  // u8 -> u16
            extend_to_u16(tmp.data(), (const uint8_t*)dense.weight.raw_data(), tmp.size(), st);
            sync_check_cuda_error();
        }
        else if (bits == 16) {
            check_cuda_error(
                cudaMemcpyAsync(tmp.raw_data(), dense.weight.raw_data(), tmp.byte_size(), cudaMemcpyDefault, st));
        }

        if (order_w == kRowMajor) {  // (k,m) -> (m,k)
            Tensor_<uint16_t> trans{{dense.output_dim, dense.input_dim}, kDEVICE};
            invokeTransposeAxis01(trans.data(), tmp.data(), dense.input_dim, dense.output_dim, 1, st);
            tmp = trans;
        }

        MatrixLayout w_desc{
            dense.data_type,
            order_w,
            (int)dense.output_dim,  // M
            (int)dense.input_dim,   // K
            order_w == kRowMajor ? (int)dense.input_dim : (int)dense.output_dim,
        };

        if (is_B) {
            std::swap(w_desc.rows, w_desc.cols);
            w_desc.order = ~w_desc.order;
        }

        MatrixLayout k_desc = w_desc;
        k_desc.type         = dense.weight_type;
        // Converter does not recognize e2m1 / e4m3
        if (bits == 4) {
            k_desc.type = data_type_v<uint4_t>;
        }
        else if (bits == 8) {
            k_desc.type = data_type_v<uint8_t>;
        }
        k_desc.pack = conv_w->pack;

        check_cuda_error(cudaMemsetAsync(dense.weight.raw_data(), 0, dense.weight.byte_size(), st));

        TM_CHECK(conv_w->Convert(tmp.data(), w_desc, dense.weight.raw_data(), k_desc, st) == 0);

        sync_check_cuda_error();

        k_desc.type = dense.weight_type;
        if (is_A) {
            k_desc = transpose(k_desc);
        }
        dense.k_desc = k_desc;
    }

    if (conv_s) {
        const auto order_s = conv_s->order;
        const auto pack_s  = conv_s->pack;
        const bool is_A    = get_operand_tag(conv_s->pack) == OPERAND_U;
        const bool is_B    = !is_A;

        Tensor   tmp_q;
        DataType scale_type;

        if (dense.zeros) {  // AWQ/GPTQ fuse scales and zeros
            tmp_q = {{dense.scales.size(), 2}, kHalf, kDEVICE};
            fuse_scales_and_zeros(
                tmp_q.data<half>(), dense.scales.data<half>(), dense.zeros.data<half>(), dense.scales.size(), st);
            scale_type   = kUint32;  // half2
            dense.zeros  = {};
            dense.scales = empty_like(tmp_q);
        }
        else if (dense.weight_type == kFloat8_e4m3) {  // e4m3
            tmp_q = empty_like(dense.scales);
            Copy(dense.scales, tmp_q);
            scale_type = kUint16;  // bf16
        }
        else {  // mxfp4
            tmp_q = empty_like(dense.scales);
            Copy(dense.scales, tmp_q);
            scale_type = kUint8;  // ue8m0
        }

        if (dense.data_type == kHalf && dense.weight_type == kFloat4_e2m1) {  // mxfp4
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
            s_desc.order = ~s_desc.order;
        }

        MatrixLayout q_desc = s_desc;
        q_desc.pack         = pack_s;

        TM_CHECK(conv_s->Convert(tmp_q.raw_data(), s_desc, dense.scales.raw_data(), q_desc, st) == 0);
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

void LlamaDenseWeight::prepare(bool fused_moe)
{
    if (!weight) {
        return;
    }

    auto stream = core::Context::stream().handle();

    if (weight_type == kFloat8_e4m3 && input_type == kFloat8_e4m3) {
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

void LlamaAttentionWeight::prepare()
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
        w->preprocess();
        w->prepare();
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

    if (gating.input_type == kFloat8_e4m3) {  // SM90 FP8*FP8 GEMM, can't fuse
        this->is_fused_silu = false;
    }

    register_module("w1", gating, tp_rank);
    register_module("w3", intermediate, tp_rank);
    register_module("w2", output, tp_rank);
}

static void Interleave(const Tensor& a, const Tensor& b, Tensor& c, cudaStream_t st)
{
    TM_CHECK(a.layout() == b.layout());
    int M, K;
    if (a.ndim() == 2) {
        std::tie(K, M) = a.shapes(0, 1);
    }
    else {
        M = a.shape(0);
        K = 1;
    }
    auto a_ = a.raw_data();
    auto b_ = b.raw_data();
    auto c_ = c.raw_data();

    const int bits = byte_size(a.dtype(), 8);
    if (bits == 4) {
        Buffer_<uint8_t> ta{a.size(), kDEVICE};
        Buffer_<uint8_t> tb{b.size(), kDEVICE};
        Buffer_<uint8_t> tc{c.size(), kDEVICE};
        extend_to_u8(ta.data(), (uint4_t*)a_, a.size(), st);
        extend_to_u8(tb.data(), (uint4_t*)b_, b.size(), st);
        interleave_output_dims(tc.data(), ta.data(), tb.data(), M, K, st);
        compact_to_u4((uint4_t*)c_, tc.data(), c.size(), st);
    }
    else if (bits == 8) {
        interleave_output_dims((uint8_t*)c_, (uint8_t*)a_, (uint8_t*)b_, M, K, st);
    }
    else if (bits == 16) {
        interleave_output_dims((uint16_t*)c_, (uint16_t*)a_, (uint16_t*)b_, M, K, st);
    }
    else if (bits == 32) {
        interleave_output_dims((uint32_t*)c_, (uint32_t*)a_, (uint32_t*)b_, M, K, st);
    }
    else {
        TM_CHECK(0);
    }
}

void interleave(LlamaDenseWeight& c, LlamaDenseWeight& a, LlamaDenseWeight& b, DataType data_type, cudaStream_t st)
{
    TM_CHECK_EQ(c.input_dim, a.input_dim);
    TM_CHECK_EQ(c.input_dim, b.input_dim);
    TM_CHECK_EQ(c.output_dim, a.output_dim * 2);
    TM_CHECK_EQ(c.output_dim, b.output_dim * 2);
    TM_CHECK_EQ(c.group_size, a.group_size);
    TM_CHECK_EQ(c.group_size, b.group_size);

    Interleave(a.weight, b.weight, c.weight, st);
    sync_check_cuda_error();

    if (a.scales) {
        Interleave(a.scales, b.scales, c.scales, st);
        sync_check_cuda_error();
    }
    if (a.zeros) {
        Interleave(a.zeros, b.zeros, c.zeros, st);
        sync_check_cuda_error();
    }
    if (a.bias) {
        Interleave(a.bias, b.bias, c.bias, st);
        sync_check_cuda_error();
    }
}

static void Chunk(const Tensor& a, const Tensor& b, Tensor& c, cudaStream_t st)
{
    TM_CHECK(a.layout() == b.layout());
    int M, K, spitch, dpitch;
    if (a.ndim() == 2) {
        std::tie(K, M) = a.shapes(0, 1);
        spitch         = byte_size(a.dtype(), a.stride(0));
        dpitch         = byte_size(c.dtype(), c.stride(0));
    }
    else {
        M      = a.shape(0);
        K      = 1;
        spitch = byte_size(a.dtype(), M);
        dpitch = byte_size(c.dtype(), c.shape(0));
    }
    int height = K;
    int width  = byte_size(a.dtype(), M);
    check_cuda_error(cudaMemcpy2DAsync((char*)c.raw_data(),  //
                                       dpitch,
                                       (const char*)a.raw_data(),
                                       spitch,
                                       width,
                                       height,
                                       cudaMemcpyDefault,
                                       st));
    check_cuda_error(cudaMemcpy2DAsync((char*)c.raw_data() + width,  //
                                       dpitch,
                                       (const char*)b.raw_data(),
                                       spitch,
                                       width,
                                       height,
                                       cudaMemcpyDefault,
                                       st));
}

void chunk(LlamaDenseWeight& c, LlamaDenseWeight& a, LlamaDenseWeight& b, DataType data_type, cudaStream_t st)
{
    TM_CHECK_EQ(c.input_dim, a.input_dim);
    TM_CHECK_EQ(c.input_dim, b.input_dim);
    TM_CHECK_EQ(c.output_dim, a.output_dim * 2);
    TM_CHECK_EQ(c.output_dim, b.output_dim * 2);
    TM_CHECK_EQ(c.group_size, a.group_size);
    TM_CHECK_EQ(c.group_size, b.group_size);

    Chunk(a.weight, b.weight, c.weight, st);
    sync_check_cuda_error();

    if (a.scales) {
        Chunk(a.scales, b.scales, c.scales, st);
        sync_check_cuda_error();
    }
    if (a.zeros) {
        Chunk(a.zeros, b.zeros, c.zeros, st);
        sync_check_cuda_error();
    }
    if (a.bias) {
        Chunk(a.bias, b.bias, c.bias, st);
        sync_check_cuda_error();
    }
}

void LlamaFfnWeight::prepare(bool fused_moe)
{
    const auto data_type = gating.data_type;

    auto stream = core::Context().stream().handle();

    gating.preprocess();
    intermediate.preprocess();

    if (fuse_up_and_gate) {
        auto& gate_and_up = fused_gating_intermediate;

        gate_and_up.emplace(gating.input_dim,  //
                            gating.output_dim * 2,
                            gating.data_type,
                            (bool)gating.bias,
                            gating.weight_type,
                            gating.group_size);
        gate_and_up.preprocess();
        register_module("w1w3", gate_and_up, this->tp_rank);

        if (is_fused_silu) {
            interleave(gate_and_up, gating, intermediate, data_type, stream);
            gate_and_up.epilogue = gemm::Epilogue::kGatedSilu;
        }
        else {
            chunk(gate_and_up, gating, intermediate, data_type, stream);
        }

        fused_gating_intermediate.prepare(fused_moe);

        gating       = {};
        intermediate = {};
    }
    else {
        gating.prepare(fused_moe);
        intermediate.prepare(fused_moe);
    }

    output.preprocess();
    output.prepare(fused_moe);
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

    method = param.method;

    const bool is_cublas_gemm = method == MoeParam::kNaive && byte_size(weight_type, 8) == 16;
    if (is_cublas_gemm || mlp_bias) {
        fuse_silu_act = false;
    }

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

void MoeFfnWeight::prepare()
{
    const auto fused_moe = method == MoeParam::kFused;

    gate.prepare();
    shared_gate.prepare();

    for (auto& e : experts) {
        e->prepare(fused_moe);
    }

    const int n = experts.size();
    LinkExperts([&](int i) { return &experts[i]->fused_gating_intermediate; }, n, block.fused_gating_intermediate);
    LinkExperts([&](int i) { return &experts[i]->output; }, n, block.output);

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
        if (e.scales) {
            scales.emplace_back(e.scales.raw_data(), e.q_desc.ld);
        }
        if (e.bias) {
            Copy(e.bias, d.bias.slice(i, 1).squeeze(0));
        }
    }

    auto stream = core::Context::stream().handle();

    if (d.weight_type == kFloat8_e4m3 && d.input_type == kFloat8_e4m3) {
        auto make_blocked_ptr = [&](const auto& ptrs) {
            return std::shared_ptr<void>{gemm::MakeBlockedPtrs(ptrs, stream), [](auto p) { cudaFree(p); }};
        };
        d.weight = Tensor{make_blocked_ptr(weights), {n}, e.weight.dtype(), kDEVICE};
        d.scales = Tensor{make_blocked_ptr(scales), {n}, e.scales.dtype(), kDEVICE};
        // This is needed to be recognized as blocked striding mode
        d.k_desc.offsets = d.q_desc.offsets = (int*)1;
    }
    else {
        auto make_strided_ptr = [&](const auto& ptrs) {
            return std::shared_ptr<void>{gemm::MakeStridedPtrs(ptrs, stream), [](auto p) { cudaFree(p); }};
        };
        d.weight = Tensor{make_strided_ptr(weights), {n}, d.weight_type, kDEVICE};
        if (e.scales) {
            d.scales = Tensor{make_strided_ptr(scales), {n}, e.scales.dtype(), kDEVICE};
        }
        // pre-sm90 grouped GEMM need `ld == 0 to resolve strided_ptr
        d.k_desc.ld = d.q_desc.ld = 0;
    }
}

}  // namespace turbomind
