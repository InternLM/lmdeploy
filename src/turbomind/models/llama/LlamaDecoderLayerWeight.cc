/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.cc

#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/core/buffer.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <filesystem>
namespace turbomind {

static bool is_fuse_silu_act()
{
    static const bool value = [] {
        const auto str = std::getenv("TM_FUSE_SILU_ACT");
        if (str) {
            try {
                auto v = std::stoi(str) != 0;
                TM_LOG_INFO("TM_FUSE_SILU_ACT=%d", (int)v);
                return v;
            }
            catch (...) {
            }
        }
        TM_LOG_INFO("TM_FUSE_SILU_ACT=1");
        return true;
    }();
    return value;
}

LlamaDecoderLayerWeight::LlamaDecoderLayerWeight(DataType           data_type,
                                                 int                layer_id,
                                                 const ModelParam&  model,
                                                 const EngineParam& engine,
                                                 const LoraParam&   lora_param,
                                                 const MoeParam&    moe_param):
    head_num_(model.head_num),
    kv_head_num_(model.kv_head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    inter_size_(model.inter_size.at(layer_id)),
    data_type_{data_type},
    weight_type_(model.weight_type),
    attn_bias_(model.attn_bias),
    attn_tp_size_(engine.attn_tp_size),
    attn_tp_rank_(engine.attn_tp_rank),
    mlp_tp_size_(engine.mlp_tp_size),
    mlp_tp_rank_(engine.mlp_tp_rank)
{
    self_attn_weights = LlamaAttentionWeight{hidden_units_,
                                             size_per_head_,
                                             head_num_,
                                             kv_head_num_,
                                             model.mla,
                                             attn_bias_,
                                             model.qk_norm,
                                             attn_tp_size_,
                                             data_type_,
                                             weight_type_,
                                             model.group_size};

    ffn_weights = LlamaFfnWeight{
        hidden_units_,
        inter_size_,
        mlp_tp_size_,
        data_type_,
        weight_type_,
        model.group_size,
        weight_type_ == TYPE_UINT4 && is_fuse_silu_act(),
    };

    moe_weights = MoeFfnWeight{layer_id,
                               moe_param,
                               hidden_units_,
                               data_type_,
                               weight_type_,
                               model.group_size,
                               mlp_tp_size_,
                               is_fuse_silu_act()};

    if (lora_param.policy == LoraPolicy::kPlora) {
        std::vector<std::string> keys = {
            "attention.w_qkv", "attention.wo", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"};
        std::vector<LlamaDenseWeight*> weights = {&self_attn_weights.qkv,
                                                  &self_attn_weights.output,
                                                  &ffn_weights.gating,
                                                  &ffn_weights.output,
                                                  &ffn_weights.intermediate};
        for (int i = 0; i < keys.size(); i++) {
            const auto& name      = keys[i];
            auto&       weight    = *weights[i];
            int         rank      = lora_param.r;
            float       scale     = lora_param.scale;
            std::string full_name = "layers." + std::to_string(layer_id) + "." + name;

            for (const auto& [re, pr] : lora_param.rank_pattern) {
                if (std::regex_search(full_name, pr.first)) {
                    rank = pr.second;
                    TM_LOG_DEBUG("find rank, pattern=%s, name=%s, value=%d", re.c_str(), full_name.c_str(), rank);
                    break;
                }
            }
            for (const auto& [re, pr] : lora_param.scale_pattern) {
                if (std::regex_search(full_name, pr.first)) {
                    scale = pr.second;
                    TM_LOG_DEBUG("find scale pattern=%s, name=%s, value=%f", re.c_str(), full_name.c_str(), scale);
                    break;
                }
            }
            if (rank) {
                weight.lora.r      = rank;
                weight.lora.scale  = scale;
                weight.lora.policy = lora_param.policy;
            }
        }
    }

    fused_up_and_gate_ = ffn_weights.gating.lora.policy != LoraPolicy::kPlora;
}

void LlamaDecoderLayerWeight::malloc()
{
    self_attn_norm = core::Buffer{hidden_units_, data_type_, MEMORY_GPU};
    ffn_norm       = core::Buffer{hidden_units_, data_type_, MEMORY_GPU};

    self_attn_weights.malloc();

    if (inter_size_) {
        ffn_weights.malloc();
    }

    if (!moe_weights.experts.empty()) {
        moe_weights.malloc();
    }
}

size_t LlamaDecoderLayerWeight::workspace_size() const noexcept
{
    // Space to hold the largest weight in full precision

    auto get_size = [](const auto& w) { return (size_t)w.input_dim * w.output_dim; };

    size_t size = 0;

    size = std::max(size, get_size(self_attn_weights.qkv));
    size = std::max(size, get_size(self_attn_weights.output));
    size = std::max(size, get_size(ffn_weights.gating));
    size = std::max(size, get_size(ffn_weights.fused_gating_intermediate));

    for (const auto& e : moe_weights.experts) {
        size = std::max(size, get_size(e.gating));
        size = std::max(size, get_size(e.fused_gating_intermediate));
    }

    return size * sizeof(uint16_t);
}

template<typename FirstArg, typename... Args>
std::string concat(FirstArg&& first, Args&&... args)
{
    std::stringstream stream;
    stream << first;
    ((stream << "." << args), ...);
    return stream.str();
}

void getWeightTensor(LlamaDenseWeight& dense, bool bias, const std::string& prefix, core::TensorMap& output)
{
    auto get_name = [=](const std::string& name) { return concat(prefix, name); };

    TM_CHECK_EQ(bias, bool(dense.bias));
    if (bias) {
        output.emplace(get_name("bias"), dense.bias);
    }

    const size_t bit_size = core::get_byte_size(dense.weight_type, 8);
    if (bit_size >= 16) {
        output.emplace(get_name("weight"), dense.weight);
    }
    else {
        output.emplace(get_name("qweight"), dense.weight);
        output.emplace(get_name("scales"), dense.scales);
        output.emplace(get_name("zeros"), dense.zeros);
    }
}

void LlamaDecoderLayerWeight::free()
{
    self_attn_norm = {};
    ffn_norm       = {};

    self_attn_weights.free();

    if (inter_size_) {
        ffn_weights.free();
    }

    if (!moe_weights.experts.empty()) {
        moe_weights.free();
    }
}

LlamaDecoderLayerWeight::~LlamaDecoderLayerWeight() = default;

void getMLATensor(LlamaAttentionWeight& w, const std::string& p, core::TensorMap& m, int tp_rank)
{
    if (w.q_proj.output_dim) {
        getWeightTensor(w.q_proj, false, concat(p, "attention.q_proj", tp_rank), m);
    }
    else {
        getWeightTensor(w.q_a_proj, false, concat(p, "attention.q_a_proj"), m);
        getWeightTensor(w.q_b_proj, false, concat(p, "attention.q_b_proj", tp_rank), m);
        m.emplace(concat(p, "attention.q_a_layernorm"), w.q_a_layernorm);
    }
    getWeightTensor(w.kv_a_proj, false, concat(p, "attention.kv_a_proj"), m);
    getWeightTensor(w.kv_b_proj, false, concat(p, "attention.kv_b_proj", tp_rank), m);
    m.emplace(concat(p, "attention.kv_a_layernorm"), w.kv_a_layernorm);
}

core::TensorMap LlamaDecoderLayerWeight::getParams(std::string prefix)
{
    core::TensorMap output;

    output.emplace(concat(prefix, "attention_norm.weight"), self_attn_norm);
    output.emplace(concat(prefix, "ffn_norm.weight"), ffn_norm);

    auto get_attn = [=](std::string_view name) { return concat(prefix, name, attn_tp_rank_); };

    if (self_attn_weights.qkv.output_dim) {
        getWeightTensor(self_attn_weights.qkv, attn_bias_, get_attn("attention.w_qkv"), output);

        if (self_attn_weights.qk_norm) {
            output.emplace(concat(prefix, "attention.q_norm"), self_attn_weights.q_a_layernorm);
            output.emplace(concat(prefix, "attention.k_norm"), self_attn_weights.kv_a_layernorm);
        }
    }
    else {
        getMLATensor(self_attn_weights, prefix, output, attn_tp_rank_);
    }
    getWeightTensor(self_attn_weights.output, attn_bias_, get_attn("attention.wo"), output);

    auto get_mlp = [=](std::string_view name) { return concat(prefix, name, mlp_tp_rank_); };

    if (inter_size_) {
        getWeightTensor(ffn_weights.gating, false, get_mlp("feed_forward.w1"), output);
        getWeightTensor(ffn_weights.intermediate, false, get_mlp("feed_forward.w3"), output);
        getWeightTensor(ffn_weights.output, false, get_mlp("feed_forward.w2"), output);
    }

    if (!moe_weights.experts.empty()) {
        output.emplace(concat(prefix, "moe_ffn.gate.weight"), moe_weights.gate.weight);
        auto& experts = moe_weights.experts;
        for (size_t i = 0; i < experts.size(); ++i) {
            const std::string name = "moe_ffn.experts." + std::to_string(i);
            getWeightTensor(experts[i].gating, false, get_mlp(concat(name, "w1")), output);
            getWeightTensor(experts[i].intermediate, false, get_mlp(concat(name, "w3")), output);
            getWeightTensor(experts[i].output, false, get_mlp(concat(name, "w2")), output);
        }
        if (moe_weights.shared_gate.weight) {
            output.emplace(concat(prefix, "moe_ffn.shared_gate.weight"), moe_weights.shared_gate.weight);
        }
    }

    return output;
}

static void
convert_u4(LlamaDenseWeight& dense, bool is_fused_moe, void* workspace, size_t size, bool use_simt, cudaStream_t st)
{
    FT_CHECK(dense.weight_type == TYPE_UINT4);

    using namespace gemm;

    auto [order_b, pack_b, order_v, pack_v] =
        get_weight_and_scales_layout(gemm::DataType::U4, is_fused_moe, getSMVersion(), use_simt);

    if (order_b == kColMajor) {
        transpose_u4(
            (uint4_t*)workspace, (const uint4_t*)dense.weight.raw_data(), dense.input_dim, dense.output_dim, st);
        cudaMemcpyAsync(
            dense.weight.raw_data(), workspace, dense.input_dim * dense.output_dim / 2, cudaMemcpyDefault, st);
    }

    extend_to_u16(
        (uint16_t*)workspace, (const uint4_t*)dense.weight.raw_data(), dense.input_dim * dense.output_dim, st);
    sync_check_cuda_error();

    MatrixLayout w_desc{
        gemm::DataType::F16,
        order_b,
        (int)dense.input_dim,   // k
        (int)dense.output_dim,  // n
        order_b == kRowMajor ? (int)dense.output_dim : (int)dense.input_dim,
    };

    MatrixLayout k_desc = w_desc;
    k_desc.type         = gemm::DataType::U4;
    k_desc.pack         = pack_b;

    cudaMemsetAsync(dense.weight.raw_data(), 0, dense.input_dim * dense.output_dim / 2, st);

    FT_CHECK(Convert(workspace, w_desc, dense.weight.raw_data(), k_desc, st) == 0);
    sync_check_cuda_error();

    const int scale_count = (dense.input_dim / dense.group_size) * dense.output_dim;

    // std::cout << "fuse_scales_and_zeros\n";
    fuse_scales_and_zeros((half*)workspace, dense.scales.data<half>(), dense.zeros.data<half>(), scale_count, st);
    sync_check_cuda_error();

    dense.scales = {};
    dense.zeros  = {};

    deviceMalloc((half**)&dense.scales_zeros, scale_count * 2, st);

    MatrixLayout s_desc{
        gemm::DataType::U32,
        order_v,
        (int)dense.input_dim / dense.group_size,  // k
        (int)dense.output_dim,                    // n
        (int)dense.output_dim,
    };

    MatrixLayout q_desc = s_desc;
    q_desc.pack         = pack_v;

    FT_CHECK(Convert(workspace, s_desc, dense.scales_zeros.raw_data(), q_desc, st) == 0);
    sync_check_cuda_error();

    dense.k_desc = k_desc;
    dense.q_desc = q_desc;
}

static void
convert_fp(LlamaDenseWeight& dense, bool is_fused_moe, void* workspace, size_t size, bool use_simt, cudaStream_t st)
{
    using namespace gemm;

    if (!is_fused_moe) {
        return;
    }

    /// TODO: unify data types
    auto data_type = dense.weight_type == TYPE_BF16 ? get_data_type_v<nv_bfloat16> : get_data_type_v<half>;

    const auto [order_b, pack_b, order_v, pack_v] =
        get_weight_and_scales_layout(data_type, is_fused_moe, getSMVersion(), use_simt);

    const int input_dim  = dense.input_dim;
    const int output_dim = dense.output_dim;

    TM_CHECK(dense.weight.is_contiguous());

    if (order_b == kColMajor) {
        invokeTransposeAxis01((uint16_t*)workspace, (uint16_t*)dense.weight.raw_data(), input_dim, output_dim, 1, st);
        sync_check_cuda_error();
    }
    else {
        check_cuda_error(
            cudaMemcpyAsync(workspace, dense.weight.raw_data(), dense.weight.byte_size(), cudaMemcpyDefault, st));
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
        FT_CHECK(Convert(workspace, src, dense.weight.raw_data(), dst, st) == 0);
        sync_check_cuda_error();
    }
    else {
        check_cuda_error(
            cudaMemcpyAsync(dense.weight.raw_data(), workspace, dense.weight.byte_size(), cudaMemcpyDefault, st));
    }

    dense.k_desc = dst;
}

static void convert(LlamaDenseWeight& dense,
                    bool              is_fused_moe,
                    DataType          data_type,
                    void*             workspace,
                    size_t            size,
                    bool              use_simt,
                    cudaStream_t      st)
{
    if (dense.weight_type == TYPE_UINT4) {
        TM_CHECK_EQ(data_type, TYPE_FP16);
        convert_u4(dense, is_fused_moe, workspace, size, use_simt, st);
    }
    else {
        convert_fp(dense, is_fused_moe, workspace, size, use_simt, st);
    }
}

void interleave(LlamaDenseWeight& c,
                LlamaDenseWeight& a,
                LlamaDenseWeight& b,
                DataType          data_type,
                void*             workspace,
                size_t            size,
                cudaStream_t      st)
{
    FT_CHECK(c.input_dim == a.input_dim);
    FT_CHECK(c.input_dim == b.input_dim);
    FT_CHECK(c.output_dim == a.output_dim * 2);
    FT_CHECK(c.output_dim == b.output_dim * 2);
    FT_CHECK(c.group_size == a.group_size);
    FT_CHECK(c.group_size == b.group_size);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        if (a.weight_type == TYPE_UINT4) {
            uint8_t* tmp_a = (uint8_t*)workspace;
            uint8_t* tmp_b = tmp_a + a.output_dim * a.input_dim;
            uint8_t* tmp_c = tmp_b + b.output_dim * b.input_dim;

            const auto sentinel = tmp_c + c.output_dim * c.input_dim;
            FT_CHECK(sentinel <= (uint8_t*)workspace + size);

            extend_to_u8(tmp_a, (const uint4_t*)a.weight.raw_data(), a.output_dim * a.input_dim, st);
            extend_to_u8(tmp_b, (const uint4_t*)b.weight.raw_data(), b.output_dim * b.input_dim, st);

            interleave_output_dims(tmp_c, tmp_a, tmp_b, a.output_dim, a.input_dim, st);

            compact_to_u4((uint4_t*)c.weight.raw_data(), tmp_c, c.output_dim * c.input_dim, st);

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

    switch (data_type) {
        case TYPE_BF16:
            return invoke(nv_bfloat16{});
        case TYPE_FP16:
            return invoke(half{});
        default:
            TM_CHECK(0) << "not implemented";
    }
}

void chunk(
    LlamaDenseWeight& c, LlamaDenseWeight& a, LlamaDenseWeight& b, DataType data_type, void*, size_t, cudaStream_t st)
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
        if (c.weight_type == TYPE_UINT4) {
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

    switch (data_type) {
        case TYPE_BF16:
            return invoke(nv_bfloat16{});
        case TYPE_FP16:
            return invoke(half{});
        default:
            TM_CHECK(0) << "not implemented";
    }
}

void LlamaDecoderLayerWeight::prepare(void* workspace, size_t size, const cudaDeviceProp& prop, cudaStream_t st)
{
    const bool is_16xx = is_16xx_series(prop.name);

    convert(self_attn_weights.qkv, false, data_type_, workspace, size, is_16xx, st);
    convert(self_attn_weights.output, false, data_type_, workspace, size, is_16xx, st);

    auto process_ffn = [&](LlamaFfnWeight& ffn, bool is_fused_moe) {
        if (fused_up_and_gate_) {
            auto& fused_up_and_gate = ffn.fused_gating_intermediate;

            fused_up_and_gate.malloc(st);

            if (ffn.is_fused_silu) {
                interleave(fused_up_and_gate, ffn.gating, ffn.intermediate, data_type_, workspace, size, st);
            }
            else {
                chunk(fused_up_and_gate, ffn.gating, ffn.intermediate, data_type_, workspace, size, st);
            }

            convert(ffn.fused_gating_intermediate, is_fused_moe, data_type_, workspace, size, is_16xx, st);

            ffn.gating.free();
            ffn.intermediate.free();
        }
        else {
            convert(ffn.gating, is_fused_moe, data_type_, workspace, size, is_16xx, st);
            convert(ffn.intermediate, is_fused_moe, data_type_, workspace, size, is_16xx, st);
        }

        convert(ffn.output, is_fused_moe, data_type_, workspace, size, is_16xx, st);
    };

    if (inter_size_) {
        // std::cerr << "process FFN\n";
        process_ffn(ffn_weights, false);
    }

    if (!moe_weights.experts.empty()) {
        // std::cerr << "process MoE\n";
        std::vector<std::pair<void*, int>> fused_ptrs;
        std::vector<std::pair<void*, int>> output_ptrs;
        std::vector<std::pair<void*, int>> fused_param_ptrs;
        std::vector<std::pair<void*, int>> output_param_ptrs;

        for (auto& e : moe_weights.experts) {

            process_ffn(e, moe_weights.method == MoeParam::kFused);

            auto& fused  = e.fused_gating_intermediate;
            auto& output = e.output;

            fused_ptrs.push_back({fused.weight.raw_data(), fused.k_desc.ld});
            output_ptrs.push_back({output.weight.raw_data(), output.k_desc.ld});

            if (e.fused_gating_intermediate.scales_zeros) {
                fused_param_ptrs.emplace_back(fused.scales_zeros.raw_data(), fused.q_desc.ld);
                output_param_ptrs.emplace_back(output.scales_zeros.raw_data(), output.q_desc.ld);
            }
        }

        // Note: This assumes all experts has the same shape
        moe_weights.block = moe_weights.experts.at(0);

        auto& fused  = moe_weights.block.fused_gating_intermediate;
        auto& output = moe_weights.block.output;

        const auto weight_type = fused.weight_type;

        // Note: these are dummy tensors to hold the blocked ptrs
        // TODO: free these ptrs
        fused.weight  = core::Tensor{gemm::make_blocked_ptrs(fused_ptrs, st), {1}, weight_type_, MEMORY_GPU};
        output.weight = core::Tensor{gemm::make_blocked_ptrs(output_ptrs, st), {1}, weight_type_, MEMORY_GPU};

        if (!fused_param_ptrs.empty()) {
            fused.scales_zeros =
                core::Tensor{gemm::make_blocked_ptrs(fused_param_ptrs, st), {1}, data_type_, MEMORY_GPU};
            output.scales_zeros =
                core::Tensor{gemm::make_blocked_ptrs(output_param_ptrs, st), {1}, data_type_, MEMORY_GPU};
        }

        fused.k_desc.ld = output.k_desc.ld = 0;
        fused.k_desc.num = output.k_desc.num = moe_weights.experts.size();

        fused.q_desc.ld = output.q_desc.ld = 0;
        fused.q_desc.num = output.q_desc.num = moe_weights.experts.size();
    }
}

}  // namespace turbomind
