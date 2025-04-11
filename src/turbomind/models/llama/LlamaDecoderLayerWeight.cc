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
    self_attn_weights.reset(new LlamaAttentionWeight{hidden_units_,
                                                     size_per_head_,
                                                     head_num_,
                                                     kv_head_num_,
                                                     model.mla,
                                                     attn_bias_,
                                                     model.qk_norm,
                                                     attn_tp_size_,
                                                     attn_tp_rank_,
                                                     data_type_,
                                                     weight_type_,
                                                     model.group_size});
    register_module("attention", *self_attn_weights);

    if (inter_size_) {
        ffn_weights.reset(new LlamaFfnWeight{
            hidden_units_,
            inter_size_,
            mlp_tp_size_,
            mlp_tp_rank_,
            data_type_,
            weight_type_,
            model.group_size,
            weight_type_ == TYPE_UINT4 && is_fuse_silu_act(),
        });
        register_module("feed_forward", *ffn_weights);
    }

    if (layer_id < moe_param.expert_num.size() && moe_param.expert_num[layer_id]) {
        moe_weights.reset(new MoeFfnWeight{layer_id,
                                           moe_param,
                                           hidden_units_,
                                           data_type_,
                                           weight_type_,
                                           model.group_size,
                                           mlp_tp_size_,
                                           mlp_tp_rank_,
                                           is_fuse_silu_act()});
        register_module("moe_ffn", *moe_weights);
    }

    fused_up_and_gate_ = ffn_weights->gating.lora.policy != LoraPolicy::kPlora;

    self_attn_norm = core::Tensor{{hidden_units_}, data_type_, MEMORY_GPU};
    ffn_norm       = core::Tensor{{hidden_units_}, data_type_, MEMORY_GPU};
    register_parameter("attention_norm.weight", self_attn_norm);
    register_parameter("ffn_norm.weight", ffn_norm);
}

LlamaDecoderLayerWeight::~LlamaDecoderLayerWeight() = default;

static void convert_u4(LlamaDenseWeight& dense, bool is_fused_moe, bool use_simt, cudaStream_t st)
{
    FT_CHECK(dense.weight_type == TYPE_UINT4);

    using namespace gemm;

    auto [order_b, pack_b, order_v, pack_v] =
        get_weight_and_scales_layout(gemm::DataType::U4, is_fused_moe, getSMVersion(), use_simt);

    if (order_b == kColMajor) {
        core::Buffer trans{dense.input_dim * dense.output_dim, TYPE_UINT4, MEMORY_GPU};
        transpose_u4(
            (uint4_t*)trans.raw_data(), (const uint4_t*)dense.weight.raw_data(), dense.input_dim, dense.output_dim, st);
        cudaMemcpyAsync(
            dense.weight.raw_data(), trans.raw_data(), dense.input_dim * dense.output_dim / 2, cudaMemcpyDefault, st);
    }

    core::Buffer_<uint16_t> tmp_w{dense.input_dim * dense.output_dim, MEMORY_GPU};
    extend_to_u16(tmp_w.data(), (const uint4_t*)dense.weight.raw_data(), dense.input_dim * dense.output_dim, st);
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

    FT_CHECK(Convert(tmp_w.data(), w_desc, dense.weight.raw_data(), k_desc, st) == 0);
    sync_check_cuda_error();

    const int scale_count = (dense.input_dim / dense.group_size) * dense.output_dim;

    core::Buffer_<half> tmp_q{scale_count * 2, MEMORY_GPU};
    fuse_scales_and_zeros(tmp_q.data(), dense.scales.data<half>(), dense.zeros.data<half>(), scale_count, st);
    sync_check_cuda_error();

    dense.scales = {};
    dense.zeros  = {};

    dense.scales_zeros = core::Tensor_<half>{{scale_count, 2}, MEMORY_GPU};

    MatrixLayout s_desc{
        gemm::DataType::U32,
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
    auto data_type = dense.weight_type == TYPE_BF16 ? get_data_type_v<nv_bfloat16> : get_data_type_v<half>;

    const auto [order_b, pack_b, order_v, pack_v] =
        get_weight_and_scales_layout(data_type, is_fused_moe, getSMVersion(), use_simt);

    const int input_dim  = dense.input_dim;
    const int output_dim = dense.output_dim;

    TM_CHECK(dense.weight.is_contiguous());

    core::Buffer_<uint16_t> tmp{input_dim * output_dim, MEMORY_GPU};

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

static void convert(LlamaDenseWeight& dense, bool is_fused_moe, DataType data_type, bool use_simt, cudaStream_t st)
{
    if (dense.weight_type == TYPE_UINT4) {
        TM_CHECK_EQ(data_type, TYPE_FP16);
        convert_u4(dense, is_fused_moe, use_simt, st);
    }
    else {
        convert_fp(dense, is_fused_moe, use_simt, st);
    }
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
        if (a.weight_type == TYPE_UINT4) {
            core::Buffer_<uint8_t> tmp_a{a.weight.size(), MEMORY_GPU};
            core::Buffer_<uint8_t> tmp_b{b.weight.size(), MEMORY_GPU};
            core::Buffer_<uint8_t> tmp_c{c.weight.size(), MEMORY_GPU};

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

    switch (data_type) {
        case TYPE_BF16:
            return invoke(nv_bfloat16{});
        case TYPE_FP16:
            return invoke(half{});
        default:
            TM_CHECK(0) << "not implemented";
    }
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

void LlamaDecoderLayerWeight::prepare(const cudaDeviceProp& prop, cudaStream_t st)
{
    const bool is_16xx = is_16xx_series(prop.name);

    convert(self_attn_weights->qkv, false, data_type_, is_16xx, st);
    convert(self_attn_weights->output, false, data_type_, is_16xx, st);

    auto process_ffn = [&](LlamaFfnWeight& ffn, bool is_fused_moe) {
        if (fused_up_and_gate_) {
            auto& fused_up_and_gate = ffn.fused_gating_intermediate;

            fused_up_and_gate.emplace(ffn.gating.input_dim,
                                      ffn.gating.output_dim * 2,
                                      data_type_,
                                      false,
                                      weight_type_,
                                      ffn.gating.group_size);

            if (ffn.is_fused_silu) {
                interleave(fused_up_and_gate, ffn.gating, ffn.intermediate, data_type_, st);
            }
            else {
                chunk(fused_up_and_gate, ffn.gating, ffn.intermediate, data_type_, st);
            }

            convert(ffn.fused_gating_intermediate, is_fused_moe, data_type_, is_16xx, st);

            ffn.gating       = {};
            ffn.intermediate = {};
        }
        else {
            convert(ffn.gating, is_fused_moe, data_type_, is_16xx, st);
            convert(ffn.intermediate, is_fused_moe, data_type_, is_16xx, st);
        }

        convert(ffn.output, is_fused_moe, data_type_, is_16xx, st);
    };

    if (inter_size_) {
        // std::cerr << "process FFN\n";
        process_ffn(*ffn_weights, false);
    }

    if (moe_weights) {
        // std::cerr << "process MoE\n";
        std::vector<std::pair<void*, int>> fused_ptrs;
        std::vector<std::pair<void*, int>> output_ptrs;
        std::vector<std::pair<void*, int>> fused_param_ptrs;
        std::vector<std::pair<void*, int>> output_param_ptrs;

        for (auto& e : moe_weights->experts) {

            process_ffn(*e, moe_weights->method == MoeParam::kFused);

            auto& fused  = e->fused_gating_intermediate;
            auto& output = e->output;

            fused_ptrs.push_back({fused.weight.raw_data(), fused.k_desc.ld});
            output_ptrs.push_back({output.weight.raw_data(), output.k_desc.ld});

            if (e->fused_gating_intermediate.scales_zeros) {
                fused_param_ptrs.emplace_back(fused.scales_zeros.raw_data(), fused.q_desc.ld);
                output_param_ptrs.emplace_back(output.scales_zeros.raw_data(), output.q_desc.ld);
            }
        }
#if 0
        // Note: This assumes all experts has the same shape
        auto& b_ = moe_weights->block;
        auto& e_ = *moe_weights->experts.at(0);
       

        auto& fused  = moe_weights->block.fused_gating_intermediate;
        auto& output = moe_weights->block.output;

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
#endif
    }
}

}  // namespace turbomind
