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
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/llama/LlamaDenseWeight.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"
#include <cstdlib>
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

template<typename T>
LlamaDecoderLayerWeight<T>::LlamaDecoderLayerWeight(int        layer_idx,
                                                    size_t     head_num,
                                                    size_t     kv_head_num,
                                                    size_t     size_per_head,
                                                    size_t     hidden_units,
                                                    size_t     inter_size,
                                                    WeightType weight_type,
                                                    int        group_size,
                                                    LoraParam  lora_param,
                                                    bool       attn_bias,
                                                    MoeParam   moe_param,
                                                    size_t     tensor_para_size,
                                                    size_t     tensor_para_rank):
    head_num_(head_num),
    kv_head_num_(kv_head_num),
    size_per_head_(size_per_head),
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    weight_type_(weight_type),
    attn_bias_(attn_bias),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank)
{
    if (lora_param.policy == LoraPolicy::kPlora) {
        std::vector<std::string> keys = {
            "attention.w_qkv", "attention.wo", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"};
        std::vector<LlamaDenseWeight<T>*> weights = {&self_attn_weights.qkv,
                                                     &self_attn_weights.output,
                                                     &ffn_weights.gating,
                                                     &ffn_weights.output,
                                                     &ffn_weights.intermediate};
        for (int i = 0; i < keys.size(); i++) {
            const auto& name      = keys[i];
            auto&       weight    = *weights[i];
            int         rank      = lora_param.r;
            float       scale     = lora_param.scale;
            std::string full_name = "layers." + std::to_string(layer_idx) + "." + name;

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

    self_attn_weights.qkv.input_dims  = hidden_units_;
    self_attn_weights.qkv.output_dims = (head_num + 2 * kv_head_num) * size_per_head / tensor_para_size_;
    self_attn_weights.qkv.type        = weight_type;
    self_attn_weights.qkv.group_size  = group_size;

    self_attn_weights.output.input_dims  = (head_num * size_per_head) / tensor_para_size_;
    self_attn_weights.output.output_dims = hidden_units_;
    self_attn_weights.output.type        = weight_type;
    self_attn_weights.output.group_size  = group_size;

    ffn_weights = LlamaFfnWeight<T>{
        hidden_units_,
        inter_size_,
        tensor_para_size_,
        weight_type_,
        group_size,
        weight_type_ == WeightType::kINT4 && is_fuse_silu_act(),
    };

    moe_weights = MoeFfnWeight<T>{hidden_units_,
                                  moe_param.inter_size,
                                  moe_param.expert_num,
                                  moe_param.method,
                                  moe_param.shared_gate,
                                  tensor_para_size_,
                                  weight_type,
                                  group_size,
                                  is_fuse_silu_act()};

    mallocWeights();
}

template<typename T>
size_t LlamaDecoderLayerWeight<T>::workspace_size() const noexcept
{
    // Space to hold the largest weight in full precision

    auto get_size = [](const auto& w) { return (size_t)w.input_dims * w.output_dims; };

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

template<typename T>
void freeWeights(LlamaDenseWeight<T>& weights)
{
    cudaFree(weights.kernel);
    cudaFree(weights.bias);
    cudaFree(weights.scales);
    cudaFree(weights.zeros);

    weights.kernel = nullptr;
    weights.bias   = nullptr;
    weights.scales = nullptr;
    weights.zeros  = nullptr;

    {
        cudaFree(weights.lora.a);
        cudaFree(weights.lora.b);
        weights.lora.a = nullptr;
        weights.lora.b = nullptr;
    }
}

template<typename T>
void LlamaDecoderLayerWeight<T>::mallocWeights(LlamaDenseWeight<T>& weights, bool bias)
{
    if (bias) {
        deviceMalloc((T**)&weights.bias, weights.output_dims);
    }
    const size_t bit_size = getBitSize(weights.type);
    if (bit_size >= 16) {  // fp16, fp32
        deviceMalloc((T**)&weights.kernel, weights.input_dims * weights.output_dims);
    }
    else {  // int8, int4
        const int factor = sizeof(float) * 8 / bit_size;
        FT_CHECK(weights.input_dims % factor == 0);
        deviceMalloc((int**)&weights.kernel, weights.input_dims * weights.output_dims / factor);
        deviceMemSetZero((int*)weights.kernel, weights.input_dims * weights.output_dims / factor);
        deviceMalloc((T**)&weights.scales, weights.input_dims / weights.group_size * weights.output_dims);
        deviceMalloc((T**)&weights.zeros, weights.input_dims / weights.group_size * weights.output_dims);
    }

    if (weights.lora.r > 0) {
        deviceMalloc((T**)&weights.lora.a, weights.input_dims * weights.lora.r);
        deviceMalloc((T**)&weights.lora.b, weights.lora.r * weights.output_dims);
    }
}

template<typename FirstArg, typename... Args>
std::string concat(FirstArg&& first, Args&&... args)
{
    std::stringstream stream;
    stream << first;
    ((stream << "." << args), ...);
    return stream.str();
}

template<typename T>
void getWeightTensor(LlamaDenseWeight<T>& weights, bool bias, const std::string& prefix, TensorMap& output)
{
    auto get_name = [=](const std::string& name) { return concat(prefix, name); };

    if (bias) {
        output.insert(get_name("bias"), Tensor{MEMORY_GPU, getTensorType<T>(), {weights.bias_size()}, weights.bias});
    }

    const size_t bit_size = getBitSize(weights.type);
    if (bit_size >= 16) {
        output.insert(get_name("weight"),
                      Tensor{MEMORY_GPU, getTensorType<T>(), {weights.kernel_size()}, weights.kernel});
    }
    else {
        output.insert(get_name("qweight"), Tensor{MEMORY_GPU, TYPE_INT32, {weights.kernel_size()}, weights.kernel});
        output.insert(get_name("scales"),
                      Tensor{MEMORY_GPU, getTensorType<T>(), {weights.scales_size()}, weights.scales});
        output.insert(get_name("zeros"),
                      Tensor{MEMORY_GPU, getTensorType<T>(), {weights.scales_size()}, weights.zeros});
    }

    if (weights.lora.r) {
        auto n = prefix.rfind(".");

        std::string _prefix = prefix.substr(0, n);
        std::string _num    = prefix.substr(n + 1);

        output.insert(concat(_prefix, "lora_a", _num, "weight"),
                      Tensor{MEMORY_GPU, getTensorType<T>(), {weights.lora_size().first}, weights.lora.a});
        output.insert(concat(_prefix, "lora_b", _num, "weight"),
                      Tensor{MEMORY_GPU, getTensorType<T>(), {weights.lora_size().second}, weights.lora.b});

        TM_LOG_DEBUG("allocate lora weight, layer_name=%s input_dims=%d, output_dims=%d, lora_r=%d",
                     get_name("weight").c_str(),
                     weights.input_dims,
                     weights.output_dims,
                     weights.lora.r);
    }
}

template<typename T>
void loadWeights(
    LlamaDenseWeight<T>& w, std::string prefix, int rank, FtCudaDataType model_file_type, size_t tensor_para_size)
{
    auto weight_file  = prefix + "." + std::to_string(tensor_para_size - 1) + ".weight";
    auto qweight_file = prefix + "." + std::to_string(tensor_para_size - 1) + ".qweight";

    if (!std::filesystem::exists(weight_file) && !std::filesystem::exists(qweight_file)) {
        TM_LOG_ERROR("%s and %s does not exist", weight_file.c_str(), qweight_file.c_str());
        FT_CHECK(false);
    }

    prefix += "." + std::to_string(rank);

    size_t     dim0 = w.input_dims;
    size_t     dim1 = w.output_dims;
    const auto type = model_file_type;

    if (w.bias) {
        loadWeightFromBin((T*)w.bias, {1, dim1}, prefix + ".bias", type);
    }
    const size_t bit_size = getBitSize(w.type);
    if (bit_size >= 16) {  // fp16, fp32
        loadWeightFromBin((T*)w.kernel, {dim0, dim1}, prefix + ".weight", type);
    }
    else {  // int8, int4
        const int factor = sizeof(float) * 8 / bit_size;

        FT_CHECK(dim1 % factor == 0);

        std::vector<size_t> w_shape{dim0, dim1 / factor * sizeof(uint32_t)};
        loadWeightFromBin((int8_t*)w.kernel, w_shape, prefix + ".qweight", FtCudaDataType::INT8);

        const size_t group_count = w.group_size > 0 ? dim0 / w.group_size : 1;

        loadWeightFromBin((half*)w.scales, {group_count, dim1}, prefix + ".scales", type);
        loadWeightFromBin((half*)w.zeros, {group_count, dim1}, prefix + ".zeros", type);
    }
}

template<typename T>
void loadWeights(LlamaDenseWeight<T>& w, std::string prefix, FtCudaDataType model_file_type)
{
    auto weight_file  = prefix + ".weight";
    auto qweight_file = prefix + ".qweight";

    if (!std::filesystem::exists(weight_file) && !std::filesystem::exists(qweight_file)) {
        TM_LOG_ERROR("%s and %s does not exist", weight_file.c_str(), qweight_file.c_str());
        FT_CHECK(false);
    }

    size_t     dim0 = w.input_dims;
    size_t     dim1 = w.output_dims;
    const auto type = model_file_type;

    if (w.bias) {
        loadWeightFromBin((T*)w.bias, {1, dim1}, prefix + ".bias", type);
    }
    const size_t bit_size = getBitSize(w.type);
    if (bit_size >= 16) {  // fp16, fp32
        loadWeightFromBin((T*)w.kernel, {dim0, dim1}, prefix + ".weight", type);
    }
    else {  // int8, int4
        const int factor = sizeof(float) * 8 / bit_size;

        FT_CHECK(dim1 % factor == 0);

        std::vector<size_t> w_shape{dim0, dim1 / factor * sizeof(uint32_t)};
        loadWeightFromBin((int8_t*)w.kernel, w_shape, prefix + ".qweight", FtCudaDataType::INT8);

        const size_t group_count = w.group_size > 0 ? dim0 / w.group_size : 1;

        loadWeightFromBin((half*)w.scales, {group_count, dim1}, prefix + ".scales", type);
        loadWeightFromBin((half*)w.zeros, {group_count, dim1}, prefix + ".zeros", type);
    }
}

template<typename T>
void LlamaDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc((T**)&self_attn_norm_weights, hidden_units_);
    deviceMalloc((T**)&ffn_norm_weights, hidden_units_);

    mallocWeights(self_attn_weights.qkv, attn_bias_);
    mallocWeights(self_attn_weights.output, attn_bias_);

    if (inter_size_) {
        mallocWeights(ffn_weights.gating, false);
        mallocWeights(ffn_weights.intermediate, false);
        mallocWeights(ffn_weights.output, false);
    }

    if (!moe_weights.experts.empty()) {
        mallocWeights(moe_weights.gate, false);
        for (auto& e : moe_weights.experts) {
            mallocWeights(e.gating, false);
            mallocWeights(e.intermediate, false);
            mallocWeights(e.output, false);
        }
        if (moe_weights.shared_gate.output_dims) {
            mallocWeights(moe_weights.shared_gate, false);
        }
    }
}

template<typename T>
LlamaDecoderLayerWeight<T>::~LlamaDecoderLayerWeight()
{
    cudaFree((void*)self_attn_norm_weights);
    cudaFree((void*)ffn_norm_weights);
    self_attn_norm_weights = nullptr;
    ffn_norm_weights       = nullptr;

    freeWeights(self_attn_weights.qkv);
    freeWeights(self_attn_weights.output);

    if (inter_size_) {
        freeWeights(ffn_weights.fused_gating_intermediate);
        freeWeights(ffn_weights.gating);
        freeWeights(ffn_weights.intermediate);
        freeWeights(ffn_weights.output);
    }

    if (!moe_weights.experts.empty()) {
        freeWeights(moe_weights.gate);
        for (auto& e : moe_weights.experts) {
            freeWeights(e.fused_gating_intermediate);
            freeWeights(e.gating);
            freeWeights(e.intermediate);
            freeWeights(e.output);
        }
        if (moe_weights.shared_gate.kernel) {
            freeWeights(moe_weights.shared_gate);
        }
    }
}

template<typename T>
void LlamaDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    const auto rank_spec = std::to_string(tensor_para_rank_);
    const auto type      = model_file_type;

    loadWeightFromBin(
        (T*)self_attn_norm_weights, {hidden_units_}, dir_path + ".attention_norm.weight", model_file_type);
    loadWeightFromBin((T*)ffn_norm_weights, {hidden_units_}, dir_path + ".ffn_norm.weight", model_file_type);

    loadWeights(self_attn_weights.qkv, dir_path + ".attention.w_qkv", tensor_para_rank_, type, tensor_para_size_);

    loadWeights(self_attn_weights.output, dir_path + ".attention.wo", tensor_para_rank_, type, tensor_para_size_);
    if (moe_weights.experts.empty()) {
        loadWeights(ffn_weights.gating, dir_path + ".feed_forward.w1", tensor_para_rank_, type, tensor_para_size_);
        loadWeights(
            ffn_weights.intermediate, dir_path + ".feed_forward.w3", tensor_para_rank_, type, tensor_para_size_);
        loadWeights(ffn_weights.output, dir_path + ".feed_forward.w2", tensor_para_rank_, type, tensor_para_size_);
    }
    else {
        loadWeights(moe_weights.gate, dir_path + ".moe_ffn.gate", type);
        for (size_t i = 0; i < moe_weights.experts.size(); ++i) {
            std::string weight_name = dir_path + ".moe_ffn.experts." + std::to_string(i);
            loadWeights(moe_weights.experts[i].gating, weight_name + ".w1", tensor_para_rank_, type, tensor_para_size_);
            loadWeights(
                moe_weights.experts[i].intermediate, weight_name + ".w3", tensor_para_rank_, type, tensor_para_size_);
            loadWeights(moe_weights.experts[i].output, weight_name + ".w2", tensor_para_rank_, type, tensor_para_size_);
        }
    }
}

template<typename T>
TensorMap LlamaDecoderLayerWeight<T>::getParams(std::string prefix)
{
    TensorMap output;

    output.insert(concat(prefix, "attention_norm.weight"),
                  Tensor{MEMORY_GPU, getTensorType<T>(), {hidden_units_ * sizeof(T)}, self_attn_norm_weights});

    output.insert(concat(prefix, "ffn_norm.weight"),
                  Tensor{MEMORY_GPU, getTensorType<T>(), {hidden_units_ * sizeof(T)}, ffn_norm_weights});

    auto get_prefix = [=](std::string_view name) { return concat(prefix, name, tensor_para_rank_); };

    getWeightTensor(self_attn_weights.qkv, attn_bias_, get_prefix("attention.w_qkv"), output);
    getWeightTensor(self_attn_weights.output, attn_bias_, get_prefix("attention.wo"), output);

    if (inter_size_) {
        getWeightTensor(ffn_weights.gating, false, get_prefix("feed_forward.w1"), output);
        getWeightTensor(ffn_weights.intermediate, false, get_prefix("feed_forward.w3"), output);
        getWeightTensor(ffn_weights.output, false, get_prefix("feed_forward.w2"), output);
    }

    if (!moe_weights.experts.empty()) {
        output.insert(
            concat(prefix, "moe_ffn.gate.weight"),
            Tensor{MEMORY_GPU, getTensorType<T>(), {moe_weights.gate.kernel_size()}, moe_weights.gate.kernel});
        auto& experts = moe_weights.experts;
        for (size_t i = 0; i < experts.size(); ++i) {
            const std::string name = "moe_ffn.experts." + std::to_string(i);
            getWeightTensor(experts[i].gating, false, get_prefix(concat(name, "w1")), output);
            getWeightTensor(experts[i].intermediate, false, get_prefix(concat(name, "w3")), output);
            getWeightTensor(experts[i].output, false, get_prefix(concat(name, "w2")), output);
        }
        if (moe_weights.shared_gate.kernel) {
            output.insert(concat(prefix, "moe_ffn.shared_gate.weight"),
                          Tensor{MEMORY_GPU,
                                 getTensorType<T>(),
                                 {moe_weights.shared_gate.kernel_size()},
                                 moe_weights.shared_gate.kernel});
        }
    }

    return output;
}

// template<class T>
static void convert_u4(LlamaDenseWeight<half>& weight, bool is_fused_moe, void* workspace, size_t size, bool use_simt)
{
    FT_CHECK(weight.type == WeightType::kINT4);

    using namespace gemm;

    auto [order_b, pack_b, order_v, pack_v] =
        get_weight_and_scales_layout(gemm::DataType::U4, is_fused_moe, getSMVersion(), use_simt);

    if (order_b == kColMajor) {
        transpose_u4((uint4_t*)workspace, (const uint4_t*)weight.kernel, weight.input_dims, weight.output_dims);
        cudaMemcpy(weight.kernel, workspace, weight.input_dims * weight.output_dims / 2, cudaMemcpyDefault);
    }

    extend_to_u16((uint16_t*)workspace, (const uint4_t*)weight.kernel, weight.input_dims * weight.output_dims);
    sync_check_cuda_error();

    MatrixLayout w_desc{
        gemm::DataType::F16,
        order_b,
        (int)weight.input_dims,   // k
        (int)weight.output_dims,  // n
        order_b == kRowMajor ? (int)weight.output_dims : (int)weight.input_dims,
    };

    MatrixLayout k_desc = w_desc;
    k_desc.type         = gemm::DataType::U4;
    k_desc.pack         = pack_b;

    cudaMemset(weight.kernel, 0, weight.input_dims * weight.output_dims / 2);

    FT_CHECK(Convert(workspace, w_desc, weight.kernel, k_desc, 0) == 0);
    sync_check_cuda_error();

    const int scale_count = (weight.input_dims / weight.group_size) * weight.output_dims;

    // std::cout << "fuse_scales_and_zeros\n";
    fuse_scales_and_zeros((half*)workspace, weight.scales, weight.zeros, scale_count);
    // cudaMemset((T*)workspace, 0, sizeof(T) * scale_count * 2);
    sync_check_cuda_error();

    cudaDeviceSynchronize();

    cudaFree(weight.scales);
    cudaFree(weight.zeros);
    weight.scales = weight.zeros = nullptr;

    deviceMalloc((half**)&weight.scales_zeros, scale_count * 2);

    MatrixLayout s_desc{
        gemm::DataType::U32,
        order_v,
        (int)weight.input_dims / weight.group_size,  // k
        (int)weight.output_dims,                     // n
        (int)weight.output_dims,
    };

    MatrixLayout q_desc = s_desc;
    q_desc.pack         = pack_v;

    FT_CHECK(Convert(workspace, s_desc, weight.scales_zeros, q_desc, 0) == 0);
    sync_check_cuda_error();

    weight.k_desc = k_desc;
    weight.q_desc = q_desc;

    // FT_CHECK(0);
}

template<class T>
static void convert_fp(LlamaDenseWeight<T>& weight, bool is_fused_moe, void* workspace, size_t size, bool use_simt)
{
    using namespace gemm;

    if (!is_fused_moe) {
        return;
    }

    const auto [order_b, pack_b, order_v, pack_v] =
        get_weight_and_scales_layout(get_data_type_v<T>, is_fused_moe, getSMVersion(), use_simt);

    const int input_dim  = weight.input_dims;
    const int output_dim = weight.output_dims;

    if (order_b == kColMajor) {
        invokeTransposeAxis01((uint16_t*)workspace, (uint16_t*)weight.kernel, input_dim, output_dim, 1, nullptr);
        sync_check_cuda_error();
        // FT_CHECK(0);
    }
    else {
        check_cuda_error(cudaMemcpy(workspace, weight.kernel, sizeof(T) * input_dim * output_dim, cudaMemcpyDefault));
    }

    MatrixLayout src{
        get_data_type_v<T>,
        order_b,
        input_dim,   // k
        output_dim,  // n
        order_b == kRowMajor ? output_dim : input_dim,
    };

    MatrixLayout dst = src;
    dst.pack         = pack_b;

    if (pack_b) {
        FT_CHECK(Convert(workspace, src, weight.kernel, dst, nullptr) == 0);
        sync_check_cuda_error();
        // FT_CHECK(0);
    }
    else {
        check_cuda_error(cudaMemcpy(weight.kernel, workspace, sizeof(T) * input_dim * output_dim, cudaMemcpyDefault));
    }

    weight.k_desc = dst;
}

template<class T>
static void convert(LlamaDenseWeight<T>& weight, bool is_fused_moe, void* workspace, size_t size, bool use_simt)
{
    if (weight.type == WeightType::kINT4) {
        if constexpr (std::is_same_v<T, half>) {
            convert_u4(weight, is_fused_moe, workspace, size, use_simt);
        }
        else {
            FT_CHECK(0);
        }
    }
    else {
        convert_fp(weight, is_fused_moe, workspace, size, use_simt);
    }
}

template<class T>
void interleave(LlamaDenseWeight<T>& c, LlamaDenseWeight<T>& a, LlamaDenseWeight<T>& b, void* workspace, size_t size)
{
    FT_CHECK(c.input_dims == a.input_dims);
    FT_CHECK(c.input_dims == b.input_dims);
    FT_CHECK(c.output_dims == a.output_dims * 2);
    FT_CHECK(c.output_dims == b.output_dims * 2);
    FT_CHECK(c.group_size == a.group_size);
    FT_CHECK(c.group_size == b.group_size);

    if (a.type == WeightType::kINT4) {
        uint8_t* tmp_a = (uint8_t*)workspace;
        uint8_t* tmp_b = tmp_a + a.output_dims * a.input_dims;
        uint8_t* tmp_c = tmp_b + b.output_dims * b.input_dims;

        const auto sentinel = tmp_c + c.output_dims * c.input_dims;
        FT_CHECK(sentinel <= (uint8_t*)workspace + size);

        extend_to_u8(tmp_a, (const uint4_t*)a.kernel, a.output_dims * a.input_dims);
        extend_to_u8(tmp_b, (const uint4_t*)b.kernel, b.output_dims * b.input_dims);

        interleave_output_dims(tmp_c, tmp_a, tmp_b, a.output_dims, a.input_dims, 0);

        compact_to_u4((uint4_t*)c.kernel, tmp_c, c.output_dims * c.input_dims);

        interleave_output_dims(c.scales, a.scales, b.scales, a.output_dims, a.input_dims / a.group_size, 0);
        interleave_output_dims(c.zeros, a.zeros, b.zeros, a.output_dims, a.input_dims / a.group_size, 0);
    }
    else {
        interleave_output_dims((T*)c.kernel, (const T*)a.kernel, (const T*)b.kernel, a.output_dims, a.input_dims, 0);
    }

    // Check at function level
    sync_check_cuda_error();
}

template<class T>
void chunk(LlamaDenseWeight<T>& c, LlamaDenseWeight<T>& a, LlamaDenseWeight<T>& b, void*, size_t)
{
    FT_CHECK(c.input_dims == a.input_dims);
    FT_CHECK(c.input_dims == b.input_dims);
    FT_CHECK(c.output_dims == a.output_dims * 2);
    FT_CHECK(c.output_dims == b.output_dims * 2);
    FT_CHECK(c.group_size == a.group_size);
    FT_CHECK(c.group_size == b.group_size);

    auto _chunks = [](auto c, auto a, auto b, int height, int width) {
        check_cuda_error(cudaMemcpy2D((char*)c + 0x000, width * 2, a, width, width, height, cudaMemcpyDefault));
        check_cuda_error(cudaMemcpy2D((char*)c + width, width * 2, b, width, width, height, cudaMemcpyDefault));
    };

    if (c.type == WeightType::kINT4) {
        _chunks(c.kernel, a.kernel, b.kernel, a.input_dims, 4 * a.output_dims / 8);
        _chunks(c.scales, a.scales, b.scales, a.input_dims / a.group_size, sizeof(T) * a.output_dims);
        _chunks(c.zeros, a.zeros, b.zeros, a.input_dims / a.group_size, sizeof(T) * a.output_dims);
    }
    else {
        _chunks(c.kernel, a.kernel, b.kernel, a.input_dims, sizeof(T) * a.output_dims);
    }

    // Check at function level
    sync_check_cuda_error();
}

template<typename T>
void LlamaDecoderLayerWeight<T>::prepare(void* workspace, size_t size, const cudaDeviceProp& prop)
{
    const bool is_16xx = is_16xx_series(prop.name);

    convert(self_attn_weights.qkv, false, workspace, size, is_16xx);
    convert(self_attn_weights.output, false, workspace, size, is_16xx);

    auto process_ffn = [&](LlamaFfnWeight<T>& ffn, bool is_fused_moe) {
        if (fused_up_and_gate_) {
            auto& fused_up_and_gate = ffn.fused_gating_intermediate;

            mallocWeights(fused_up_and_gate, false);

            if (ffn.is_fused_silu) {
                interleave(fused_up_and_gate, ffn.gating, ffn.intermediate, workspace, size);
            }
            else {
                chunk(fused_up_and_gate, ffn.gating, ffn.intermediate, workspace, size);
            }

            convert(ffn.fused_gating_intermediate, is_fused_moe, workspace, size, is_16xx);

            freeWeights(ffn.gating);
            freeWeights(ffn.intermediate);
        }
        else {
            convert(ffn.gating, is_fused_moe, workspace, size, is_16xx);
            convert(ffn.intermediate, is_fused_moe, workspace, size, is_16xx);
        }

        convert(ffn.output, is_fused_moe, workspace, size, is_16xx);
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

            process_ffn(e, moe_weights.method);

            const auto& fused  = e.fused_gating_intermediate;
            const auto& output = e.output;

            fused_ptrs.push_back({fused.kernel, fused.k_desc.ld});
            output_ptrs.push_back({output.kernel, output.k_desc.ld});

            if (e.fused_gating_intermediate.scales_zeros) {
                fused_param_ptrs.emplace_back(fused.scales_zeros, fused.q_desc.ld);
                output_param_ptrs.emplace_back(output.scales_zeros, output.q_desc.ld);
            }
        }

        // Note: This assumes all experts has the same shape
        moe_weights.block = moe_weights.experts.at(0);

        auto& fused  = moe_weights.block.fused_gating_intermediate;
        auto& output = moe_weights.block.output;

        // TODO: free these ptrs
        fused.kernel  = gemm::make_blocked_ptrs(fused_ptrs, nullptr);
        output.kernel = gemm::make_blocked_ptrs(output_ptrs, nullptr);

        if (!fused_param_ptrs.empty()) {
            fused.scales_zeros  = (T*)gemm::make_blocked_ptrs(fused_param_ptrs, nullptr);
            output.scales_zeros = (T*)gemm::make_blocked_ptrs(output_param_ptrs, nullptr);
        }

        fused.k_desc.ld = output.k_desc.ld = 0;
        fused.k_desc.num = output.k_desc.num = moe_weights.experts.size();

        fused.q_desc.ld = output.q_desc.ld = 0;
        fused.q_desc.num = output.q_desc.num = moe_weights.experts.size();
    }
}

#ifdef ENABLE_FP32
template struct LlamaDecoderLayerWeight<float>;
#endif
template struct LlamaDecoderLayerWeight<half>;
#ifdef ENABLE_BF16
template struct LlamaDecoderLayerWeight<__nv_bfloat16>;
#endif

}  // namespace turbomind
