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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.cc

#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<typename T>
LlamaDecoderLayerWeight<T>::LlamaDecoderLayerWeight(size_t     hidden_units,
                                                    size_t     inter_size,
                                                    WeightType weight_type,
                                                    bool       attn_bias,
                                                    size_t     tensor_para_size,
                                                    size_t     tensor_para_rank):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    weight_type_(weight_type),
    attn_bias_(attn_bias),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank)
{
    self_attn_weights.qkv.input_dims  = hidden_units_;
    self_attn_weights.qkv.output_dims = 3 * hidden_units_ / tensor_para_size_;
    self_attn_weights.qkv.type        = weight_type;

    self_attn_weights.output.input_dims  = hidden_units_ / tensor_para_size_;
    self_attn_weights.output.output_dims = hidden_units_;
    self_attn_weights.output.type        = weight_type;

    ffn_weights.gating.input_dims  = hidden_units_;
    ffn_weights.gating.output_dims = inter_size_ / tensor_para_size_;
    ffn_weights.gating.type        = weight_type;

    ffn_weights.intermediate.input_dims  = hidden_units_;
    ffn_weights.intermediate.output_dims = inter_size_ / tensor_para_size_;
    ffn_weights.intermediate.type        = weight_type;

    ffn_weights.output.input_dims  = inter_size_ / tensor_para_size_;
    ffn_weights.output.output_dims = hidden_units_;
    ffn_weights.output.type        = weight_type;
    mallocWeights();
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
}

template<typename T>
void mallocWeights(LlamaDenseWeight<T>& weights, bool bias)
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
        deviceMalloc((float**)&weights.kernel, weights.input_dims / factor * weights.output_dims);
        deviceMalloc((T**)&weights.scales, weights.output_dims);
        deviceMalloc((T**)&weights.zeros, weights.output_dims);
    }
}

template<typename T>
void loadWeights(LlamaDenseWeight<T>& w, std::string prefix, int rank, FtCudaDataType model_file_type)
{
    prefix += "." + std::to_string(rank);
    const auto type = model_file_type;

    if (w.bias) {
        loadWeightFromBin((T*)w.bias, {w.output_dims}, prefix + ".bias", type);
    }
    const size_t bit_size = getBitSize(w.type);
    if (bit_size >= 16) {  // fp16, fp32
        loadWeightFromBin((T*)w.kernel, {w.input_dims, w.output_dims}, prefix + ".weight", type);
    }
    else {  // int8, int4
        const int factor = sizeof(float) * 8 / bit_size;
        FT_CHECK(w.input_dims % factor == 0);
        const auto f32_type = FtCudaDataType::FP32;
        loadWeightFromBin((float*)w.kernel, {w.input_dims / factor, w.output_dims}, prefix + ".qweight", f32_type);
        loadWeightFromBin((T*)w.scales, {w.output_dims}, prefix + ".scales", type);
        loadWeightFromBin((T*)w.zeros, {w.output_dims}, prefix + ".zeros", type);
    }
}

template<typename T>
void LlamaDecoderLayerWeight<T>::mallocWeights()
{
    deviceMalloc((T**)&self_attn_norm_weights, hidden_units_);
    deviceMalloc((T**)&ffn_norm_weights, hidden_units_);

    turbomind::mallocWeights(self_attn_weights.qkv, attn_bias_);
    turbomind::mallocWeights(self_attn_weights.output, attn_bias_);

    turbomind::mallocWeights(ffn_weights.gating, false);
    turbomind::mallocWeights(ffn_weights.intermediate, false);
    turbomind::mallocWeights(ffn_weights.output, false);
}

template<typename T>
LlamaDecoderLayerWeight<T>::~LlamaDecoderLayerWeight()
{
    cudaFree((void*)self_attn_norm_weights);
    cudaFree((void*)ffn_norm_weights);

    freeWeights(self_attn_weights.qkv);
    freeWeights(self_attn_weights.output);
    freeWeights(ffn_weights.gating);
    freeWeights(ffn_weights.intermediate);
    freeWeights(ffn_weights.output);
}

template<typename T>
void LlamaDecoderLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    const auto rank_spec = std::to_string(tensor_para_rank_);
    const auto type      = model_file_type;

    loadWeightFromBin(
        (T*)self_attn_norm_weights, {hidden_units_}, dir_path + ".attention_norm.weight", model_file_type);
    loadWeightFromBin((T*)ffn_norm_weights, {hidden_units_}, dir_path + ".ffn_norm.weight", model_file_type);

    loadWeights(self_attn_weights.qkv, dir_path + ".attention.w_qkv", tensor_para_rank_, type);
    loadWeights(self_attn_weights.output, dir_path + ".attention.wo", tensor_para_rank_, type);
    loadWeights(ffn_weights.gating, dir_path + ".feed_forward.w1", tensor_para_rank_, type);
    loadWeights(ffn_weights.intermediate, dir_path + ".feed_forward.w3", tensor_para_rank_, type);
    loadWeights(ffn_weights.output, dir_path + ".feed_forward.w2", tensor_para_rank_, type);

    // load kv_cache quant scale
    // if file not exist, get empty vector
    std::string   scale_path = dir_path + ".past_kv_scale." + rank_spec + ".weight";
    std::ifstream in(scale_path, std::ios::in);
    if (in.is_open()) {
        in.close();
        self_attn_weights.past_kv_scale = loadArrayFromBin({2}, scale_path);
    }
    else {
        self_attn_weights.past_kv_scale = {};
    }
}

template struct LlamaDecoderLayerWeight<float>;
template struct LlamaDecoderLayerWeight<half>;

}  // namespace turbomind
