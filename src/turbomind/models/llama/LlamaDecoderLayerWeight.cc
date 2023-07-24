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
#include <filesystem>

namespace turbomind {

template<typename T>
LlamaDecoderLayerWeight<T>::LlamaDecoderLayerWeight(size_t     head_num,
                                                    size_t     kv_head_num,
                                                    size_t     size_per_head,
                                                    size_t     inter_size,
                                                    WeightType weight_type,
                                                    bool       attn_bias,
                                                    size_t     tensor_para_size,
                                                    size_t     tensor_para_rank):
    head_num_(head_num),
    kv_head_num_(kv_head_num),
    size_per_head_(size_per_head),
    hidden_units_(head_num * size_per_head),
    inter_size_(inter_size),
    weight_type_(weight_type),
    attn_bias_(attn_bias),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank)
{
    self_attn_weights.qkv.input_dims  = hidden_units_;
    self_attn_weights.qkv.output_dims = (head_num + 2 * kv_head_num) * size_per_head / tensor_para_size_;
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
void loadWeights(LlamaDenseWeight<T>& w,
                 std::string          prefix,
                 int                  rank,
                 FtCudaDataType       model_file_type,
                 size_t               tensor_para_size,
                 int                  slice_dim   = 0,
                 std::vector<size_t>  slice_shape = {})
{
    auto       max_prefix = prefix + "." + std::to_string(tensor_para_size - 1);
    const auto type       = model_file_type;

    bool enable_slice = true;
    // Disable slice if tensor param rank is 1
    if (tensor_para_size <= 1) {
        enable_slice = false;
    }
    else {
        // Disable slice if weight has already been sliced
        if (std::filesystem::exists(max_prefix + ".weight") || std::filesystem::exists(max_prefix + ".qweight")) {
            TM_LOG_DEBUG("TP weight exists. Disable runtime TP.");
            enable_slice = false;
        }
    }

    size_t dim0 = w.input_dims;
    size_t dim1 = w.output_dims;
    if (enable_slice) {
        // multiple tp size for slice stride
        if (slice_dim == 0) {
            dim0 = dim0 * tensor_para_size;
            if (slice_shape.size() == 0) {
                slice_shape = {dim0};
            }
        }
        else {
            dim1 = dim1 * tensor_para_size;
            if (slice_shape.size() == 0) {
                slice_shape = {dim1};
            }
        }

        prefix += "." + std::to_string(0);
    }
    else {
        prefix += "." + std::to_string(rank);
    }

    if (w.bias) {
        std::vector<ConcateSlice> bias_slices{};
        if (enable_slice) {
            if (slice_dim == 1) {
                size_t       start = 0;
                ConcateSlice slice0{.slices = {{0, 1}}};
                ConcateSlice slice1{.slices = {{}}};
                for (auto len : slice_shape) {
                    size_t stride = len / tensor_para_size;
                    slice1.slices.push_back({start + stride * rank, start + stride * (rank + 1)});
                    start += len;
                }
                bias_slices = {slice0, slice1};
            }
        }
        loadWeightFromBin((T*)w.bias, {1, dim1}, prefix + ".bias", type, bias_slices);
    }
    const size_t bit_size = getBitSize(w.type);
    if (bit_size >= 16) {  // fp16, fp32
        std::vector<ConcateSlice> weight_slices{};
        if (enable_slice) {
            if (slice_dim == 1) {
                size_t       start = 0;
                ConcateSlice slice0{.slices = {{0, dim0}}};
                ConcateSlice slice1{.slices = {{}}};
                for (auto len : slice_shape) {
                    size_t stride = len / tensor_para_size;
                    slice1.slices.push_back({start + stride * rank, start + stride * (rank + 1)});
                    start += len;
                }
                weight_slices = {slice0, slice1};
            }
            else {
                size_t       start = 0;
                ConcateSlice slice0{.slices = {}};
                ConcateSlice slice1{.slices = {{0, dim1}}};
                for (auto len : slice_shape) {
                    size_t stride = len / tensor_para_size;
                    slice0.slices.push_back({start + stride * rank, start + stride * (rank + 1)});
                    start += len;
                }
                weight_slices = {slice0, slice1};
            }
        }
        loadWeightFromBin((T*)w.kernel, {dim0, dim1}, prefix + ".weight", type, weight_slices);
    }
    // else {  // int8, int4
    //     const int factor = sizeof(float) * 8 / bit_size;
    //     FT_CHECK(dim0 % factor == 0);
    //     const auto               f32_type = FtCudaDataType::FP32;
    //     std::vector<ConcateSlice> weight_slices{};
    //     std::vector<ConcateSlice> bias_slices{};
    //     if (enable_slice) {
    //         if (slice_dim == 1) {
    //             size_t      stride = dim1 / tensor_para_size;
    //             ConcateSlice slice0{.start = 0, .end = dim0 / factor};
    //             ConcateSlice slice1{.start = stride * rank, .end = stride * (rank + 1)};
    //             weight_slices = {slice0, slice1};

    //             ConcateSlice bias_slice0{.start = 0, .end = bias_dim0};
    //             bias_slices = {bias_slice0, slice1};
    //         }
    //         else {
    //             size_t      stride = dim0 / factor / tensor_para_size;
    //             ConcateSlice slice0{.start = stride * rank, .end = stride * (rank + 1)};
    //             ConcateSlice slice1{.start = 0, .end = dim1};
    //             weight_slices = {slice0, slice1};
    //         }
    //     }
    //     loadWeightFromBin((float*)w.kernel, {dim0 / factor, dim1}, prefix + ".qweight", f32_type, weight_slices);
    //     loadWeightFromBin((T*)w.scales, {bias_dim0, dim1}, prefix + ".scales", type, bias_slices);
    //     loadWeightFromBin((T*)w.zeros, {bias_dim0, dim1}, prefix + ".zeros", type, bias_slices);
    // }
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

    loadWeights(self_attn_weights.qkv,
                dir_path + ".attention.w_qkv",
                tensor_para_rank_,
                type,
                tensor_para_size_,
                1,
                {head_num_ * size_per_head_, kv_head_num_ * size_per_head_, kv_head_num_ * size_per_head_});
    loadWeights(self_attn_weights.output, dir_path + ".attention.wo", tensor_para_rank_, type, tensor_para_size_, 0);
    loadWeights(ffn_weights.gating, dir_path + ".feed_forward.w1", tensor_para_rank_, type, tensor_para_size_, 1);
    loadWeights(ffn_weights.intermediate, dir_path + ".feed_forward.w3", tensor_para_rank_, type, tensor_para_size_, 1);
    loadWeights(ffn_weights.output, dir_path + ".feed_forward.w2", tensor_para_rank_, type, tensor_para_size_, 0);

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
