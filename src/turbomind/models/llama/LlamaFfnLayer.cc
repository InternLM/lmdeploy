/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/FfnLayer.h

#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/nvtx_utils.h"

namespace turbomind {

template<typename T>
void LlamaFfnLayer<T>::allocateBuffer(size_t                     token_num,
                                      const LlamaDenseWeight<T>* gating,
                                      const LlamaDenseWeight<T>* inter)
{
    const size_t sz = token_num * inter_size_;

    const size_t sz_gate  = token_num * gating->lora.r;
    const size_t sz_inter = token_num * inter->lora.r;

    gating_buf_ = (T*)allocator_->reMalloc(gating_buf_, sizeof(T) * (sz * 2 + sz_gate + sz_inter), false);
    inter_buf_  = gating_buf_ + sz;

    // gate & inter is not fused when lora is enabled
    if (gating->lora.r) {
        inter_buf_ += sz_gate;
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaFfnLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        // allocator_->free((void**)&inter_buf_);
        allocator_->free((void**)&gating_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void LlamaFfnLayer<T>::activation(int token_num, bool is_chunked)
{
    NvtxScope scope("activation");
    if (is_chunked) {
        invokeGenericActivation_v2<SiluActivation>(
            gating_buf_, gating_buf_ + inter_size_, inter_size_ * 2, token_num, inter_size_, stream_);
        sync_check_cuda_error();
    }
    else {
        invokeGenericActivation_v2<SiluActivation>(
            gating_buf_, inter_buf_, inter_size_, token_num, inter_size_, stream_);
        sync_check_cuda_error();
    }
}

template<typename T>
void LlamaFfnLayer<T>::forward(TensorMap*               output_tensors,
                               const TensorMap*         input_tensors,
                               const LlamaFfnWeight<T>* weights)
{
    /**
     * input_tensors:
     *   \param ffn_input [token_num, hidden_dimension]
     *
     * output_tensors:
     *   \param ffn_output [token_num, hidden_dimension]
     */

    NvtxScope scope("ffn");

    const size_t num_token = input_tensors->at("ffn_input").shape[0];
    const int    layer_id  = input_tensors->getVal<int>("layer_id");
    // LOG(WARNING);

    allocateBuffer(num_token, &weights->gating, &weights->intermediate);

    const T* ffn_input_data  = input_tensors->at("ffn_input").getPtr<T>();
    T*       ffn_output_data = output_tensors->at("ffn_output").getPtr<T>();
    int*     lora_mask = input_tensors->at("lora_mask", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<int>();

    if (weights->fused_gating_intermediate.kernel) {
        NvtxScope scope("fused_silu_ffn");

        const auto type = weights->is_fused_silu ? LlamaLinear<T>::kFusedSiluFfn : LlamaLinear<T>::kGemm;

        linear_->forward(gating_buf_, ffn_input_data, num_token, weights->fused_gating_intermediate, type);
        sync_check_cuda_error();

        if (!weights->is_fused_silu) {
            activation(num_token, true);
        }

        count_and_fix(gating_buf_, num_token * weights->output.input_dims, Concat("w1_w3_silu", layer_id), 3);
    }
    else {
        {  // w1(x)
            NvtxScope scope("w1");
            linear_->forward(gating_buf_, ffn_input_data, num_token, weights->gating, LlamaLinear<T>::kGemm, lora_mask);
            sync_check_cuda_error();
        }
        count_and_fix(gating_buf_, num_token * weights->gating.output_dims, Concat("w1", layer_id), 3);

        {  // w3(x)
            NvtxScope scope("w3");
            linear_->forward(
                inter_buf_, ffn_input_data, num_token, weights->intermediate, LlamaLinear<T>::kGemm, lora_mask);
            sync_check_cuda_error();
        }
        count_and_fix(inter_buf_, num_token * weights->intermediate.output_dims, Concat("w3", layer_id), 3);

        // silu(w1(x)) * w3(x)
        activation(num_token, false);

        count_and_fix(gating_buf_, num_token * weights->output.input_dims, Concat("act", layer_id), 3);
    }

    {  // w2(x)
        NvtxScope scope("w2");
        const int pitch = (weights->fused_gating_intermediate.kernel && !weights->is_fused_silu) ? inter_size_ * 2 : 0;
        linear_->forward(
            ffn_output_data, {gating_buf_, pitch}, num_token, weights->output, LlamaLinear<T>::kGemm, lora_mask);
        sync_check_cuda_error();
    }

    count_and_fix(ffn_output_data, num_token * weights->output.output_dims, Concat("w2", layer_id), 3);

    if (tensor_para_.world_size_ > 1) {
        NcclGuard nccl_guard(tensor_para_, stream_);
        ftNcclAllReduceSum(ffn_output_data, ffn_output_data, num_token * hidden_units_, tensor_para_, stream_);
        sync_check_cuda_error();
    }

    if (is_free_buffer_after_forward_) {
        freeBuffer();
    }
    // LOG(WARNING);
}

#ifdef ENABLE_FP32
template class LlamaFfnLayer<float>;
#endif
template class LlamaFfnLayer<half>;
#ifdef ENABLE_BF16
template class LlamaFfnLayer<__nv_bfloat16>;
#endif

}  // namespace turbomind
