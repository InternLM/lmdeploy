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

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/layers/FfnLayer.h

#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/kernels/activation_kernels.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/nvtx_utils.h"
// #include <glog/logging.h>

namespace turbomind {

template<typename T>
void LlamaFfnLayer<T>::allocateBuffer(size_t token_num)
{
    inter_buf_          = (T*)allocator_->reMalloc(inter_buf_, sizeof(T) * token_num * inter_size_, false);
    gating_buf_         = (T*)allocator_->reMalloc(gating_buf_, sizeof(T) * token_num * inter_size_, false);
    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaFfnLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)&inter_buf_);
        allocator_->free((void**)&gating_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void LlamaFfnLayer<T>::activation(int num_token)
{
    NvtxScope scope("activation");
    invokeGenericActivation<SiluActivation>(gating_buf_,
                                            (const T*)nullptr,  // bias
                                            inter_buf_,
                                            (const T*)nullptr,  // gated_bias
                                            nullptr,            // ia3_tasks
                                            (const T*)nullptr,  // ia3_weights
                                            num_token,          // m
                                            inter_size_,        // n
                                            0,                  // int8_mode
                                            nullptr,            // activation_in
                                            nullptr,            // activation_out
                                            nullptr,            // padding_offset
                                            0,                  // seq_len
                                            stream_);
    sync_check_cuda_error();
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
    // LOG(WARNING);

    allocateBuffer(num_token);

    const T* ffn_input_data  = input_tensors->at("ffn_input").getPtr<T>();
    T*       ffn_output_data = output_tensors->at("ffn_output").getPtr<T>();

    if (weights->fused_gating_intermediate.kernel) {
        NvtxScope scope("fused_silu_ffn");
        linear_.forward(
            gating_buf_, ffn_input_data, num_token, weights->fused_gating_intermediate, LlamaLinear<T>::kFusedSiluFfn);
    }
    else {
        {  // w1(x)
            NvtxScope scope("w1");
            linear_.forward(gating_buf_, ffn_input_data, num_token, weights->gating);
        }
        {  // w3(x)
            NvtxScope scope("w3");
            linear_.forward(inter_buf_, ffn_input_data, num_token, weights->intermediate);
        }
        // silu(w1(x)) * w3(x)
        activation(num_token);
    }

    {  // w2(x)
        NvtxScope scope("w2");
        linear_.forward(ffn_output_data, gating_buf_, num_token, weights->output);
    }

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
