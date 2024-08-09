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
#include "src/turbomind/kernels/marlin_qqq_gemm/marlin_qqq_gemm_kernel.h"
#include "src/turbomind/kernels/quant_kernels.h"
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
    size_t sz       = sizeof(T) * token_num * inter_size_;
    size_t sz_gate  = (gating->lora.r > 0) ? sz + sz / inter_size_ * gating->lora.r : sz;
    size_t sz_inter = (inter->lora.r > 0) ? sz + sz / inter_size_ * inter->lora.r : sz;
    inter_buf_      = (T*)allocator_->reMalloc(inter_buf_, sz_inter, false);
    gating_buf_     = (T*)allocator_->reMalloc(gating_buf_, sz_gate, false);
    if (quantization_ == QuantMethod::QQQ) {
        quant_buf_     = (int8_t*)allocator_->reMalloc(quant_buf_, sizeof(int8_t) * token_num * inter_size_, false);
        act_scale_buf_ = (float*)allocator_->reMalloc(act_scale_buf_, sizeof(float) * token_num, false);
    }
    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaFfnLayer<T>::allocateWorkspace()
{
    if (quantization_ == QuantMethod::QQQ) {
        size_t max_dims     = std::max(inter_size_, hidden_units_);
        size_t sz_reduction = sizeof(int) * marlin_qqq::max_par * 64 * max_dims;
        size_t sz_workspace = sizeof(int) * marlin_qqq::max_par * (max_dims / marlin_qqq::min_thread_n);

        auto [reduce_buf, workspace_buf] = linear_.getQQQBuffer();
        reduce_buf                       = (int*)allocator_->malloc(sz_reduction, false);
        workspace_buf                    = (int*)allocator_->malloc(sz_workspace, true);
        linear_.setQQQBuffer(reduce_buf, workspace_buf);
    }
    is_allocate_workspace_ = true;
}

template<typename T>
void LlamaFfnLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        allocator_->free((void**)&inter_buf_);
        allocator_->free((void**)&gating_buf_);
        allocator_->free((void**)&quant_buf_);
        allocator_->free((void**)&act_scale_buf_);
        is_allocate_buffer_ = false;
    }
}

template<typename T>
void LlamaFfnLayer<T>::freeWorkspace()
{
    if (is_allocate_workspace_) {
        // free qqq workspace
        auto [reduce_buf, workspace_buf] = linear_.getQQQBuffer();
        allocator_->free((void**)&reduce_buf);
        allocator_->free((void**)&workspace_buf);

        is_allocate_workspace_ = false;
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
                                            quant_buf_,         // quant_out
                                            act_scale_buf_,     // quant_scale
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
     *   \param ffn_quant_input [token_num, hidden_dimension]
     *   \param ffn_quant_scale [token_num, hidden_dimension]
     * output_tensors:
     *   \param ffn_output [token_num, hidden_dimension]
     */

    NvtxScope scope("ffn");

    const size_t num_token = input_tensors->at("ffn_input").shape[0];
    const int    layer_id  = input_tensors->getVal<int>("layer_id");
    // LOG(WARNING);

    allocateBuffer(num_token, &weights->gating, &weights->intermediate);

    const T* ffn_input_data       = input_tensors->at("ffn_input").getPtr<T>();
    int8_t*  ffn_quant_input_data = input_tensors->at("ffn_quant_input").getPtr<int8_t>();
    float*   ffn_quant_scale_data = input_tensors->at("ffn_quant_scale").getPtr<float>();
    T*       ffn_output_data      = output_tensors->at("ffn_output").getPtr<T>();
    int*     lora_mask = input_tensors->at("lora_mask", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<int>();

    if (weights->fused_gating_intermediate.kernel) {
        NvtxScope scope("fused_silu_ffn");
        linear_.forward(gating_buf_,
                        ffn_input_data,
                        ffn_quant_input_data,
                        ffn_quant_scale_data,
                        num_token,
                        weights->fused_gating_intermediate,
                        LlamaLinear<T>::kFusedSiluFfn);

        count_and_fix(gating_buf_, num_token * weights->output.input_dims, Concat("w1_w3_silu", layer_id), 3);
    }
    else {
        {  // w1(x)
            NvtxScope scope("w1");
            linear_.forward(gating_buf_,
                            ffn_input_data,
                            ffn_quant_input_data,
                            ffn_quant_scale_data,
                            num_token,
                            weights->gating,
                            LlamaLinear<T>::kGemm,
                            lora_mask);
        }

        {  // w3(x)
            NvtxScope scope("w3");
            linear_.forward(inter_buf_,
                            ffn_input_data,
                            ffn_quant_input_data,
                            ffn_quant_scale_data,
                            num_token,
                            weights->intermediate,
                            LlamaLinear<T>::kGemm,
                            lora_mask);
        }

        count_and_fix(gating_buf_, num_token * weights->gating.output_dims, Concat("w1", layer_id), 3);
        count_and_fix(inter_buf_, num_token * weights->intermediate.output_dims, Concat("w3", layer_id), 3);

        // silu(w1(x)) * w3(x)
        activation(num_token);

        count_and_fix(gating_buf_, num_token * weights->output.input_dims, Concat("act", layer_id), 3);
    }

    {  // w2(x)
        NvtxScope scope("w2");
        linear_.forward(ffn_output_data,
                        gating_buf_,
                        quant_buf_,
                        act_scale_buf_,
                        num_token,
                        weights->output,
                        LlamaLinear<T>::kGemm,
                        lora_mask);
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
