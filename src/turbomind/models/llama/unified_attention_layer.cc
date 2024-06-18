/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/turbomind/layers/attention_layers/GptContextAttentionLayer.cc

#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/debug_utils.h"
#include "src/turbomind/utils/logger.h"
#include <algorithm>
#include <math.h>

namespace turbomind {

template<typename T>

void UnifiedAttentionLayer<T>::allocateBuffer(size_t            q_count,
                                              size_t            k_count,
                                              size_t            batch_size,
                                              const WeightType* weights)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    if (weights->qkv.lora.r) {
        size_t sz = sizeof(T) * q_count * (local_q_kv_head_num * size_per_head_ + weights->qkv.lora.r);
        qkv_buf_  = (T*)allocator_->reMalloc(qkv_buf_, sz, false);
    }
    else {
        qkv_buf_ =
            (T*)allocator_->reMalloc(qkv_buf_, sizeof(T) * q_count * local_q_kv_head_num * size_per_head_, false);
    }

    qkv_buf_3_ = (T*)allocator_->reMalloc(qkv_buf_3_, sizeof(T) * q_count * local_head_num_ * size_per_head_, false);

    // Pad the tmp buffer for linear KV cache by `MAX_CTA_S` to avoid illegal accesses
    tmp_kv_buf_ = (T*)allocator_->reMalloc(
        tmp_kv_buf_, sizeof(T) * local_kv_head_num_ * 2 * (k_count + MAX_CTA_S) * size_per_head_, false);

    is_allocate_buffer_ = true;
}

template<typename T>
void UnifiedAttentionLayer<T>::allocateWorkspace()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(!is_allocate_workspace_);
    partial_M_ = (float*)allocator_->malloc(sizeof(float) * kMaxWorkspaceTokens * local_head_num_);
    partial_L_ = (float*)allocator_->malloc(sizeof(float) * kMaxWorkspaceTokens * local_head_num_);
    partial_O_ = (float*)allocator_->malloc(sizeof(float) * kMaxWorkspaceTokens * local_head_num_ * size_per_head_);
    split_cnt_ = (int*)allocator_->malloc(sizeof(int) * kMaxWorkspaceTokens);
    barriers_  = (int*)allocator_->malloc(sizeof(int) * kMaxWorkspaceTokens * local_head_num_, true, false);
    is_allocate_workspace_ = true;
}

template<typename T>
void UnifiedAttentionLayer<T>::freeWorkspace()
{
    if (is_allocate_workspace_) {
        TM_LOG_DEBUG(__PRETTY_FUNCTION__);

        allocator_->free((void**)&partial_M_);
        allocator_->free((void**)&partial_L_);
        allocator_->free((void**)&partial_O_);
        allocator_->free((void**)&split_cnt_);
        allocator_->free((void**)&barriers_);

        is_allocate_workspace_ = false;
    }
}

template<typename T>
void UnifiedAttentionLayer<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        TM_LOG_DEBUG(__PRETTY_FUNCTION__);

        allocator_->free((void**)&qkv_buf_);
        allocator_->free((void**)&qkv_buf_3_);
        allocator_->free((void**)&tmp_kv_buf_);

        is_allocate_buffer_ = false;
    }
}

template<typename T>
inline void UnifiedAttentionLayer<T>::forward(TensorMap* outputs, const TensorMap* inputs, const WeightType* weights)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    /**
     * input_tensors:
     *   \param input_query [token_num, hidden_dim]
     *   \param cu_q_len [batch_size+1], int
     *   \param cu_k_len [batch_size+1], int
     *   \param cu_block_counts [batch_size+1], int
     *   \param finished [batch_size], bool
     *   \param rope_theta [batch_size], float
     *   \param h_q_len [batch_size], int on cpu
     *   \param h_k_len [batch_size], int on cpu
     *   \param h_cu_q_len [batch_size+1], int on cpu
     *   \param h_cu_k_len [batch_size+1], int on cpu
     *   \param dc_batch_size [1], int on cpu
     *   \param pf_batch_size [1], int on cpu
     *   \param layer_id [1], int on cpu
     *
     * output_tensors:
     *   \param hidden_features [token_num, hidden_dim], float
     *   \param block_ptrs [total_block_counts], void*
     */

    /////////////////////////////////////////////
    /// parse inputs
    const int token_num = inputs->at("input_query").shape[0];
    const int layer_id  = inputs->getVal<int>("layer_id");

    const int dc_batch_size = inputs->getVal<int>("dc_batch_size");
    const int pf_batch_size = inputs->getVal<int>("pf_batch_size");
    const int batch_size    = dc_batch_size + pf_batch_size;

    int* h_q_len    = inputs->getPtr<int>("h_q_len");
    int* h_k_len    = inputs->getPtr<int>("h_k_len");
    int* cu_q_len   = inputs->getPtr<int>("cu_q_len");
    int* cu_k_len   = inputs->getPtr<int>("cu_k_len");
    int* h_cu_q_len = inputs->getPtr<int>("h_cu_q_len");
    int* h_cu_k_len = inputs->getPtr<int>("h_cu_k_len");

    bool*  is_finished = inputs->getPtr<bool>("finished");
    float* rope_theta  = inputs->getPtr<float>("rope_theta");

    void** block_ptrs     = outputs->getPtr<void*>("block_ptrs");
    int*   cu_block_count = inputs->getPtr<int>("cu_block_counts");

    T* attention_input = inputs->getPtr<T>("input_query");
    T* attention_out   = outputs->getPtr<T>("hidden_features");

    /////////////////////////////////////////////
    /// allocate buffers
    allocateBuffer(token_num,                                           // shared
                   h_cu_k_len[batch_size] - h_cu_k_len[dc_batch_size],  // prefill
                   batch_size,
                   weights);

    // [L, 2, H, s, D]
    const size_t layer_offset = layer_id * 2 * local_kv_head_num_ * kv_cache_block_len_ * size_per_head_;

    // static int count = 0;

    // if (layer_id == 0 && count == 0) {
    //     Compare(attention_input, num_token * weights->qkv.input_dims, "qkv_input", kCmpRead, stream_);
    // }

    int* lora_mask = inputs->at("lora_mask", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<int>();
    //////////////////////////////////////////////
    /// qkv gemm
    // [token_num, hidden_dim] -> [token_num, 3, local_hidden_dim]
    linear_.forward(qkv_buf_, attention_input, token_num, weights->qkv, LlamaLinear<T>::kGemm, lora_mask);

    count_and_fix(qkv_buf_, token_num * weights->qkv.output_dims, Concat("qkv", layer_id), 3);

    // if (layer_id == 0 && count == 0) {
    //     Compare(qkv_buf_, num_token * weights->qkv.output_dims, "qkv_buf", kCmpRead, stream_);
    // }

    auto stream_ptr = streams_.data();

    auto CreateParams = [&](int offset, int batch_size, int max_kv_splits, cudaStream_t stream) {
        AttentionParams<T> params{};

        // Batch offset for `out` and `q` are computed inside the kernel
        params.out = qkv_buf_3_;

        params.q      = (T*)qkv_buf_;
        params.k      = params.q + local_head_num_ * size_per_head_;
        params.v      = params.k + local_kv_head_num_ * size_per_head_;
        params.stride = (local_head_num_ + 2 * local_kv_head_num_) * size_per_head_;

        if (weights->qkv.bias) {
            params.q_bias = weights->qkv.bias;
            params.k_bias = params.q_bias + local_head_num_ * size_per_head_;
            params.v_bias = params.k_bias + local_kv_head_num_ * size_per_head_;
        }

        params.token_num  = h_cu_q_len[offset + batch_size] - h_cu_q_len[offset];
        params.batch_size = batch_size;
        params.max_q_len  = *std::max_element(h_q_len + offset, h_q_len + offset + batch_size);
        params.max_k_len  = *std::max_element(h_k_len + offset, h_k_len + offset + batch_size);

        // Decoding use only
        params.block_iter_params = BlockIteratorParams{(char**)block_ptrs,  //
                                                       (int*)cu_block_count + offset,
                                                       layer_id,
                                                       (int)kv_cache_block_len_};

        // Prefilling use only
        const int sum_k_len       = h_cu_k_len[offset + pf_batch_size] - h_cu_k_len[offset];
        params.linear_iter_params = LinearIteratorParams{tmp_kv_buf_,  //
                                                         int(2 * sum_k_len * size_per_head_),
                                                         int(sum_k_len * size_per_head_)};

        params.finished   = is_finished + offset;
        params.cu_q_len   = cu_q_len + offset;
        params.cu_k_len   = cu_k_len + offset;
        params.rope_theta = rope_theta + offset;

        params.num_heads     = local_head_num_;
        params.num_kv_heads  = local_kv_head_num_;
        params.size_per_head = size_per_head_;
        // MSVC does not have M_LOG2E
        params.inv_sqrt_dh = (float)std::log2(expf(1.)) / std::sqrt((float)params.size_per_head);

        params.rotary_embedding_dim    = size_per_head_;
        params.rotary_embedding_base   = params_.rotary_embedding_base;
        params.max_position_embeddings = params_.max_position_embeddings;
        params.rope_ti_scale           = 1.f;
        if (!params_.use_dynamic_ntk && params_.rope_scaling_factor) {
            params.rope_ti_scale /= params_.rope_scaling_factor;
        }

        params.use_logn_attn = params_.use_logn_attn;

        // Decoding use only for now
        FT_CHECK(barriers_);
        params.split_cnt   = split_cnt_;
        params.partial_L   = partial_L_;
        params.partial_M   = partial_M_;
        params.partial_O   = partial_O_;
        params.locks       = barriers_;
        params.max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), max_kv_splits);

        params.arch   = arch_;
        params.stream = stream;

        params.quant_policy = quant_policy_;

        return params;
    };

    cudaStream_t pf_stream = stream_;
    cudaStream_t dc_stream = stream_;

    if (pf_batch_size && dc_batch_size) {
        pf_stream = aux_stream_;
        check_cuda_error(cudaEventRecord(qkv_event_, stream_));
        check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));
    }

    if (pf_batch_size) {
        const int offset    = dc_batch_size;
        const int sum_k_len = h_cu_k_len[offset + pf_batch_size] - h_cu_k_len[offset];
        // We are executing prefill & decoding kernels concurrently, but only have 1 workspace
        // disable split kv for prefill for now
        auto params = CreateParams(offset, pf_batch_size, 1, pf_stream);
        if constexpr (sizeof(T) == 2) {
            invokeProcessKV_v2_(params);
            /// TODO: skip flattening for `sm_80`
            invokeFlattenKV_v2_(params, sum_k_len);
            dispatchAttention(params);
        }
    }

    if (dc_batch_size) {
        auto params = CreateParams(0, dc_batch_size, kMaxKVSplits, dc_stream);
        if constexpr (sizeof(T) == 2) {
            dispatchDecoding<T>(params);
        }
    }

    if (pf_batch_size && dc_batch_size) {
        check_cuda_error(cudaEventRecord(aux_event_, aux_stream_));
        check_cuda_error(cudaStreamWaitEvent(stream_, aux_event_));
    }

    // if (layer_id == 0 && count == 0) {
    //     Compare(qkv_buf_3_, num_token * weights->output.input_dims, "qkv_buf_3", kCmpRead, stream_);

    //     dump(qkv_buf_3_, num_token * weights->output.input_dims, stream_, "qkv_buf_3");
    // }

    count_and_fix(qkv_buf_3_, token_num * weights->output.input_dims, Concat("attn", layer_id), 3);

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>
    linear_.forward(attention_out, qkv_buf_3_, token_num, weights->output, LlamaLinear<T>::kGemm, lora_mask);

    // ++count;

    count_and_fix(attention_out, token_num * weights->output.output_dims, Concat("wo", layer_id), 3);

    if (tensor_para_.world_size_ > 1) {
        NcclGuard nccl_guard(tensor_para_, stream_);
        ftNcclAllReduceSum(attention_out, attention_out, token_num * hidden_units_, tensor_para_, stream_);
        sync_check_cuda_error();
    }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();
}

#ifdef ENABLE_FP32
template class UnifiedAttentionLayer<float>;
#endif
template class UnifiedAttentionLayer<half>;
#ifdef ENABLE_BF16
template class UnifiedAttentionLayer<__nv_bfloat16>;
#endif  // ENABLE_BF16

}  // namespace turbomind
