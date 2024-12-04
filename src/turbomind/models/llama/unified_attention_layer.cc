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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.cc

#include <algorithm>
#include <math.h>

#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/mla_utils.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<class T>
UnifiedAttentionLayer<T>::UnifiedAttentionLayer(const ModelParam&     model,
                                                const AttentionParam& attn,
                                                const LoraParam&      lora,
                                                const NcclParam&      tp,
                                                const Context<T>&     ctx):
    head_num_(model.head_num),
    kv_head_num_(model.kv_head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    local_head_num_(head_num_ / tp.world_size_),
    local_kv_head_num_(model.kv_head_num / tp.world_size_),
    param_(attn),
    model_param_(model),
    lora_param_(lora),
    tensor_para_(tp),
    context_(ctx),
    stream_(ctx.stream),
    linear_(ctx.linear.get()),
    allocator_(ctx.allocator.get()),
    arch_(getSMVersion())
{
    FT_CHECK(head_num_ % kv_head_num_ == 0);

    check_cuda_error(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    check_cuda_error(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    check_cuda_error(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    streams_[0] = stream_;
    streams_[1] = aux_stream_;

    allocateWorkspace();
}

template<typename T>
void UnifiedAttentionLayer<T>::allocateBuffer(size_t q_count, size_t k_count, size_t batch_size, size_t qkv_lora_rank)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const int local_q_kv_head_num = local_head_num_ + 2 * local_kv_head_num_;

    if (qkv_lora_rank) {
        size_t sz = sizeof(T) * q_count * (local_q_kv_head_num * size_per_head_ + qkv_lora_rank);
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
                   weights->qkv.lora.r);

    // [L, 2, H, s, D]
    const size_t layer_offset = layer_id * 2 * local_kv_head_num_ * param_.cache_block_seq_len * size_per_head_;

    // static int count = 0;

    // if (tensor_para_.rank_ == 0) {
    //     Compare(attention_input, token_num * hidden_units_, Concat("qkv_input", layer_id), compare_mode, stream_);
    // }

    int* lora_mask = inputs->at("lora_mask", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<int>();

    if (weights->qkv.output_dims) {
        //////////////////////////////////////////////
        /// qkv gemm
        // [token_num, hidden_dim] -> [token_num, 3, local_hidden_dim]
        linear_->forward(qkv_buf_, attention_input, token_num, weights->qkv, LlamaLinear<T>::kGemm, lora_mask);
        sync_check_cuda_error();
    }
    else {
        forward_mla(attention_input, token_num, *weights);
    }

    // std::cerr << layer_id << " " << count << " " << tensor_para_.rank_ << "\n";

    count_and_fix(qkv_buf_, token_num * weights->qkv.output_dims, Concat("qkv", layer_id), 3);

    // std::cerr << "token num: " << token_num << "\n";

    // if (layer_id == 0 && count == 0 && tensor_para_.rank_ == 0) {
    //     Compare(qkv_buf_, token_num * (3 * local_head_num_ * size_per_head_), "qkv_buf", CMP_MODE, stream_);
    // }

    if constexpr (0) {
        std::vector<T> tmp(token_num * weights->qkv.output_dims);
        cudaMemcpyAsync(tmp.data(), qkv_buf_, sizeof(T) * tmp.size(), cudaMemcpyDefault, stream_);
        cudaStreamSynchronize(stream_);
        int i = 0;
        for (auto& x : tmp) {
            std::cout << (float)x << " ";
            if (++i == 256) {
                break;
            }
        }
        std::cout << "\n";
        i = 0;
        for (auto it = tmp.rbegin(); it != tmp.rend(); ++it) {
            std::cout << (float)*it << " ";
            if (++i == 256) {
                break;
            }
        }
        std::cout << "\n";
    }

    // FT_CHECK(0);

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
                                                       (int)param_.cache_block_seq_len};

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
        params.inv_sqrt_dh = (float)std::log2(expf(1.));
        if (param_.softmax_scale) {  // model predefined softmax scale
            params.inv_sqrt_dh *= param_.softmax_scale;
        }
        else {  // default value
            params.inv_sqrt_dh /= std::sqrt((float)params.size_per_head);
        }

        params.rotary_embedding_dim    = param_.rotary_embedding_dim;
        params.rotary_embedding_base   = param_.rotary_embedding_base;
        params.max_position_embeddings = param_.max_position_embeddings;
        params.rope_scaling_factor     = param_.rope_scaling_factor;
        params.attention_scaling       = 1.0;
        params.rope_ti_scale           = 1.f;
        if (param_.rope_scaling_type == "linear") {
            params.rope_ti_scale /= param_.rope_scaling_factor;
        }
        if (param_.rope_scaling_type == "llama3") {
            const double PI                   = 3.14159265358979323846;
            float        inv_diff_freq_factor = 1.0 / (param_.high_freq_factor - param_.low_freq_factor);
            params.llama3_inv_scaling_factor  = 1.0 / param_.rope_scaling_factor;
            params.llama3_alpha = param_.original_max_position_embeddings / (2 * PI) * inv_diff_freq_factor;
            params.llama3_beta  = param_.low_freq_factor * inv_diff_freq_factor;
        }
        if (param_.rope_scaling_type == "yarn") {
            const double PI                  = 3.14159265358979323846;
            auto         find_correction_dim = [&](float num_rotations) {
                return (param_.rotary_embedding_dim
                        * std::log(param_.max_position_embeddings / (num_rotations * 2 * PI)))
                       / (2 * std::log(param_.rotary_embedding_base));
            };
            auto find_correction_range = [&](float low_rot, float high_rot, float& low, float& high) {
                low  = std::floor(find_correction_dim(low_rot));
                high = std::ceil(find_correction_dim(high_rot));
                low  = std::max(low, 0.f);
                high = std::min(high, param_.rotary_embedding_dim - 1.f);
            };
            float low, high;
            find_correction_range(param_.beta_fast, param_.beta_slow, low, high);
            // https://github.com/huggingface/transformers/blob/6c3f168b36882f0beebaa9121eafa1928ba29633/src/transformers/modeling_rope_utils.py#L216
            if (low == high) {
                high += 0.001f;
            }
            params.yarn_ramp_inv_factor_div_2   = 1.0 / (high - low) / 2.0;
            params.yarn_ramp_inv_factor_mul_min = 1.0 / (high - low) * low;
            params.yarn_inv_scaling_factor      = (1 - 1.0 / param_.rope_scaling_factor);
            if (param_.attention_factor < 0) {
                params.attention_scaling = 0.1 * std::log(param_.rope_scaling_factor) + 1.0;
            }
            else {
                params.attention_scaling = param_.attention_factor;
            }
        }

        params.use_logn_attn = param_.use_logn_attn;

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

        params.quant_policy = model_param_.quant_policy;
        return params;
    };

    cudaStream_t pf_stream = stream_;
    cudaStream_t dc_stream = stream_;

    if (pf_batch_size && dc_batch_size) {
        pf_stream = aux_stream_;
        check_cuda_error(cudaEventRecord(qkv_event_, stream_));
        check_cuda_error(cudaStreamWaitEvent(aux_stream_, qkv_event_));
    }

    if (pf_batch_size && !isTuning()) {
        const int offset    = dc_batch_size;
        const int sum_k_len = h_cu_k_len[offset + pf_batch_size] - h_cu_k_len[offset];
        // We are executing prefill & decoding kernels concurrently, but only have 1 workspace
        // disable split kv for prefill for now
        auto params = CreateParams(offset, pf_batch_size, 1, pf_stream);
        if constexpr (sizeof(T) == 2) {
            invokeProcessKV_v2_(params);
            sync_check_cuda_error();

            /// TODO: skip flattening for `sm_80`
            invokeFlattenKV_v2_(params, sum_k_len);
            sync_check_cuda_error();

            dispatchAttention(params);
            sync_check_cuda_error();
        }
    }

    if (dc_batch_size && !isTuning()) {
        auto params = CreateParams(0, dc_batch_size, kMaxKVSplits, dc_stream);
        if constexpr (sizeof(T) == 2) {
            dispatchDecoding<T>(params);
            sync_check_cuda_error();
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

    if (isTuning()) {
        rng_.set_stream(stream_);
        rng_.GenerateUniform(qkv_buf_3_, token_num * weights->output.input_dims, .02f, -.01f);
    }

    count_and_fix(qkv_buf_3_, token_num * weights->output.input_dims, Concat("attn", layer_id), 3);

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>
    linear_->forward(attention_out, qkv_buf_3_, token_num, weights->output, LlamaLinear<T>::kGemm, lora_mask);
    sync_check_cuda_error();

    count_and_fix(attention_out, token_num * weights->output.output_dims, Concat("wo", layer_id), 3);

    if (tensor_para_.world_size_ > 1) {
        NcclGuard nccl_guard(tensor_para_, stream_);
        ftNcclAllReduceSum(attention_out, attention_out, token_num * hidden_units_, tensor_para_, stream_);
        sync_check_cuda_error();
    }

    // if (tensor_para_.rank_ == 0) {
    //     Compare(attention_out, token_num * hidden_units_, Concat("attn_out", layer_id), compare_mode, stream_);
    //     // dump(qkv_buf_3_, num_token * weights->output.input_dims, stream_, "qkv_buf_3");
    // }

    if (is_free_buffer_after_forward_ == true) {
        freeBuffer();
    }
    sync_check_cuda_error();

    // ++count;
}

template<typename T>
void UnifiedAttentionLayer<T>::forward_mla(const T* inputs, int token_num, const WeightType& w)
{
    const int q_lora_rank  = w.q_a_proj.output_dims;
    const int kv_lora_rank = w.kv_b_proj.input_dims;
    const int qk_rope_dim  = w.kv_a_proj.output_dims - kv_lora_rank;
    const int qk_nope_dim  = std::max(w.q_b_proj.output_dims, w.q_proj.output_dims) / local_head_num_ - qk_rope_dim;
    const int v_head_dim   = w.kv_b_proj.output_dims / local_head_num_ - qk_nope_dim;

    T* q{};

    if (w.q_proj.kernel) {
        deviceMalloc((T**)&q, (size_t)token_num * w.q_proj.output_dims, stream_);
        linear_->forward(q, inputs, token_num, w.q_proj);
        sync_check_cuda_error();
    }
    else {
        T* q_a{};
        deviceMalloc((T**)&q_a, (size_t)token_num * q_lora_rank, stream_);

        linear_->forward(q_a, inputs, token_num, w.q_a_proj);
        sync_check_cuda_error();

        invokeRMSNorm(q_a,
                      q_lora_rank,
                      q_a,
                      q_lora_rank,
                      w.q_a_layernorm,
                      q_lora_rank,
                      token_num,
                      model_param_.norm_eps,
                      stream_);
        sync_check_cuda_error();

        deviceMalloc((T**)&q, (size_t)token_num * w.q_b_proj.output_dims, stream_);
        linear_->forward(q, q_a, token_num, w.q_b_proj);
        sync_check_cuda_error();

        deviceFree(q_a, stream_);
    }

    T*        kv_a{};
    const int kv_a_dim = w.kv_a_proj.output_dims;
    deviceMalloc((T**)&kv_a, (size_t)token_num * kv_a_dim, stream_);

    linear_->forward(kv_a, inputs, token_num, w.kv_a_proj);
    sync_check_cuda_error();

    invokeRMSNorm(
        kv_a, kv_a_dim, kv_a, kv_a_dim, w.kv_a_layernorm, kv_lora_rank, token_num, model_param_.norm_eps, stream_);
    sync_check_cuda_error();

    T* kv_b{};
    deviceMalloc((T**)&kv_b, (size_t)token_num * w.kv_b_proj.output_dims, stream_);
    sync_check_cuda_error();

    linear_->forward(kv_b, {kv_a, kv_a_dim}, token_num, w.kv_b_proj);
    sync_check_cuda_error();

    dispatchMLACopyQKV(qkv_buf_,
                       q,
                       kv_a,
                       kv_b,
                       token_num,
                       local_head_num_,
                       qk_nope_dim,
                       qk_rope_dim,
                       kv_lora_rank,
                       v_head_dim,
                       stream_);
    sync_check_cuda_error();

    deviceFree(q, stream_);
    deviceFree(kv_a, stream_);
    deviceFree(kv_b, stream_);
}

#ifdef ENABLE_FP32
template class UnifiedAttentionLayer<float>;
#endif
template class UnifiedAttentionLayer<half>;
#ifdef ENABLE_BF16
template class UnifiedAttentionLayer<__nv_bfloat16>;
#endif  // ENABLE_BF16

}  // namespace turbomind
