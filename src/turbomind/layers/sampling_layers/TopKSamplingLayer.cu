/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <float.h>
#include <sstream>

#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"
#include "src/turbomind/layers/sampling_layers/TopKSamplingLayer.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

template<uint TOP_K_MAX>
__global__ void setup_topk_runtime_args(int    batch_size,
                                        uint   top_k,
                                        uint*  top_ks,
                                        int    top_ks_size,
                                        float  top_p,
                                        float* top_ps,
                                        int    top_ps_size,
                                        bool*  skip_decode)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < batch_size; i += gridDim.x * blockDim.x) {
        uint  k = top_ks_size > 1 ? top_ks[i] : top_k;
        float p = top_ps_size > 1 ? top_ps[i] : top_p;
        if (k == 0 && p == 0.0f) {
            // FT's topp implementation does not support topp = 0.0f, but it equivalent to greedy search.
            // So, we set the topk = 1 as an alternative solution.
            k = 1;
        }
        if (k > 0 && p == 0.0f) {
            // for compatibility <= FT5.0.
            // This case corresponds to the old topk sampling, which is equivalent to
            // the old topk_topp sampling with topp=1.0f. TopKSamplingLayer and
            // TopKTopPSamplingLayer are now merged by TopKSamplingLayer. Thus, we
            // replace the case topk>0 and topp=0.0f by topk>0 and topp=1.0f for the
            // compatibility.
            p = 1.0f;
        }
        // Clip k value. A topk sampling kernel supports up to TOP_K_MAX=64.
        top_ks[i] = k > TOP_K_MAX ? TOP_K_MAX : k;
        if (k > TOP_K_MAX) {
            printf("[WARNING] topk (%d) is larger than max supported number (%d) for token %d"
                   " clip to max supported number %d. \n",
                   k,
                   TOP_K_MAX,
                   i,
                   top_ks[i]);
        }
        // Clip p value if it is out of range. range = [0.0, 1.0].
        top_ps[i] = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        if (p < 0.0f || p > 1.0f) {
            printf("[WARNING] topp (%f) is out of range ([0.0, 1.0f]) for token %d"
                   " clip to closest number %f.\n",
                   p,
                   i,
                   top_ps[i]);
        }
        skip_decode[i] = k == 0;
    }
}

template<typename T>
void TopKSamplingLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void TopKSamplingLayer<T>::allocateBuffer(size_t batch_size, Tensor top_k, Tensor top_p)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::allocateBuffer(batch_size, top_k, top_p);
    uint max_top_k = top_k.size() > 0 ? top_k.max<uint>() : 1;
    if (max_top_k == 0) {
        // for safety. TopKSamplingLayer handles a case of top_k=0 and top_p=0 as
        // a greedy decode, i.e. top_k=1, although such case has max_top_k=0.
        max_top_k = 1;
    }
    invokeTopKSampling<T>(nullptr,
                          sampling_workspace_size_,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          nullptr,
                          max_top_k,
                          1.0f,
                          vocab_size_padded_,
                          nullptr,
                          stream_,
                          batch_size,
                          skip_decode_buf_);
    sampling_workspace_ = allocator_->reMalloc(sampling_workspace_, sampling_workspace_size_, false);
    runtime_top_k_buf_ =
        reinterpret_cast<uint*>(allocator_->reMalloc(runtime_top_k_buf_, sizeof(uint) * batch_size, false));
    runtime_top_p_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(runtime_top_p_buf_, sizeof(float) * batch_size, false));
    is_allocate_buffer_ = true;
}

template<typename T>
void TopKSamplingLayer<T>::freeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)(&sampling_workspace_));
        allocator_->free((void**)(&runtime_top_k_buf_));
        allocator_->free((void**)(&runtime_top_p_buf_));
    }
    BaseSamplingLayer<T>::freeBuffer();
    is_allocate_buffer_ = false;
}

template<typename T>
void TopKSamplingLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args)
{
    // Setup runtime topk and topp arguments.
    //
    // runtime_args:
    //     runtime_top_k [1] or [batch_size] on cpu, optional, uint.
    //     runtime_top_p [1] or [batch_size] on cpu, optional, float.
    //     temperature [1] or [batch_size] on cpu, optional
    //     repetition_penalty [1] or [batch_size] on cpu, optional
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::setup(batch_size, beam_width, runtime_args);

    uint         tmp_top_k     = 0;
    const Tensor runtime_top_k = runtime_args->isExist("runtime_top_k") ?
                                     runtime_args->at("runtime_top_k") :
                                     Tensor(MEMORY_CPU, TYPE_UINT32, {1}, &tmp_top_k);
    const Tensor runtime_top_p = runtime_args->isExist("runtime_top_p") ? runtime_args->at("runtime_top_p") : Tensor();
    const size_t runtime_top_k_size = runtime_top_k.size();
    const size_t runtime_top_p_size = runtime_top_p.size();

    uint  top_k = runtime_top_k.max<uint>();
    float top_p = runtime_top_p_size == 0 ? 0.0f : runtime_top_p.getVal<float>();

    if (runtime_top_k_size > 1) {
        FT_CHECK_WITH_INFO(
            runtime_top_k.size() == batch_size,
            fmtstr("runtime_top_k.size() (%d) == batch_size (%d) is not satisfied!", runtime_top_k.size(), batch_size));
        cudaAutoCpy(runtime_top_k_buf_, runtime_top_k.getPtr<uint>(), batch_size, stream_);
    }
    if (runtime_top_p_size > 1) {
        FT_CHECK_WITH_INFO(
            runtime_top_p.size() == batch_size,
            fmtstr("runtime_top_p.size() (%d) == batch_size (%d) is not satisfied!", runtime_top_p.size(), batch_size));
        cudaAutoCpy(runtime_top_p_buf_, runtime_top_p.getPtr<float>(), batch_size, stream_);
    }

    dim3 block(std::min((int)batch_size, 256));
    dim3 grid(div_up((int)batch_size, (int)block.x));
    // support top_k up to 1024.
    setup_topk_runtime_args<1024><<<grid, block, 0, stream_>>>(batch_size,
                                                               top_k,
                                                               runtime_top_k_buf_,
                                                               runtime_top_k_size,
                                                               top_p,
                                                               runtime_top_p_buf_,
                                                               runtime_top_p_size,
                                                               skip_decode_buf_);
    cudaAutoCpy(skip_decode_, skip_decode_buf_, batch_size, stream_);
    uint* runtime_top_ks = new uint[batch_size];
    cudaAutoCpy(runtime_top_ks, runtime_top_k_buf_, batch_size, stream_);
    runtime_max_top_k_ = static_cast<int>(*std::max_element(runtime_top_ks, runtime_top_ks + batch_size));
    {
    }
    delete[] runtime_top_ks;
}

template<typename T>
void TopKSamplingLayer<T>::runSampling(TensorMap* output_tensors, TensorMap* input_tensors)
{
    // input_tensors:
    //      logits [local_batch_size, vocab_size_padded]
    //      embedding_bias [vocab_size_padded], optional
    //      step [1] on cpu
    //      max_input_length [1] on cpu
    //      input_lengths [local_batch_size], optional
    //      ite [1] on cpu

    // output_tensors:
    //      output_ids [max_seq_len, batch_size]
    //      curand_state [local_batch_size]
    //      finished [local_batch_size], optional
    //      sequence_length [local_batch_size], optional
    //      cum_log_probs [batch_size], must be float*, optional
    //          The cumultative log probability of generated tokens.
    //      output_log_probs [local_batch_size], must be float*, optional
    //          The log probs at the current step.

    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(input_tensors->size() >= 4);
    FT_CHECK(output_tensors->size() >= 1);

    const int batch_size       = output_tensors->at("output_ids").shape[1];
    const int local_batch_size = input_tensors->at("logits").shape[0];
    const int ite              = input_tensors->at("ite").getVal<int>();
    const int step             = input_tensors->at("step").getVal<int>();

    // in case of skip any, the logit value is already copied and processed.
    T* logits = !skip_any_ ? input_tensors->at("logits").getPtr<T>() : runtime_logits_buf_;

    invokeAddBiasEndMask(logits,
                         (T*)(nullptr),
                         input_tensors->at("end_id").getPtr<const int>(),
                         output_tensors->at("finished", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<bool>(),
                         local_batch_size,
                         vocab_size_,
                         vocab_size_padded_,
                         stream_);
    sync_check_cuda_error();

    float* cum_log_probs =
        output_tensors->isExist("cum_log_probs") ? output_tensors->at("cum_log_probs").getPtr<float>() : nullptr;
    float* output_log_probs =
        output_tensors->isExist("output_log_probs") ? output_tensors->at("output_log_probs").getPtr<float>() : nullptr;

    if (cum_log_probs != nullptr || output_log_probs != nullptr) {
        invokeAddBiasSoftMax(
            logits,
            (T*)(nullptr),
            input_tensors->at("end_id").getPtr<int>(),
            output_tensors->at("finished", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<bool>(),
            local_batch_size,
            vocab_size_padded_,
            vocab_size_,
            stream_);
        sync_check_cuda_error();
    }

    float* sampled_logprobs =
        output_tensors->isExist("sampled_logprobs") ? output_tensors->at("sampled_logprobs").getPtr<float>() : nullptr;
    uint32_t* sampled_indexes =
        output_tensors->isExist("sampled_indexes") ? output_tensors->at("sampled_indexes").getPtr<uint32_t>() : nullptr;
    uint32_t* sampled_nums =
        output_tensors->isExist("sampled_nums") ? output_tensors->at("sampled_nums").getPtr<uint32_t>() : nullptr;

    invokeBatchTopKSampling(
        sampling_workspace_,
        sampling_workspace_size_,
        logits,
        output_tensors->at("output_ids").getPtrWithOffset<int>(step * batch_size + ite * local_batch_size),
        output_tensors->at("sequence_length", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<int>(),
        output_tensors->at("finished", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<bool>(),
        cum_log_probs,
        output_log_probs,
        sampled_logprobs,
        sampled_indexes,
        sampled_nums,
        output_tensors->at("curand_state").getPtr<curandState_t>() + ite * local_batch_size,
        (int)runtime_max_top_k_,  // useless because runtime_top_k_buf_ is never nullptr. Keep for legacy.
        (int*)(runtime_top_k_buf_ + ite * local_batch_size),
        1.0f,  // useless because runtime_top_p_buf_ is never nullptr. Keep for legacy.
        runtime_top_p_buf_ + ite * local_batch_size,
        vocab_size_padded_,
        input_tensors->at("end_id").getPtr<int>(),
        stream_,
        local_batch_size,
        skip_decode_buf_ + ite * local_batch_size);
    sync_check_cuda_error();
}

template<typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(size_t             max_batch_size,
                                        size_t             vocab_size,
                                        size_t             vocab_size_padded,
                                        int                end_id,
                                        size_t             top_k,
                                        unsigned long long random_seed,
                                        float              temperature,
                                        float              len_penalty,
                                        float              repetition_penalty,
                                        cudaStream_t       stream,
                                        cublasMMWrapper*   cublas_wrapper,
                                        IAllocator*        allocator,
                                        bool               is_free_buffer_after_forward):
    BaseSamplingLayer<T>(max_batch_size,
                         vocab_size,
                         vocab_size_padded,
                         end_id,
                         top_k,
                         0.0f,
                         random_seed,
                         temperature,
                         len_penalty,
                         repetition_penalty,
                         stream,
                         cublas_wrapper,
                         allocator,
                         is_free_buffer_after_forward,
                         nullptr)
{
}

template<typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(TopKSamplingLayer<T> const& top_k_sampling_layer):
    BaseSamplingLayer<T>(top_k_sampling_layer)
{
}

template<typename T>
TopKSamplingLayer<T>::~TopKSamplingLayer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    freeBuffer();
}

template class TopKSamplingLayer<float>;
// template class TopKSamplingLayer<half>;

}  // namespace turbomind
