/*
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

#include "src/turbomind/layers/sampling_layers/SamplingLayer.h"
#include "src/turbomind/kernels/sampling_kernels.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"
#include "src/turbomind/utils/memory_utils.h"

namespace turbomind {

void set_runtime_args(int    batch_size,
                      int    top_k,
                      int*   top_ks,
                      int    top_ks_size,
                      int*   runtime_top_k,
                      float  top_p,
                      float* top_ps,
                      int    top_ps_size,
                      float* runtime_top_p,
                      float  min_p,
                      float* min_ps,
                      int    min_ps_size,
                      float* runtime_min_p)
{
    for (int i = 0; i < batch_size; i++) {
        int   topk = top_ks_size > 1 ? top_ks[i] : top_k;
        float topp = top_ps_size > 1 ? top_ps[i] : top_p;
        float minp = min_ps_size > 1 ? min_ps[i] : min_p;

        if (topk == 0 && topp == 0.f) {
            topk = 1;
        }

        if (topk < 0 || topk > 1024) {
            TM_LOG_WARNING("topk (%d) is out of range [0, 1024]", topk);
            topk = std::max(0, std::min(topk, 1024));
        }
        if (topp < 0.f || topp > 1.f) {
            TM_LOG_WARNING("topp (%f) is out of range [0.0, 1.0f]", topp);
            topp = std::max(0.f, std::min(topp, 1.f));
        }
        if (minp < 0.f || minp > 1.f) {
            TM_LOG_WARNING("minp (%f) is out of range [0.0, 1.0f]", minp);
            minp = std::max(0.f, std::min(minp, 1.f));
        }
        runtime_top_k[i] = topk;
        runtime_top_p[i] = topp;
        runtime_min_p[i] = minp;
    }
}

template<typename T>
void SamplingLayer<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void SamplingLayer<T>::allocateBuffer(const size_t batch_size)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    runtime_top_k_buf_ =
        reinterpret_cast<int*>(allocator_->reMalloc(runtime_top_k_buf_, sizeof(int) * batch_size, false));
    runtime_top_p_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(runtime_top_p_buf_, sizeof(float) * batch_size, false));
    runtime_min_p_buf_ =
        reinterpret_cast<float*>(allocator_->reMalloc(runtime_min_p_buf_, sizeof(float) * batch_size, false));

    indices_ = reinterpret_cast<int*>(
        allocator_->reMalloc(indices_, batch_size * sizeof(int) * args_.vocab_size_padded, false));
    kept_ = reinterpret_cast<int*>(allocator_->reMalloc(kept_, batch_size * sizeof(int), false));

    {
        // topk buffer
        TopKSortFilterParams params{};
        params.batch_size = batch_size;
        params.max_top_k  = max_topk_;
        invokeTopKSortFilter<T>(params, stream_);
        topk_ws_size_ = params.workspace_size;
        topk_ws_      = allocator_->reMalloc(topk_ws_, topk_ws_size_, false);
    }

    {
        // topp buffer
        TopPSortParams params{};
        params.batch_size        = batch_size;
        params.vocab_size        = args_.vocab_size;
        params.vocab_size_padded = args_.vocab_size_padded;
        invokeTopPSort<T>(params, stream_);
        topp_ws_size_ = params.workspace_size;
        topp_ws_      = allocator_->reMalloc(topp_ws_, topp_ws_size_, false);
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void SamplingLayer<T>::freeBuffer()
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    kept_n_        = {};
    runtime_top_k_ = {};
    runtime_top_p_ = {};
    runtime_min_p_ = {};

    allocator_->free((void**)&runtime_top_k_buf_);
    allocator_->free((void**)&runtime_top_p_buf_);
    allocator_->free((void**)&runtime_min_p_buf_);
    allocator_->free((void**)&topk_ws_);
    allocator_->free((void**)&topp_ws_);

    allocator_->free((void**)&indices_);
    allocator_->free((void**)&kept_);
    logits_ = nullptr;

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
SamplingLayer<T>::~SamplingLayer()
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    freeBuffer();

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void SamplingLayer<T>::forward(TensorMap* output_tensors, TensorMap* input_tensors)
{
    // step1:
    //  - use topk / topp_minp kernel to sort and filter the scores
    //  - softmax the left score
    // step2:
    //  - sampling from left and sorted scores

    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    Tensor    logits     = input_tensors->at("logits");
    const int batch_size = logits.shape[0];
    const int step       = input_tensors->at("step").getVal<int>();
    logits_              = logits.getPtr<T>();

    cudaAutoCpy(kept_, kept_n_.data(), batch_size, stream_);

    // use topk sort if some request use topk filter
    if (max_topk_ > 0) {
        // TODO: top_k >= 64 is much slower than torch.topk()
        TopKSortFilterParams params{};
        params.workspace         = topk_ws_;
        params.workspace_size    = topk_ws_size_;
        params.logits            = logits_;
        params.sorted_logits     = logits_;
        params.sorted_indices    = indices_;
        params.kept              = kept_;
        params.top_ks            = runtime_top_k_buf_;
        params.max_top_k         = max_topk_;
        params.batch_size        = batch_size;
        params.vocab_size        = args_.vocab_size;
        params.vocab_size_padded = args_.vocab_size_padded;
        invokeTopKSortFilter<T>(params, stream_);
    }

    // use topp sort if some request skip topk filter
    if (min_topk_ == 0) {
        invokeSoftmax<T>(logits_, args_.vocab_size_padded, args_.vocab_size, batch_size, kept_, stream_);

        TopPSortParams params{};
        params.workspace         = topp_ws_;
        params.workspace_size    = topp_ws_size_;
        params.logits            = logits_;
        params.sorted_logits     = logits_;
        params.sorted_indices    = indices_;
        params.kept              = kept_;
        params.top_ks            = runtime_top_k_buf_;
        params.top_ps            = runtime_top_p_buf_;
        params.batch_size        = batch_size;
        params.vocab_size        = args_.vocab_size;
        params.vocab_size_padded = args_.vocab_size_padded;
        invokeTopPSort<T>(params, stream_);
    }

    // apply topp minp filter
    if (max_minp_ != 0.f || min_topp_ != 1.f) {
        TopPMinPFilterParams params{};
        params.sorted_logits     = logits_;
        params.sorted_indices    = indices_;
        params.kept              = kept_;
        params.top_ps            = runtime_top_p_buf_;
        params.min_ps            = runtime_min_p_buf_;
        params.batch_size        = batch_size;
        params.vocab_size        = args_.vocab_size;
        params.vocab_size_padded = args_.vocab_size_padded;
        invokeTopPMinPFilter<T>(params, stream_);
    }

    // sample
    {
        SamplingParams params{};
        params.logits      = logits.getPtr<T>();
        params.stride      = args_.vocab_size_padded;
        params.indices     = indices_;
        params.kept        = kept_;
        params.curandstate = output_tensors->at("curand_state").getPtr<curandState_t>();
        params.batch_size  = batch_size;
        params.output_ids  = output_tensors->at("output_ids").getPtrWithOffset<int>(step * batch_size);
        params.sequence_length =
            output_tensors->at("sequence_length", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<int>();
        params.sampled_logprobs =
            output_tensors->at("sampled_logprobs", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<float>();
        params.sampled_indexes =
            output_tensors->at("sampled_indexes", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<uint32_t>();
        params.sampled_nums =
            output_tensors->at("sampled_nums", Tensor{MEMORY_GPU, TYPE_INVALID, {}, nullptr}).getPtr<uint32_t>();

        invokeSampling<T>(params, stream_);
        sync_check_cuda_error();
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void SamplingLayer<T>::setup(const size_t batch_size, const size_t beam_width, TensorMap* runtime_args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    const Tensor runtime_top_k = runtime_args->isExist("runtime_top_k") ? runtime_args->at("runtime_top_k") : Tensor();
    const Tensor runtime_top_p = runtime_args->isExist("runtime_top_p") ? runtime_args->at("runtime_top_p") : Tensor();
    const Tensor runtime_min_p = runtime_args->isExist("runtime_min_p") ? runtime_args->at("runtime_min_p") : Tensor();

    kept_n_.resize(batch_size);
    runtime_top_k_.resize(batch_size);
    runtime_top_p_.resize(batch_size);
    runtime_min_p_.resize(batch_size);

    int   top_k = runtime_top_k.size() > 0 ? runtime_top_k.getVal<int>() : 0;
    float top_p = runtime_top_p.size() > 0 ? runtime_top_p.getVal<float>() : 0.0f;
    float min_p = runtime_min_p.size() > 0 ? runtime_min_p.getVal<float>() : 0.0f;
    set_runtime_args(batch_size,
                     top_k,
                     runtime_top_k.getPtr<int>(),
                     runtime_top_k.size(),
                     runtime_top_k_.data(),
                     top_p,
                     runtime_top_p.getPtr<float>(),
                     runtime_top_p.size(),
                     runtime_top_p_.data(),
                     min_p,
                     runtime_min_p.getPtr<float>(),
                     runtime_min_p.size(),
                     runtime_min_p_.data());

    max_topk_ = *std::max_element(runtime_top_k_.begin(), runtime_top_k_.end());
    min_topk_ = *std::min_element(runtime_top_k_.begin(), runtime_top_k_.end());
    min_topp_ = *std::min_element(runtime_top_p_.begin(), runtime_top_p_.end());
    max_minp_ = *std::max_element(runtime_min_p_.begin(), runtime_min_p_.end());

    allocateBuffer(batch_size);

    // kept
    std::fill_n(kept_n_.data(), batch_size, args_.vocab_size);

    cudaAutoCpy(runtime_top_k_buf_, runtime_top_k_.data(), batch_size, stream_);
    cudaAutoCpy(runtime_top_p_buf_, runtime_top_p_.data(), batch_size, stream_);
    cudaAutoCpy(runtime_min_p_buf_, runtime_min_p_.data(), batch_size, stream_);

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template class SamplingLayer<float>;

}  // namespace turbomind
