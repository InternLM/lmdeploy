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
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/kernels/sampling_kernels.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

template<typename T>
SamplingLayer<T>::SamplingLayer(const BaseParam& param): BaseDynamicDecodeLayer{param}
{
    top_k_ = {max_batch_size_, kCPUpinned};
    top_p_ = {max_batch_size_, kCPUpinned};
    min_p_ = {max_batch_size_, kCPUpinned};
    kept_  = {max_batch_size_, kCPUpinned};

    // constant array
    std::fill_n(kept_.data(), max_batch_size_, vocab_size_);

    top_k_buf_ = {max_batch_size_, kDEVICE};
    top_p_buf_ = {max_batch_size_, kDEVICE};
    min_p_buf_ = {max_batch_size_, kDEVICE};
    kept_buf_  = {max_batch_size_, kDEVICE};
}

template<typename T>
void SamplingLayer<T>::Forward(TensorMap& args)
{
    // step1:
    //  - use topk / topp_minp kernel to sort and filter the scores
    //  - softmax the left score
    // step2:
    //  - sampling from left and sorted scores

    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    Tensor_<T> logits = args.at("logits");

    const auto bsz = logits.shape(0);

    const int step = *args.at("step").data<int>();

    core::Copy(kept_.data(), bsz, kept_buf_.data());

    // use topk sort if some request use topk filter
    if (max_topk_ > 0) {
        // TODO: top_k >= 64 is much slower than torch.topk()
        TopKSortFilterParams params{};
        params.workspace         = topk_ws_.data();
        params.workspace_size    = topk_ws_.size();
        params.logits            = logits.data();
        params.sorted_logits     = logits.data();
        params.sorted_indices    = indices_.data();
        params.kept              = kept_buf_.data();
        params.top_ks            = top_k_buf_.data();
        params.max_top_k         = max_topk_;
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopKSortFilter<T>(params, stream_);
    }

    // use topp sort if some request skip topk filter
    if (min_topk_ == 0) {
        invokeSoftmax<T>(logits.data(), vocab_size_padded_, vocab_size_, bsz, kept_buf_.data(), stream_);

        TopPSortParams params{};
        params.workspace         = topp_ws_.data();
        params.workspace_size    = topp_ws_.size();
        params.logits            = logits.data();
        params.sorted_logits     = logits.data();
        params.sorted_indices    = indices_.data();
        params.kept              = kept_buf_.data();
        params.top_ks            = top_k_buf_.data();
        params.top_ps            = top_p_buf_.data();
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopPSort<T>(params, stream_);
    }

    // apply topp minp filter
    if (max_minp_ != 0.f || min_topp_ != 1.f) {
        TopPMinPFilterParams params{};
        params.sorted_logits     = logits.data();
        params.sorted_indices    = indices_.data();
        params.kept              = kept_buf_.data();
        params.top_ps            = top_p_buf_.data();
        params.min_ps            = min_p_buf_.data();
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopPMinPFilter<T>(params, stream_);
    }

    // sample
    {
        SamplingParams params{};
        params.logits          = logits.data();
        params.stride          = vocab_size_padded_;
        params.indices         = indices_.data();
        params.kept            = kept_buf_.data();
        params.curandstate     = (curandState_t*)args.at("curand_state").raw_data();
        params.batch_size      = bsz;
        params.output_ids      = args.at("output_ids").data<int>() + step * bsz;
        params.sequence_length = args.at("sequence_length").data<int>();

        if (auto sampled_logprobs = args.try_("sampled_logprobs")) {
            params.sampled_logprobs = sampled_logprobs->data<T>();
            params.sampled_indexes  = args.at("sampled_indexes").data<uint32_t>();
            params.sampled_nums     = args.at("sampled_nums").data<uint32_t>();
        }

        invokeSampling<T>(params, stream_);
        sync_check_cuda_error();
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void SamplingLayer<T>::Setup(const std::vector<const Request*>& rs, const TensorMap&)
{
    const auto bsz = rs.size();

    for (int i = 0; i < bsz; ++i) {
        top_k_[i] = rs[i]->gen_cfg.top_k;
        top_p_[i] = rs[i]->gen_cfg.top_p;
        min_p_[i] = rs[i]->gen_cfg.min_p;
    }

    max_topk_ = *std::max_element(top_k_.begin(), top_k_.end());
    min_topk_ = *std::min_element(top_k_.begin(), top_k_.end());
    min_topp_ = *std::min_element(top_p_.begin(), top_p_.end());
    max_minp_ = *std::max_element(min_p_.begin(), min_p_.end());

    indices_ = Buffer_<int>(bsz * vocab_size_padded_, kDEVICE);

    {
        // topk buffer
        TopKSortFilterParams params{};
        params.batch_size = bsz;
        params.max_top_k  = max_topk_;
        invokeTopKSortFilter<T>(params, stream_);
        topk_ws_ = {(ssize_t)params.workspace_size, kDEVICE};
    }

    {
        // topp buffer
        TopPSortParams params{};
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopPSort<T>(params, stream_);
        topp_ws_ = {(ssize_t)params.workspace_size, kDEVICE};
    }

    core::Copy(top_k_.data(), bsz, top_k_buf_.data());
    core::Copy(top_p_.data(), bsz, top_p_buf_.data());
    core::Copy(min_p_.data(), bsz, min_p_buf_.data());
}

template class SamplingLayer<float>;

}  // namespace turbomind
