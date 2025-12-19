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

#include "src/turbomind/generation/sampling.h"

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/sampling_kernels.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"

#include "src/turbomind/engine/request.h"

#include "src/turbomind/utils/logger.h"

namespace turbomind {

struct SamplingData {

    explicit SamplingData(int max_batch_size, DeviceType device)
    {
        top_k_buf = {max_batch_size, device};
        top_p_buf = {max_batch_size, device};
        min_p_buf = {max_batch_size, device};
        kept_buf  = {max_batch_size, device};
    }

    int   max_topk;
    int   min_topk;
    float min_topp;
    float max_minp;

    Buffer_<int>   top_k_buf;
    Buffer_<float> top_p_buf;
    Buffer_<float> min_p_buf;

    Buffer_<int> kept_buf;  // kept sample
};

Sampling::Sampling(const BaseGenerationParam& base, int phases): BaseGenerationParam{base}
{
    top_k_ = {max_batch_size_, kCPUpinned};
    top_p_ = {max_batch_size_, kCPUpinned};
    min_p_ = {max_batch_size_, kCPUpinned};
    kept_  = {max_batch_size_, kCPUpinned};

    // constant array
    std::fill_n(kept_.data(), max_batch_size_, vocab_size_);

    for (int i = 0; i < phases; ++i) {
        data_.push_back(std::make_shared<SamplingData>(max_batch_size_, kDEVICE));
    }
}

void Sampling::Forward(int phase, TensorMap& args)
{
    // step1:
    //  - use topk / topp_minp kernel to sort and filter the scores
    //  - softmax the left score
    // step2:
    //  - sampling from left and sorted scores

    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto& d = *data_.at(phase);

    Tensor_<float> logits = args.at("logits");

    const auto bsz = logits.shape(0);

    Buffer_<int> indices(bsz * vocab_size_padded_, kDEVICE);

    auto stream = core::Context::stream().handle();

    // use topk sort if some request use topk filter
    if (d.max_topk > 0) {
        // TODO: top_k >= 64 is much slower than torch.topk()
        TopKSortFilterParams params{};
        params.logits            = logits.data();
        params.sorted_logits     = logits.data();
        params.sorted_indices    = indices.data();
        params.kept              = d.kept_buf.data();
        params.top_ks            = d.top_k_buf.data();
        params.max_top_k         = d.max_topk;
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopKSortFilter<float>(params, stream);
    }

    // use topp sort if some request skip topk filter
    if (d.min_topk == 0) {
        invokeSoftmax<float>(logits.data(), vocab_size_padded_, vocab_size_, bsz, d.kept_buf.data(), stream);

        TopPSortParams params{};
        params.logits            = logits.data();
        params.sorted_logits     = logits.data();
        params.sorted_indices    = indices.data();
        params.kept              = d.kept_buf.data();
        params.top_ks            = d.top_k_buf.data();
        params.top_ps            = d.top_p_buf.data();
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopPSort<float>(params, stream);
    }

    // apply topp minp filter
    if (d.max_minp != 0.f || d.min_topp != 1.f) {
        TopPMinPFilterParams params{};
        params.sorted_logits     = logits.data();
        params.sorted_indices    = indices.data();
        params.kept              = d.kept_buf.data();
        params.top_ps            = d.top_p_buf.data();
        params.min_ps            = d.min_p_buf.data();
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        invokeTopPMinPFilter<float>(params, stream);
    }

    // sample
    {
        SamplingParams params{};
        params.logits          = logits.data();
        params.stride          = vocab_size_padded_;
        params.indices         = indices.data();
        params.kept            = d.kept_buf.data();
        params.curandstate     = (curandState_t*)args.at("curand_state").raw_data();
        params.batch_size      = bsz;
        params.output_ids      = args.at("output_ids").data<int>();  // (B, 1)
        params.sequence_length = args.at("sequence_length").data<int>();

        if (auto sampled_logprobs = args.try_("sampled_logprobs")) {
            params.sampled_logprobs = sampled_logprobs->data<float>();
            params.sampled_indexes  = args.at("sampled_indexes").data<uint32_t>();
            params.sampled_nums     = args.at("sampled_nums").data<uint32_t>();
        }

        invokeSampling<float>(params, stream);
        sync_check_cuda_error();
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void Sampling::Setup(int phase, TensorMap& env)
{
    Buffer_<const RequestCache*> rc = env.at("requests").buffer();

    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    const auto bsz = rc.size();

    for (int i = 0; i < bsz; ++i) {
        top_k_[i] = rc[i]->gen_cfg.top_k;
        top_p_[i] = rc[i]->gen_cfg.top_p;
        min_p_[i] = rc[i]->gen_cfg.min_p;
    }

    auto& d = *data_.at(phase);

    d.max_topk = *std::max_element(top_k_.begin(), top_k_.begin() + bsz);
    d.min_topk = *std::min_element(top_k_.begin(), top_k_.begin() + bsz);
    d.min_topp = *std::min_element(top_p_.begin(), top_p_.begin() + bsz);
    d.max_minp = *std::max_element(min_p_.begin(), min_p_.begin() + bsz);

    copy(top_k_.data(), bsz, d.top_k_buf.data());
    copy(top_p_.data(), bsz, d.top_p_buf.data());

    copy(min_p_.data(), bsz, d.min_p_buf.data());
    copy(kept_.data(), bsz, d.kept_buf.data());
}

}  // namespace turbomind
