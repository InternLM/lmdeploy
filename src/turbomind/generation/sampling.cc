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

#include "src/turbomind/kernels/sampling_kernels.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/kernels/sampling_topp_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/utils/constant.h"

namespace turbomind {

struct SamplingData {

    struct LogprobOutput {
        int                      row;
        int                      offset;
        std::shared_ptr<Request> request;
    };

    explicit SamplingData(int max_batch_size, DeviceType device)
    {
        top_k_buf = {max_batch_size, device};
        top_p_buf = {max_batch_size, device};
        min_p_buf = {max_batch_size, device};
        kept_buf  = {max_batch_size, device};

        sampled_logprobs = {max_batch_size * (ssize_t)kMaxLogProb, device};
        sampled_indices  = {max_batch_size * (ssize_t)kMaxLogProb, device};
        sampled_nums     = {max_batch_size, device};
    }

    int   max_topk = 0;
    int   min_topk = 0;
    float min_topp = 0;
    float max_minp = 0;

    Buffer_<int>   top_k_buf;
    Buffer_<float> top_p_buf;
    Buffer_<float> min_p_buf;

    Buffer_<int> kept_buf;  // kept sample

    int                        generation_size = 0;
    bool                       output_logprobs = 0;
    std::vector<LogprobOutput> logprob_outputs;

    Buffer_<float> sampled_logprobs;
    Buffer_<int>   sampled_indices;
    Buffer_<int>   sampled_nums;
};

Sampling::Sampling(const BaseGenerationParam& base, int phases, int tp_rank):
    BaseGenerationParam{base}, tp_rank_{tp_rank}
{
    top_k_ = {max_batch_size_, kCPUpinned};
    top_p_ = {max_batch_size_, kCPUpinned};
    min_p_ = {max_batch_size_, kCPUpinned};
    kept_  = {max_batch_size_, kCPUpinned};

    sampled_logprobs_buf_ = {max_batch_size_ * (ssize_t)kMaxLogProb, kCPUpinned};
    sampled_indices_buf_  = {max_batch_size_ * (ssize_t)kMaxLogProb, kCPUpinned};
    sampled_nums_buf_     = {max_batch_size_, kCPUpinned};

    // constant array
    std::fill_n(kept_.data(), max_batch_size_, vocab_size_);

    for (int i = 0; i < phases; ++i) {
        data_.push_back(std::make_shared<SamplingData>(max_batch_size_, kDEVICE));
    }
}

void Sampling::Forward(int phase, TensorMap& args)
{
    TM_FUNCTION_SCOPE();
    // step1:
    //  - use topk / topp_minp kernel to sort and filter the scores
    //  - softmax the left score
    // step2:
    //  - sampling from left and sorted scores

    TM_LOG_DEBUG("{} start", __PRETTY_FUNCTION__);

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
        TM_SCOPE_CALL(invokeTopKSortFilter<float>(params, stream));
    }

    // use topp sort if some request skip topk filter
    if (d.min_topk == 0) {
        TM_SCOPE_CALL(
            invokeSoftmax<float>(logits.data(), vocab_size_padded_, vocab_size_, bsz, d.kept_buf.data(), stream));

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
        TM_SCOPE_CALL(invokeTopPSort<float>(params, stream));
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
        TM_SCOPE_CALL(invokeTopPMinPFilter<float>(params, stream));
    }

    // sample
    {
        SamplingParams params{};
        params.logits              = logits.data();
        params.stride              = vocab_size_padded_;
        params.indices             = indices.data();
        params.kept                = d.kept_buf.data();
        params.curandstate         = (curandState_t*)args.at("curand_state").raw_data();
        params.curandstate_indices = args.at("curand_state_indices").data<int>();
        params.batch_size          = bsz;
        params.output_ids          = args.at("output_ids").data<int>();  // (B, 1)
        params.sequence_length     = args.at("sequence_length").data<int>();

        if (d.output_logprobs) {
            params.sampled_logprobs = d.sampled_logprobs.data();
            params.sampled_indexes  = d.sampled_indices.data();
            params.sampled_nums     = d.sampled_nums.data();
        }

        TM_SCOPE_CALL(invokeSampling<float>(params, stream));
    }

    TM_LOG_DEBUG("{} stop", __PRETTY_FUNCTION__);
}

void Sampling::Setup(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();

    // const auto& rc   = env.at("batch").data<BatchData*>()[0]->rc;
    Buffer_<Sequence*> rc = env.at("requests").buffer();

    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    auto& d = *data_.at(phase);

    d.generation_size = 0;
    d.output_logprobs = false;
    d.logprob_outputs.clear();

    for (int i = 0; i < rc.size(); ++i) {
        auto& c = *rc[i];
        if (!c.generating) {
            continue;
        }

        const int row = d.generation_size++;

        top_k_[row] = c.gen_cfg.top_k;
        top_p_[row] = c.gen_cfg.top_p;
        min_p_[row] = c.gen_cfg.min_p;

        if (c.gen_cfg.output_logprobs) {
            d.output_logprobs = true;
            d.logprob_outputs.push_back({row, c.seq_len + c.inflight_new_tokens - c.prompt_len, c.req});
        }
    }

    const int bsz = d.generation_size;
    if (bsz == 0) {
        d.max_topk = d.min_topk = 0;
        d.min_topp              = 0.f;
        d.max_minp              = 0.f;
        return;
    }

    d.max_topk = *std::max_element(top_k_.begin(), top_k_.begin() + bsz);
    d.min_topk = *std::min_element(top_k_.begin(), top_k_.begin() + bsz);
    d.min_topp = *std::min_element(top_p_.begin(), top_p_.begin() + bsz);
    d.max_minp = *std::max_element(min_p_.begin(), min_p_.begin() + bsz);

    copy(top_k_.data(), bsz, d.top_k_buf.data());
    copy(top_p_.data(), bsz, d.top_p_buf.data());

    copy(min_p_.data(), bsz, d.min_p_buf.data());
    copy(kept_.data(), bsz, d.kept_buf.data());
}

void Sampling::Fetch(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    auto& d    = *data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    if (d.output_logprobs) {
        copy(d.sampled_logprobs, d.generation_size * kMaxLogProb, sampled_logprobs_buf_);
        copy(d.sampled_indices, d.generation_size * kMaxLogProb, sampled_indices_buf_);
        copy(d.sampled_nums, d.generation_size, sampled_nums_buf_);
    }
}

void Sampling::Update(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();
    (void)env;

    if (tp_rank_ != 0) {
        return;
    }

    auto& d = *data_.at(phase);
    if (!d.output_logprobs) {
        return;
    }

    float* logprob_buf = sampled_logprobs_buf_.data();
    int*   indices_buf = sampled_indices_buf_.data();
    int*   n_buf       = sampled_nums_buf_.data();

    for (const auto& x : d.logprob_outputs) {
        auto logprob_out = x.request->outputs.at("logprob_vals").data<float>();
        auto indices_out = x.request->outputs.at("logprob_indexes").data<int>();
        auto n_out       = x.request->outputs.at("logprob_nums").data<int>();

        const int n = n_buf[x.row];
        std::copy_n(logprob_buf + x.row * kMaxLogProb, n, logprob_out + x.offset * kMaxLogProb);
        std::copy_n(indices_buf + x.row * kMaxLogProb, n, indices_out + x.offset * kMaxLogProb);
        n_out[x.offset] = n;
    }
}

}  // namespace turbomind
