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
#include "src/turbomind/kernels/speculative_sampling_kernels.h"
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

    explicit SamplingData(int parameter_capacity, int max_batch_size, DeviceType device)
    {
        top_k_buf = {parameter_capacity, device};
        top_p_buf = {parameter_capacity, device};
        min_p_buf = {parameter_capacity, device};
        kept_buf  = {parameter_capacity, device};
        greedy    = {parameter_capacity, device};

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

    Buffer_<int>  kept_buf;  // kept sample
    Buffer_<bool> greedy;

    int                        generation_size = 0;
    bool                       output_logprobs = 0;
    std::vector<LogprobOutput> logprob_outputs;

    Buffer_<float> sampled_logprobs;
    Buffer_<int>   sampled_indices;
    Buffer_<int>   sampled_nums;
};

Sampling::Sampling(const BaseGenerationParam& base, int phases, int tp_rank, int parameter_capacity):
    BaseGenerationParam{base}, tp_rank_{tp_rank}, parameter_capacity_{parameter_capacity}
{
    top_k_  = {parameter_capacity_, kCPUpinned};
    top_p_  = {parameter_capacity_, kCPUpinned};
    min_p_  = {parameter_capacity_, kCPUpinned};
    kept_   = {parameter_capacity_, kCPUpinned};
    greedy_ = {parameter_capacity_, kCPUpinned};

    sampled_logprobs_buf_ = {max_batch_size_ * (ssize_t)kMaxLogProb, kCPUpinned};
    sampled_indices_buf_  = {max_batch_size_ * (ssize_t)kMaxLogProb, kCPUpinned};
    sampled_nums_buf_     = {max_batch_size_, kCPUpinned};

    // constant array
    std::fill_n(kept_.data(), parameter_capacity_, vocab_size_);

    for (int i = 0; i < phases; ++i) {
        data_.push_back(std::make_shared<SamplingData>(parameter_capacity_, max_batch_size_, kDEVICE));
    }
}

void Sampling::ProcessDistributions(int phase, Tensor_<float> probabilities, Buffer_<int> token_indices)
{
    auto& d = *data_.at(phase);

    const auto bsz = probabilities.shape(0);

    auto stream = core::Context::stream().handle();

    // use topk sort if some request use topk filter
    if (d.max_topk > 0) {
        // TODO: top_k >= 64 is much slower than torch.topk()
        TopKSortFilterParams params{};
        params.logits            = probabilities.data();
        params.sorted_logits     = probabilities.data();
        params.sorted_indices    = token_indices.data();
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
        TM_SCOPE_CALL(invokeSoftmax<float>(
            probabilities.data(), vocab_size_padded_, vocab_size_, bsz, d.kept_buf.data(), stream));

        TopPSortParams params{};
        params.logits            = probabilities.data();
        params.sorted_logits     = probabilities.data();
        params.sorted_indices    = token_indices.data();
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
        params.sorted_logits     = probabilities.data();
        params.sorted_indices    = token_indices.data();
        params.kept              = d.kept_buf.data();
        params.top_ps            = d.top_p_buf.data();
        params.min_ps            = d.min_p_buf.data();
        params.batch_size        = bsz;
        params.vocab_size        = vocab_size_;
        params.vocab_size_padded = vocab_size_padded_;
        TM_SCOPE_CALL(invokeTopPMinPFilter<float>(params, stream));
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

    ProcessDistributions(phase, logits, indices);

    // sample
    {
        SamplingParams params{};
        params.probabilities       = logits.data();
        params.stride              = vocab_size_padded_;
        params.indices             = indices.data();
        params.kept                = d.kept_buf.data();
        params.curandstate         = (curandState_t*)args.at("curand_state").raw_data();
        params.curandstate_indices = args.at("curand_state_indices").data<int>();
        params.sample_mask         = args.contains("sample_mask") ? args.at("sample_mask").data<bool>() : nullptr;
        params.batch_size          = bsz;
        params.selected_tokens     = args.at("output_ids").data<int>();  // (B, 1)

        if (d.output_logprobs) {
            params.sampled_logprobs = d.sampled_logprobs.data();
            params.sampled_indexes  = d.sampled_indices.data();
            params.sampled_nums     = d.sampled_nums.data();
        }

        TM_SCOPE_CALL(invokeSampling<float>(params, stream));
    }

    TM_LOG_DEBUG("{} stop", __PRETTY_FUNCTION__);
}

void Sampling::VerifyTargetBlock(int phase, Tensor_<float> probabilities, VerifyTargetBlockParams params)
{
    auto& d = *data_.at(phase);

    const int rows = probabilities.shape(0);

    Buffer_<int> token_indices(rows * (ssize_t)vocab_size_padded_, kDEVICE);

    ProcessDistributions(phase, probabilities, token_indices);

    params.probabilities         = probabilities.data();
    params.probability_stride    = probabilities.stride(0);
    params.probability_token_ids = token_indices.data();
    params.token_id_stride       = probabilities.stride(0);
    params.kept_count            = d.kept_buf.data();
    params.greedy                = d.greedy.data();

    invokeVerifyTargetBlock(params, core::Context::stream().handle());
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

    std::vector<Sequence*> generating_requests;
    generating_requests.reserve(rc.size());
    for (Sequence* request : rc) {
        auto& c = *request;
        if (!c.submitted->generating) {
            continue;
        }

        const int row = d.generation_size++;
        generating_requests.push_back(request);

        if (c.gen_cfg.output_logprobs) {
            d.output_logprobs = true;
            d.logprob_outputs.push_back({row, c.seq_len + c.inflight_new_tokens - c.prompt_len, c.req});
        }
    }

    const int G = d.generation_size;
    const int P = parameter_capacity_ == max_batch_size_ ? 1 : env.at("verification_positions").data<int>()[0];
    const int bsz = P * G;
    if (bsz == 0) {
        d.max_topk = d.min_topk = 0;
        d.min_topp              = 0.f;
        d.max_minp              = 0.f;
        return;
    }

    for (int position = 0; position < P; ++position) {
        for (int g = 0; g < G; ++g) {
            const int row = position * G + g;
            const auto& config = generating_requests[g]->gen_cfg;
            top_k_[row]  = config.top_k;
            top_p_[row]  = config.top_p;
            min_p_[row]  = config.min_p;
            greedy_[row] = config.top_k == 1;
        }
    }

    d.max_topk = *std::max_element(top_k_.begin(), top_k_.begin() + bsz);
    d.min_topk = *std::min_element(top_k_.begin(), top_k_.begin() + bsz);
    d.min_topp = *std::min_element(top_p_.begin(), top_p_.begin() + bsz);
    d.max_minp = *std::max_element(min_p_.begin(), min_p_.begin() + bsz);

    copy(top_k_.data(), bsz, d.top_k_buf.data());
    copy(top_p_.data(), bsz, d.top_p_buf.data());

    copy(min_p_.data(), bsz, d.min_p_buf.data());
    copy(kept_.data(), bsz, d.kept_buf.data());
    copy(greedy_.data(), bsz, d.greedy.data());
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
