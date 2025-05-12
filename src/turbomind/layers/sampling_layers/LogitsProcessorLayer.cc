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

#include <iostream>
#include <numeric>

#include "src/turbomind/core/check.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/kernels/ban_bad_words.h"
#include "src/turbomind/kernels/penalty_types.h"
#include "src/turbomind/kernels/sampling_penalty_kernels.h"
#include "src/turbomind/layers/sampling_layers/LogitsProcessorLayer.h"
#include "src/turbomind/layers/sampling_layers/utils.h"

namespace turbomind {

#define ALL_OF(p_, sz_, dt_, v_) (std::all_of(p_, p_ + sz_, [&](dt_ b) { return b == v_; }))

namespace {

template<typename T>
void init_host_buffer(const TensorMap& map, const std::string& key, size_t size, T* dst, T default_value)
{
    Tensor        empty{};
    const Tensor& src = map.contains(key) ? map.at(key) : empty;

    if (src) {
        if (size_t sz = src.size(); sz > size) {
            TM_LOG_ERROR("runtime_args %s has invalid size %ld vs batch_size %ld", key.c_str(), sz, size);
        }
        std::copy_n(src.data<T>(), size, dst);
    }
    else {
        std::fill_n(dst, size, default_value);
    }
}

}  // namespace

template<typename T>
LogitsProcessorLayer<T>::LogitsProcessorLayer(const BaseParam& param): BaseDynamicDecodeLayer{param}
{

    repetition_penalty_ = {max_batch_size_, kCPUpinned};
    min_lengths_        = {max_batch_size_, kCPUpinned};
    temperature_        = {max_batch_size_, kCPUpinned};
    bad_words_          = {max_batch_size_ * 2 * kMaxStopBadWordsLen, kCPUpinned};
    end_ids_            = {max_batch_size_ * kMaxEndIdsSize, kCPUpinned};

    repetition_penalty_buf_ = {max_batch_size_, kDEVICE};
    min_lengths_buf_        = {max_batch_size_, kDEVICE};
    temperature_buf_        = {max_batch_size_, kDEVICE};
    bad_words_buf_          = {max_batch_size_ * 2 * kMaxStopBadWordsLen, kDEVICE};
    end_ids_buf_            = {max_batch_size_ * kMaxEndIdsSize, kDEVICE};
}

template<typename T>
void LogitsProcessorLayer<T>::Forward(TensorMap& args)
{
    // apply repetition penalty -> ban bad words -> min length penalty -> temperature penalty
    // the order is same with transformers

    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    Tensor_<int> output_ids = args.at("output_ids");
    Tensor_<T>   logits     = args.at("logits");

    const auto bsz = logits.shape(0);

    const int step             = *args.at("step").data<int>();
    const int max_input_length = *args.at("max_input_length").data<int>();

    // repetition penalty
    if (step > 1 && repetition_penalty_type_ != RepetitionPenaltyType::None) {
        Buffer_<uint8_t> workspace(bsz * step * (sizeof(int) + sizeof(float)), kDEVICE);
        invokeBatchApplyRepetitionPenalty(logits.data(),
                                          repetition_penalty_buf_.data(),
                                          (int*)workspace.data(),
                                          output_ids.data(),
                                          bsz,
                                          bsz,
                                          vocab_size_padded_,
                                          args.at("init_context_length").data<int>(),
                                          max_input_length,
                                          step,
                                          repetition_penalty_type_,
                                          stream_);
        sync_check_cuda_error();
    }

    // ban bad words
    if (auto& bad_words = bad_words_ten_) {
        TM_CHECK_EQ(bad_words.ndim(), 3);
        const auto bad_words_len = bad_words.shape(2);
        invokeBanBadWords(logits.data(),
                          output_ids.data(),
                          nullptr,
                          bsz,
                          bsz,
                          1,
                          bad_words.data(),
                          false,
                          bad_words_len,
                          0,
                          vocab_size_padded_,
                          step,
                          stream_);

        sync_check_cuda_error();
    }

    // min length
    if (end_ids_ten_) {
        TM_CHECK_EQ(end_ids_ten_.ndim(), 2);
        auto enable = [&] {
            const int num_generated_tokens = step - max_input_length;
            auto      context_len          = args.at("context_length").data<int>();
            for (int i = 0; i < bsz; ++i) {
                if (min_lengths_[i] > context_len[i] + num_generated_tokens) {
                    return true;
                }
            }
            return false;
        }();
        if (enable) {
            invokeMinLengthPenalty(logits.data(),
                                   min_lengths_buf_.data(),
                                   args.at("sequence_length").data<int>(),
                                   vocab_size_padded_,
                                   bsz,
                                   end_ids_ten_.data(),
                                   end_ids_ten_.shape(1),
                                   stream_);
            sync_check_cuda_error();
        }
    }

    // temperature
    if (!ALL_OF(temperature_.begin(), bsz, float, 1.f)) {
        invokeBatchApplyTemperaturePenalty_v2(logits.data(),  //
                                              (T*)nullptr,
                                              temperature_buf_.data(),
                                              bsz,
                                              vocab_size_,
                                              vocab_size_padded_,
                                              stream_);
        sync_check_cuda_error();
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void LogitsProcessorLayer<T>::Setup(const std::vector<const Request*>& rs, const TensorMap& args)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    const int bsz = rs.size();

    const auto prompt_length = args.at("prompt_length").data<int>();

    repetition_penalty_type_ = RepetitionPenaltyType::None;

    for (int i = 0; i < bsz; ++i) {
        auto& c = rs[i]->gen_cfg;
        // repetition_penalty
        repetition_penalty_[i] = c.repetition_penalty;
        if (repetition_penalty_[i] != 1.f) {
            repetition_penalty_type_ = RepetitionPenaltyType::Multiplicative;
        }
        // temperature
        temperature_[i] = c.temperature;
        // min_length
        min_lengths_[i] = c.min_new_tokens + prompt_length[i];
    }

    Copy_(temperature_, bsz, temperature_buf_);
    Copy_(repetition_penalty_, bsz, repetition_penalty_buf_);
    Copy_(min_lengths_, bsz, min_lengths_buf_);

    sync_check_cuda_error();

    bad_words_ten_ = {};
    init_stop_bad_words(&GenerationConfig::bad_ids,  //
                        "bad_words",
                        rs,
                        bad_words_.data(),
                        bad_words_buf_.data(),
                        bad_words_ten_);

    {  // end ids for min length
        end_ids_ten_   = {};
        int max_length = 0;
        for (int i = 0; i < bsz; ++i) {
            max_length = std::max(max_length, (int)rs[i]->gen_cfg.eos_ids.size());
        }
        if (max_length) {
            max_length     = std::min(max_length, kMaxEndIdsSize);
            int* h_end_ids = end_ids_.data();
            std::fill(h_end_ids, h_end_ids + std::min(kMaxEndIdsSize, max_length) * bsz, -1);
            for (int i = 0; i < bsz; ++i) {
                const auto& eos_ids = rs[i]->gen_cfg.eos_ids;
                if (eos_ids.size() == 0) {
                    continue;
                }
                if (TM_UNLIKELY(eos_ids.size() > kMaxEndIdsSize)) {
                    TM_LOG_WARNING("[InitializeSampling] [%ld] eos length (%d) exceeds %d, truncated to %d",
                                   (long)rs[i]->id,
                                   (int)eos_ids.size(),
                                   kMaxEndIdsSize,
                                   kMaxEndIdsSize);
                }
                std::copy_n(eos_ids.begin(), std::min((int)eos_ids.size(), kMaxEndIdsSize), h_end_ids);
                h_end_ids += max_length;
            }
            Copy(end_ids_, bsz * max_length, end_ids_buf_);
            end_ids_ten_ = {end_ids_buf_.data(), {bsz, max_length}, kDEVICE};
        }
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template class LogitsProcessorLayer<float>;

}  // namespace turbomind
