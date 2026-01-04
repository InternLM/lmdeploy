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

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"

#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/kernels/ban_bad_words.h"
#include "src/turbomind/kernels/sampling_penalty_kernels.h"

#include "src/turbomind/generation/logits_processor.h"
#include "src/turbomind/generation/utils.h"

namespace turbomind {

struct LogitsProcessor::Data {

    Data(int max_batch_size, DeviceType device)
    {
        repetition_penalty_buf = {max_batch_size, device};
        min_lengths_buf        = {max_batch_size, device};
        temperature_buf        = {max_batch_size, device};
        bad_words_buf          = {max_batch_size * 2 * kMaxStopBadWordsLen, device};
        end_ids_buf            = {max_batch_size * kMaxEndIdsSize, device};
    }

    Buffer_<float> repetition_penalty_buf;
    Buffer_<int>   min_lengths_buf;
    Buffer_<float> temperature_buf;
    Buffer_<int>   bad_words_buf;
    Buffer_<int>   end_ids_buf;

    Tensor_<int> bad_words_ten;
    Tensor_<int> end_ids_ten;

    bool has_repetition_penalty{};
    bool has_bad_words_penalty{};
    bool has_min_length_penalty{};
    bool has_temperature_penalty{};
};

LogitsProcessor::LogitsProcessor(const BaseGenerationParam& base, int phases): BaseGenerationParam{base}
{
    buf_ = std::make_shared<Data>(max_batch_size_, kCPUpinned);
    for (int i = 0; i < phases; ++i) {
        data_.push_back(std::make_shared<Data>(max_batch_size_, kDEVICE));
    }
}

void LogitsProcessor::Forward(int phase, TensorMap& env)
{
    // apply repetition penalty -> ban bad words -> min length penalty -> temperature penalty
    // the order is same with transformerss
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    Tensor_<float>      logits          = env.at("logits");
    const Buffer_<int*> token_ids_ptrs  = env.at("token_ids_ptrs").buffer();
    const Buffer_<int>  sequence_length = env.at("sequence_length").buffer();

    const auto bsz = logits.shape(0);

    auto& d = *data_.at(phase);

    auto stream = core::Context::stream().handle();

    // repetition penalty
    if (d.has_repetition_penalty) {
        ApplyRepetitionPenalty(logits, d.repetition_penalty_buf, token_ids_ptrs, sequence_length, stream);
        sync_check_cuda_error();
    }

    // ban bad words
    if (auto& bad_words = d.bad_words_ten) {
        BanBadWords(logits, token_ids_ptrs, sequence_length, bad_words, stream);
        sync_check_cuda_error();
    }

    // min length
    if (d.has_min_length_penalty) {
        invokeMinLengthPenalty(logits.data(),
                               d.min_lengths_buf.data(),
                               sequence_length.data(),
                               vocab_size_padded_,
                               bsz,
                               d.end_ids_ten.data(),
                               d.end_ids_ten.shape(1),
                               stream);
        sync_check_cuda_error();
    }

    // temperature
    if (d.has_temperature_penalty) {
        invokeBatchApplyTemperaturePenalty_v2(logits.data(),  //
                                              (float*)nullptr,
                                              d.temperature_buf.data(),
                                              bsz,
                                              vocab_size_,
                                              vocab_size_padded_,
                                              stream);
        sync_check_cuda_error();
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void LogitsProcessor::Setup(int phase, TensorMap& env)
{
    TM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto& d = *data_.at(phase);

    const auto& rs   = env.at("batch").data<BatchData*>()[0]->rc;
    auto&       copy = *env.at("copy").data<BatchCopy*>()[0];

    const int bsz = rs.size();

    auto& repetition_penalty = buf_->repetition_penalty_buf;
    auto& temperature        = buf_->temperature_buf;
    auto& min_lengths        = buf_->min_lengths_buf;

    d.has_temperature_penalty = {};
    d.has_min_length_penalty  = {};
    d.has_repetition_penalty  = {};
    d.has_bad_words_penalty   = {};

    for (int i = 0; i < bsz; ++i) {
        auto& g = rs[i]->gen_cfg;

        // repetition_penalty
        repetition_penalty[i] = g.repetition_penalty;
        if (repetition_penalty[i] != 1.f) {
            d.has_repetition_penalty = true;
        }

        // temperature
        temperature[i] = g.temperature;
        if (g.temperature != 1.f) {
            d.has_temperature_penalty = true;
        }

        // min_length
        min_lengths[i] = rs[i]->prompt_len + g.min_new_tokens;
        if (rs[i]->seq_len + rs[i]->beta < min_lengths[i]) {
            d.has_min_length_penalty = true;
        }
    }

    if (d.has_temperature_penalty) {
        copy(temperature, bsz, d.temperature_buf);
    }

    if (d.has_repetition_penalty) {
        copy(repetition_penalty, bsz, d.repetition_penalty_buf);
    }

    if (d.has_min_length_penalty) {
        copy(min_lengths, bsz, d.min_lengths_buf);
    }

    sync_check_cuda_error();

    d.bad_words_ten = {};
    init_stop_bad_words(&GenerationConfig::bad_ids,  //
                        "bad_words",
                        rs,
                        buf_->bad_words_buf.data(),
                        d.bad_words_buf.data(),
                        d.bad_words_ten,
                        copy);

    if (d.has_min_length_penalty) {  // end ids for min length
        d.end_ids_ten  = {};
        int max_length = 0;
        for (int i = 0; i < bsz; ++i) {
            max_length = std::max(max_length, (int)rs[i]->gen_cfg.eos_ids.size());
        }
        if (max_length) {
            max_length     = std::min(max_length, kMaxEndIdsSize);
            int* h_end_ids = buf_->end_ids_buf.data();
            std::fill(h_end_ids, h_end_ids + std::min(kMaxEndIdsSize, max_length) * bsz, -1);
            for (int i = 0; i < bsz; ++i) {
                const auto& eos_ids = rs[i]->gen_cfg.eos_ids;
                if (eos_ids.size() == 0) {
                    continue;
                }
                if (TM_UNLIKELY(eos_ids.size() > kMaxEndIdsSize)) {
                    TM_LOG_WARNING("[InitializeSampling] [%ld] eos length (%d) exceeds %d, truncated to %d",
                                   (long)rs[i]->req->id,
                                   (int)eos_ids.size(),
                                   kMaxEndIdsSize,
                                   kMaxEndIdsSize);
                }
                std::copy_n(eos_ids.begin(), std::min((int)eos_ids.size(), kMaxEndIdsSize), h_end_ids);
                h_end_ids += max_length;
            }
            copy(buf_->end_ids_buf, bsz * max_length, d.end_ids_buf);
            d.end_ids_ten = {d.end_ids_buf.data(), {bsz, max_length}, kDEVICE};
        }
    }

    TM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

}  // namespace turbomind
