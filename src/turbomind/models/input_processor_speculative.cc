// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"

#include "src/turbomind/engine/request.h"

#include "src/turbomind/kernels/speculative_sequence_kernels.h"

#include "src/turbomind/models/input_processor_impl.h"

namespace turbomind {

void InputProcessor::Impl::SetupSpeculative(int phase, TensorMap& env)
{
    auto& d    = data_.at(phase);
    auto& b    = *env.at("batch").data<BatchData*>()[0];
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    Buffer_<Sequence*> rc = env.at("requests").buffer();

    input_ids_offsets_buf_[0] = 0;
    int generation_count      = 0;
    for (int i = 0; i < rc.size(); ++i) {
        const Sequence&     c       = *rc[i];
        const SubmittedRow& row     = *c.submitted;
        const int           q_begin = input_ids_offsets_buf_[i];
        const int           q_len   = row.input_len;

        input_ids_offsets_buf_[i + 1] = q_begin + q_len;
        target_ids_from_row_buf_[i]   = row.autoregres;
        if (!row.autoregres) {
            const int* src = c.token_ids + row.history_len + c.inflight_input_len;
            std::copy_n(src, q_len, input_ids_buf_.data() + q_begin);
        }
        if (row.generating) {
            ++generation_count;
        }
    }

    const int position_count = env.at("verification_positions").data<int>()[0];

    int g = 0;
    for (int i = 0; i < rc.size(); ++i) {
        const Sequence&     c         = *rc[i];
        const SubmittedRow& submitted = *c.submitted;
        if (!submitted.generating) {
            continue;
        }

        const int q_end   = input_ids_offsets_buf_[i + 1];

        for (int position = 0; position < position_count; ++position) {
            // Wide rows carry width == input_len; positions beyond a row's width clamp to its last token.
            decode_token_pos_buf_[position * generation_count + g] =
                q_end - 1 - std::max(0, submitted.verification_positions - 1 - position);
        }
        ++g;
    }

    d.selected_token_count = position_count * generation_count;

    copy(input_ids_buf_, input_ids_offsets_buf_[b.bsz], d.input_ids);
    copy(decode_token_pos_buf_, d.selected_token_count, d.selected_token_pos);
    copy(input_ids_offsets_buf_, b.bsz + 1, d.input_ids_offsets);
    copy(target_ids_from_row_buf_, b.bsz, d.target_ids_from_row);

    d.input_token_num = input_ids_offsets_buf_[b.bsz];

    env.produce("token_num", Buffer{&d.input_token_num, 1, kCPU});

    StageEmbeddingPatches(phase, rc, successor_embeddings_);
}

void InputProcessor::Impl::PrepareSpeculative(int phase, TensorMap& env)
{
    auto& d = data_.at(phase);
    auto& b = *env.at("batch").data<BatchData*>()[0];

    env.produce("input_ids", d.input_ids.slice(0, d.input_token_num));
    env.produce("q_offsets", d.input_ids_offsets.slice(0, b.bsz + 1));
    env.produce("selected_token_pos", d.selected_token_pos.slice(0, d.selected_token_count));
}

void InputProcessor::Impl::BuildTargetInputs(int phase, TensorMap& env)
{
    auto& d = data_.at(phase);
    auto& b = *env.at("batch").data<BatchData*>()[0];

    invokeBuildTargetInputs(env.at("input_ids").data<int>(),
                            d.target_key_lengths.data(),
                            reinterpret_cast<const int* const*>(env.at("request_token_ids_ptrs").data<int*>()),
                            env.at("q_offsets").data<int>(),
                            env.at("sequence_length").data<int>(),
                            d.target_ids_from_row.data(),
                            env.at("finished").data<bool>(),
                            b.bsz,
                            core::Context::stream().handle());

    env.produce("target_key_lengths", d.target_key_lengths.slice(0, b.bsz));
}

}  // namespace turbomind
