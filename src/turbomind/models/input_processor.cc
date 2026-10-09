// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/input_processor.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/core/core.h"

#include "src/turbomind/engine/request.h"

#include "src/turbomind/models/input_processor_impl.h"
#include "src/turbomind/models/vision_model.h"

namespace turbomind {

using std::vector;

InputProcessor::Impl::Impl(const EngineParam& engine,
                           int                hidden_units,
                           DataType           data_type,
                           int                phases,
                           bool               speculative,
                           int                max_verification_positions,
                           bool               successor_embeddings):
    max_batch_size_{engine.max_batch_size},
    max_forward_token_num_{engine.max_forward_token_num},
    speculative_engine_{speculative},
    successor_embeddings_{successor_embeddings}
{
    input_ids_buf_           = {max_forward_token_num_, kCPUpinned};
    input_ids_offsets_buf_   = {max_batch_size_ + 1, kCPUpinned};
    decode_token_pos_buf_    = {max_batch_size_ * max_verification_positions, kCPUpinned};
    target_ids_from_row_buf_ = {max_batch_size_, kCPUpinned};

    data_.reserve(phases);
    for (int i = 0; i < phases; ++i) {
        auto& d               = data_.emplace_back();
        d.input_ids           = empty_like(input_ids_buf_, kDEVICE);
        d.input_ids_offsets   = empty_like(input_ids_offsets_buf_, kDEVICE);
        d.selected_token_pos  = empty_like(decode_token_pos_buf_, kDEVICE);
        d.target_ids_from_row = empty_like(target_ids_from_row_buf_, kDEVICE);
        if (speculative_engine_) {
            d.target_key_lengths = {max_batch_size_, kDEVICE};
        }

        d.autoreg_ids_pos = {max_batch_size_, kCPU};  // ! CPU buffer

        /// TODO: initialize only when required
        d.input_embeds_buf = {
            {max_forward_token_num_ + max_batch_size_, hidden_units}, data_type, kCPUpinned};
    }
}

int InputProcessor::Impl::Add(Sequence& c)
{
    const auto& r = *c.req;

    // trim input embeds
    if (!c.input_embeds_offsets.empty()) {
        Interval l{0, (int)c.tokens.size()};
        using Size    = Interval::Size;
        auto& embeds  = c.input_embeds;
        auto& offsets = c.input_embeds_offsets;
        int   i       = embeds.size() - 1;
        for (; i >= 0; --i) {
            Interval r{offsets[i], Size{(int)embeds[i].shape(0)}};
            if (auto o = r & l) {
                if (o.end() < r.end()) {
                    embeds[i] = embeds[i].slice(0, o.end() - r.begin());
                }
                break;
            }
        }
        embeds.resize(i + 1);
        offsets.resize(i + 1);
    }

    if (auto ranges_ptr = r.inputs.try_("input_embedding_ranges")) {  // [n, 2]
        auto embeds = r.inputs.at("input_embeddings");                // [k, d]
        if (ranges_ptr->ndim() != 2 || embeds.ndim() != 2 || ranges_ptr->shape(1) != 2) {
            /// TODO: reject for invalid shapes
            return Request::kInvalid;
        }

        const auto [sum, dim] = embeds.shapes(0, 1);
        const auto n          = ranges_ptr->shape(0);
        const auto ranges     = ranges_ptr->data<int>();

        int offset = 0;
        int last   = c.step0;
        for (int i = 0; i < n; ++i) {
            Interval range{c.step0 + ranges[i * 2], c.step0 + ranges[i * 2 + 1]};
            auto     size = (int)range.size();
            if (range.begin() < last) {
                /// TODO: reject for non-sorted ranges
                return Request::kInvalid;
            }
            if (range.end() > c.seq_len) {
                /// TODO: reject for dst range OOB
                return Request::kInvalid;
            }
            if (offset + size > sum) {
                /// TODO: reject for src range OOB
                return Request::kInvalid;
            }
            c.input_embeds_offsets.push_back(range.begin());
            c.input_embeds.push_back(embeds.slice(offset, size));  // reference into `embeds`
            offset += size;
            last = range.end();
        }
    }

    return 0;
}

void InputProcessor::Impl::Add(int phase, TensorMap& env)
{
    const Buffer_<Sequence*> rc = env.at("requests").buffer();
    for (int i = 0; i < rc.size(); ++i) {
        auto& c = *TM_CHECK_NOTNULL(rc[i]);
        if (c.status == 0) {
            c.status = Add(c);
        }
    }
}

void InputProcessor::Impl::Setup(int phase, TensorMap& env)
{
    if (speculative_engine_) {
        return SetupSpeculative(phase, env);
    }
    return SetupOrdinary(phase, env);
}

void InputProcessor::Impl::SetupOrdinary(int phase, TensorMap& env)
{
    auto& d    = data_.at(phase);
    auto& b    = *env.at("batch").data<BatchData*>()[0];
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    Buffer_<Sequence*> rc = env.at("requests").buffer();

    input_ids_offsets_buf_[0] = 0;
    for (int i = 0; i < rc.size(); ++i) {
        const Sequence&     c       = *rc[i];
        const SubmittedRow& row     = *c.submitted;
        const int           q_begin = input_ids_offsets_buf_[i];
        const int           q_len   = row.input_len;

        input_ids_offsets_buf_[i + 1] = q_begin;
        if (TM_UNLIKELY(!row.autoregres)) {
            const int* src = c.token_ids + row.history_len + c.inflight_input_len;
            std::copy_n(src, q_len, input_ids_buf_.data() + q_begin);
            d.autoreg_ids_pos[i] = -1;
            input_ids_offsets_buf_[i + 1] += q_len;
        }
        else {
            d.autoreg_ids_pos[i] = q_begin;
            input_ids_offsets_buf_[i + 1] += 1;
        }
        decode_token_pos_buf_[i] = input_ids_offsets_buf_[i + 1] - 1;
    }

    d.selected_token_count = rc.size();

    copy(input_ids_buf_, input_ids_offsets_buf_[b.bsz], d.input_ids);
    copy(decode_token_pos_buf_, d.selected_token_count, d.selected_token_pos);
    copy(input_ids_offsets_buf_, b.bsz + 1, d.input_ids_offsets);

    d.input_token_num = input_ids_offsets_buf_[b.bsz];

    env.produce("token_num", Buffer{&d.input_token_num, 1, kCPU});

    StageEmbeddingPatches(phase, rc, false);
}

void InputProcessor::Impl::Prepare(int phase, TensorMap& env)
{
    if (speculative_engine_) {
        return PrepareSpeculative(phase, env);
    }
    return PrepareOrdinary(phase, env);
}

void InputProcessor::Impl::PrepareOrdinary(int phase, TensorMap& env)
{
    auto& d    = data_.at(phase);
    auto& b    = *env.at("batch").data<BatchData*>()[0];
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    const Buffer_<int> autoreg_ids = env.at("autoreg_ids").buffer();

    if (auto g = copy.group()) {
        for (int i = 0; i < b.bsz; ++i) {
            if (auto pos = d.autoreg_ids_pos[i]; pos >= 0) {
                TM_CHECK_LT(b.perm[i], b.bs0);
                copy(autoreg_ids.data() + b.perm[i], 1, &d.input_ids[pos]);
            }
        }
    }

    env.produce("input_ids", d.input_ids.slice(0, d.input_token_num));
    env.produce("q_offsets", d.input_ids_offsets.slice(0, b.bsz + 1));
    env.produce("selected_token_pos", d.selected_token_pos.slice(0, d.selected_token_count));
}

void InputProcessor::Impl::StageEmbeddingPatches(int                       phase,
                                                 const Buffer_<Sequence*>& rc,
                                                 bool                      stage_successor)
{
    auto& d = data_.at(phase);

    d.target_input_patches.clear();
    d.successor_input_patches.clear();
    auto embed_ptr  = (uint8_t*)d.input_embeds_buf.raw_data();
    int  staged_row = 0;
    for (int k = 0; k < rc.size(); ++k) {
        auto&               c   = *rc[k];
        const SubmittedRow& row = *c.submitted;
        if (!row.autoregres) {
            const auto& embeds  = c.input_embeds;
            const auto& offsets = c.input_embeds_offsets;
            const int begin = row.history_len + c.inflight_input_len;
            const int end   = begin + row.input_len;
            const Interval target{begin, end};
            const Interval successor{begin + 1, std::min(end + 1, c.seq_len)};
            const Interval staged{begin, std::min(end + 1, c.seq_len)};
            const int packed_begin = input_ids_offsets_buf_[k];
            for (int i = (int)offsets.size() - 1; i >= 0; --i) {
                Interval r{offsets[i], Interval::Size{(int)embeds[i].shape(0)}};
                auto     staged_overlap = r & staged;
                if (auto size = (int)staged_overlap.size()) {
                    auto src = embeds[i].slice(staged_overlap.begin() - r.begin(), size);
                    embed_ptr = std::copy_n((const uint8_t*)src.raw_data(), src.byte_size(), embed_ptr);

                    if (auto o = r & target; !o.empty()) {
                        d.target_input_patches.push_back(
                            {static_cast<int>(o.size()),
                             staged_row + o.begin() - staged_overlap.begin(),
                             packed_begin + o.begin() - target.begin()});
                    }
                    if (stage_successor) {
                        if (auto o = r & successor; !o.empty()) {
                            d.successor_input_patches.push_back(
                                {static_cast<int>(o.size()),
                                 staged_row + o.begin() - staged_overlap.begin(),
                                 packed_begin + o.begin() - successor.begin()});
                        }
                    }
                    staged_row += size;
                }
            }
        }
    }
}

void InputProcessor::Impl::ApplyPatches(const Tensor&                  source,
                                        const std::vector<EmbeddingPatch>& patches,
                                        Tensor&                        destination,
                                        BatchCopy&                     copy)
{
    for (const EmbeddingPatch& patch : patches) {
        copy(source.slice(patch.source_row, patch.row_count).buffer(),
             patch.row_count * destination.shape(1),
             destination.slice(patch.destination_row, patch.row_count).buffer());
    }
}

void InputProcessor::Impl::PatchEmbedding(int phase, Tensor& embeds, BatchCopy& copy, TensorMap& env)
{
    auto& data = data_.at(phase);
    ApplyPatches(data.input_embeds_buf, data.target_input_patches, embeds, copy);
    if (env.try_("multimodal")) {
        const auto& multimodal = *env.at("multimodal").data<MultiModalEmbeddingData*>()[0];
        ApplyPatches(multimodal.data, multimodal.target_patches, embeds, copy);
    }
}

void InputProcessor::Impl::PatchSuccessorEmbedding(int phase, Tensor& embeds, BatchCopy& copy, TensorMap& env)
{
    auto& data = data_.at(phase);
    ApplyPatches(data.input_embeds_buf, data.successor_input_patches, embeds, copy);
    if (env.try_("multimodal")) {
        const auto& multimodal = *env.at("multimodal").data<MultiModalEmbeddingData*>()[0];
        ApplyPatches(multimodal.data, multimodal.successor_patches, embeds, copy);
    }
}

InputProcessor::~InputProcessor() = default;

InputProcessor::InputProcessor(const EngineParam& engine,
                               int                hidden_units,
                               DataType           data_type,
                               int                phases,
                               bool               speculative,
                               int                max_verification_positions,
                               bool               successor_embeddings):
    impl_{std::make_unique<Impl>(
        engine, hidden_units, data_type, phases, speculative, max_verification_positions, successor_embeddings)}
{
}

void InputProcessor::Run(BatchOp op, int phase, TensorMap& env)
{
    switch (op) {
        case BatchOp::kAdd:
            return impl_->Add(phase, env);
        case BatchOp::kSetup:
            return impl_->Setup(phase, env);
        case BatchOp::kPrepare:
            return impl_->Prepare(phase, env);
        default:
            return;
    }
}

void InputProcessor::BuildTargetInputs(int phase, TensorMap& env)
{
    impl_->BuildTargetInputs(phase, env);
}

void InputProcessor::PatchEmbedding(int phase, Tensor& embeds, BatchCopy& copy, TensorMap& env)
{
    impl_->PatchEmbedding(phase, embeds, copy, env);
}

void InputProcessor::PatchSuccessorEmbedding(int phase, Tensor& embeds, BatchCopy& copy, TensorMap& env)
{
    impl_->PatchSuccessorEmbedding(phase, embeds, copy, env);
}

}  // namespace turbomind
