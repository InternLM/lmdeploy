// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/batch_status.h"

#include <algorithm>

#include "src/turbomind/core/copy.h"

namespace turbomind {

using core::BatchCopy;

struct BatchStatus::Data {
    Buffer_<int>  sequence_length;
    Buffer_<int>  readonly_block_num;
    Buffer_<bool> finished;

    Buffer_<bool> autoregres;
    Buffer_<bool> generating;

    int n_generating{};
    int verification_positions{};
};

BatchStatus::BatchStatus(int max_batch_size, int phases): max_batch_size_{max_batch_size}
{
    false_ = {max_batch_size, kDEVICE};
    Clear(false_);

    finished_buf_ = {max_batch_size, kCPUpinned};
    finished_     = {{max_batch_size}, kBool, kDEVICE};

    sequence_length_buf_    = {max_batch_size, kCPUpinned};
    readonly_block_num_buf_ = {max_batch_size, kCPUpinned};
    sequence_length_        = {{max_batch_size}, kInt, kDEVICE};

    data_.reserve(phases);
    for (int i = 0; i < phases; ++i) {
        auto d                = std::make_unique<Data>();
        d->sequence_length    = empty_like(sequence_length_buf_, kDEVICE);
        d->readonly_block_num = empty_like(readonly_block_num_buf_, kDEVICE);
        d->finished           = empty_like(finished_buf_, kDEVICE);
        d->autoregres         = {max_batch_size, kCPU};
        d->generating         = {max_batch_size, kCPU};
        data_.push_back(std::move(d));
    }
}

BatchStatus::~BatchStatus() = default;

void BatchStatus::Run(BatchOp op, int phase, TensorMap& env)
{
    switch (op) {
        case BatchOp::kSetup:
            return Setup(phase, env);
        case BatchOp::kPrepare:
            return Prepare(phase, env);
        case BatchOp::kUnprep:
            return Unprep(phase, env);
        case BatchOp::kFetch:
            return Fetch(phase, env);
        default:
            return;
    }
}

void BatchStatus::Setup(int phase, TensorMap& env)
{
    auto& d    = *data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    Buffer_<Sequence*> requests = env.at("requests").buffer();

    d.n_generating           = 0;
    d.verification_positions = 0;

    for (int i = 0; i < requests.size(); ++i) {
        const Sequence&     request = *requests[i];
        const SubmittedRow& row     = *request.submitted;

        d.autoregres[i] = row.autoregres;
        d.generating[i] = row.generating;
        d.n_generating += row.generating;

        if (row.generating) {
            d.verification_positions = std::max(d.verification_positions, row.verification_positions);
        }

        sequence_length_buf_[i]    = row.autoregres ? request.seq_len : row.key_capacity_end;
        readonly_block_num_buf_[i] = request.readonly_block_num;
    }

    copy(sequence_length_buf_, requests.size(), d.sequence_length);
    copy(readonly_block_num_buf_, requests.size(), d.readonly_block_num);

    env.produce("verification_positions", Buffer_<int>{&d.verification_positions, 1, kCPU});
}

void BatchStatus::Prepare(int phase, TensorMap& env)
{
    auto& d     = *data_.at(phase);
    auto& batch = *env.at("batch").data<BatchData*>()[0];
    auto& copy  = *env.at("copy").data<BatchCopy*>()[0];

    if (auto group = copy.group()) {
        for (int i = 0; i < batch.bsz; ++i) {
            if (const int j = batch.perm[i]; j < batch.bs0) {
                copy(finished_.front().data<bool>() + j, 1, finished_.back().data<bool>() + i);
            }
            else {
                copy(false_.data() + i, 1, finished_.back().data<bool>() + i);
            }
        }
        finished_.Swap();
    }

    if (auto group = copy.group()) {
        for (int i = 0; i < batch.bsz; ++i) {
            if (const int j = batch.perm[i]; j < batch.bs0 && d.autoregres[i]) {
                copy(sequence_length_.front().data<int>() + j, 1, sequence_length_.back().data<int>() + i);
            }
            else {
                copy(d.sequence_length.data() + i, 1, sequence_length_.back().data<int>() + i);
            }
        }
        sequence_length_.Swap();
    }

    env.produce("finished", finished_.front());
    env.produce("sequence_length", sequence_length_.front());
    env.produce("readonly_block_num", d.readonly_block_num);
}

void BatchStatus::Unprep(int phase, TensorMap& env)
{
    auto& d    = *data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    copy(sequence_length_.front().buffer(), d.sequence_length.size(), d.sequence_length);
    copy(finished_.front().buffer(), d.finished.size(), d.finished);
}

void BatchStatus::Fetch(int phase, TensorMap& env)
{
    auto& d    = *data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    copy(d.sequence_length, d.sequence_length.size(), sequence_length_buf_);
    env.produce("sequence_length", sequence_length_buf_);

    copy(d.finished, d.finished.size(), finished_buf_);
    env.produce("finished", finished_buf_);

    env.produce("generating", d.generating);
}

int BatchStatus::VerificationPositions(int phase) const
{
    return data_.at(phase)->verification_positions;
}

int BatchStatus::GeneratingCount(int phase) const
{
    return data_.at(phase)->n_generating;
}

Buffer_<int> BatchStatus::SequenceLength() const
{
    return sequence_length_.data_[0].buffer();
}

}  // namespace turbomind
