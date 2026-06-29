
#include <memory>

#include "src/turbomind/generation/generation.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/generation/guided_decoding.h"
#include "src/turbomind/generation/logits_processor.h"
#include "src/turbomind/generation/sampling.h"
#include "src/turbomind/generation/stop_criteria.h"

#include "src/turbomind/kernels/sampling_topk_kernels.h"  // InitializeRandomStates

#include "src/turbomind/models/llama/llama_kernels.h"  // invokePadLastTokenIds
#include "src/turbomind/utils/cuda_utils.h"

// #include "dbg.h"

namespace turbomind {

using std::unique_ptr;
using std::shared_ptr;
using std::vector;

struct GenerationData {
    Buffer_<uint64_t> random_seed;
    Buffer_<bool>     random_init;
    Buffer_<int>      random_state_indices;
    Buffer_<int>      max_seq_len;
    Buffer_<int*>     token_ids_ptrs;
    Buffer_<int>      output_ids;

    bool random_init_needed;
    int  generation_size;
};

struct Generation::Impl {

    // child modules
    unique_ptr<LogitsProcessor> logits_processor_;
    unique_ptr<Sampling>        sampling_;
    shared_ptr<StopCriteria>    stop_criteria_;
    unique_ptr<GuidedDecoding>  guided_decoding_;

    // persistent
    Tensor_<int> token_ids_;
    Tensor_<uint8_t> random_states_;

    // scheduling states
    vector<int> free_token_rows_;
    vector<int> free_random_state_rows_;

    // immutable states
    Buffer_<int> output_ids_;

    std::vector<std::unique_ptr<GenerationData>> data_;

    // staging buffers
    Buffer_<uint64_t> random_seed_buf_;
    Buffer_<bool>     random_init_buf_;
    Buffer_<int>      random_state_indices_buf_;
    Buffer_<int*>     token_ids_ptrs_buf_;
    Buffer_<int>      token_ids_buf_;
    Buffer_<int>      output_ids_buf_;

    const int max_batch_size_;
    const int session_len_;

    int* RowPtr(int row)
    {
        TM_CHECK_GE(row, 0);
        TM_CHECK_LT(row, max_batch_size_);
        return token_ids_.data() + row * token_ids_.stride(0);
    }

    Impl(DataType              dtype,
         int                   max_batch_size,
         int                   session_len,
         int                   vocab_size,
         int                   vocab_size_padded,
         const comm::HostComm& tp_group,
         int                   phases):
        max_batch_size_{max_batch_size}, session_len_{session_len}
    {
        TM_CHECK_EQ(dtype, kFloat32);
        BaseGenerationParam base{max_batch_size, vocab_size, vocab_size_padded};
        logits_processor_ = std::make_unique<LogitsProcessor>(base, phases);
        sampling_         = std::make_unique<Sampling>(base, phases, tp_group->rank());
        stop_criteria_    = std::make_unique<StopCriteria>(base, phases);
        guided_decoding_  = std::make_unique<GuidedDecoding>(base, tp_group, phases);

        static_assert(sizeof(curandState_t) % alignof(curandState_t) == 0);
        random_states_ = {{max_batch_size_, (int)sizeof(curandState_t)}, kDEVICE};
        token_ids_     = {{max_batch_size_, session_len_}, kDEVICE};
        output_ids_    = {max_batch_size_, kDEVICE};
        for (int i = 0; i < max_batch_size_; ++i) {
            free_token_rows_.push_back(i);
            free_random_state_rows_.push_back(i);
        }

        random_seed_buf_          = {max_batch_size_, kCPUpinned};
        random_init_buf_          = {max_batch_size_, kCPUpinned};
        random_state_indices_buf_ = {max_batch_size_, kCPUpinned};

        token_ids_ptrs_buf_ = {max_batch_size_, kCPUpinned};
        token_ids_buf_      = {max_batch_size_ * (ssize_t)session_len_, kCPUpinned};

        output_ids_buf_ = {max_batch_size_, kCPUpinned};

        for (int i = 0; i < phases; ++i) {
            auto d = std::make_unique<GenerationData>();

            d->random_seed          = empty_like(random_seed_buf_, kDEVICE);
            d->random_init          = empty_like(random_init_buf_, kDEVICE);
            d->random_state_indices = empty_like(random_state_indices_buf_, kDEVICE);
            d->token_ids_ptrs       = empty_like(token_ids_ptrs_buf_, kDEVICE);
            d->output_ids           = empty_like(output_ids_, kDEVICE);

            data_.push_back(std::move(d));
        }
    }

    void Setup(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        auto& d = *data_.at(phase);

        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        Buffer_<Sequence*> rc = env.at("requests").buffer();

        // random states
        d.random_init_needed = false;
        std::fill_n(random_init_buf_.data(), max_batch_size_, false);

        int* token_ids_buf   = token_ids_buf_.data();
        int  generation_size = 0;
        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *rc[i];
            if (!c.generating) {
                continue;
            }

            if (c.generation_random_state_row < 0) {
                TM_CHECK(!free_random_state_rows_.empty());

                c.generation_random_state_row = free_random_state_rows_.back();
                free_random_state_rows_.pop_back();

                random_init_buf_[c.generation_random_state_row] = true;
                random_seed_buf_[c.generation_random_state_row] = c.gen_cfg.random_seed;
                d.random_init_needed = true;
            }

            if (c.generation_token_ids_row < 0) {
                TM_CHECK(!free_token_rows_.empty());

                c.generation_token_ids_row = free_token_rows_.back();
                free_token_rows_.pop_back();

                auto* dst = RowPtr(c.generation_token_ids_row);
                std::copy_n(c.token_ids, c.seq_len, token_ids_buf);
                copy(token_ids_buf, c.seq_len, dst);
                token_ids_buf += c.seq_len;
            }

            random_state_indices_buf_[generation_size] = c.generation_random_state_row;
            token_ids_ptrs_buf_[generation_size++]     = RowPtr(c.generation_token_ids_row);
        }

        if (d.random_init_needed) {
            copy(random_init_buf_, max_batch_size_, d.random_init);
            copy(random_seed_buf_, max_batch_size_, d.random_seed);
        }

        copy(token_ids_ptrs_buf_, generation_size, d.token_ids_ptrs);
        copy(random_state_indices_buf_, generation_size, d.random_state_indices);
        d.generation_size = generation_size;
        // dbg(d.generation_size);

        logits_processor_->Setup(phase, env);
        sampling_->Setup(phase, env);
        stop_criteria_->Setup(phase, env);
        guided_decoding_->Setup(phase, env);
    }

    void Del(TensorMap& env)
    {
        Buffer_<Sequence*> rc = env.at("requests").buffer();

        for (int i = 0; i < rc.size(); ++i) {
            auto& token_row = rc[i]->generation_token_ids_row;
            if (token_row >= 0) {
                free_token_rows_.push_back(token_row);
                token_row = -1;
            }

            auto& random_row = rc[i]->generation_random_state_row;
            if (random_row >= 0) {
                free_random_state_rows_.push_back(random_row);
                random_row = -1;
            }
        }
    }

    void Prepare(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        (void)phase;
        (void)env;
    }

    void Unprep(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        auto& d    = *data_.at(phase);
        auto& b    = *env.at("batch").data<BatchData*>()[0];
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        copy(output_ids_, b.bsz, d.output_ids);
    }

    void Fetch(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        auto& d    = *data_.at(phase);
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        copy(d.output_ids, d.output_ids.size(), output_ids_buf_);
        env.produce("output_ids", output_ids_buf_);

        sampling_->Fetch(phase, env);
    }

    void Update(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        sampling_->Update(phase, env);
    }

    void Forward(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        auto& d = *data_.at(phase);

        const auto stream = core::Context::stream().handle();

        if (d.random_init_needed) {
            InitializeRandomStates((curandState_t*)random_states_.raw_data(),
                                   d.random_seed.data(),
                                   d.random_init.data(),
                                   max_batch_size_,
                                   stream);
        }

        env.emplace("output_ids", output_ids_);              // out
        env.emplace("curand_state", random_states_);         // inout

        if (const int gs = d.generation_size) {

            env.emplace("token_ids_ptrs", d.token_ids_ptrs.slice(0, gs));
            env.emplace("curand_state_indices", d.random_state_indices.slice(0, gs));

            auto logits = env.consume("logits");

            if (logits.dtype() != kFloat32) {
                auto tmp = empty_like(logits, kFloat32);
                TM_SCOPE_CALL(invokeCastFloat2D(logits, tmp, stream));
                logits = std::move(tmp);
            }

            env.produce("logits", logits.slice(0, gs));

            Buffer_<int> output_pos{max_batch_size_, kDEVICE};
            Copy(env.at("sequence_length").buffer(), gs, output_pos);

            logits_processor_->Forward(phase, env);

            guided_decoding_->FillMask(phase, env);
            guided_decoding_->ApplyMask(phase, env);

            sampling_->Forward(phase, env);

            guided_decoding_->Update(phase, env);

            AppendTokenIds(d.token_ids_ptrs.data(), output_ids_.data(), output_pos.data(), gs, stream);

            stop_criteria_->Forward(phase, env);
        }
    }
};

Generation::~Generation() = default;

Generation::Generation(DataType              dtype,
                       int                   max_batch_size,
                       int                   session_len,
                       int                   vocab_size,
                       int                   vocab_size_padded,
                       const comm::HostComm& tp_group,
                       int                   phases):
    impl_{std::make_unique<Impl>(dtype, max_batch_size, session_len, vocab_size, vocab_size_padded, tp_group, phases)}
{
}

void Generation::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kSetup) {
        return impl_->Setup(phase, env);
    }
    else if (op == BatchOp::kDel) {
        return impl_->Del(env);
    }
    else if (op == BatchOp::kPrepare) {
        return impl_->Prepare(phase, env);
    }
    else if (op == BatchOp::kForward) {
        return impl_->Forward(phase, env);
    }
    else if (op == BatchOp::kUnprep) {
        return impl_->Unprep(phase, env);
    }
    else if (op == BatchOp::kFetch) {
        return impl_->Fetch(phase, env);
    }
    else if (op == BatchOp::kUpdate) {
        return impl_->Update(phase, env);
    }
}

}  // namespace turbomind
