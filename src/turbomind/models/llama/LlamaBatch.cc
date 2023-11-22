// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/kernels/decoding_kernels.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/models/llama/Request.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/debug_utils.h"
#include "src/turbomind/utils/gemm_test/gemm_func.h"
#include "src/turbomind/utils/logger.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace turbomind {

void ClearState(BatchState& s)
{
    std::fill_n(s.requests.begin(), s.size, nullptr);
    std::fill_n(s.sequences.begin(), s.size, nullptr);
    s.size = s.active_size = 0;
}

template<typename T>
void LlamaBatch<T>::RejectInvalidRequests(Requests& stop_reqs, Requests& infer_reqs)
{
    std::unordered_map<uint64_t, int> occurrence;

    auto count_occurrence = [&occurrence](const Requests& rs) {
        for (const auto& r : rs) {
            ++occurrence[r->id];
        }
    };

    auto reject = [](const char* type, std::shared_ptr<Request>& req, int ec) {
        TM_LOG_WARNING(
            "[RejectInvalidRequests] Skipping invalid %s request for id %ld, code = %d", type, (long)req->id, ec);
        req->signal.set_value(ec);
        req.reset();
    };

    auto handle_conflict_or_invalid = [this, &occurrence, &reject](Requests& rs, const char* type) {
        for (auto& r : rs) {
            if (r) {
                int ec = 0;

                const int  input_length = r->inputs[rank_].getVal<int>("input_lengths", 0);
                const auto get_offset   = [&](int token_count) {
                    return std::max(0, std::min(token_count, r->inputs[rank_].getVal<int>("step", token_count)));
                };

                if (occurrence[r->id] != 1) {
                    ec = Request::kConflict;
                }
                else if (r->start_flag && r->stop_flag) {
                    ec = Request::kInvalid;
                }
                else if (input_length > session_len_) {
                    ec = Request::kTooLong;
                }
                else if (!r->start_flag) {
                    if (auto seq = sequence_manager_->Get(r->id); seq == nullptr) {
                        ec = Request::kInvalid;
                    }
                    else if (get_offset(seq->tokens.size()) + input_length > session_len_) {
                        ec = Request::kTooLong;
                    }
                }

                if (ec) {
                    reject(type, r, ec);
                }
            }
        }
    };

    auto drop_invalid = [](Requests& rs) {
        int count = 0;
        for (int i = 0; i < rs.size(); ++i) {
            if (rs[i]) {
                rs[count++] = std::move(rs[i]);
            }
        }
        rs.resize(count);
    };

    count_occurrence(stop_reqs);
    count_occurrence(infer_reqs);

    if (!stop_reqs.empty()) {
        handle_conflict_or_invalid(stop_reqs, "stop");

        // invalidate stop-only requests for inactive sequences
        for (auto& r : stop_reqs) {
            if (r && r->end_flag == false) {
                int ec = Request::kInactive;
                for (int i = 0; i < state_->size; ++i) {
                    if (state_->requests[i] && state_->requests[i]->id == r->id) {
                        ec = 0;
                        break;
                    }
                }
                if (ec) {
                    reject("stop", r, ec);
                }
            }
        }

        drop_invalid(stop_reqs);
    }

    if (!infer_reqs.empty()) {
        handle_conflict_or_invalid(infer_reqs, "infer");

        // invalidate requests for busy sequences
        for (auto& r : infer_reqs) {
            if (r) {
                for (int i = 0; i < state_->size; ++i) {
                    if (state_->requests[i] && state_->requests[i]->id == r->id) {
                        reject("infer", r, Request::kBusy);
                        break;
                    }
                }
            }
        }

        drop_invalid(infer_reqs);
    }
}

template<typename T>
auto LlamaBatch<T>::ProcessStopRequests(const Requests& requests) -> std::vector<Signal>
{
    NvtxScope           scope("stop_request");
    std::vector<Signal> signals;
    int                 count = 0;
    for (const auto& r : requests) {
        int ec = Request::kFail;
        // find matching active sequence
        for (int i = 0; i < state_->size; ++i) {
            // stop & optionally erase active sequence
            if (state_->requests[i] && state_->requests[i]->id == r->id) {
                ec = 0;
                signals.push_back(Interrupt(i, true, r->end_flag));
                ++count;
                break;
            }
        }
        // mismatch, try erase inactive sequence, in this case there is no active request to interrupt
        if (ec && r->end_flag) {
            if (sequence_manager_->Erase(r->id)) {
                ec = 0;
            }
        }
        signals.push_back([=] {
            if (rank_ == 0) {
                r->signal.set_value(ec);
            }
        });
    }
    if (count) {
        check_cuda_error(cudaStreamSynchronize(stream_));
    }
    return signals;
}

template<typename T>
void LlamaBatch<T>::ProcessInferRequests(const Requests& requests)
{
    NvtxScope scope("infer_request");
    auto&     state = *incoming_;

    FT_CHECK(state.size == 0);
    FT_CHECK(state.active_size == 0);

    std::vector<int> existing_idx;

    int idx = 0;
    for (const auto& r : requests) {
        FT_CHECK(!state.requests[idx]);

        if (rank_ == 0) {
            TM_LOG_WARNING("[ProcessInferRequests] Request for %ld received.", (long)r->id);
        }

        state.requests[idx] = r;

        // get sequence for the request
        state.sequences[idx] = r->start_flag ? sequence_manager_->Create(r->id) : sequence_manager_->Get(r->id);
        FT_CHECK(state.sequences[idx]);

        auto& seq = *state.sequences[idx];

        if (int step = r->inputs[rank_].getVal<int>("step", -1); step >= 0) {
            if (step <= seq.tokens.size()) {
                seq.tokens.resize(step);
                seq.cache_len = std::min(seq.cache_len, step);
            }
            else if (rank_ == 0) {
                TM_LOG_WARNING(
                    "[ProcessInferRequests] Skipping invalid step (%d) setting for ID %ld", step, (long)seq.id);
            }
        }

        const int  input_length = r->inputs[rank_].getVal<int>("input_lengths");
        const int* input_ids    = r->inputs[rank_].getPtr<int>("input_ids");

        // `output_ids` contains all token ids of the sequences
        const auto output_ids_base = state.output_ids + session_len_ * idx;
        auto       output_ids      = output_ids_base;

        // copy history tokens
        if (!seq.tokens.empty()) {
            output_ids = Copy(seq.tokens.data(), seq.tokens.size(), output_ids);
        }

        // copy input tokens
        if (input_length) {
            output_ids = Copy(input_ids, input_length, output_ids);
        }

        // total context length (history + input)
        state.h_context_length[idx] = output_ids - output_ids_base;
        state.h_finished[idx]       = false;

        const int request_output_len = state.requests[idx]->inputs[rank_].getVal<int>("request_output_len");
        state.seq_len_limit[idx]     = state.h_context_length[idx] + request_output_len;
        // `length_criterion` sets finish flag when step >= seq_limit_len, however when step == seq_limit_len
        // the actual sequence length is seq_limit_len + 1, hence seq_limit_len must truncated to session_len - 1
        if (state.seq_len_limit[idx] >= session_len_) {
            state.seq_len_limit[idx] = session_len_ - 1;
            if (rank_ == 0) {
                const int trunc_output_len = state.seq_len_limit[idx] - state.h_context_length[idx];
                TM_LOG_WARNING(
                    "[ProcessInferRequests] [%ld] total sequence length (%d + %d) exceeds `session_len` (%d), `request_output_len` is truncated to %d",
                    (long)seq.id,
                    state.h_context_length[idx],
                    request_output_len,
                    (int)session_len_,
                    trunc_output_len);
            }
        }

        // compute rope scaling factor
        if (r->start_flag) {
            seq.rope_theta      = model_->attn_params_.rotary_embedding_base;
            auto scaling_factor = 1.f;
            if (r->inputs[rank_].isExist("rope_scaling_factor")) {  // runtime scaling factor
                scaling_factor = r->inputs[rank_].getVal<float>("rope_scaling_factor");
            }
            else if (model_->attn_params_.rope_scaling_factor >= 1.f) {  // infer by `seq_len_limit`
                scaling_factor   = model_->attn_params_.rope_scaling_factor;
                auto max_seq_len = state.seq_len_limit[idx];
                auto max_pos_emb = model_->attn_params_.max_position_embeddings;
                if (max_seq_len > max_pos_emb) {
                    scaling_factor = scaling_factor * max_seq_len / max_pos_emb - (scaling_factor - 1);
                    // scaling_factor = std::max(exp2f(ceilf(log2f((float)max_seq_len / max_pos_emb) + 1.f))
                    // - 1.f, 1.f);
                }
            }
            if (scaling_factor != 1.f) {
                float rope_dim = model_->attn_params_.rotary_embedding_dim;
                seq.rope_theta *= powf(scaling_factor, rope_dim / (rope_dim - 2.f));
                TM_LOG_INFO("[ProcessInferRequests] %ld rope_scaling_factor: %f, rope_theta = %f",
                            (long)seq.id,
                            scaling_factor,
                            seq.rope_theta);
            }
        }
        state.h_rope_theta[idx] = seq.rope_theta;

        if (r->start_flag) {
            // prepare to initialize random state for new sequence
            h_random_seed_[idx] = r->inputs[rank_].getVal<unsigned long long>("random_seed", 0);
        }
        else {
            // Recover device states if not a new sequence
            h_curand_state_[existing_idx.size()] = *(curandState_t*)seq.random_state.data();
            existing_idx.push_back(idx);
        }

        // ! SHARED STATE IS MODIFIED, BARRIER SYNCHRONIZATION REQUIRED
        // assign priority based on arrival time
        if (rank_ == 0) {
            r->priority = request_count_++;
        }

        // increment pointer
        idx++;
    }

    state.size = idx;

    // when there are new sequences
    if (state.size != existing_idx.size()) {
        // copy random seeds to device
        Copy(h_random_seed_, state.size, d_random_seed_);
        // initialize random states
        invokeCurandBatchInitialize(state.curand_state, state.size, d_random_seed_, stream_);
        sync_check_cuda_error();
    }

    if (!existing_idx.empty()) {
        // copy existing curand states to device
        Copy(h_curand_state_, existing_idx.size(), d_curand_state_);
        // insert the states to their correct positions in the batch
        IndexedCopy({}, existing_idx, std::tuple{d_curand_state_, state.curand_state, 1});
    }
}

template<typename T>
bool LlamaBatch<T>::Initialize()
{
    NvtxScope                                scope("initialize");
    std::vector<const Sequence*>             sequences;
    std::vector<Sequence::Status>            status;
    std::vector<uint64_t>                    priorities;
    std::vector<int>                         context_lengths;
    std::vector<std::pair<BatchState*, int>> coords;

    // count the holes introduced by finished requests in from previous iteration or stop requests from
    // current iteration
    int holes{};
    int active_holes{};
    for (int i = 0; i < state_->size; ++i) {
        if (!state_->requests[i]) {
            ++holes;
            if (i < state_->active_size) {
                ++active_holes;
            }
        }
    }

    // dbg(holes, active_holes);

    auto process = [&](BatchState* state) {
        for (int i = 0; i < state->size; ++i) {
            if (auto& r = state->requests[i]) {
                sequences.push_back(state->sequences[i]);
                status.push_back(state->sequences[i]->status);
                priorities.push_back(r->priority);
                context_lengths.push_back(state->h_context_length[i]);
                coords.emplace_back(state, i);
                // clear swap-in flags
                state->is_swap_in[i] = 0;
            }
        }
    };

    process(state_);
    process(incoming_);

    auto outcome = sequence_manager_->Materialize(sequences, context_lengths, priorities, step_length_);

    if (outcome.allocation || outcome.swap_in || outcome.swap_out) {
        dbg(outcome);
    }

    bool exchange = outcome.swap_in + outcome.swap_out > 0;

    std::vector<int> idxs(sequences.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    if (exchange || holes || incoming_->size) {
        // put active ones first
        auto active_end = std::stable_partition(idxs.begin(), idxs.end(), [&](int idx) {
            return sequences[idx]->status == Sequence::kActive;  // present status
        });

        // all blocks are not enough to hold a single sequence
        if (!sequences.empty()) {
            FT_CHECK_WITH_INFO(active_end != idxs.begin(), "No enough blocks.");
        }

        // move swap-ins to the back
        auto swapin_beg = std::stable_partition(idxs.begin(), active_end, [&](int idx) {
            return status[idx] == Sequence::kActive;  // past status
        });

        // sort swap-ins according to missing length
        if (swapin_beg != active_end) {
            std::vector<int> missing_len(sequences.size());
            for (int i = 0; i < sequences.size(); ++i) {
                missing_len[i] = context_lengths[i] - sequences[i]->cache_len;
            }
            std::stable_sort(swapin_beg, active_end, [&](int i, int j) { return missing_len[i] < missing_len[j]; });
        }

        // Copy sequence states to back buffer
        FT_CHECK(back_->size == 0 && back_->active_size == 0);
        std::vector<std::tuple<BatchState*, BatchState*, int, int>> cpys;
        for (const auto& i : idxs) {
            auto& s = *sequences[i];
            if (exchange) {
                const auto& [state, idx] = coords[i];
                // mark swap-ins
                if (status[i] != Sequence::kActive && s.status == Sequence::kActive) {
                    state->is_swap_in[idx] = 1;
                }
            }
            if (s.status == Sequence::kActive) {
                ++back_->active_size;
            }
            cpys.emplace_back(coords[i].first, back_, coords[i].second, back_->size++);
        }
        CopyState(cpys);
        // Swap the buffers
        std::swap(state_, back_);

        ClearState(*back_);
        ClearState(*incoming_);
    }

    FT_CHECK(state_->size <= max_batch_size_);

    /// Update block ptrs when there were
    //  1. swap-in or swap-out
    //  2. holes in the active buffer
    //  3. new allocations (for existing active sequences)
    if (exchange || active_holes || outcome.allocation) {
        // Prepare intermediate buffers
        h_cu_block_counts_[0] = 0;

        auto k_ptrs = h_k_block_ptrs_;
        auto v_ptrs = h_v_block_ptrs_;

        const int batch_size = state_->active_size;

        for (int i = 0; i < batch_size; ++i) {
            const auto& seq = *state_->sequences[i];

            // cumulative num of blocks
            h_cu_block_counts_[i + 1] = h_cu_block_counts_[i] + seq.blocks.size();

            FT_CHECK_WITH_INFO(h_cu_block_counts_[i + 1] <= sequence_manager_->max_block_count(),
                               std::to_string(h_cu_block_counts_[i + 1]));

            k_ptrs = std::transform(seq.blocks.cbegin(), seq.blocks.cend(), k_ptrs, [&](auto p) {
                return reinterpret_cast<uintptr_t>(sequence_manager_->OffsetKey(p->data));
            });
            v_ptrs = std::transform(seq.blocks.cbegin(), seq.blocks.cend(), v_ptrs, [&](auto p) {
                return reinterpret_cast<uintptr_t>(sequence_manager_->OffsetVal(p->data));
            });
        }

        static_assert(sizeof(uintptr_t) == sizeof(void*));

        Copy(h_cu_block_counts_, batch_size + 1, cu_block_counts_);
        Copy(h_k_block_ptrs_, h_cu_block_counts_[batch_size], k_block_ptrs_);
        Copy(h_v_block_ptrs_, h_cu_block_counts_[batch_size], v_block_ptrs_);
    }

    /// Layout of the buffers is changed, generation & sampling need to be re-initialized for correctness when there
    /// were
    //  1. swap-in or swap-out
    //  2. holes in the active buffer
    return exchange || active_holes;
}

template<typename T>
void LlamaBatch<T>::CopyState(const std::vector<std::tuple<BatchState*, BatchState*, int, int>>& desc)
{
    std::vector<int> idxs(desc.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) { return desc[i] < desc[j]; });

    auto get_signature = [&](int i) -> std::pair<BatchState*, BatchState*> {
        return std::make_pair(std::get<0>(desc[idxs[i]]), std::get<1>(desc[idxs[i]]));
    };

    std::vector<int> offsets;
    auto             current = get_signature(0);
    offsets.push_back(0);
    for (int i = 0; i < idxs.size(); ++i) {
        if (auto signature = get_signature(i); signature != current) {
            current = signature;
            offsets.push_back(i);
        }
    }
    offsets.push_back(idxs.size());

    for (int bi = 1; bi < offsets.size(); ++bi) {
        int beg = offsets[bi - 1];
        int end = offsets[bi];

        if (beg == end) {
            continue;
        }

        auto [s, d] = get_signature(beg);

        std::vector<int> s_idx;
        std::vector<int> d_idx;
        for (int i = beg; i < end; ++i) {
            s_idx.push_back(std::get<2>(desc[idxs[i]]));
            d_idx.push_back(std::get<3>(desc[idxs[i]]));
        }

        IndexedCopy(s_idx,
                    d_idx,
                    std::tuple{s->output_ids, d->output_ids, session_len_},
                    std::tuple{s->curand_state, d->curand_state, 1});
    }

    for (const auto& [s, d, si, di] : desc) {
        d->h_context_length[di] = s->h_context_length[si];
        d->h_finished[di]       = s->h_finished[si];
        d->h_rope_theta[di]     = s->h_rope_theta[si];
        d->seq_len_limit[di]    = s->seq_len_limit[si];
        d->sequences[di]        = s->sequences[si];
        d->is_swap_in[di]       = s->is_swap_in[si];
        d->requests[di]         = s->requests[si];
    }
}

template<typename T>
void LlamaBatch<T>::AllocateBuffer(size_t batch_size, size_t session_len)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t batchxbeam = batch_size;

    const size_t hidden_units      = model_->hidden_units_;
    const size_t vocab_size        = model_->vocab_size_padded_;
    const size_t head_dim          = model_->size_per_head_;
    const size_t local_kv_head_num = model_->local_kv_head_num_;
    // +1 padding, BlockIterator does not use predicate
    const size_t max_block_count = sequence_manager_->max_block_count() + 1;

    context_decoder_input_buf_ =
        (T*)allocator_->reMalloc(context_decoder_input_buf_, sizeof(T) * max_context_token_num_ * hidden_units, false);
    context_decoder_output_buf_ =
        (T*)allocator_->reMalloc(context_decoder_output_buf_, sizeof(T) * max_context_token_num_ * hidden_units, false);
    context_decoder_ids_buf_ =
        (int*)allocator_->reMalloc(context_decoder_ids_buf_, sizeof(int) * max_context_token_num_, false);

    tmp_k_cache_buf_ = (T*)allocator_->reMalloc(
        tmp_k_cache_buf_, sizeof(T) * max_context_token_num_ * local_kv_head_num * head_dim, false);
    tmp_v_cache_buf_ = (T*)allocator_->reMalloc(
        tmp_v_cache_buf_, sizeof(T) * max_context_token_num_ * local_kv_head_num * head_dim, false);

    tmp_k_ptrs_ = (void**)allocator_->reMalloc(tmp_k_ptrs_, sizeof(void*) * batch_size, false);
    tmp_v_ptrs_ = (void**)allocator_->reMalloc(tmp_v_ptrs_, sizeof(void*) * batch_size, false);

    decoder_input_buf_  = (T*)allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * hidden_units, false);
    decoder_output_buf_ = (T*)allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units, false);

    input_ids_buf_      = (int*)allocator_->reMalloc(input_ids_buf_, sizeof(int) * batchxbeam * session_len, true);
    input_length_buf_   = (int*)allocator_->reMalloc(input_length_buf_, sizeof(int) * batchxbeam);
    context_length_buf_ = (int*)allocator_->reMalloc(context_length_buf_, sizeof(int) * batchxbeam);

    sequence_lengths_ = (int*)allocator_->reMalloc(sequence_lengths_, sizeof(int) * batchxbeam, false);

    cu_block_counts_ = (int*)allocator_->reMalloc(cu_block_counts_, sizeof(int) * (batch_size + 1));
    k_block_ptrs_    = (uintptr_t*)allocator_->reMalloc(k_block_ptrs_, sizeof(uintptr_t) * max_block_count);
    v_block_ptrs_    = (uintptr_t*)allocator_->reMalloc(v_block_ptrs_, sizeof(uintptr_t) * max_block_count);

    logits_buf_       = (float*)allocator_->reMalloc(logits_buf_, sizeof(float) * batchxbeam * vocab_size, false);
    local_logits_buf_ = (float*)allocator_->reMalloc(local_logits_buf_, sizeof(float) * batchxbeam * vocab_size, false);

    token_ids_buf_ = (int*)allocator_->reMalloc(token_ids_buf_, sizeof(int) * batchxbeam * session_len * 2, true);

    finished_buf_  = (bool*)allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false);
    seq_limit_len_ = (uint32_t*)allocator_->reMalloc(seq_limit_len_, sizeof(uint32_t) * batch_size, false);

    request_output_ids_ptrs_ = (int**)allocator_->reMalloc(request_output_ids_ptrs_, sizeof(int*) * batch_size, true);
    request_output_ids_lens_ = (int*)allocator_->reMalloc(request_output_ids_lens_, sizeof(int) * batch_size, true);
    request_seqlen_ptrs_     = (int**)allocator_->reMalloc(request_seqlen_ptrs_, sizeof(int*) * batch_size, true);

    rope_theta_ = (float*)allocator_->reMalloc(rope_theta_, sizeof(float) * batch_size, false);

    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaBatch<T>::AllocatePersistantBuffer(size_t max_batch_size)
{
    d_stop_words_ = (int*)allocator_->reMalloc(d_stop_words_, sizeof(int) * max_batch_size * kMaxStopBadWordsLen, true);
    d_bad_words_  = (int*)allocator_->reMalloc(d_bad_words_, sizeof(int) * max_batch_size * kMaxStopBadWordsLen, true);
    h_stop_words_ =
        (int*)allocator_->reMalloc(h_stop_words_, sizeof(int) * max_batch_size * kMaxStopBadWordsLen, true, true);
    h_bad_words_ =
        (int*)allocator_->reMalloc(h_bad_words_, sizeof(int) * max_batch_size * kMaxStopBadWordsLen, true, true);

    h_runtime_top_k_ = (int*)allocator_->reMalloc(h_runtime_top_k_, sizeof(int) * max_batch_size, true, true);
    h_runtime_top_p_ = (float*)allocator_->reMalloc(h_runtime_top_p_, sizeof(float) * max_batch_size, true, true);
    h_temperature_   = (float*)allocator_->reMalloc(h_temperature_, sizeof(float) * max_batch_size, true, true);
    h_repetition_penalty_ =
        (float*)allocator_->reMalloc(h_repetition_penalty_, sizeof(float) * max_batch_size, true, true);

    h_random_seed_ = (unsigned long long*)allocator_->reMalloc(
        h_random_seed_, sizeof(unsigned long long) * max_batch_size, true, true);
    d_random_seed_ = (unsigned long long*)allocator_->reMalloc(
        d_random_seed_, sizeof(unsigned long long) * max_batch_size, true, false);

    h_curand_state_ =
        (curandState_t*)allocator_->reMalloc(h_curand_state_, sizeof(curandState_t) * max_batch_size, true, true);
    d_curand_state_ =
        (curandState_t*)allocator_->reMalloc(d_curand_state_, sizeof(curandState_t) * max_batch_size, true, false);

    d_end_ids_buf_ = (int*)allocator_->reMalloc(d_end_ids_buf_, sizeof(int) * max_batch_size, false);
    h_end_ids_buf_ = (int*)allocator_->reMalloc(h_end_ids_buf_, sizeof(int) * max_batch_size, false, true);

    sampling_params_ = {
        {"stop_words_list", (std::byte*)h_stop_words_, (std::byte*)d_stop_words_},
        {"bad_words_list", (std::byte*)h_bad_words_, (std::byte*)d_bad_words_},
        {"runtime_top_k", (std::byte*)h_runtime_top_k_, nullptr},
        {"runtime_top_p", (std::byte*)h_runtime_top_p_, nullptr},
        {"temperature", (std::byte*)h_temperature_, nullptr},
        {"repetition_penalty", (std::byte*)h_repetition_penalty_, nullptr},
    };

    for (auto& s : states_) {
        s.output_ids = (int*)allocator_->reMalloc(s.output_ids, sizeof(int) * max_batch_size * session_len_, true);
        s.curand_state =
            (curandState_t*)allocator_->reMalloc(s.curand_state, sizeof(curandState_t) * max_batch_size, true);
    }

    const size_t max_block_count = sequence_manager_->max_block_count();

    {
        NcclGuard barrier(model_->tensor_para_, stream_, true);
        h_input_ids_buf_ =
            (int*)allocator_->reMalloc(h_input_ids_buf_, sizeof(int) * max_batch_size * session_len_, false, true);
        h_input_length_buf_ =
            (int*)allocator_->reMalloc(h_input_length_buf_, sizeof(int) * max_batch_size, false, true);

        h_tmp_k_ptrs_ = (void**)allocator_->reMalloc(h_tmp_k_ptrs_, sizeof(void*) * max_batch_size, false, true);
        h_tmp_v_ptrs_ = (void**)allocator_->reMalloc(h_tmp_v_ptrs_, sizeof(void*) * max_batch_size, false, true);

        h_cu_block_counts_ =
            (int*)allocator_->reMalloc(h_cu_block_counts_, sizeof(int) * (max_batch_size + 1), false, true);
        h_k_block_ptrs_ =
            (uintptr_t*)allocator_->reMalloc(h_k_block_ptrs_, sizeof(uintptr_t) * max_block_count, false, true);
        h_v_block_ptrs_ =
            (uintptr_t*)allocator_->reMalloc(h_v_block_ptrs_, sizeof(uintptr_t) * max_block_count, false, true);

        for (auto& s : states_) {
            s.h_context_length =
                (int*)allocator_->reMalloc(s.h_context_length, sizeof(int) * max_batch_size, false, true);
            s.h_finished   = (bool*)allocator_->reMalloc(s.h_finished, sizeof(bool) * max_batch_size * 2, false, true);
            s.h_rope_theta = (float*)allocator_->reMalloc(s.h_rope_theta, sizeof(float) * max_batch_size, false, true);
        }

        h_seq_limit_len_ =
            (uint32_t*)allocator_->reMalloc(h_seq_limit_len_, sizeof(uint32_t) * max_batch_size, false, true);

        h_request_output_ids_ptrs_ =
            (int**)allocator_->reMalloc(h_request_output_ids_ptrs_, sizeof(int*) * max_batch_size, true, true);
        h_request_output_ids_lens_ =
            (int*)allocator_->reMalloc(h_request_output_ids_lens_, sizeof(int) * max_batch_size, true, true);
        h_request_seqlen_ptrs_ =
            (int**)allocator_->reMalloc(h_request_seqlen_ptrs_, sizeof(int*) * max_batch_size, true, true);

        h_output_ids_ =
            (int*)allocator_->reMalloc(h_output_ids_, sizeof(int) * max_batch_size * session_len_, false, true);
    }

    is_allocate_persistant_buffer_ = true;
}

template<typename T>
void LlamaBatch<T>::FreeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        allocator_->free((void**)&context_decoder_input_buf_);
        allocator_->free((void**)&context_decoder_output_buf_);
        allocator_->free((void**)&context_decoder_ids_buf_);

        allocator_->free((void**)&tmp_k_cache_buf_);
        allocator_->free((void**)&tmp_v_cache_buf_);
        allocator_->free((void**)&tmp_k_ptrs_);
        allocator_->free((void**)&tmp_v_ptrs_);

        allocator_->free((void**)&decoder_input_buf_);
        allocator_->free((void**)&decoder_output_buf_);

        allocator_->free((void**)&input_ids_buf_);
        allocator_->free((void**)&input_length_buf_);
        allocator_->free((void**)&context_length_buf_);

        allocator_->free((void**)&sequence_lengths_);

        allocator_->free((void**)&cu_block_counts_);
        allocator_->free((void**)&k_block_ptrs_);
        allocator_->free((void**)&v_block_ptrs_);

        allocator_->free((void**)&logits_buf_);
        allocator_->free((void**)&local_logits_buf_);

        if (local_context_logits_buf_) {
            allocator_->free((void**)&local_context_logits_buf_);
        }
        if (context_logits_buf_) {
            allocator_->free((void**)&context_logits_buf_);
        }

        allocator_->free((void**)&token_ids_buf_);

        allocator_->free((void**)&d_end_ids_buf_);
        allocator_->free((void**)&h_end_ids_buf_, true);

        allocator_->free((void**)&finished_buf_);
        allocator_->free((void**)&seq_limit_len_);

        allocator_->free((void**)&request_output_ids_ptrs_);
        allocator_->free((void**)&request_output_ids_lens_);
        allocator_->free((void**)&request_seqlen_ptrs_);

        allocator_->free((void**)&rope_theta_);

        is_allocate_buffer_ = false;
    }

    if (is_allocate_persistant_buffer_) {

        allocator_->free((void**)&d_stop_words_);
        allocator_->free((void**)&h_stop_words_, true);
        allocator_->free((void**)&d_bad_words_);
        allocator_->free((void**)&h_bad_words_, true);
        allocator_->free((void**)&d_random_seed_);
        allocator_->free((void**)&h_random_seed_, true);
        allocator_->free((void**)&d_curand_state_);
        allocator_->free((void**)&h_curand_state_, true);

        for (auto& s : states_) {
            allocator_->free((void**)&s.h_context_length, true);
            allocator_->free((void**)&s.h_finished, true);
            allocator_->free((void**)&s.h_rope_theta, true);
            allocator_->free((void**)&s.output_ids);
            allocator_->free((void**)&s.curand_state);
        }
        allocator_->free((void**)&h_tmp_k_ptrs_, true);
        allocator_->free((void**)&h_tmp_v_ptrs_, true);
        allocator_->free((void**)&h_cu_block_counts_, true);
        allocator_->free((void**)&h_k_block_ptrs_, true);
        allocator_->free((void**)&h_v_block_ptrs_, true);
        allocator_->free((void**)&h_input_ids_buf_, true);
        allocator_->free((void**)&h_input_length_buf_, true);
        allocator_->free((void**)&h_seq_limit_len_, true);

        allocator_->free((void**)&h_request_output_ids_ptrs_, true);
        allocator_->free((void**)&h_request_output_ids_lens_, true);
        allocator_->free((void**)&h_request_seqlen_ptrs_, true);

        allocator_->free((void**)&h_output_ids_, true);

        is_allocate_persistant_buffer_ = false;
    }
}

template<typename T>
LlamaBatch<T>::LlamaBatch(int                              max_batch_size,
                          int                              max_context_token_num,
                          int                              session_len,
                          std::unique_ptr<SequenceManager> sequence_manager,
                          LlamaV2<T>*                      model):
    max_batch_size_(max_batch_size),
    max_context_token_num_(max_context_token_num),
    session_len_(session_len),
    rank_(model->tensor_para_.rank_),
    debug_(model->debug_),
    step_length_(model->step_length_),
    sequence_manager_(std::move(sequence_manager)),
    model_(model),
    data_type_(getTensorType<T>())
{
    stream_         = model_->stream_;
    allocator_      = model_->allocator_;
    cublas_wrapper_ = model_->cublas_wrapper_;

    for (auto& s : states_) {
        s.requests.resize(max_batch_size);
        s.sequences.resize(max_batch_size);
        s.seq_len_limit.resize(max_batch_size);
        s.is_swap_in.resize(max_batch_size);
    }

    state_    = &states_[0];
    back_     = &states_[1];
    incoming_ = &states_[2];

    AllocateBuffer(max_batch_size, session_len_);
    AllocatePersistantBuffer(max_batch_size);
}

template<typename T>
void LlamaBatch<T>::InitializeSampling()
{
    NvtxScope _("InitSampling");
    const int batch_size = state_->active_size;
    TensorMap inputs;
    for (const auto& [name, h_ptr, d_ptr] : sampling_params_) {
        // find an exemplar that matches the param name
        const Tensor* ptr{};
        for (int i = 0; i < batch_size; ++i) {
            if (state_->requests[i]->inputs[rank_].isExist(name)) {
                ptr = &state_->requests[i]->inputs[rank_].at(name);
                break;
            }
        }
        // fill the batch of the param
        if (ptr) {
            const auto& ref   = *ptr;
            auto        shape = ref.shape;
            FT_CHECK(shape[0] == 1);
            shape[0]                = batch_size;
            const int size_in_bytes = ref.sizeBytes();
            memset(h_ptr, 0, size_in_bytes * batch_size);
            for (int i = 0; i < batch_size; ++i) {
                if (state_->requests[i]->inputs[rank_].isExist(name)) {
                    Tensor& src = state_->requests[i]->inputs[rank_].at(name);
                    FT_CHECK(ref.shape == src.shape);
                    std::copy_n(src.getPtr<std::byte>(), size_in_bytes, h_ptr + size_in_bytes * i);
                }
            }
            if (d_ptr) {
                Copy(h_ptr, batch_size * size_in_bytes, d_ptr);
            }
            inputs.insert({name, {d_ptr ? MEMORY_GPU : MEMORY_CPU, ref.type, shape, d_ptr ? d_ptr : h_ptr}});
            if (debug_ && rank_ == 0) {
                TM_LOG_INFO("[initializeSampling] %s", format({name, inputs.at(name)}).c_str());
            }
        }
    }

    // init for eos
    std::fill_n(h_end_ids_buf_, batch_size, model_->end_id_);
    Copy(h_end_ids_buf_, batch_size, d_end_ids_buf_);
    inputs.insert({"end_id", {MEMORY_GPU, TYPE_INT32, {(size_t)batch_size}, d_end_ids_buf_}});

    inputs_ = std::move(inputs);

    model_->dynamic_decode_layer_->setup(batch_size, 1, &inputs_);
}

template<typename T>
auto LlamaBatch<T>::InitializeGeneration() -> GenerationState
{
    NvtxScope _("InitGen");
    const int batch_size      = state_->active_size;
    const int max_context_len = *std::max_element(state_->h_context_length, state_->h_context_length + batch_size);

    Copy(state_->h_context_length, batch_size, context_length_buf_);  // also referenced in `SetOutputTensors`
    Copy(context_length_buf_, batch_size, sequence_lengths_);
    // `sequence_lengths_` will be increased by dynamic decode
    // note that in decoder and in output "sequence length" has different semantic
    // - in decoder it means length of sequence that has kv cache already computed
    // - in output it means length of all tokens (the last generated token does not have k/v cache computed yet)
    invokePlusScalar(sequence_lengths_, -1, batch_size, stream_);
    sync_check_cuda_error();

    Clear(token_ids_buf_, batch_size * session_len_);
    invokeTransposeAxis01(token_ids_buf_, state_->output_ids, batch_size, session_len_, 1, stream_);
    sync_check_cuda_error();

    // token_ids_buf_[s, b]
    // ABCDe            ABCDe     e
    // ABCDEFGHIJk      ABCDEFGHIJk
    // ABCDEFGHi    ->  ABCDEFGHi i
    // ABCDEFGh         ABCDEFGh  h
    // ABCd             ABCd      d
    invokePadLastTokenIds(token_ids_buf_, context_length_buf_, max_context_len, batch_size, stream_);
    sync_check_cuda_error();

    // used for dispatching split-k decoding kernels
    const int sum_seq_len =
        std::accumulate(state_->h_context_length, state_->h_context_length + batch_size, -batch_size);
    const int max_seq_len = *std::max_element(state_->h_context_length, state_->h_context_length + batch_size) - 1;

    // seq_limit_len_, will be compared to `step` instead of `sequence_length`, so padding len should be accounted
    // for
    for (int i = 0; i < batch_size; ++i) {
        h_seq_limit_len_[i] = state_->seq_len_limit[i] + (max_context_len - state_->h_context_length[i]);
    }
    Copy(h_seq_limit_len_, batch_size, seq_limit_len_);
    Copy(state_->h_finished, batch_size, finished_buf_);

    for (int i = 0; i < batch_size; ++i) {
        Tensor& output_ids         = state_->requests[i]->outputs[rank_].at("output_ids");
        int*    req_output_ids_ptr = output_ids.getPtr<int>();
        int*    req_seqlen_ptr     = state_->requests[i]->outputs[rank_].getPtr<int>("sequence_length");

        h_request_output_ids_ptrs_[i] = req_output_ids_ptr;
        h_request_output_ids_lens_[i] = output_ids.shape.at(2);
        h_request_seqlen_ptrs_[i]     = req_seqlen_ptr;

        FT_CHECK(h_request_output_ids_ptrs_[i]);
        FT_CHECK(h_request_output_ids_lens_[i]);
        FT_CHECK(h_request_seqlen_ptrs_[i]);
    }
    Copy(h_request_output_ids_ptrs_, batch_size, request_output_ids_ptrs_);
    Copy(h_request_output_ids_lens_, batch_size, request_output_ids_lens_);
    Copy(h_request_seqlen_ptrs_, batch_size, request_seqlen_ptrs_);

    Copy(state_->h_rope_theta, batch_size, rope_theta_);

    // ! range of step_ [1, 2 * session_len]
    // consider a sequence with context_len == session_len and another sequence with context_len == 1 and
    // request_output_len == session_len - 1 => step_ will loop in [session_len, 2 * session_len)
    const int start_step = max_context_len;

    if (rank_ == 0) {
        TM_LOG_INFO("[initGen] batch_size = %d", (int)batch_size);
        TM_LOG_INFO("[initGen] max_context_len = %d", (int)max_context_len);

        if (debug_) {
            TM_LOG_INFO("[initGen] slot  sequence_id  context_len  seq_limit_len  finished");
            for (int i = 0; i < batch_size; ++i) {
                TM_LOG_INFO("[initGen] %4d  %11ld  %11d  %13d  %8d",
                            i,
                            (long)state_->sequences[i]->id,
                            state_->h_context_length[i],
                            (int)h_seq_limit_len_[i],
                            (int)state_->h_finished[i]);
            }
        }
    }

    return GenerationState{max_context_len, start_step, sum_seq_len, max_seq_len};
}

template<typename T>
bool LlamaBatch<T>::Generate(GenerationState& g)
{
    NvtxScope scope("Generate");
    const int batch_size = state_->active_size;

    constexpr int kLogInterval = 10;
    if (rank_ == 0 && (g.step - 1) % kLogInterval == 0) {
        TM_LOG_INFO("------------------------- step = %d -------------------------", g.step - 1);
    }

    const bool is_first_step = (g.step == g.max_init_ctx_len);

    std::vector<int> prev;
    if (debug_ && rank_ == 0 && is_first_step) {
        prev.resize(batch_size);
        Copy(token_ids_buf_ + (g.step - 1) * batch_size, batch_size, prev.data());
    }

    // embeddingLookup(step_ - 1);
    model_->embeddingLookup(decoder_input_buf_,  //
                            token_ids_buf_,
                            batch_size,
                            g.step - 1);

    model_->decoderForward(decoder_output_buf_,
                           k_block_ptrs_,
                           v_block_ptrs_,
                           decoder_input_buf_,
                           sequence_lengths_,
                           finished_buf_,
                           cu_block_counts_,
                           rope_theta_,
                           g.step,
                           0,
                           g.sum_seq_len,
                           g.max_seq_len,
                           batch_size);

    model_->postDecodeEmbedding(logits_buf_,  //
                                local_logits_buf_,
                                decoder_output_buf_,
                                batch_size);

    /// sync for better NVTX visualization, THIS IS NOT NEEDED
    // check_cuda_error(cudaStreamSynchronize(stream_));

    // stop-words & bad-words require the matched tokens to be contiguous, so item size > 1 is
    // not supported yet.
    bool should_stop{};
    model_->dynamicDecode(token_ids_buf_,
                          finished_buf_,
                          sequence_lengths_,
                          &should_stop,
                          state_->curand_state,
                          &inputs_,
                          &outputs_,
                          logits_buf_,
                          seq_limit_len_,
                          context_length_buf_,
                          d_end_ids_buf_,
                          g.step,
                          0,
                          g.max_init_ctx_len,
                          session_len_ * 2,
                          batch_size);

    if (debug_ && rank_ == 0) {
        std::vector<int> curr(batch_size);

        Copy(token_ids_buf_ + g.step * batch_size, batch_size, curr.data());
        cudaStreamSynchronize(stream_);

        if (is_first_step) {
            std::stringstream sprev;
            for (int k = 0; k < prev.size(); ++k) {
                sprev << std::setw(6) << prev[k];
            }
            TM_LOG_INFO("[ lookup ] step = %d, [%s]", g.step - 1, sprev.str().c_str());
        }

        std::stringstream scurr;
        for (int k = 0; k < curr.size(); ++k) {
            scurr << std::setw(6) << curr[k];
        }
        TM_LOG_INFO("[generate] step = %d, [%s]", g.step - 1, scurr.str().c_str());
    }

    ////////////////////////////////////////////////
    /// ! increase the counters
    g.step += 1;
    g.max_seq_len += 1;
    g.sum_seq_len += batch_size;

    return !should_stop;
}

template<typename T>
void LlamaBatch<T>::ContextDecode()
{
    NvtxScope  _("prefill");
    const auto batch_size = state_->active_size;

    int base = -1;
    for (int i = 0; i < batch_size; ++i) {
        if (state_->is_swap_in[i]) {
            const auto& seq = *state_->sequences[i];
            dbg(std::tuple(i, state_->h_context_length[i], seq.cache_len));
            if (const int missing = state_->h_context_length[i] - seq.cache_len; missing > 1) {
                base = base < 0 ? i : base;
                dbg(seq.tokens, seq.cache_len);
                Copy(state_->output_ids + i * session_len_ + seq.cache_len, missing, input_ids_buf_ + i * session_len_);
                // subtract input/context len by 1 to skip last input token (will process with decoder later)
                h_input_length_buf_[i] = missing - 1;
            }
        }
    }
    if (base < 0) {
        // TM_LOG_INFO("[decodeContext] Context decoding is not needed.");
        return;
    }

    const int context_decode_count = batch_size - base;

    Copy(state_->h_context_length, batch_size, context_length_buf_);
    Copy(state_->h_rope_theta, batch_size, rope_theta_);
    Copy(h_input_length_buf_, batch_size, input_length_buf_);

    // check_cuda_error(cudaStreamSynchronize(stream_));
    // const auto tick = std::chrono::high_resolution_clock::now();

    if (rank_ == 0) {
        TM_LOG_INFO("[decodeContext] base = %d, count = %d", base, context_decode_count);
    }
    // subtract input/context len by 1 to skip last input token (will process with decoder later)
    invokePlusScalar(context_length_buf_ + base, -1, context_decode_count, stream_);

    // find sub-batch offsets
    std::vector<int> offsets{base};
    std::vector<int> max_context_cnts;
    int              accum_size        = 0;
    int              accum_input_count = 0;
    int              max_context_count = 0;
    for (int i = base; i < batch_size; ++i) {
        int size          = accum_size + 1;
        int input_count   = accum_input_count + h_input_length_buf_[i];
        int context_count = std::max(max_context_count, state_->h_context_length[i] - 1);
        // we have `cu_seqlens` on q so no padding for input is needed
        // kernels are expecting uniform k/v cache length -> `max_context_count * size <= max_context_token_num_`
        if (input_count <= max_context_token_num_ && context_count * size <= max_context_token_num_) {
            accum_size        = size;
            accum_input_count = input_count;
            max_context_count = context_count;
        }
        else {
            offsets.push_back(i);
            max_context_cnts.push_back(max_context_count);
            accum_size        = 1;
            accum_input_count = h_input_length_buf_[i];
            max_context_count = state_->h_context_length[i] - 1;
        }
    }
    offsets.push_back(batch_size);
    max_context_cnts.push_back(max_context_count);

    dbg(offsets, max_context_cnts);

    // context decode on sub-batches
    for (int k = 0; k < offsets.size() - 1; ++k) {
        int              first          = offsets[k];
        int              last           = offsets[k + 1];
        int              sub_batch_size = last - first;
        T*               k_ptr          = tmp_k_cache_buf_;
        T*               v_ptr          = tmp_v_cache_buf_;
        std::vector<int> decode_indices{};
        std::vector<int> decode_lengths{};
        int              max_input_len{};
        auto             input_ids = context_decoder_ids_buf_;
        TM_LOG_INFO("first = %d, last = %d", first, last);
        for (int i = first; i < last; ++i) {
            // TM_LOG_INFO("session_len = %d, input_length = %d", session_len_, h_input_length_buf_[i]);
            input_ids = Copy(input_ids_buf_ + i * session_len_, h_input_length_buf_[i], input_ids);
            dbg(i, h_input_length_buf_[i]);
            h_tmp_k_ptrs_[i] = k_ptr;
            h_tmp_v_ptrs_[i] = v_ptr;
            k_ptr += model_->local_kv_head_num_ * max_context_cnts[k] * model_->size_per_head_;
            v_ptr += model_->local_kv_head_num_ * max_context_cnts[k] * model_->size_per_head_;
            decode_indices.push_back(i);
            decode_lengths.push_back(h_input_length_buf_[i]);
            max_input_len = std::max(max_input_len, h_input_length_buf_[i]);
        }
        int token_count = input_ids - context_decoder_ids_buf_;
        dbg(token_count, max_input_len, max_context_cnts[k]);

        Copy(h_tmp_k_ptrs_ + first, sub_batch_size, tmp_k_ptrs_ + first);
        Copy(h_tmp_v_ptrs_ + first, sub_batch_size, tmp_v_ptrs_ + first);

        if (rank_ == 0) {
            TM_LOG_INFO(
                "[decodeContext] offset = %d, batch_size = %d, token_num = %d, max_input_len = %d, max_context_len = %d",
                base,
                sub_batch_size,
                token_count,
                max_input_len,
                max_context_cnts[k]);
        }

        dbg(first, last);
        dbg(k_block_ptrs_, v_block_ptrs_);

        model_->contextDecode(nullptr,
                              k_block_ptrs_,
                              v_block_ptrs_,
                              tmp_k_ptrs_ + first,
                              tmp_v_ptrs_ + first,
                              context_decoder_input_buf_,
                              context_decoder_output_buf_,
                              context_decoder_ids_buf_,
                              input_length_buf_ + first,
                              context_length_buf_ + first,
                              cu_block_counts_ + first,
                              rope_theta_ + first,
                              token_count,
                              max_input_len,
                              max_context_cnts[k],
                              max_context_cnts[k],
                              sub_batch_size);

        // compute logits of inputs if requested
        OutputContextLogits(context_decoder_output_buf_, decode_indices, decode_lengths);
    }

    invokePlusScalar(context_length_buf_ + base, 1, context_decode_count, stream_);

    std::fill(h_input_length_buf_ + base, h_input_length_buf_ + batch_size, 0);

    // `SequenceManager` needs real-time value of cache length
    for (int i = base; i < batch_size; ++i) {
        if (state_->requests[i]) {
            FT_CHECK(state_->sequences[i]);
            state_->sequences[i]->cache_len = state_->h_context_length[i] - 1;  // -1 since we skip last token
        }
    }

    // check_cuda_error(cudaStreamSynchronize(stream_));
    // const auto tock = std::chrono::high_resolution_clock::now();
    // if (rank_ == 0) {
    //     TM_LOG_INFO("[decodeContext] %.2f ms", std::chrono::duration<float, std::milli>(tock - tick).count());
    // }
}

template<typename T>
void LlamaBatch<T>::OutputContextLogits(T*                      context_decoder_output,
                                        const std::vector<int>& indices,
                                        const std::vector<int>& lengths)
{
    std::vector<float*> output_logits;
    int                 num_token = 0;
    {
        bool is_return_logits = false;
        for (int k = 0; k < indices.size(); ++k) {
            auto& request = state_->requests[indices[k]];
            output_logits.push_back(request->outputs[rank_].getPtr<float>("logits", nullptr));
            num_token += lengths[k];
            if (output_logits.back()) {
                is_return_logits = true;
            }
        }
        if (!is_return_logits) {
            return;
        }
    }

    if (context_logits_buf_ == nullptr) {
        NcclGuard guard(model_->tensor_para_, stream_, true);
        context_logits_buf_ =
            (float*)allocator_->malloc(sizeof(float) * model_->vocab_size_padded_ * max_context_token_num_);
        const auto tp = model_->tensor_para_.world_size_;
        if (tp > 1) {
            FT_CHECK(model_->vocab_size_padded_ % tp == 0);
            const auto local_vocab_size = model_->vocab_size_padded_ / tp;
            local_context_logits_buf_ =
                (float*)allocator_->malloc(sizeof(float) * local_vocab_size * max_context_token_num_);
        }
    }

    model_->postDecodeEmbedding(context_logits_buf_, local_context_logits_buf_, context_decoder_output, num_token);

    auto logits = context_logits_buf_;

    for (int k = 0; k < indices.size(); ++k) {
        if (output_logits[k]) {
            Copy(logits, model_->vocab_size_ * lengths[k], output_logits[k]);
        }
        logits += model_->vocab_size_padded_ * lengths[k];
    }
}

template<typename T>
auto LlamaBatch<T>::Finish(GenerationState& g, int& finished_count) -> std::vector<Signal>
{
    NvtxScope scope("Finish");
    const int batch_size = state_->active_size;

    // [s,b] -> [b,s] and skip padding in [context_len, max_context_len)
    invokeGatherOutput(state_->output_ids,
                       token_ids_buf_,
                       context_length_buf_,
                       g.max_init_ctx_len,
                       g.step,
                       session_len_,
                       batch_size,
                       stream_);
    sync_check_cuda_error();

    Copy(state_->output_ids, batch_size * session_len_, h_output_ids_);
    Copy(finished_buf_, batch_size, state_->h_finished);
    Copy(sequence_lengths_, batch_size, state_->h_context_length);

    check_cuda_error(cudaStreamSynchronize(stream_));

    // `SequenceManager` needs real-time value of cache length
    // ! Must be done before incrementing `h_context_length` because the generated token is NOT kv-cached yet
    for (int i = 0; i < batch_size; ++i) {
        if (state_->requests[i]) {
            FT_CHECK(state_->sequences[i]);
            state_->sequences[i]->cache_len = state_->h_context_length[i];
        }
    }

    // invariant: context_length = sequence_length + 1, so that h_context_length include all (including the one just
    // generated) tokens
    for (int i = 0; i < batch_size; ++i) {
        ++state_->h_context_length[i];
    }

    {  // set output tokens ids and sequence length
        int* output_ptr = h_output_ids_;
        for (int i = 0; i < batch_size; ++i) {
            if (state_->requests[i] && (state_->requests[i]->stream_cb || state_->h_finished[i])) {
                const int count = state_->h_context_length[i];
                // TODO: sync history output tokens at when receiving the request and copy only the last token here
                std::copy(output_ptr, output_ptr + count, h_request_output_ids_ptrs_[i]);
                *h_request_seqlen_ptrs_[i] = count;
            }
            output_ptr += session_len_;
        }
    }

    if (debug_ && rank_ == 0) {
        std::stringstream ss;
        for (int i = 0; i < batch_size; ++i) {
            ss << (i ? ", " : "") << "(" << state_->h_context_length[i] << "," << state_->h_finished[i] << ")";
        }
        TM_LOG_INFO("[finish] [%s]", ss.str().c_str());
    }

    std::vector<Signal> signals;
    {
        NvtxScope _("stream_and_completion_signal");
        for (int i = 0; i < batch_size; ++i) {
            if (state_->requests[i]) {
                if (state_->h_finished[i]) {
                    // Interrupt finished sequences and move the request handle into the signal closure
                    signals.push_back(Interrupt(i));
                    ++finished_count;
                }
                else if (state_->requests[i]->stream_cb) {
                    // Create signals by copying the request handles for non-finished streaming requests
                    signals.push_back([this, r = state_->requests[i]] {
                        if (rank_ == 0) {
                            r->stream_cb(&r->outputs[rank_].get());
                        }
                    });
                }
            }
        }
        if (finished_count) {
            // synchronize for interrupted sequences
            check_cuda_error(cudaStreamSynchronize(stream_));
        }
    }
    return signals;
}

template<typename T>
auto LlamaBatch<T>::Interrupt(int index, bool force_stop, bool force_end) -> Signal
{
    if (rank_ == 0) {
        TM_LOG_INFO("[Interrupt] slot = %d, id = %lu", index, (long)state_->requests[index]->id);
    }

    if (debug_ && rank_ == 0) {
        std::vector<int> tokens(state_->h_context_length[index]);
        Copy(state_->output_ids + index * session_len_, tokens.size(), tokens.data());
        cudaStreamSynchronize(stream_);
        std::stringstream ss;
        for (const auto& t : tokens) {
            ss << " " << t;
        }
        TM_LOG_INFO("[Interrupt] slot %d, tokens [%s]", index, ss.str().c_str());
    }

    if (state_->requests[index]->end_flag || force_end) {
        // Sequence is ending this round or a stop request is issued to end it
        FT_CHECK(sequence_manager_->Erase(state_->requests[index]->id));
    }
    else {
        const int output_len = state_->h_context_length[index];
        auto&     seq        = *state_->sequences[index];

        // Update token IDs
        seq.tokens.resize(output_len);
        const auto output_ids_data = state_->requests[index]->outputs[rank_].at("output_ids").getPtr<int>();
        std::copy_n(output_ids_data, output_len, seq.tokens.data());

        // Save random state in host memory
        seq.random_state.resize(sizeof(curandState_t));
        // This async copy must be synchronized by the caller
        Copy(state_->curand_state + index, 1, (curandState_t*)seq.random_state.data());

        // Set unlock flag for corresponding blocks, will be unlocked in the next `Materialize()`
        sequence_manager_->UpdateAndSetUnlock(seq);
    }

    state_->sequences[index] = nullptr;

    // move the request handle into the signal
    return [this, r = std::move(state_->requests[index])] {
        if (rank_ == 0) {
            r->signal.set_value(0);
        }
    };
}

template<typename T>
void LlamaBatch<T>::InternalThreadEntry(int device_id)
{
    // TM_LOG_INFO("[InternalThreadEntry] %d", (int)rank_);
    check_cuda_error(cudaSetDevice(device_id));

    auto& shared_state = model_->shared_state_;

    auto& request_queue  = shared_state->request_queue;
    auto& infer_requests = shared_state->infer_requests;
    auto& stop_requests  = shared_state->stop_requests;

    // sequences that are removed but still counted in state's size
    int finished_count = 0;

    GenerationState g{};

    constexpr int request_interval = 1;
    long          request_counter  = 0;

    while (1) {
        if (rank_ == 0) {
            const int  free_slot_count = max_batch_size_ - state_->size + finished_count;
            const bool is_empty        = (free_slot_count == max_batch_size_);
            stop_requests.clear();
            infer_requests.clear();
            if (is_empty || request_counter % request_interval == 0) {
                // Block if batch is empty
                request_queue.dequeue(stop_requests, infer_requests, free_slot_count, is_empty, shared_state->abort);
                if (!shared_state->abort) {
                    RejectInvalidRequests(stop_requests, infer_requests);
                }
            }
        }

        NvtxScope scope("mainloop");

        // wait while rank-0 is dequeueing
        shared_state->barrier->wait();

        if (shared_state->abort) {
            TM_LOG_INFO("[InternalThreadEntry] stop requested.");
            return;
        }

        auto signals = ProcessStopRequests(stop_requests);

        // Shared `priority` field will be assigned by rank-0
        ProcessInferRequests(infer_requests);

        // Wait while shared `requests` is being used
        shared_state->barrier->wait();

        SendSignals(std::move(signals));

        auto modified = Initialize();
        // finished sequences is handled by `Initialize()`
        finished_count = 0;

        ContextDecode();

        if (state_->active_size) {
            if (modified) {
                g = InitializeGeneration();
                InitializeSampling();
            }
            for (int i = 0; i < step_length_; ++i) {
                if (!Generate(g)) {
                    break;
                }
            }
            if (auto signals = Finish(g, finished_count); !signals.empty()) {
                if (finished_count) {
                    // Finished requests and corresponding output tensors will be released when notified
                    // wait for all ranks to ensure no rank (except for output thread) will access related
                    // resources
                    shared_state->barrier->wait();
                }
                SendSignals(std::move(signals));
            }
        }

        ++request_counter;
    }

    FT_CHECK(0);
}

template<typename T>
void LlamaBatch<T>::SendSignals(std::vector<Signal> signals)
{
    if (rank_ != 0 || signals.empty()) {
        return;
    }
    {
        std::lock_guard lock{output_mutex_};
        output_signals_.insert(output_signals_.end(),  //
                               std::move_iterator{signals.begin()},
                               std::move_iterator{signals.end()});
    }
    output_cv_.notify_one();
}

template<typename T>
void LlamaBatch<T>::Start()
{
    TM_LOG_INFO("LlamaBatch<T>::Start()");
    int device_id = -1;
    check_cuda_error(cudaGetDevice(&device_id));
    internal_thread_ = std::thread(&LlamaBatch::InternalThreadEntry, this, device_id);
    if (rank_ == 0) {
        output_thread_ = std::thread(&LlamaBatch::OutputThreadEntry, this);
    }
}

template<typename T>
void LlamaBatch<T>::OutputThreadEntry()
{
    while (true) {
        std::vector<Signal> signals;
        {
            // Wait for signals to come
            std::unique_lock lock(output_mutex_);
            output_cv_.wait(lock, [&] { return !output_signals_.empty() || output_stop_token_; });
            if (output_stop_token_) {
                TM_LOG_INFO("[OutputThreadEntry] stop requested.");
                return;
            }
            signals = std::move(output_signals_);
        }
        if (rank_ == 0 && model_->ffi_lock_) {
            model_->ffi_lock_(1);
        }
        // invoke stream cbs & signals
        for (const auto& s : signals) {
            s();
        }
        if (rank_ == 0 && model_->ffi_lock_) {
            model_->ffi_lock_(0);
        }
    }
}

template class LlamaBatch<half>;
template class LlamaBatch<float>;

}  // namespace turbomind
