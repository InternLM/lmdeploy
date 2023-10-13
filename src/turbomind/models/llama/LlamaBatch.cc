// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/kernels/decoding_kernels.h"
#include "src/turbomind/macro.h"
#include "src/turbomind/models/llama/LlamaNcclGuard.h"
#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/models/llama/Request.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/Tensor.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/dbg.h"
#include "src/turbomind/utils/logger.h"
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <math.h>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace turbomind {

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

                if (occurrence[r->id] != 1) {
                    ec = Request::kConflict;
                }
                else if (r->start_flag && r->stop_flag) {
                    ec = Request::kInvalid;
                }
                else if (!r->start_flag && !sequence_manager_->Contains(r->id)) {
                    ec = Request::kInvalid;
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
    std::vector<Signal> signals;
    for (const auto& r : requests) {
        int ec = Request::kFail;
        // find matching active sequence
        for (int i = 0; i < state_->size; ++i) {
            // stop & optionally erase active sequence
            if (state_->requests[i] && state_->requests[i]->id == r->id) {
                ec = 0;
                CompleteRequest(i, true, r->end_flag);
                break;
            }
        }
        // mismatch, try erase inactive sequence, in this case there is no active request to finish
        if (ec && r->end_flag) {
            ec = 0;
            sequence_manager_->Erase(r->id);
        }
        // clear output buffers (prevent leaking conversations) if request is successful
        if (ec == 0) {

            if (rank_ == 0) {
                std::unique_lock lock{output_mutex_};
                output_cv_.wait(lock, [&] { return output_reqs_.empty(); });
            }

            auto& output_ids      = r->outputs[rank_].at("output_ids");
            auto& sequence_length = r->outputs[rank_].at("sequence_length");
            Clear(output_ids.getPtr<int>(), output_ids.shape.at(2));
            Clear(sequence_length.getPtr<int>(), 1);
            check_cuda_error(cudaStreamSynchronize(stream_));
        }
        signals.push_back([=] { r->signal.set_value(ec); });
    }
    return signals;
}

template<typename T>
void LlamaBatch<T>::ProcessInferRequests(const Requests& requests)
{
    auto& state = *incoming_;

    state.size = state.active_size = 0;

    int i = 0;
    for (const auto& r : requests) {

        // sanity check, incoming request in previous iter should have been moved to `state_`
        FT_CHECK(!state.requests[i]);

        state.requests[i] = r;

        // get sequence for the request
        state.sequences[i] = r->start_flag ? sequence_manager_->Create(r->id) : sequence_manager_->Get(r->id);

        auto& seq = *state.sequences[i];

        if (int step = r->inputs[rank_].getVal<int>("step", -1); step >= 0) {
            /// TODO: revise step setting
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
        const auto output_ids_base = state.output_ids + session_len_ * i;
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
        state.h_context_length[i] = output_ids - output_ids_base;
        state.h_finished[i]       = false;

        const int request_output_len = state.requests[i]->inputs[rank_].getVal<int>("request_output_len");
        state.seq_len_limit[i]       = state.h_context_length[i] + request_output_len;
        // `length_criterion` sets finish flag when step >= seq_limit_len, however when step == seq_limit_len
        // the actual sequence length is seq_limit_len + 1, hence seq_limit_len must truncated to session_len - 1
        if (state.seq_len_limit[i] >= session_len_) {
            state.seq_len_limit[i] = session_len_ - 1;
            if (rank_ == 0) {
                const int trunc_output_len = state.seq_len_limit[i] - state.h_context_length[i];
                TM_LOG_WARNING(
                    "[initialize] [%ld] total sequence length (%d + %d) exceeds `session_len` (%d), `request_output_len` is truncated to %d",
                    (long)seq.id,
                    state.h_context_length[i],
                    request_output_len,
                    (int)session_len_,
                    trunc_output_len);
            }
        }

        // recover random state HtoD if not a new sequence
        if (!r->start_flag) {
            Copy((curandState_t*)seq.random_state.data() + 0, 1, (curandState_t*)state.top_k_curand_state);
            Copy((curandState_t*)seq.random_state.data() + 1, 1, (curandState_t*)state.top_p_curand_state);
        }

        // assign priority based on arrival time
        r->priority = request_count_++;

        // increment pointer
        i++;
    }

    incoming_->size = i;
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

    auto process = [&](BatchState* state) {
        // dbg(state->size);
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

    // dbg(sequences);
    // dbg(context_lengths);
    // dbg(priorities);
    // dbg(step_length_);

    auto outcome = sequence_manager_->Materialize(sequences, context_lengths, priorities, step_length_);
    if (outcome.allocation || outcome.swap_in || outcome.swap_out) {
        dbg(outcome);
    }

    bool exchange = outcome.swap_in + outcome.swap_out > 0;

    std::vector<int> idxs(sequences.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    if (exchange) {
        // put active ones first
        auto active_end = std::stable_partition(idxs.begin(), idxs.end(), [&](int idx) {
            return sequences[idx]->status == Sequence::kActive;  // present status
        });

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
    }

    if (exchange || holes) {
        // Copy sequence states to the back state buffer
        back_->size = back_->active_size = 0;
        for (const auto& i : idxs) {
            auto& s = *sequences[i];
            if (exchange) {
                const auto& [state, idx] = coords[i];
                // backup random states from dynamic decode layers for swap-outs
                if (status[i] == Sequence::kActive && s.status != Sequence::kActive) {
                    SaveRandomState(*state, idx);
                }
                if (status[i] != Sequence::kActive && s.status == Sequence::kActive) {
                    state->is_swap_in[idx] = 1;
                }
            }
            if (s.status == Sequence::kActive) {
                ++back_->active_size;
            }
            CopyState(coords[i], {back_, back_->size++});
        }
        // Swap the buffers
        std::swap(state_, back_);
    }

    const int batch_size = state_->active_size;

    if (exchange || outcome.allocation) {
        // Prepare intermediate buffers
        h_cu_block_counts_[0] = 0;

        auto k_ptrs = h_k_block_ptrs_;
        auto v_ptrs = h_v_block_ptrs_;

        for (int i = 0; i < batch_size; ++i) {
            const auto& seq = *state_->sequences[i];

            // cumulative num of blocks
            h_cu_block_counts_[i + 1] = h_cu_block_counts_[i] + seq.blocks.size();

            k_ptrs = std::transform(seq.blocks.cbegin(), seq.blocks.cend(), k_ptrs, [&](auto p) {
                return reinterpret_cast<uintptr_t>(sequence_manager_->OffsetKey(p->data));
            });
            v_ptrs = std::transform(seq.blocks.cbegin(), seq.blocks.cend(), v_ptrs, [&](auto p) {
                return reinterpret_cast<uintptr_t>(sequence_manager_->OffsetVal(p->data));
            });
        }

        if (1) {
            std::vector cu_block_cnts(h_cu_block_counts_, h_cu_block_counts_ + batch_size + 1);
            dbg(cu_block_cnts);
        }
        // dbg(std::vector(h_k_block_ptrs_, h_k_block_ptrs_ + h_cu_block_counts_[batch_size]));
        // dbg(std::vector(h_v_block_ptrs_, h_v_block_ptrs_ + h_cu_block_counts_[batch_size]));
        // dbg(h_cu_block_counts_[batch_size]);

        Copy(h_cu_block_counts_, batch_size + 1, cu_block_counts_);
        Copy(h_k_block_ptrs_, h_cu_block_counts_[batch_size], k_block_ptrs_);
        Copy(h_v_block_ptrs_, h_cu_block_counts_[batch_size], v_block_ptrs_);

        static_assert(sizeof(uintptr_t) == sizeof(void*));
    }

    // in case of swap-in/swap-out or there are holes in active buffer, layout of the buffers is changed
    // generation & sampling need to be re-initialized for correctness
    return exchange || active_holes;
}

template<typename T>
void LlamaBatch<T>::CopyState(const std::pair<BatchState*, int> _src, const std::pair<BatchState*, int>& _dst)
{
    const auto& [src, i] = _src;
    const auto& [dst, j] = _dst;

    FT_CHECK((bool)src->requests[i]);
    FT_CHECK(!(bool)dst->requests[j]);

    dst->h_context_length[j] = src->h_context_length[i];
    dst->h_finished[j]       = src->h_finished[i];
    dst->seq_len_limit[j]    = src->seq_len_limit[i];
    dst->sequences[j]        = src->sequences[i];
    dst->is_swap_in[i]       = src->is_swap_in[i];
    dst->requests[j]         = std::move(src->requests[i]);

    Copy(src->output_ids + i * session_len_, src->h_context_length[i], dst->output_ids + j * session_len_);

    Copy((curandState_t*)src->top_k_curand_state + i, 1, (curandState_t*)dst->top_k_curand_state + j);
    Copy((curandState_t*)src->top_p_curand_state + i, 1, (curandState_t*)dst->top_p_curand_state + j);
}

template<typename T>
void LlamaBatch<T>::SaveRandomState(BatchState& state, int idx)
{
    Copy(model_->GetTopKState(idx), 1, (curandState_t*)state.top_k_curand_state + idx);
    Copy(model_->GetTopPState(idx), 1, (curandState_t*)state.top_k_curand_state + idx);
}

template<typename T>
void LlamaBatch<T>::LoadRandomState(BatchState& state, int idx)
{
    dbg(idx);
    Copy((curandState_t*)state.top_k_curand_state + idx, 1, model_->GetTopKState(idx));
    Copy((curandState_t*)state.top_p_curand_state + idx, 1, model_->GetTopPState(idx));
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

    end_ids_buf_   = (int*)allocator_->reMalloc(end_ids_buf_, sizeof(int) * batch_size, false);
    finished_buf_  = (bool*)allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false);
    seq_limit_len_ = (uint32_t*)allocator_->reMalloc(seq_limit_len_, sizeof(uint32_t) * batch_size, false);

    is_allocate_buffer_ = true;
}

template<typename T>
void LlamaBatch<T>::AllocatePersistantBuffer(size_t max_batch_size)
{
    stop_words_buf_ =
        (int*)allocator_->reMalloc(stop_words_buf_, sizeof(int) * max_batch_size * kMaxStopBadWordsLen, true);
    bad_words_buf_ =
        (int*)allocator_->reMalloc(bad_words_buf_, sizeof(int) * max_batch_size * kMaxStopBadWordsLen, true);

    h_runtime_top_k_ = (int*)allocator_->reMalloc(h_runtime_top_k_, sizeof(int) * max_batch_size, true, true);
    h_runtime_top_p_ = (float*)allocator_->reMalloc(h_runtime_top_p_, sizeof(float) * max_batch_size, true, true);
    h_temperature_   = (float*)allocator_->reMalloc(h_temperature_, sizeof(float) * max_batch_size, true, true);
    h_repetition_penalty_ =
        (float*)allocator_->reMalloc(h_repetition_penalty_, sizeof(float) * max_batch_size, true, true);
    h_random_seed_ = (uint64_t*)allocator_->reMalloc(h_random_seed_, sizeof(uint64_t) * max_batch_size, true, true);

    sampling_params_ = {{"stop_words_list", stop_words_buf_},
                        {"bad_words_list", bad_words_buf_},
                        {"runtime_top_k", h_runtime_top_k_},
                        {"runtime_top_p", h_runtime_top_p_},
                        {"temperature", h_temperature_},
                        {"repetition_penalty", h_repetition_penalty_},
                        {"random_seed", h_random_seed_}};

    for (auto& s : states_) {
        s.output_ids = (int*)allocator_->reMalloc(s.output_ids, sizeof(int) * max_batch_size * session_len_, true);
        s.top_k_curand_state = allocator_->reMalloc(s.top_k_curand_state, sizeof(curandState_t) * max_batch_size, true);
        s.top_p_curand_state = allocator_->reMalloc(s.top_p_curand_state, sizeof(curandState_t) * max_batch_size, true);
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
            s.h_finished = (bool*)allocator_->reMalloc(s.h_finished, sizeof(bool) * max_batch_size * 2, false, true);
        }

        h_seq_limit_len_ =
            (uint32_t*)allocator_->reMalloc(h_seq_limit_len_, sizeof(uint32_t) * max_batch_size, false, true);
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

        allocator_->free((void**)&end_ids_buf_);
        allocator_->free((void**)&finished_buf_);
        allocator_->free((void**)&seq_limit_len_);

        is_allocate_buffer_ = false;
    }

    if (is_allocate_persistant_buffer_) {
        for (auto& s : states_) {
            allocator_->free((void**)&s.h_context_length, true);
            allocator_->free((void**)&s.h_finished, true);
            allocator_->free((void**)&s.output_ids);
        }
        allocator_->free((void**)&h_tmp_k_ptrs_, true);
        allocator_->free((void**)&h_tmp_v_ptrs_, true);
        allocator_->free((void**)&h_cu_block_counts_, true);
        allocator_->free((void**)&h_k_block_ptrs_, true);
        allocator_->free((void**)&h_v_block_ptrs_, true);
        allocator_->free((void**)&h_input_ids_buf_, true);
        allocator_->free((void**)&h_input_length_buf_, true);
        allocator_->free((void**)&h_seq_limit_len_, true);
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
    const int batch_size = state_->active_size;
    TensorMap inputs;
    for (const auto& param : sampling_params_) {
        // find an exemplar that matches the param name
        const Tensor* ptr{};
        for (int i = 0; i < batch_size; ++i) {
            if (state_->requests[i]->inputs[rank_].isExist(param.first)) {
                ptr = &state_->requests[i]->inputs[rank_].at(param.first);
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
            Clear((std::byte*)param.second, size_in_bytes * batch_size);
            for (int i = 0; i < batch_size; ++i) {
                if (state_->requests[i]->inputs[rank_].isExist(param.first)) {
                    auto& src = state_->requests[i]->inputs[rank_].at(param.first);
                    FT_CHECK(ref.shape == src.shape);
                    Copy(src.getPtr<std::byte>(), size_in_bytes, (std::byte*)param.second + size_in_bytes * i);
                }
            }
            inputs.insert({param.first, {ref.where, ref.type, shape, param.second}});
            if (debug_ && rank_ == 0) {
                TM_LOG_INFO("[initializeSampling] %s", format({param.first, inputs.at(param.first)}).c_str());
            }
        }
    }

    inputs_ = std::move(inputs);

    model_->dynamic_decode_layer_->setup(batch_size, 1, &inputs_);

    // recover random states if not a new request
    for (int i = 0; i < batch_size; ++i) {
        if (!state_->requests[i]->start_flag && state_->is_swap_in[i]) {
            LoadRandomState(*state_, i);
        }
    }

    handleOptArg(&inputs_, "end_id", end_ids_buf_, model_->end_id_, batch_size);
    cudaStreamSynchronize(0);
}

template<typename T>
auto LlamaBatch<T>::InitializeGeneration() -> GenerationState
{
    const int batch_size      = state_->active_size;
    const int max_context_len = *std::max_element(state_->h_context_length, state_->h_context_length + batch_size);

    Clear(token_ids_buf_, batch_size * session_len_);
    invokeTransposeAxis01(token_ids_buf_, state_->output_ids, batch_size, session_len_, 1, stream_);
    sync_check_cuda_error();

    // token_ids_buf_[s, b]
    // ABCDe            ABCDe     e
    // ABCDEFGHIJk      ABCDEFGHIJk
    // ABCDEFGHi    ->  ABCDEFGHi i
    // ABCDEFGh         ABCDEFGh  h
    // ABCd             ABCd      d
    for (int i = 0; i < batch_size; ++i) {
        auto token_ids = token_ids_buf_ + i;
        auto p_src     = state_->h_context_length[i] - 1;
        auto p_dst     = max_context_len - 1;
        if (p_src != p_dst) {  // dst and src of `cudaMemcpyAsync` must not overlap
            Copy(token_ids + p_src * batch_size, 1, token_ids + p_dst * batch_size);
        }
    }

    Copy(state_->h_context_length, batch_size, context_length_buf_);  // also referenced in `SetOutputTensors`
    Copy(context_length_buf_, batch_size, sequence_lengths_);
    // `sequence_lengths_` will be increased by dynamic decode
    // note that in decoder and in output "sequence length" has different semantic
    // - in decoder it means length of sequence that has kv cache already computed
    // - in output it means length of all tokens (the last generated token does not have k/v cache computed yet)
    invokePlusScalar(sequence_lengths_, -1, batch_size, stream_);
    sync_check_cuda_error();

    // seq_limit_len_, will be compared to `step` instead of `sequence_length`, so padding len should be accounted
    // for
    for (int i = 0; i < batch_size; ++i) {
        h_seq_limit_len_[i] = state_->seq_len_limit[i] + (max_context_len - state_->h_context_length[i]);
        // mask finished sequences
        state_->h_finished[i] = max_context_len >= h_seq_limit_len_[i];
    }
    Copy(h_seq_limit_len_, batch_size, seq_limit_len_);
    Copy(state_->h_finished, batch_size, finished_buf_);

    // ! range of step_ [1, 2 * session_len]
    // consider a sequence with context_len == session_len and another sequence with context_len == 1 and
    // request_output_len == session_len - 1 => step_ will loop in [session_len, 2 * session_len)
    const int start_step = max_context_len;

    if (rank_ == 0) {
        TM_LOG_INFO("[initGen] batch_size = %d", (int)batch_size);
        TM_LOG_INFO("[initGen] max_context_len = %d", (int)max_context_len);

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
    return GenerationState{max_context_len, start_step};
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
                           g.step,
                           0,
                           session_len_,
                           batch_size);

    model_->postDecodeEmbedding(logits_buf_,  //
                                local_logits_buf_,
                                decoder_output_buf_,
                                batch_size);

    // stop-words & bad-words require the matched tokens to be contiguous, so item size > 1 is
    // not supported yet.
    bool should_stop{};
    model_->dynamicDecode(token_ids_buf_,
                          finished_buf_,
                          sequence_lengths_,
                          &should_stop,
                          &inputs_,
                          &outputs_,
                          logits_buf_,
                          seq_limit_len_,
                          context_length_buf_,
                          end_ids_buf_,
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
    /// ! increase the step counter
    ++g.step;

    return !should_stop;
}

template<typename T>
void LlamaBatch<T>::ContextDecode()
{
    const auto batch_size = state_->active_size;

    int base = -1;
    for (int i = 0; i < batch_size; ++i) {
        if (state_->is_swap_in[i]) {
            const auto& seq = *state_->sequences[i];
            dbg(state_->h_context_length[i], seq.cache_len);
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
    Copy(h_input_length_buf_, batch_size, input_length_buf_);

    check_cuda_error(cudaStreamSynchronize(stream_));
    const auto tick = std::chrono::high_resolution_clock::now();

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
            accum_size        = 0;
            accum_input_count = 0;
            max_context_count = 0;
        }
    }
    offsets.push_back(batch_size);
    max_context_cnts.push_back(max_context_count);

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
        for (int i = first; i < last; ++i) {
            input_ids        = Copy(input_ids_buf_ + i * session_len_, h_input_length_buf_[i], input_ids);
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

        if (1) {
            std::vector<int> input_len(sub_batch_size);
            std::vector<int> context_len(sub_batch_size);
            Copy(input_length_buf_ + first, sub_batch_size, input_len.data());
            Copy(context_length_buf_ + first, sub_batch_size, context_len.data());
            cudaStreamSynchronize(stream_);
            dbg(input_len, context_len);
        }

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

    check_cuda_error(cudaStreamSynchronize(stream_));
    const auto tock = std::chrono::high_resolution_clock::now();
    if (rank_ == 0) {
        TM_LOG_INFO("[decodeContext] %.2f ms", std::chrono::duration<float, std::milli>(tock - tick).count());
    }
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
auto LlamaBatch<T>::Finish(GenerationState& g) -> std::vector<Signal>
{
    const int batch_size = state_->active_size;

    // secure info needed by `Initialize()`
    Copy(finished_buf_, batch_size, state_->h_finished);
    Copy(sequence_lengths_, batch_size, state_->h_context_length);

    if (1) {
        std::unique_lock<std::mutex> lock;
        if (rank_ == 0) {
            // wait for previous output operations
            lock = std::unique_lock{output_mutex_};
            output_cv_.wait(lock, [&] { return output_reqs_.empty(); });
        }

        SetOutputTensors(g);
        check_cuda_error(cudaStreamSynchronize(stream_));

        if (rank_ == 0) {
            // enqueue new output requests
            for (int i = 0; i < batch_size; ++i) {
                FT_CHECK(state_->requests[i] != nullptr);
                if (state_->requests[i]->stream_cb) {
                    output_reqs_.push_back(state_->requests[i]);
                }
            }
            lock.unlock();
            // notify output thread when we do have stream cbs to call
            if (!output_reqs_.empty()) {
                output_cv_.notify_one();
            }
        }
    }
    else {
        SetOutputTensors(g);
        check_cuda_error(cudaStreamSynchronize(stream_));

        if (rank_ == 0 && model_->ffi_lock_) {
            model_->ffi_lock_(1);
        }
        for (int i = 0; i < batch_size; ++i) {
            FT_CHECK(state_->requests[i] != nullptr);
            if (state_->requests[i]->stream_cb && rank_ == 0) {
                state_->requests[i]->stream_cb(&state_->requests[i]->outputs[rank_].get());
            }
        }
        if (rank_ == 0 && model_->ffi_lock_) {
            model_->ffi_lock_(0);
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
    for (int i = 0; i < batch_size; ++i) {
        if (state_->requests[i] && state_->h_finished[i]) {
            CompleteRequest(i, false, false);
            signals.push_back([r = std::move(state_->requests[i])] { r->signal.set_value(0); });
        }
    }
    return signals;
}

template<typename T>
void LlamaBatch<T>::SetOutputTensors(const GenerationState& g)
{
    // dbg(g.max_init_ctx_len);
    const auto batch_size = state_->active_size;
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

    /// TODO: fuse the loop into a single kernel
    for (int i = 0; i < batch_size; ++i) {
        if (state_->requests[i]) {
            auto& output_ids      = state_->requests[i]->outputs[rank_].at("output_ids");
            auto& sequence_length = state_->requests[i]->outputs[rank_].at("sequence_length");
            Copy(state_->output_ids + i * session_len_, output_ids.shape.at(2), output_ids.getPtr<int>());
            Copy(sequence_lengths_ + i, 1, sequence_length.getPtr<int>());
            if (g.step > g.max_init_ctx_len) {  // +1 for newly generated token
                invokePlusScalar(sequence_length.getPtr<int>(), 1, 1, stream_);
            }
        }
    }
}

template<typename T>
void LlamaBatch<T>::CompleteRequest(int index, bool is_stop_request, bool is_force_end)
{
    if (rank_ == 0) {
        TM_LOG_INFO("[CompleteRequest] slot = %d, id = %lu", index, (long)state_->requests[index]->id);
    }

    if (debug_ && rank_ == 0) {
        std::vector<int> tokens(state_->h_context_length[index] + 1);
        Copy(state_->output_ids + index * session_len_, tokens.size(), tokens.data());
        cudaStreamSynchronize(stream_);
        std::stringstream ss;
        for (const auto& t : tokens) {
            ss << " " << t;
        }
        TM_LOG_INFO("[CompleteRequest] slot %d, tokens [%s]", index, ss.str().c_str());
    }

    if (state_->requests[index]->end_flag || is_force_end) {
        sequence_manager_->Erase(state_->requests[index]->id);
    }
    else {
        const int cache_len  = state_->h_context_length[index];
        const int output_len = !is_stop_request ? cache_len + 1 : cache_len;

        auto& seq = *state_->sequences[index];

        seq.cache_len = cache_len;

        // update token IDs
        seq.tokens.resize(output_len);

        const auto output_ids_data = state_->requests[index]->outputs[rank_].at("output_ids").getPtr<int>();
        Copy(output_ids_data, output_len, seq.tokens.data());

        // update random states
        seq.random_state.resize(sizeof(curandState_t) * 2);

        // save random state in host memory
        if (auto ptr = (curandState_t*)seq.random_state.data()) {
            ptr = Copy(model_->GetTopKState(index), 1, ptr);
            ptr = Copy(model_->GetTopPState(index), 1, ptr);
        }

        check_cuda_error(cudaStreamSynchronize(stream_));

        sequence_manager_->Release(seq);
    }

    state_->sequences[index] = nullptr;
}

template<typename T>
void LlamaBatch<T>::InternalThreadEntry(int device_id)
{
    TM_LOG_INFO("[InternalThreadEntry] %d", (int)rank_);
    check_cuda_error(cudaSetDevice(device_id));

    auto& shared_state = model_->shared_state_;

    auto& request_queue  = shared_state->request_queue;
    auto& infer_requests = shared_state->infer_requests;
    auto& stop_requests  = shared_state->stop_requests;

    int finished_count = 0;

    GenerationState g{};

    while (1) {
        if (rank_ == 0) {
            const int  free_slot_count = max_batch_size_ - state_->size + finished_count;
            const bool is_empty        = (free_slot_count == max_batch_size_);

            // will block if batch is empty
            request_queue.dequeue(stop_requests, infer_requests, free_slot_count, is_empty, shared_state->abort);

            if (!shared_state->abort) {
                RejectInvalidRequests(stop_requests, infer_requests);
            }
        }

        NvtxScope scope("mainloop");

        // wait while rank-0 is dequeueing
        shared_state->barrier->wait();

        if (shared_state->abort) {
            TM_LOG_INFO("[InternalThreadEntry] stop requested.");
            // if (state_->size && rank_ == 0) {
            //     TM_LOG_WARNING("Active request(s) present (%d) while exiting.", state_->size);
            // }
            return;
        }

        auto signals = ProcessStopRequests(stop_requests);
        BarrierSignalRequests(*shared_state->barrier, signals);

        ProcessInferRequests(infer_requests);

        // wait while shared stop/infer_requests is being used
        shared_state->barrier->wait();

        auto modified = Initialize();

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
            auto signals   = Finish(g);
            finished_count = signals.size();
            BarrierSignalRequests(*shared_state->barrier, signals);
        }
    }

    FT_CHECK(0);
}

template<typename T>
void LlamaBatch<T>::BarrierSignalRequests(Barrier& barrier, const std::vector<Signal>& signals)
{
    if (!signals.empty()) {
        barrier.wait();
        if (rank_ == 0) {
            std::for_each(signals.cbegin(), signals.cend(), [](auto& s) { s(); });
        }
        barrier.wait();
    }
}

template<typename T>
void LlamaBatch<T>::Start()
{
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
        {
            // wait for requests with stream cbs
            std::unique_lock lock(output_mutex_);
            output_cv_.wait(lock, [&] { return !output_reqs_.empty() || output_stop_token_; });

            if (rank_ == 0 && model_->ffi_lock_) {
                model_->ffi_lock_(1);
            }
            // invoke stream cbs
            for (const auto& r : output_reqs_) {
                r->stream_cb(&r->outputs[rank_].get());
            }
            if (rank_ == 0 && model_->ffi_lock_) {
                model_->ffi_lock_(0);
            }
            output_reqs_.clear();

            // stop requested
            if (output_stop_token_) {
                TM_LOG_INFO("[OutputThreadEntry] stop requested.");
                break;
            }
        }
        FT_CHECK(output_reqs_.empty());
        // notify infer thread 0
        output_cv_.notify_one();
    }
}

template class LlamaBatch<half>;
template class LlamaBatch<float>;

}  // namespace turbomind
