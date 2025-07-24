// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <utility>

#include <cuda_runtime.h>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/comm/host_comm.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/buffer.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/tensor.h"

#include "src/turbomind/macro.h"

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/kernels/decoding_kernels.h"
#include "src/turbomind/kernels/gemm/tuner/params.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"

#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/copy.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"

#include "src/turbomind/comm/serialize.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/constant.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/debug_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

void PrintDecodeTokens(
    const int* token_ids, int max_seq_len, int batch_sizse, cudaStream_t stream, const std::string& msg)
{
    // tokens in [S, B] layout
    std::vector<int> tokens(max_seq_len * batch_sizse);
    check_cuda_error(cudaMemcpyAsync(tokens.data(), token_ids, sizeof(int) * tokens.size(), cudaMemcpyDefault, stream));
    check_cuda_error(cudaStreamSynchronize(stream));

    printf("[%s] ", msg.c_str());
    for (int j = 0; j < max_seq_len; ++j) {
        printf("%5d ", j);
    }
    printf("\n");
    for (int i = 0; i < batch_sizse; ++i) {
        printf("[%s] ", msg.c_str());
        for (int j = 0; j < max_seq_len; ++j) {
            // std::cout << sb_tokens[j * batch_size + i] << " ";
            printf("%5d ", tokens[j * batch_sizse + i]);
        }
        printf("\n");
    }
}
void ClearState(BatchState& s)
{
    std::fill_n(s.requests.begin(), s.size, nullptr);
    std::fill_n(s.sequences.begin(), s.size, nullptr);
    std::fill_n(s.errors.begin(), s.size, 0);
    s.size = s.active_size = 0;
}

void DropEmbeddings(const Sequence& seq)
{
    int    seq_len = seq.tokens.size();
    int    num_emb = seq.input_embeddings.size();
    size_t sz      = num_emb;
    for (; sz >= 1; sz--) {
        if (seq.input_embedding_ranges[sz - 1].second <= seq_len) {
            break;
        }
    }
    // should we keep part of embedding?
    seq.input_embeddings.resize(sz);
    seq.input_embedding_ranges.resize(sz);
}

void LlamaBatch::DisableInvalidRequests(Requests& infer_reqs, Requests& kill_reqs)
{
    NvtxScope _("disable invalid");

    std::pmr::monotonic_buffer_resource    mbr;
    std::pmr::unordered_map<uint64_t, int> occur(&mbr);

    auto count = [&occur](const auto& reqs) {
        for (const auto& r : reqs) {
            ++occur[r->id];
        }
    };

    auto validate = [&occur](auto& reqs, const char* type) {
        for (const auto& r : reqs) {
            if (occur[r->id] > 1) {
                TM_LOG_ERROR("Skip conflicting %s request for ID %lu", type, r->id);
                r->ec = Request::kConflict;
            }
        }
    };

    // Current batch
    for (int i = 0; i < state_->size; ++i) {
        if (state_->requests[i]) {
            ++occur[state_->requests[i]->id];
        }
    }

    count(kill_reqs);
    count(infer_reqs);

    validate(kill_reqs, "kill");
    validate(infer_reqs, "infer");

    // New requests that never get a chance to start
    for (auto& r : infer_reqs) {
        if (r && r->cancel_flag.load(std::memory_order_acquire) == -1) {
            r->ec = Request::kCancel;
        }
    }
}

void LlamaBatch::FindCanceledIndices(std::vector<int>& indices)
{
    for (int i = 0; i < state_->size; ++i) {  // current batch
        const auto& r = state_->requests[i];
        if (r && r->cancel_flag.load(std::memory_order_acquire) == -1) {
            indices.push_back(i);
        }
    }
}

void LlamaBatch::ProcessCancelRequests(std::vector<int>& indices, std::vector<Signal>& signals)
{
    int count = 0;

    for (const auto& i : indices) {
        if (auto& r = state_->requests[i]) {
            ++count;
            signals.push_back(Interrupt(i, true));
            // Interrupt should reset r
            FT_CHECK(!r);
        }
    }

    if (count) {
        // Still need this sync after `Interrupt`?
        check_cuda_error(cudaStreamSynchronize(stream_));
    }
}

void LlamaBatch::ProcessKillRequests(const Requests& kill_reqs, std::vector<Signal>& signals)
{
    for (auto& r : kill_reqs) {
        if (r) {
            int ec = r->ec;
            if (!ec) {
                if (!sequence_manager_->Erase(r->id)) {
                    ec = Request::kInvalid;
                }
            }
            signals.push_back([=] {
                if (r->end_cb) {
                    r->end_cb(ec);
                }
            });
        }
    }
}

void LlamaBatch::ProcessInferRequests(const Requests& reqs, std::vector<Signal>& signals)
{
    NvtxScope scope("infer_request");
    auto&     state = *incoming_;

    FT_CHECK(state.size == 0);
    FT_CHECK(state.active_size == 0);

    std::vector<int> existing_idx;

    int idx = 0;
    for (const auto& r : reqs) {

        if (tp_rank_ == 0) {
            TM_LOG_INFO("[ProcessInferRequests] Request for %ld received.", (long)r->id);
        }

        if (r->ec) {
            signals.push_back([r] { UpdateState(*r, r->ec, 0); });
            continue;
        }

        const int input_length = r->inputs.at("input_ids").shape(0);

        if (input_length > session_len_) {
            signals.push_back([r] { UpdateState(*r, Request::kTooLong, 0); });
            continue;
        }

        auto ptr = r->session.start_flag ? sequence_manager_->Create(r->id) : sequence_manager_->Get(r->id);
        if (!ptr) {
            signals.push_back([r] { UpdateState(*r, Request::kInvalid, 0); });
            continue;
        }

        const int step = [&] {
            int s = r->session.step;
            if (s < 0) {
                s = ptr->tokens.size();
            }
            else if (s > ptr->tokens.size()) {
                if (tp_rank_ == 0) {
                    TM_LOG_WARNING("[ProcessInferRequests] Skipping invalid step (%d) setting for ID %lu", s, ptr->id);
                }
                s = ptr->tokens.size();
            }
            return s;
        }();

        if (step + input_length > session_len_) {
            signals.push_back([r] { UpdateState(*r, Request::kTooLong, 0); });
            continue;
        }

        FT_CHECK(!state.requests[idx]);

        state.requests[idx]  = r;
        state.sequences[idx] = ptr;

        auto& seq = *state.sequences[idx];

        if (step < seq.tokens.size()) {
            // resize sequence tokens to match step
            seq.tokens.resize(step);
            seq.cache_len = std::min(seq.cache_len, step);
            DropEmbeddings(seq);
        }

        const int* input_ids = r->inputs.at("input_ids").data<int>();

        {
            // `output_ids` contains all token ids of the sequences
            const auto output_ids_base = state.output_ids.data() + session_len_ * idx;
            auto       d_output_ids    = output_ids_base;
            auto       h_output_ids    = r->output_ids.data();
            // copy history tokens
            if (!seq.tokens.empty()) {
                d_output_ids = core::Copy(seq.tokens.data(), seq.tokens.size(), d_output_ids);
                h_output_ids = std::copy_n(seq.tokens.data(), seq.tokens.size(), h_output_ids);
            }

            // copy input tokens
            if (input_length) {
                d_output_ids = core::Copy(input_ids, input_length, d_output_ids);
                h_output_ids = std::copy_n(input_ids, input_length, h_output_ids);
            }

            // total context length (history + input)
            state.h_prompt_length[idx]  = d_output_ids - output_ids_base;
            state.h_context_length[idx] = d_output_ids - output_ids_base;
            state.h_finished[idx]       = false;
        }

        // copy input tokens to prompt for prefix matching
        if (input_length && r->session.start_flag && !r->inputs.contains("input_embedding_ranges")) {
            // TODO: truncate prompt to enable prefix caching for VLM
            seq.prompt.resize(input_length);
            std::copy_n(input_ids, input_length, seq.prompt.data());
        }

        const int elem_size = byte_size(data_type_);

        // copy input embeddings
        if (r->inputs.contains("input_embedding_ranges")) {
            const auto& range_tensor = r->inputs.at("input_embedding_ranges");
            const auto& emb_tensor   = r->inputs.at("input_embeddings");
            const int*  ranges       = range_tensor.data<int>();

            auto check_embeddings = [&](int& num_valid_embeddings) {
                if (range_tensor.ndim() != 3 || range_tensor.shape(2) % 2 != 0) {
                    return false;
                }
                int embedding_count  = range_tensor.shape(1);
                int embedding_length = 0;
                int pre_end          = -1;

                for (size_t i = 0; i < embedding_count; i++) {
                    int begin = ranges[i * 2];
                    int end   = ranges[i * 2 + 1];
                    embedding_length += (end - begin);
                    if (begin < 0 || end < 0) {
                        break;
                    }
                    if (begin >= end || end > input_length || begin < pre_end
                        || embedding_length * model_->hidden_units_ * elem_size > emb_tensor.shape(1)) {
                        return false;
                    }
                    pre_end              = end;
                    num_valid_embeddings = i + 1;
                }
                return true;
            };

            int num_valid_embeddings = 0;
            if (!check_embeddings(num_valid_embeddings)) {
                TM_LOG_WARNING("[ImageFeature] Skip invalid input embeddings, id = %ld, input_length = %d",
                               (long)seq.id,
                               input_length);
            }
            else {
                const std::byte* emb_tensor_ptr = (const std::byte*)emb_tensor.raw_data();
                for (size_t i = 0; i < num_valid_embeddings; i++) {
                    int    begin = ranges[i * 2];
                    int    end   = ranges[i * 2 + 1];
                    size_t count = (end - begin) * model_->hidden_units_ * elem_size;
                    seq.input_embeddings.emplace_back(emb_tensor_ptr, emb_tensor_ptr + count);
                    seq.input_embedding_ranges.emplace_back(begin + seq.tokens.size(), end + seq.tokens.size());
                    emb_tensor_ptr += count;
                }
            }
        }

        const int max_new_tokens = state.requests[idx]->gen_cfg.max_new_tokens;
        state.seq_len_limit[idx] = state.h_context_length[idx] + max_new_tokens;
        // `length_criterion` sets finish flag when step >= seq_limit_len, however when step == seq_limit_len
        // the actual sequence length is seq_limit_len + 1, hence seq_limit_len must truncated to session_len - 1
        if (state.seq_len_limit[idx] >= session_len_) {
            state.seq_len_limit[idx] = session_len_ - 1;
            if (tp_rank_ == 0) {
                const int trunc_output_len = state.seq_len_limit[idx] - state.h_context_length[idx];
                TM_LOG_WARNING(
                    "[ProcessInferRequests] [%ld] total sequence length (%d + %d) exceeds `session_len` (%d), `max_new_tokens` is truncated to %d",
                    (long)seq.id,
                    state.h_context_length[idx],
                    max_new_tokens,
                    (int)session_len_,
                    trunc_output_len);
            }
        }

        // compute rope scaling factor
        if (r->session.start_flag) {
            seq.rope_theta = model_->attn_param_.rope.base;
            if (model_->attn_param_.rope.type == RopeType::kDynamic) {
                auto scaling_factor = model_->attn_param_.rope.factor;
                if (scaling_factor >= 1.f) {  // infer by current context length
                    auto max_seq_len = state.h_context_length[idx];
                    auto max_pos_emb = model_->attn_param_.rope.max_position_embeddings;
                    if (max_seq_len > max_pos_emb) {
                        scaling_factor = scaling_factor * max_seq_len / max_pos_emb - (scaling_factor - 1);
                        float rope_dim = model_->attn_param_.rope.dim;
                        seq.rope_theta *= powf(scaling_factor, rope_dim / (rope_dim - 2.f));
                        TM_LOG_INFO("[ProcessInferRequests] %ld rope_scaling_factor: %f, rope_theta = %f",
                                    (long)seq.id,
                                    scaling_factor,
                                    seq.rope_theta);
                    }
                }
            }
        }
        state.h_rope_theta[idx] = seq.rope_theta;

        if (r->session.start_flag) {
            // prepare to initialize random state for new sequence
            h_random_seed_[idx] = r->gen_cfg.random_seed;
        }
        else {
            // Recover device states if not a new sequence
            ((curandState_t*)h_curand_state_.data())[existing_idx.size()] = *(curandState_t*)seq.random_state.data();
            existing_idx.push_back(idx);
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
        invokeCurandBatchInitialize(
            (curandState_t*)state.curand_state.data(), state.size, d_random_seed_.data(), stream_);
        sync_check_cuda_error();
    }

    if (!existing_idx.empty()) {
        // copy existing curand states to device
        core::Copy((curandState_t*)h_curand_state_.data(), existing_idx.size(), (curandState_t*)h_curand_state_.data());
        // insert the states to their correct positions in the batch
        IndexedCopy({},
                    existing_idx,
                    std::tuple{(curandState_t*)d_curand_state_.data(), (curandState_t*)state.curand_state.data(), 1});
    }
}

int LlamaBatch::AdjustMaxInputCount(GenerationState&                    g,
                                    const std::vector<const Sequence*>& sequences,
                                    const std::vector<int>&             context_length)
{
    int input_count = 0;
    for (int i = 0; i < sequences.size(); ++i) {
        input_count += context_length[i] - sequences[i]->cache_len;
    }
    const int batch_size = sequences.size();
    input_count -= batch_size;

    // min tokens per iter for satisfying max prefill iters constraint
    input_count = (input_count + max_prefill_iters_ - 1) / max_prefill_iters_;

    if (g.min_input_count.empty()) {
        g.min_input_count.resize(max_prefill_iters_);
    }
    g.min_input_count.pop_front();
    g.min_input_count.push_back(input_count);
    /// TODO: sub-optimal when there are inactive sequences due to memory constraint
    for (auto& x : g.min_input_count) {
        x = std::max(x, input_count);
    }

    // Enlarge to satisfy `max_prefill_iters_`
    input_count = std::max(g.min_input_count.front() + batch_size, num_tokens_per_iter_);
    // Clamp to conform memory constraint
    input_count = std::min(input_count, max_forward_token_num_);

    return input_count;
}

void LlamaBatch::Initialize(GenerationState& g)
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
        for (int i = 0; i < state->size; ++i) {
            if (auto& r = state->requests[i]) {
                sequences.push_back(state->sequences[i]);
                status.push_back(state->sequences[i]->status);
                priorities.push_back(r->unique_id);
                context_lengths.push_back(state->h_context_length[i]);
                coords.emplace_back(state, i);
            }
        }
    };

    process(state_);
    process(incoming_);

    auto adjust = [this, &g](const Sequences& sequences, const std::vector<int>& context_length) -> int {
        return AdjustMaxInputCount(g, sequences, context_length);
    };

    // TM_LOG_INFO("max_input_count %d", max_input_count);
    auto outcome = sequence_manager_->Materialize(sequences, context_lengths, priorities, 1, adjust);

    if (outcome.allocation || outcome.swap_in || outcome.swap_out) {
        dbg(outcome);
    }

    bool exchange = outcome.swap_in + outcome.swap_out > 0;

    std::vector<int> idxs(sequences.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    if (exchange || holes || incoming_->size) {
        // put active ones first
        auto active_end = std::stable_partition(idxs.begin(), idxs.end(), [&](int idx) {
            return sequences[idx]->status == Sequence::kActive;  // current status
        });

        // all blocks are not enough to hold a single sequence
        if (!sequences.empty()) {
            FT_CHECK_WITH_INFO(active_end != idxs.begin(), "No enough blocks.");
        }

        // move the partial seq to the back
        auto partial_beg = std::stable_partition(idxs.begin(), active_end, [&](int i) {
            return sequences[i]->cache_len + sequences[i]->input_length == context_lengths[i];
        });
        FT_CHECK(active_end - partial_beg <= 1);

        auto swapin_beg = std::stable_partition(idxs.begin(), partial_beg, [&](int i) {
            return status[i] == Sequence::kActive;  // past status
        });

        // sort swap-ins according to input length
        if (swapin_beg != partial_beg) {
            std::stable_sort(swapin_beg, partial_beg, [&](int i, int j) {
                return sequences[i]->input_length < sequences[j]->input_length;
            });
        }

        // Copy sequence states to back buffer
        FT_CHECK(back_->size == 0 && back_->active_size == 0);
        std::vector<std::tuple<BatchState*, BatchState*, int, int>> cpys;
        for (const auto& i : idxs) {
            auto& s = *sequences[i];
            if (s.status == Sequence::kActive) {
                ++back_->active_size;
            }
            cpys.emplace_back(coords[i].first, back_, coords[i].second, back_->size++);
        }
        CopyState(cpys);
        // Swap the buffers
        SwapState(state_, back_);

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

        auto block_ptrs = h_block_ptrs_.data();

        const int batch_size = state_->active_size;

        for (int i = 0; i < batch_size; ++i) {
            const auto& seq = *state_->sequences[i];

            // cumulative num of blocks
            h_cu_block_counts_[i + 1] = h_cu_block_counts_[i] + seq.blocks.size();

            block_ptrs = std::transform(seq.blocks.cbegin(), seq.blocks.cend(), block_ptrs, [&](int block_id) {
                return reinterpret_cast<uintptr_t>(sequence_manager_->GetBlockPtr(block_id));
            });
        }

        static_assert(sizeof(uintptr_t) == sizeof(void*));

        Copy(h_cu_block_counts_, batch_size + 1, cu_block_counts_);
        Copy(h_block_ptrs_, h_cu_block_counts_[batch_size], block_ptrs_);
    }

    const int batch_size = state_->active_size;

    // check if the last sequence is partial
    int partial     = 0;
    int partial_len = -1;
    if (state_->active_size) {
        const int i = state_->active_size - 1;
        partial = state_->sequences[i]->cache_len + state_->sequences[i]->input_length != state_->h_context_length[i];
        if (partial) {
            // backup full context length of partial
            partial_len = state_->h_context_length[i];
            // replace with partial context length
            state_->h_context_length[i] = state_->sequences[i]->cache_len + state_->sequences[i]->input_length;
        }
    }

    const int max_context_len =
        *std::max_element(state_->h_context_length.data(), state_->h_context_length.data() + batch_size);

    std::vector<uint64_t> unique_ids(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        unique_ids[i] = state_->requests[i]->unique_id;
    }

    // Real-time context length that will change during generation
    Copy_(state_->h_context_length, batch_size, state_->context_length_buf);
    Copy_(state_->h_finished, batch_size, state_->finished_buf);
    Copy_(state_->h_rope_theta, batch_size, state_->rope_theta);

    bool skip_init_sampling = std::equal(g.unique_ids.begin(),  //
                                         g.unique_ids.end() - g.partial,
                                         unique_ids.begin(),
                                         unique_ids.end() - partial);

    g.partial                = partial;
    g.partial_context_legnth = partial_len;
    g.unique_ids             = std::move(unique_ids);
    g.finished_count         = 0;
    g.skip_init_sampling     = skip_init_sampling;

    // TM_LOG_ERROR("[Initialize] batch size: %d, active size: %d", state_->size, state_->active_size);

    if (!skip_init_sampling) {
        g.max_init_ctx_len       = max_context_len;
        g.step                   = max_context_len;
        state_->pp_init_sampling = true;
    }
}

void LlamaBatch::CopyState(const std::vector<std::tuple<BatchState*, BatchState*, int, int>>& desc)
{
    if (desc.empty()) {
        return;
    }

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
                    std::tuple{s->output_ids.data(), d->output_ids.data(), session_len_},
                    std::tuple{(curandState_t*)s->curand_state.data(), (curandState_t*)d->curand_state.data(), 1});
    }

    for (const auto& [s, d, si, di] : desc) {
        d->h_prompt_length[di]  = s->h_prompt_length[si];
        d->h_context_length[di] = s->h_context_length[si];
        d->h_finished[di]       = s->h_finished[si];
        d->h_rope_theta[di]     = s->h_rope_theta[si];
        d->seq_len_limit[di]    = s->seq_len_limit[si];
        d->sequences[di]        = s->sequences[si];
        d->requests[di]         = s->requests[si];
        d->errors[di]           = s->errors[si];
    }
}

void LlamaBatch::SwapState(BatchState*& a, BatchState*& b)
{
    std::swap(a, b);

    if (param_.pp_size > 1) {
        ClearState(*b);
        std::vector<std::tuple<BatchState*, BatchState*, int, int>> cpys;
        FT_CHECK(b->size == 0 && b->active_size == 0);
        for (int i = 0; i < a->size; ++i) {
            cpys.emplace_back(a, b, b->size, b->size);
            b->size++;
        }
        b->active_size = a->active_size;
        CopyState(cpys);
        std::swap(a, b);
    }
    else {
        // shared buffers between state_ and back_
        a->init_context_length = b->init_context_length;
        a->context_length_buf  = b->context_length_buf;
        a->sequence_lengths    = b->sequence_lengths;
        a->rope_theta          = b->rope_theta;
        a->h_seq_limit_len     = b->h_seq_limit_len;
        a->seq_limit_len       = b->seq_limit_len;
        a->finished_buf        = b->finished_buf;
        a->token_ids_buf       = b->token_ids_buf;
        a->input_ids_buf       = b->input_ids_buf;
        a->h_input_length_buf  = b->h_input_length_buf;
    }
}

void LlamaBatch::AllocateBuffer(ssize_t batch_size, ssize_t session_len, int cache_block_seq_len)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const ssize_t batchxbeam = batch_size;

    const ssize_t hidden_units      = model_->hidden_units_;
    const ssize_t vocab_size        = model_->vocab_size_padded_;
    const ssize_t head_dim          = model_->size_per_head_;
    const ssize_t local_kv_head_num = model_->local_kv_head_num_;
    // +1 padding, BlockIterator does not use predicate
    const ssize_t max_batch_block_count =
        batch_size * ((session_len + cache_block_seq_len - 1) / cache_block_seq_len) + 1;

    decoder_output_buf_ = {{batchxbeam, hidden_units}, data_type_, kDEVICE};

    cu_block_counts_ = {batch_size + 1, kDEVICE};
    block_ptrs_      = {max_batch_block_count, kDEVICE};

    sampled_logprobs_ = {batchxbeam * kMaxLogProb, kDEVICE};
    sampled_indexes_  = {batchxbeam * kMaxLogProb, kDEVICE};
    sampled_nums_     = {batchxbeam, kDEVICE};

    sampling_logits_ = {{(ssize_t)max_batch_size_, (ssize_t)model_->vocab_size_padded_}, kDEVICE};

    h_random_seed_ = {batch_size, kCPUpinned};
    Clear(h_random_seed_);

    d_random_seed_ = {batch_size, kDEVICE};
    Clear(d_random_seed_);

    h_curand_state_ = {{batch_size, sizeof(curandState_t)}, kCPUpinned};
    Clear(h_curand_state_.buffer());

    d_curand_state_ = {{batch_size, sizeof(curandState_t)}, kDEVICE};
    Clear(d_curand_state_.buffer());

    for (auto& s : states_) {
        s.output_ids = {{batch_size, session_len_}, kDEVICE};
        Clear(s.output_ids.buffer());

        s.curand_state = {{batch_size, sizeof(curandState_t)}, kDEVICE};
        Clear(s.curand_state.buffer());
    }

    h_cu_block_counts_ = {batch_size + 1, kCPUpinned};
    h_block_ptrs_      = {(ssize_t)max_batch_block_count, kCPUpinned};

    for (auto& s : states_) {
        s.h_prompt_length  = {batch_size, kCPUpinned};
        s.h_context_length = {batch_size, kCPUpinned};
        s.h_finished       = {batch_size * 2, kCPUpinned};
        s.h_rope_theta     = {batch_size, kCPUpinned};
    }

    for (int i = 0; i < states_.size() - 2; ++i) {
        auto& s               = state_[i];
        s.context_length_buf  = {batchxbeam, kDEVICE};
        s.init_context_length = {batchxbeam, kDEVICE};
        s.sequence_lengths    = {batchxbeam, kDEVICE};
        s.rope_theta          = {batch_size, kDEVICE};
        s.token_ids_buf       = {ssize_t(session_len * 2 * batchxbeam), kDEVICE};
        s.finished_buf        = {(int)batchxbeam, kDEVICE};
        s.h_seq_limit_len     = {batch_size, kCPUpinned};
        s.seq_limit_len       = {batch_size, kDEVICE};
        s.input_ids_buf       = {max_forward_token_num_, kDEVICE};
        s.h_input_length_buf  = {batch_size, kCPUpinned};
    }

    h_output_ids_ = {batch_size * session_len_, kCPUpinned};

    h_sampled_logprobs_ = {batch_size * kMaxLogProb, kCPUpinned};
    h_sampled_indexes_  = {batch_size * kMaxLogProb, kCPUpinned};
    h_sampled_nums_     = {batch_size, kCPUpinned};
}

void LlamaBatch::AllocSymmBuffers()
{
    const ssize_t hidden_units      = model_->hidden_units_;
    const ssize_t vocab_size_padded = model_->vocab_size_padded_;

    // Native comm fuses allreduce & rmsnorm in token granularity
    TM_CHECK(max_forward_token_num_ % tp_size_ == 0);

    symm_hidden_states_buf_ = {{max_forward_token_num_ * param_.attn_dp_size, hidden_units}, data_type_, symm_alloc_};
    symm_logits_buf_        = {{max_batch_size_, vocab_size_padded}, data_type_, symm_alloc_};

    if (param_.pp_size > 1) {
        symm_residual_buf_ = {{max_forward_token_num_ * param_.attn_dp_size, hidden_units}, data_type_, symm_alloc_};
    }
}

void LlamaBatch::FreeSymmBuffers()
{
    symm_hidden_states_buf_ = {};
    symm_logits_buf_        = {};

    symm_residual_buf_ = {};
}

LlamaBatch::~LlamaBatch()
{
    TM_LOG_DEBUG("~LlamaBatch()");

    internal_thread_.join();

    // The dtor maybe called from unknown thread, set device id before CUDA calls
    cudaSetDevice(device_id_);
    cudaStreamSynchronize(stream_);

    model_.reset();
    sequence_manager_.reset();
    context_.reset();  // This destroy all objects in context except for `stream`
}

LlamaBatch::LlamaBatch(DataType                 data_type,
                       const EngineParam&       param,
                       std::unique_ptr<LlamaV2> model,  // ! This is moved
                       std::unique_ptr<Context> ctx,    // ! This is moved
                       std::shared_ptr<Gateway> gateway,
                       int                      device_id,
                       int                      dp_rank):
    param_(param),
    gateway_(gateway),
    max_batch_size_(param.max_batch_size),
    max_forward_token_num_(param.max_forward_token_num),
    max_context_token_num_(param.max_context_token_num),
    num_tokens_per_iter_(param.num_tokens_per_iter),
    max_prefill_iters_(param.max_prefill_iters),
    device_id_(device_id),
    dp_rank_(dp_rank),
    tp_size_(model->tp_size_),
    tp_rank_(model->tp_rank_),
    data_type_(data_type),
    debug_(isDebug()),
    stream_(ctx->stream),
    context_(std::move(ctx)),
    model_(std::move(model)),
    comm_(context_->comm),
    session_len_(param.session_len)
{
    const auto cache_block_seq_len = model_->attn_param_.cache_block_seq_len;

    const int dbits = byte_size(data_type, 8);

    const auto quant_policy = model_->param_.quant_policy;
    const int  elem_bits    = quant_policy ? quant_policy : dbits;

    SequenceManager::BlockConfig block_config{
        (int)model_->size_per_head_,
        (int)model_->local_kv_head_num_,
        cache_block_seq_len,
        elem_bits == dbits ? 0 : dbits,
        elem_bits,
    };

    const auto get_free_size = [&] {  //
        size_t free{}, total{};
        check_cuda_error(cudaMemGetInfo(&free, &total));
        free = AllReduce(model_->comm_->h_tp_group, free, comm::RedOp::kMin);
        if (param_.pp_size > 1) {
            free = AllReduce(model_->comm_->h_pp_group, free, comm::RedOp::kMin);
        }
        return free;
    };

    const size_t layer_num = model_->layer_num_ / param_.pp_size + (model_->layer_num_ % param_.pp_size != 0);
    sequence_manager_.reset(new SequenceManager{layer_num,
                                                block_config,
                                                param.cache_max_block_count,
                                                param.cache_chunk_size,
                                                param.enable_prefix_caching,
                                                tp_rank_,
                                                core::Context::alloc(kDEVICE),
                                                get_free_size});

    const size_t max_session_len = sequence_manager_->max_block_count() * cache_block_seq_len;
    if (max_session_len < session_len_) {
        if (tp_rank_ == 0) {
            TM_LOG_WARNING("No enough blocks for `session_len` (%d), `session_len` truncated to %d.",
                           session_len_,
                           max_session_len);
        }
        session_len_ = max_session_len;
    }

    FT_CHECK(max_context_token_num_ >= session_len_);
    FT_CHECK(max_forward_token_num_ >= max_batch_size_);

    const int state_size = param_.pp_size + 2;
    states_.resize(state_size);
    for (auto& s : states_) {
        s.requests.resize(max_batch_size_);
        s.sequences.resize(max_batch_size_);
        s.seq_len_limit.resize(max_batch_size_);
        s.errors.resize(max_batch_size_);
    }

    state_    = &states_[0];
    back_     = &states_[state_size - 2];
    incoming_ = &states_[state_size - 1];

    gs_.resize(param_.pp_size);
    for (int i = 0; i < param_.pp_size; ++i) {
        slots_.push_back({&states_[i], &gs_[i]});
    }

    symm_alloc_ = core::SimpleAllocator::Create([this](ssize_t size) { return SymmAlloc(size, true); },
                                                [this](void* p, ssize_t size) { return SymmFree(p, size, true); },
                                                kDEVICE);

    AllocSymmBuffers();

    AllocateBuffer(max_batch_size_, session_len_, cache_block_seq_len);

    // Wait for allocations
    check_cuda_error(cudaStreamSynchronize(stream_));
}

void LlamaBatch::InitializeSampling(const GenerationState& g)
{
    NvtxScope _("InitSampling");

    const int batch_size = state_->active_size - g.partial;

    if (batch_size == 0) {
        return;
    }

    if (param_.pp_size == 0 || state_->pp_init_sampling) {
        state_->pp_init_sampling = false;  // updated by Initialize(g)

        // Context length at initialization, will stay constant until re-initialziation
        Copy(state_->context_length_buf, batch_size, state_->init_context_length);

        Copy(state_->context_length_buf, batch_size, state_->sequence_lengths);
        // `sequence_lengths_` will be increased by dynamic decode
        // note that in decoder and in output "sequence length" has different semantic
        // - in decoder it means length of sequence that has kv cache already computed
        // - in output it means length of all tokens (the last generated token does not have k/v cache computed yet)
        invokePlusScalar(state_->sequence_lengths.data(), -1, batch_size, stream_);
        sync_check_cuda_error();

        Clear(state_->token_ids_buf.slice(0, batch_size * session_len_));
        invokeTranspose2D(state_->token_ids_buf.data(), state_->output_ids.data(), batch_size, session_len_, stream_);
        sync_check_cuda_error();

        // token_ids_buf_[s, b]
        // ABCDe            ABCDe     e
        // ABCDEFGHIJk      ABCDEFGHIJk
        // ABCDEFGHi    ->  ABCDEFGHi i
        // ABCDEFGh         ABCDEFGh  h
        // ABCd             ABCd      d
        invokePadLastTokenIds(
            state_->token_ids_buf.data(), state_->init_context_length.data(), g.max_init_ctx_len, batch_size, stream_);
        sync_check_cuda_error();

        // seq_limit_len_, will be compared to `step` instead of `sequence_length`, so padding len should be accounted
        // for
        for (int i = 0; i < batch_size; ++i) {
            state_->h_seq_limit_len[i] = state_->seq_len_limit[i] + (g.max_init_ctx_len - state_->h_context_length[i]);
        }
        Copy(state_->h_seq_limit_len, batch_size, state_->seq_limit_len);
    }

    std::vector<const Request*> rs;
    rs.reserve(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        rs.push_back(state_->requests[i].get());
    }

    model_->dynamic_decode_->Setup(rs, {{"prompt_length", {state_->h_prompt_length, {batch_size}}}});

    sync_check_cuda_error();
}

void LlamaBatch::ComputeAndOutputLogits(const Tensor& hidden_states, int first, int last)
{
    auto enable = [&] {
        for (int i = first; i < last; ++i) {
            if (state_->requests[i]->gen_cfg.output_logits == GenerationConfig::kAll) {
                const auto& s = *state_->sequences[i];
                // Skip when the seq is filling missed cache only
                if (s.cache_len + state_->h_input_length_buf[i] > s.tokens.size()) {
                    return true;
                }
            }
        }
        return false;
    }();

    if (!enable) {
        return;
    }

    const int vocab_size_padded = model_->vocab_size_padded_;
    const int token_num         = hidden_states.shape(0);

    if (symm_logits_buf_.shape(0) < token_num) {
        if (tp_size_ > 1) {
            check_cuda_error(cudaStreamSynchronize(stream_));
            comm_.h_tp_group->Sync();
        }
        symm_logits_buf_ = {{token_num, vocab_size_padded}, data_type_, symm_alloc_};
        if (tp_size_ > 1) {
            check_cuda_error(cudaStreamSynchronize(stream_));
            comm_.h_tp_group->Sync();
        }
    }

    auto logits = model_->postDecodeEmbedding(hidden_states, symm_logits_buf_.buffer());

    if (tp_rank_ == 0) {
        OutputLogits(logits, first, last, GenerationConfig::kAll);
    }
}

void LlamaBatch::OutputLogits(const Tensor& logits, int first, int last, GenerationConfig::OutType out_type)
{
    const auto& src_buf   = logits.buffer();
    const auto  elem_size = byte_size(logits.dtype(), 1);
    // when `is_all` is true, logits only contains last token of the sequences
    const bool is_all = out_type == GenerationConfig::kAll;

    int base = 0;

    for (int i = first; i < last; ++i) {

        const int input_len = state_->h_input_length_buf[i];  // input lenght for this iter

        if (state_->requests[i]->gen_cfg.output_logits == out_type) {

            auto& dst_buf = state_->requests[i]->outputs.at("logits").buffer();

            const int cache_len   = state_->sequences[i]->cache_len;
            const int history_len = state_->sequences[i]->tokens.size();

            // ----------H------I-------P-----------
            //      C        C      C         C

            // offset to the last token prompt
            const int offset = is_all ? 0 : state_->requests[i]->inputs.at("input_ids").shape(0) - 1;

            int diff = (history_len + offset) - cache_len;

            const int valid_len = input_len - std::max(0, (history_len + offset) - cache_len);

            // TM_LOG_ERROR("%d %d   %d %d  %d  %d %d",
            //              history_len,
            //              offset,
            //              cache_len,
            //              input_len,
            //              valid_len,
            //              std::max(0, diff),
            //              std::max(0, -diff));

            if (valid_len <= 0) {
                continue;
            }

            int src_base = base;

            if (is_all) {
                // Skip invalid tokens caused by cache miss
                src_base += std::max(0, (history_len + offset) - cache_len);
            }
            // Skip previous chunks
            int dst_base = std::max(0, cache_len - (history_len + offset));

            check_cuda_error(cudaMemcpy2DAsync(dst_buf.raw_data(dst_base * model_->vocab_size_),
                                               elem_size * model_->vocab_size_,
                                               src_buf.raw_data(src_base * model_->vocab_size_padded_),
                                               elem_size * model_->vocab_size_padded_,
                                               elem_size * model_->vocab_size_,
                                               valid_len,
                                               cudaMemcpyDefault,
                                               stream_));
        }

        base += is_all ? input_len : 1;
    }
}

void LlamaBatch::OutputLastHiddenState(const Tensor& hidden_states, int first, int last)
{
    if (tp_rank_ != 0) {
        return;
    }

    const auto& src_buf   = hidden_states.buffer();
    const auto  data_type = src_buf.dtype();
    int         base      = 0;

    for (int i = first; i < last; ++i) {
        const int input_len = state_->h_input_length_buf[i];  // input lenght for this iter

        if (auto out_type = state_->requests[i]->gen_cfg.output_last_hidden_state) {

            const bool is_all = out_type == GenerationConfig::kAll;

            auto& dst_buf = state_->requests[i]->outputs.at("last_hidden_state").buffer();

            const int cache_len   = state_->sequences[i]->cache_len;
            const int history_len = state_->sequences[i]->tokens.size();

            // offset to the last prompt token
            const int offset = is_all ? 0 : state_->requests[i]->inputs.at("input_ids").shape(0) - 1;

            const int valid_len = input_len - std::max(0, (history_len + offset) - cache_len);

            // TM_LOG_ERROR("%d %d %d %d %d", history_len, offset, cache_len, input_len, valid_len);

            if (valid_len > 0) {
                // Skip invalid tokens caused by cache miss
                int src_base = std::max(0, (history_len + offset) - cache_len) + base;
                // Skip previous chunks
                int dst_base = std::max(0, cache_len - (history_len + offset));

                core::Copy(src_buf.raw_data(src_base * model_->hidden_units_),
                           byte_size(data_type, valid_len * model_->hidden_units_),
                           dst_buf.raw_data(dst_base * model_->hidden_units_));
            }
        }

        // hidden_states += input_len * model_->hidden_units_;
        base += input_len;
    }
}

void LlamaBatch::Finish(GenerationState& g, std::vector<Signal>& signals)
{
    NvtxScope scope("Finish");
    const int batch_size = state_->active_size;

    signals.reserve(batch_size);

    if (batch_size - g.partial) {
        FT_CHECK(g.step >= 0);

        // [s,b] -> [b,s] and skip padding in [context_len, max_context_len)
        invokeGatherOutput(state_->output_ids.data(),
                           state_->token_ids_buf.data(),
                           state_->init_context_length.data(),
                           g.max_init_ctx_len,
                           g.step,
                           session_len_,
                           batch_size - g.partial,
                           stream_);
        sync_check_cuda_error();
    }

    Copy(state_->token_ids_buf.slice((g.step - 1) * (batch_size - g.partial), -1),
         batch_size - g.partial,
         h_output_ids_);
    Copy(state_->finished_buf, batch_size, state_->h_finished);
    Copy(state_->sequence_lengths, batch_size, state_->h_context_length);

    bool output_logprobs = false;
    for (int i = 0; i < batch_size - g.partial; ++i) {
        if (state_->requests[i]->gen_cfg.output_logprobs) {
            output_logprobs = true;
            break;
        }
    }
    if (output_logprobs) {
        Copy(sampled_logprobs_, batch_size * kMaxLogProb, h_sampled_logprobs_);
        Copy(sampled_indexes_, batch_size * kMaxLogProb, h_sampled_indexes_);
        Copy(sampled_nums_, batch_size, h_sampled_nums_);
    }

    check_cuda_error(cudaStreamSynchronize(stream_));

    // invariant: context_length = sequence_length + 1, so that h_context_length include all (including the one just
    // generated) tokens
    for (int i = 0; i < batch_size; ++i) {
        ++state_->h_context_length[i];
    }

    // ! Only rank-0 writes to output
    if (tp_rank_ == 0 && output_logprobs) {
        NvtxScope scope("logprobs");
        float*    sampled_logprobs_ptr = h_sampled_logprobs_.data();
        uint32_t* sampled_indexes_ptr  = h_sampled_indexes_.data();
        uint32_t* sampled_nums_ptr     = h_sampled_nums_.data();
        for (int i = 0; i < batch_size - g.partial; ++i) {
            if (state_->requests[i] && state_->requests[i]->gen_cfg.output_logprobs) {
                auto logprob_vals    = state_->requests[i]->outputs.at("logprob_vals").data<float>();
                auto logprob_indexes = state_->requests[i]->outputs.at("logprob_indexes").data<int32_t>();
                auto logprob_nums    = state_->requests[i]->outputs.at("logprob_nums").data<int32_t>();

                int offset = state_->h_context_length[i] - state_->h_prompt_length[i] - 1;
                std::copy(sampled_logprobs_ptr,
                          sampled_logprobs_ptr + *sampled_nums_ptr,
                          logprob_vals + offset * kMaxLogProb);
                std::copy(sampled_indexes_ptr,
                          sampled_indexes_ptr + *sampled_nums_ptr,
                          logprob_indexes + offset * kMaxLogProb);
                *(logprob_nums + offset) = *sampled_nums_ptr;
            }
            sampled_logprobs_ptr += kMaxLogProb;
            sampled_indexes_ptr += kMaxLogProb;
            sampled_nums_ptr++;
        }
    }

    // ! Only rank-0 writes to output
    if (tp_rank_ == 0) {
        NvtxScope scope("output_ids");
        for (int i = 0; i < batch_size - g.partial; ++i) {
            if (auto& r = state_->requests[i]) {
                auto      output_ids  = r->output_ids.data();
                auto      output_len  = r->sequence_length.data();
                const int count       = state_->h_context_length[i];
                output_ids[count - 1] = h_output_ids_[i];
                *output_len           = count;
            }
        }
    }

    // Cache computed blocks to block trie
    sequence_manager_->CacheIfEnabled(state_->sequences, batch_size);

    if (debug_ && tp_rank_ == 0) {
        for (int i = 0; i < batch_size; ++i) {
            // ss << (i ? ", " : "") << "(" << state_->h_context_length[i] << "," << state_->h_finished[i] << ")";
            std::vector<int> tokens(state_->h_context_length[i]);
            core::Copy(state_->output_ids.data() + i * session_len_, tokens.size(), tokens.data());
            cudaStreamSynchronize(stream_);
            std::stringstream ss;
            for (const auto& t : tokens) {
                ss << " " << t;
            }
            TM_LOG_INFO("[Finish] slot %d, tokens [%s]", i, ss.str().c_str());
        }
    }

    {
        NvtxScope _("count and sync");
        bool      need_sync = false;
        for (int i = 0; i < batch_size - g.partial; ++i) {
            if (state_->h_finished[i]) {
                ++g.finished_count;
                if (!state_->requests[i]->session.end_flag) {
                    need_sync = true;
                }
            }
        }
        if (need_sync) {
            // Release updates on request output buffers to all ranks (`Interrupt` will use it)
            comm_.h_tp_group->Sync();
        }
    }

    {
        NvtxScope _("stream_and_completion_signal");
        for (int i = 0; i < batch_size - g.partial; ++i) {
            auto& r = state_->requests[i];
            if (state_->h_finished[i]) {
                // Interrupt finished sequences and move the request handle into the signal closure
                signals.push_back(Interrupt(i));
                // Interrupt should reset r
                FT_CHECK(!r);
            }
            else if (r->stream_output && tp_rank_ == 0) {
                const auto seq_len = *r->sequence_length.data();
                // Create signals by copying the request handles for non-finished streaming requests
                signals.push_back([this, r, seq_len] {  //
                    UpdateState(*r, Request::kOk, seq_len);
                });
            }
        }
    }

    if (g.finished_count) {
        // synchronize for interrupted sequences
        check_cuda_error(cudaStreamSynchronize(stream_));
    }

    if (g.partial) {
        const int i = batch_size - 1;
        // recover full context length of partial
        state_->h_context_length[i] = g.partial_context_legnth;
    }
}

auto LlamaBatch::Interrupt(int index, bool force_stop, bool force_end) -> Signal
{
    if (tp_rank_ == 0) {
        TM_LOG_INFO("[Interrupt] slot %d, request %lu, stop %d, end %d",
                    index,
                    (long)state_->requests[index]->id,
                    force_stop,
                    force_end);
    }

    if (debug_ && tp_rank_ == 0) {
        std::vector<int> tokens(state_->h_context_length[index]);
        core::Copy(state_->output_ids.data() + index * session_len_, tokens.size(), tokens.data());
        cudaStreamSynchronize(stream_);
        std::stringstream ss;
        for (const auto& t : tokens) {
            ss << " " << t;
        }
        TM_LOG_INFO("[Interrupt] slot %d, tokens [%s]", index, ss.str().c_str());
    }

    if (state_->requests[index]->session.end_flag || force_end) {
        // Sequence is ending this round or a stop request is issued to end it
        FT_CHECK(sequence_manager_->Erase(state_->requests[index]->id));
    }
    else {
        const int output_len = state_->h_context_length[index];
        auto&     seq        = *state_->sequences[index];

        // Update token IDs
        seq.tokens.resize(output_len);

        // output_ids is updated & synced in `Finish`
        const auto output_ids = state_->requests[index]->output_ids.data();
        std::copy_n(output_ids, output_len, seq.tokens.data());

        // Save random state in host memory
        seq.random_state.resize(sizeof(curandState_t));
        // This async copy must be synchronized by the caller
        core::Copy((curandState_t*)state_->curand_state.data() + index, 1, (curandState_t*)seq.random_state.data());

        // Set unlock flag for corresponding blocks, will be unlocked in the next `Materialize()`
        sequence_manager_->UpdateAndSetUnlock(seq);
    }

    state_->sequences[index] = nullptr;

    auto ec = std::exchange(state_->errors[index], Request::kOk);

    const auto len = *state_->requests[index]->sequence_length.data();
    // move the request handle into the signal
    return [this, len, force_stop, r = std::move(state_->requests[index])] {  //
        UpdateState(*r, force_stop ? Request::kCancel : Request::kFinish, len);
    };
}

namespace {

struct RequestData {
    std::vector<std::shared_ptr<Request>> infer;  // incoming inference request
    std::vector<std::shared_ptr<Request>> kill;   // incoming kill request

    std::vector<int> cancel;  // canceled indices in current batch
    bool             abort;
};

}  // namespace

#ifdef BUILD_MULTI_GPU
namespace comm {

void serialize(std::ostream& os, const RequestData& req)
{
    // std::vector<std::shared_ptr<Request>> infer;
    serialize(os, (int)req.infer.size());
    for (const auto& r : req.infer) {
        serialize(os, *r);
    }
    // std::vector<std::shared_ptr<Request>> kill;
    serialize(os, (int)req.kill.size());
    for (const auto& r : req.kill) {
        serialize(os, *r);
    }

    serialize(os, req.cancel);  // std::vector<int> cancel;
    serialize(os, req.abort);   // bool             abort;
}

template<>
void serialize(const std::shared_ptr<RequestData>* req, int n, std::vector<char>& vec)
{
    std::stringstream ss;
    for (int i = 0; i < n; ++i) {
        const auto& r = req[i];
        if (r != nullptr) {
            serialize(ss, *r);
        }
    }
    vec = streambuf_to_vector(ss.rdbuf());
}

void deserialize(std::istream& is, RequestData& req)
{
    auto process = [](std::istream& is, std::vector<std::shared_ptr<Request>>& vec) {
        int size;
        deserialize(is, size);
        vec.resize(size);
        for (auto& r : vec) {
            r = std::make_shared<Request>();
            deserialize(is, *r);
        }
    };
    process(is, req.infer);
    process(is, req.kill);
    deserialize(is, req.cancel);
    deserialize(is, req.abort);
}

template<>
void deserialize(std::shared_ptr<RequestData>* req, int n, const std::vector<char>& vec)
{
    std::stringstream ss;
    ss.write(vec.data(), vec.size());
    for (int i = 0; i < n; ++i) {
        auto& r = req[i];
        if (r == nullptr) {
            r = std::make_shared<RequestData>();
        }
        deserialize(ss, *r);
    }
}

}  // namespace comm

#endif  // BUILD_MULTI_GPU

void LlamaBatch::InternalThreadEntry()
{
    // TM_LOG_INFO("[InternalThreadEntry] %d", (int)rank_);
    check_cuda_error(cudaSetDevice(device_id_));

    core::ContextGuard guard{context_->core_stream, context_->allocator};

    // Initialize `AnomalyHandler`
    AnomalyHandler::instance().Init(tp_rank_, model_->vocab_size_padded_, 0, max_batch_size_, stream_);

    GenerationState* g = &gs_[0];

    while (1) {
        if (param_.pp_size > 1 && param_.pp_rank == 0) {
            std::tie(state_, g) = slots_.front();
        }

        auto req = std::make_shared<RequestData>();

        if (tp_rank_ == 0 && param_.pp_rank == 0) {
            req = std::make_shared<RequestData>();
            {
                NvtxScope  _("pop");
                const int  free_slot_count = max_batch_size_ - state_->size + g->finished_count;
                const bool is_empty        = (free_slot_count == max_batch_size_);
                // Block if batch is empty AND no silbings are ready AND comm in same node
                const bool blocking = is_empty && comm_.h_comm->is_same_process() && param_.pp_size == 1;
                int        wait     = 0;
                do {
                    gateway_->pop(req->infer, req->kill, free_slot_count, blocking, req->abort, dp_rank_);
                    if (!comm_.h_comm->is_same_process()) {
                        bool empty_pop = req->infer.size() == 0 && req->kill.size() == 0 && req->abort == false;
                        wait           = is_empty && empty_pop;
                        wait           = AllReduce(comm_.h_comm, wait, comm::RedOp::kSum) == comm_.h_comm->n_ranks();
                    }
                } while (wait);
            }
            // Mark reqs to the same session_id as invalid (which are dangerous to the engine)
            DisableInvalidRequests(req->infer, req->kill);
            FindCanceledIndices(req->cancel);
        }

        NvtxScope scope("mainloop");

        // 1. Wait while rank-0 is dequeueing
        // 2. Broadcast `ec` from rank-0
        // shared_state_->barrier->wait();
        // comm_.h_comm->Sync(comm_.h_comm_tp_group);
        if (comm_.h_tp_group->n_ranks() > 1 && param_.pp_rank == 0) {
            Broadcast(comm_.h_tp_group, req, 0);
        }

        if (!comm_.h_comm->is_same_process() && param_.pp_rank == 0) {
            req->abort = AllReduce(comm_.h_comm, (int)req->abort, comm::RedOp::kSum) > 0;
        }

        if (req->abort || pp_abort_) {
            if (param_.pp_size == 1 || (batch_que_.empty() && pp_abort_)) {
                TM_LOG_ERROR("[InternalThreadEntry] stop requested.");
                break;
            }
            pp_abort_ = true;
        }

        std::vector<Signal> signals;

        ProcessKillRequests(req->kill, signals);

        // Shared `priority` field will be assigned by rank-0
        ProcessInferRequests(req->infer, signals);

        // 1. Wait while shared `requests` is being used
        // 2. Broadcast modifcations from rank-0
        // comm_.h_comm->Sync(comm_.h_comm_tp_group);

        ProcessCancelRequests(req->cancel, signals);

        if (tp_rank_ == 0 && param_.pp_rank == 0) {
            gateway_->notify(std::move(signals));
        }

        Initialize(*g);

        const int n_active = AllReduce(comm_.h_dp_group, state_->active_size, comm::RedOp::kSum);

        if (n_active || param_.pp_size > 1) {
            //
            if (!Forward(g)) {
                continue;
            }

            Finish(*g, signals);

            if (g->finished_count) {
                // Finished requests and corresponding output tensors will be released when notified
                // wait for all ranks to ensure no rank (except for output thread) will access related
                // resources
                comm_.h_tp_group->Sync();
            }

            if (tp_rank_ == 0) {
                gateway_->notify(std::move(signals));
            }
        }
    }

    // barrier synchronization inside
    DestroyCommunicators();
}

void LlamaBatch::Start()
{
    TM_LOG_INFO("LlamaBatch<T>::Start()");
    internal_thread_ = std::thread([this] {
        try {
            InternalThreadEntry();
        }
        catch (const std::exception& e) {
            TM_LOG_ERROR("[Engine] %s", e.what());
            std::abort();
        }
    });
}

bool LlamaBatch::Forward(GenerationState*& g)
{
    NvtxScope _("Forward");

    FT_CHECK(max_context_token_num_ >= max_batch_size_);

    int active_size = state_->active_size;

    constexpr int kLogInterval = 10;
    if (tp_rank_ == 0 && (g->step - 1) % kLogInterval == 0 && param_.pp_rank == 0) {
        TM_LOG_INFO("------------------------- step = %d -------------------------", g->step - 1);
    }

    int               pf_offset = -1;
    std::vector<int*> input_d_ptrs(active_size);

    for (int i = 0; i < active_size; ++i) {
        const auto& seq = *state_->sequences[i];
        // const int   missing = state_->h_context_length[i] - seq.cache_len;
        FT_CHECK(seq.input_length >= 1);
        state_->h_input_length_buf[i] = seq.input_length;
        input_d_ptrs[i]               = state_->output_ids.data() + i * session_len_ + seq.cache_len;
        if (seq.input_length > 1 && pf_offset < 0) {
            pf_offset = i;
        }
    }
    if (pf_offset < 0) {
        pf_offset = active_size;
    }

    // Find mini-batch offsets: input length > 1 ? prefill() : decode()
    // Constraints on mini-batches
    //   sum(Q) <= `max_forward_token_num` && sum(K) <= `max_context_token_num`
    std::vector<int> offsets{0};
    // initialize first mini-batch with decode tokens
    int sum_q = pf_offset;
    int sum_k = 0;  // only for prefill
    for (int i = pf_offset; i < active_size; ++i) {
        FT_CHECK(state_->h_input_length_buf[i] <= max_forward_token_num_);
        const int q = sum_q + state_->h_input_length_buf[i];
        const int k = sum_k + state_->h_context_length[i];
        if (q <= max_forward_token_num_ && k <= max_context_token_num_) {
            sum_q = q;
            sum_k = k;
        }
        else {
            offsets.push_back(i);
            sum_q = state_->h_input_length_buf[i];
            sum_k = state_->h_context_length[i];
        }
    }
    offsets.push_back(active_size);

    // Synchronize mini batch count with sync DP ranks
    int n_batches = AllReduce(comm_.h_dp_group, (int)offsets.size(), comm::RedOp::kMax);

    // Populate empty batches
    while (offsets.size() < n_batches) {
        offsets.push_back(offsets.back());
    }

    // prepare inputs
    Buffer_<int>     input_ids_buf;
    Tensor           hidden_states;
    Tensor           residual;
    int              batch_size{};
    IntermediateData inter{};  // pipeline parallel intermediate data

    if (param_.pp_rank == 0) {
        TM_CHECK(n_batches == 2) << "pipeline parallel only support n_batches=1";  // TODO:This modification relies on
                                                                                   // the removal of mini-batch.
        const int first = offsets[0];
        const int last  = offsets[1];
        TM_CHECK(last - first == state_->active_size);
        int* input_ids = state_->input_ids_buf.data();

        BatchedCopy batched_copy;
        int         sum_k = 0;
        for (int i = first; i < last; ++i) {
            input_ids = batched_copy.Add(input_d_ptrs[i], state_->h_input_length_buf[i], input_ids);
            if (state_->h_input_length_buf[i] > 1) {
                sum_k += state_->h_context_length[i];
            }
        }
        int sum_q = input_ids - state_->input_ids_buf.data();
        batched_copy.Submit(stream_);
        state_->dc_batch_size = pf_offset;
        state_->pf_batch_size = state_->active_size - state_->dc_batch_size;

        if (tp_rank_ == 0) {
            if (state_->pf_batch_size) {
                const auto max_q = *std::max_element(state_->h_input_length_buf.data() + first,
                                                     state_->h_input_length_buf.data() + last);
                const auto max_k =
                    *std::max_element(state_->h_context_length.data() + first, state_->h_context_length.data() + last);
                TM_LOG_INFO("[Forward] [%d, %d), dc=%d, pf=%d, sum_q=%d, sum_k=%d, max_q=%d, max_k=%d",
                            first,
                            last,
                            state_->dc_batch_size,
                            state_->pf_batch_size,
                            sum_q,
                            sum_k,
                            max_q,
                            max_k);
            }
        }

        // Synchronize batch token num with sync DP ranks
        state_->local_token_nums = AllGather(comm_.h_dp_group, sum_q);
        state_->global_token_num = std::accumulate(state_->local_token_nums.begin(), state_->local_token_nums.end(), 0);

        input_ids_buf = state_->input_ids_buf.slice(0, sum_q);
    }
    else {
        RecvIntermediateData(inter);
        PostProcessIntermediateData(inter);
    }

    batch_size = state_->pf_batch_size + state_->dc_batch_size;

    // forward logits
    if (batch_size) {
        state_->hidden_states = symm_hidden_states_buf_.slice(0, state_->global_token_num).borrow();
        state_->residual =
            (param_.pp_size > 1) ? symm_residual_buf_.slice(0, state_->global_token_num).borrow() : Tensor{};

        model_->Forward(input_ids_buf,          // temp
                        state_->hidden_states,  // temp
                        state_->residual,       // used by pipeline parallel
                        decoder_output_buf_.slice(0, batch_size),
                        block_ptrs_,
                        cu_block_counts_.slice(0, batch_size + 1),
                        state_->h_input_length_buf.slice(0, batch_size),
                        state_->h_context_length.slice(0, batch_size),
                        state_->rope_theta.slice(0, batch_size),
                        state_->finished_buf.slice(0, batch_size),
                        Buffer(state_->local_token_nums.data(), state_->local_token_nums.size(), kCPU),
                        lora_mask_buf_,
                        state_->dc_batch_size,
                        state_->pf_batch_size,
                        state_->sequences.data());
    }

    if (param_.pp_size > 1) {
        // for pipeline parallel
        // - pp_rank 0 ~ pp_size - 1 should send intermediate data to next pp_rank
        // - pp_rank 0 should receive intermediate data from last pp_rank and output logits/hidden states
        if (param_.pp_rank == 0) {
            IntermediateData last_inter{};
            BatchState*      last_state{};
            GenerationState* last_g{};

            // receive
            if (batch_que_.size() == param_.pp_size - 1 || (batch_que_.size() > 0 && batch_size == 0)) {
                RecvIntermediateData(last_inter);  // logits
                std::tie(last_state, last_g) = batch_que_.front();
                batch_que_.pop();
            }

            if (batch_size > 0 || (batch_size == 0 && batch_que_.empty())) {
                // send, maybe dummy
                PreProcessIntermediateData(inter);
                SendIntermediateData(inter);
            }

            if (batch_size > 0) {
                batch_que_.push({state_, g});
                slots_.pop_front();
            }

            if (!last_state) {
                return false;
            }

            state_ = last_state;
            g      = last_g;
            slots_.push_front({state_, g});
        }
        else {
            if (param_.pp_rank != param_.pp_size - 1 || batch_size > 0) {
                SendIntermediateData(inter);
            }
            return false;
        }
    }

    // forward logits & dynamic decode
    if (const auto bsz = state_->pf_batch_size + state_->dc_batch_size; bsz > 0) {
        ComputeAndOutputLogits(state_->hidden_states, 0, bsz);
        OutputLastHiddenState(state_->hidden_states, 0, bsz);
    }

    if (const auto bsz = state_->active_size - g->partial; bsz > 0) {

        auto logits = model_->postDecodeEmbedding(decoder_output_buf_.slice(0, bsz), symm_logits_buf_.buffer());

        // AnomalyHandler::instance().FixLogits(logits.data<nv_bfloat16>(), bsz, 1);

        OutputLogits(logits, 0, bsz, GenerationConfig::kGeneration);

        TM_CHECK_GE(g->step, 0);

        if (!g->skip_init_sampling || param_.pp_size > 1) {
            InitializeSampling(*g);
        }

        bool output_logprobs = [&] {
            for (int i = 0; i < bsz; ++i) {
                if (state_->requests[i]->gen_cfg.output_logprobs) {
                    return true;
                }
            }
            return false;
        }();

        auto sampling_logits = sampling_logits_.slice(0, bsz);
        invokeCastFloat2D(logits, sampling_logits, stream_);
        sync_check_cuda_error();

        // stop-words & bad-words require the matched tokens to be contiguous, so item size > 1 is not supported.
        model_->dynamicDecode(state_->token_ids_buf,
                              state_->finished_buf,
                              state_->sequence_lengths,
                              state_->curand_state,
                              sampling_logits,  // <- batch size indicator
                              state_->seq_limit_len,
                              state_->init_context_length,
                              state_->h_context_length,
                              state_->h_prompt_length,
                              output_logprobs ? sampled_logprobs_ : Buffer{},  // <- indicator
                              sampled_indexes_,
                              sampled_nums_,
                              g->step,
                              g->max_init_ctx_len);
    }

    std::fill(state_->h_input_length_buf.data(), state_->h_input_length_buf.data() + state_->active_size, 0);

    // `SequenceManager` needs real-time value of cache length
    for (int i = 0; i < state_->active_size; ++i) {
        FT_CHECK((bool)state_->requests[i]);
        FT_CHECK(state_->sequences[i]);
        state_->sequences[i]->cache_len += state_->sequences[i]->input_length;
    }

    AnomalyHandler::instance().Summarize([&](const int* is_anomaly, int batch_size) {
        for (int i = 0; i < batch_size; ++i) {
            if (is_anomaly[i]) {
                TM_LOG_WARNING("[Forward] Abnormal logits detected for request (%s)",
                               std::to_string(state_->sequences[i]->id).c_str());
                state_->errors[i] = Request::kFail;
            }
        }
    });
    AnomalyHandler::instance().Reset();

    if (debug_ && tp_rank_ == 0) {
        std::vector<int> curr(state_->active_size);
        core::Copy(state_->token_ids_buf.data() + g->step * state_->active_size, state_->active_size, curr.data());
        cudaStreamSynchronize(stream_);
        std::stringstream scurr;
        for (int k = 0; k < curr.size(); ++k) {
            scurr << std::setw(10) << curr[k];
        }
        TM_LOG_INFO("[Forward] step = %d, [%s]", g->step - 1, scurr.str().c_str());
    }

    ////////////////////////////////////////////////
    /// ! increase the counters
    g->step += 1;

    return true;
}

namespace {

template<class First, class Last>
std::string Join(First first, Last last, const std::string& delim)
{
    if (first == last) {
        return {};
    }
    std::ostringstream oss;
    oss << *first++;
    while (first != last) {
        oss << delim << *first++;
    }
    return oss.str();
}

struct TuningContext {
    LlamaLinear& linear_;
    cudaStream_t stream_;
    TuningContext(LlamaLinear& linear, cudaStream_t stream): linear_{linear}, stream_{stream}
    {
        isTuning() = true;
        linear_.set_measure(true);
    }
    ~TuningContext()
    {
        linear_.set_measure(false);
        isTuning() = false;
    }
};

}  // namespace

void LlamaBatch::Warmup()
{
    auto& linear = *context_->linear;
    if (auto str = std::getenv("TM_GEMM_IMPORT")) {
        std::ifstream ifs(str);
        const int     n_imported = linear.Import(ifs);
        if (tp_rank_ == 0) {
            TM_LOG_INFO("[Gemm2] %d records imported", n_imported);
        }
        return;
    }

    std::vector<int> bss = linear.GetTuningSeq();
    if (bss.empty()) {
        bss = gemm::GenerateTuningSequence(gemm::GetDefaultTuningGenerators());
    }

    // remove bs that is too large
    bss.erase(std::remove_if(bss.begin(), bss.end(), [&](auto x) { return x > max_forward_token_num_; }), bss.end());

    if (bss.empty() || bss.back() < max_forward_token_num_) {
        bss.push_back(max_forward_token_num_);
    }

    if (tp_rank_ == 0) {
        auto str = Join(bss.begin(), bss.end(), ", ");
        TM_LOG_INFO("[Gemm2] Tuning sequence: %s", str.c_str());
    }

    if (!bss.empty()) {
        const auto                         max_bs = *std::max_element(bss.begin(), bss.end());
        Buffer_<int>                       input_ids(max_bs, kCPU);
        Buffer_<int>                       input_ids_buf(max_bs, kDEVICE);
        std::mt19937                       g{};
        std::uniform_int_distribution<int> d{0, (int)model_->vocab_size_ - 1};
        for (auto& x : input_ids) {
            x = d(g);
        }

        Copy(input_ids, input_ids_buf);
        check_cuda_error(cudaStreamSynchronize(stream_));

        TuningContext context{linear, stream_};

        auto tick = std::chrono::steady_clock::now();

        /// NOTE: No explicit barrier can be used here as internal threads are waiting on it now
        for (auto token_num : bss) {
            if (tp_rank_ == 0) {
                TM_LOG_INFO("[Gemm2] %d", token_num);
            }

            int  input_length     = token_num;
            auto local_token_nums = AllGather(comm_.h_dp_group, token_num);

            const auto bsz = 1;

            // A single sequence containing `token_num` prefill tokens
            model_->Forward(state_->input_ids_buf.slice(0, token_num),
                            symm_hidden_states_buf_.slice(0, token_num * param_.attn_dp_size),
                            {},  // residual
                            decoder_output_buf_.slice(0, bsz),
                            block_ptrs_,
                            cu_block_counts_.slice(0, bsz + 1),
                            Buffer{&input_length, 1, kCPU},
                            Buffer{&input_length, 1, kCPU},
                            state_->rope_theta.slice(0, bsz),
                            state_->finished_buf.slice(0, bsz),
                            Buffer{local_token_nums.data(), (int)local_token_nums.size(), kCPU},
                            Buffer{},
                            0,
                            bsz,
                            nullptr);
        }

        auto tock = std::chrono::steady_clock::now();

        if (tp_rank_ == 0) {
            TM_LOG_INFO("[Gemm2] Tuning finished in %.2f seconds.",
                        std::chrono::duration<float, std::ratio<1, 1>>(tock - tick).count());
        }
    }

    // This will catch async errors during tuning
    check_cuda_error(cudaStreamSynchronize(stream_));

    // Only rank-0 exports the dispatch cache
    if (tp_rank_ == 0) {
        if (auto path = std::getenv("TM_GEMM_EXPORT")) {
            std::ofstream ofs(path);
            const auto    n_records = context_->linear->Export(ofs);
            TM_LOG_INFO("[Gemm2] %d records exported.", n_records);
        }
    }
}

void* LlamaBatch::SymmAlloc(size_t size, bool register_)
{
    if (auto& comm = model_->comm_->d_comm) {
        auto ptr = comm->Allocate(size);
        if (register_) {
            comm->Register(ptr, size);
        }
        return ptr;
    }
    else {
        return context_->allocator->allocate(size);
    }
}

void LlamaBatch::SymmFree(void* ptr, size_t size, bool deregister)
{
    if (!ptr) {
        return;
    }
    if (auto& comm = comm_.d_comm) {
        if (deregister) {
            comm->Deregister(ptr);
        }
        comm->Free(ptr);
    }
    else {
        context_->allocator->deallocate(ptr, size);
    }
}

void LlamaBatch::DestroyCommunicators()
{
    cudaStreamSynchronize(stream_);
    comm_.h_comm->Sync();

    FreeSymmBuffers();
    comm_.h_comm->Sync();

    // Destroy device communicator
    comm_.d_comm = {};
    if (param_.pp_size > 1) {
        comm_.d_pp_comm = {};
    }

    cudaStreamSynchronize(stream_);
    comm_.h_comm->Sync();
}

void LlamaBatch::PreProcessIntermediateData(IntermediateData& inter)
{
    if (param_.pp_rank == 0) {
        for (int i = 0; i < state_->active_size; ++i) {
            const auto& seq = *state_->sequences[i];
            inter.blocks.push_back(seq.blocks);
        }

        inter.abort              = pp_abort_;
        inter.h_cu_block_counts  = h_cu_block_counts_;
        inter.h_input_length_buf = state_->h_input_length_buf;
        inter.h_context_length   = state_->h_context_length;
        inter.h_rope_theta       = state_->h_rope_theta;
        inter.h_finished         = state_->h_finished;
        inter.dc_batch_size      = state_->dc_batch_size;
        inter.pf_batch_size      = state_->pf_batch_size;
        inter.local_token_nums   = state_->local_token_nums;
        inter.global_token_num   = state_->global_token_num;
    }
}

void LlamaBatch::PostProcessIntermediateData(IntermediateData& inter)
{
    // state should always copy
    if (param_.pp_rank > 0) {
        state_->pf_batch_size    = inter.pf_batch_size;
        state_->dc_batch_size    = inter.dc_batch_size;
        state_->local_token_nums = inter.local_token_nums;
        state_->global_token_num = inter.global_token_num;
        pp_abort_                = inter.abort;
    }

    // early exist as there is no data to process
    const int batch_size = inter.pf_batch_size + inter.dc_batch_size;
    if (batch_size == 0 || param_.pp_rank == 0) {
        return;
    }

    // cpu
    std::copy_n(inter.h_input_length_buf.data(), batch_size, state_->h_input_length_buf.data());
    std::copy_n(inter.h_context_length.data(), batch_size, state_->h_context_length.data());

    // device
    Copy(inter.h_cu_block_counts, batch_size + 1, cu_block_counts_);
    Copy(inter.h_rope_theta, batch_size, state_->rope_theta);
    Copy(inter.h_finished, batch_size, state_->finished_buf);

    h_cu_block_counts_[0] = 0;
    auto block_ptrs       = h_block_ptrs_.data();
    for (int i = 0; i < batch_size; ++i) {
        const auto& seq    = *state_->sequences[i];
        const auto& blocks = inter.blocks[i];

        // cumulative num of blocks
        h_cu_block_counts_[i + 1] = h_cu_block_counts_[i] + blocks.size();

        block_ptrs = std::transform(blocks.cbegin(), blocks.cend(), block_ptrs, [&](int block_id) {
            return reinterpret_cast<uintptr_t>(sequence_manager_->GetBlockPtr(block_id));
        });
    }
    Copy(h_cu_block_counts_, batch_size + 1, cu_block_counts_);
    Copy(h_block_ptrs_, h_cu_block_counts_[batch_size], block_ptrs_);
}

void LlamaBatch::SendIntermediateData(IntermediateData& inter)
{
    const int dst        = (param_.pp_rank + 1) % param_.pp_size;
    const int batch_size = inter.dc_batch_size + inter.pf_batch_size;

    Send(comm_.h_pp_group, inter, dst);
    if (batch_size == 0) {
        // no device data to send
        return;
    }

    if (param_.pp_rank < param_.pp_size - 1) {
        // for [0, pp_rank - 1), send hidden & residual
        Tensor hidden   = symm_hidden_states_buf_.slice(0, inter.global_token_num);
        Tensor residual = symm_residual_buf_.slice(0, inter.global_token_num);
        comm_.d_pp_comm->Send(hidden.raw_data(), hidden.size(), hidden.dtype(), dst, 0, stream_);
        comm_.d_pp_comm->Send(residual.raw_data(), residual.size(), residual.dtype(), dst, 0, stream_);
    }
    else {
        // for pp_rank - 1, send logits
        Tensor logits = decoder_output_buf_.slice(0, batch_size);
        comm_.d_pp_comm->Send(logits.raw_data(), logits.size(), logits.dtype(), dst, 0, stream_);
    }
}

void LlamaBatch::RecvIntermediateData(IntermediateData& inter)
{
    const int src = (param_.pp_rank - 1 + param_.pp_size) % param_.pp_size;
    Recv(comm_.h_pp_group, inter, src);

    const int batch_size = inter.dc_batch_size + inter.pf_batch_size;
    if (batch_size == 0) {
        // no device data to receive
        return;
    }

    if (param_.pp_rank > 0) {
        // for [1, pp_rank - 1], recv hidden & residual
        Tensor hidden   = symm_hidden_states_buf_.slice(0, inter.global_token_num);
        Tensor residual = symm_residual_buf_.slice(0, inter.global_token_num);
        comm_.d_pp_comm->Recv(hidden.raw_data(), hidden.size(), hidden.dtype(), src, 0, stream_);
        comm_.d_pp_comm->Recv(residual.raw_data(), residual.size(), residual.dtype(), src, 0, stream_);
    }
    else {
        // for pp_rank 0, recv logits
        Tensor logits = decoder_output_buf_.slice(0, batch_size);
        comm_.d_pp_comm->Recv(logits.raw_data(), logits.size(), logits.dtype(), src, 0, stream_);
    }
}

}  // namespace turbomind
