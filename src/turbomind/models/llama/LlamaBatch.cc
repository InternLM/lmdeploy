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

#include "src/turbomind/kernels/core/data_type.h"
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

#include "src/turbomind/utils/Tensor.h"
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

        const int input_length = r->inputs.at("input_ids").shape[0];

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

        const int* input_ids = r->inputs.getPtr<int>("input_ids");

        {
            // `output_ids` contains all token ids of the sequences
            const auto output_ids_base = state.output_ids.data() + session_len_ * idx;
            auto       d_output_ids    = output_ids_base;
            auto       h_output_ids    = r->output_ids.getPtr<int>();
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
        if (input_length && r->session.start_flag && !r->inputs.isExist("input_embedding_ranges")) {
            // TODO: truncate prompt to enable prefix caching for VLM
            seq.prompt.resize(input_length);
            std::copy_n(input_ids, input_length, seq.prompt.data());
        }

        const int elem_size = core::get_byte_size(data_type_);

        // copy input embeddings
        if (r->inputs.isExist("input_embedding_ranges")) {
            const auto range_tensor = r->inputs.at("input_embedding_ranges");
            const auto emb_tensor   = r->inputs.at("input_embeddings");
            const int* ranges       = range_tensor.getPtr<int>();

            auto check_embeddings = [&](int& num_valid_embeddings) {
                if (range_tensor.shape.size() != 3 || range_tensor.shape[2] % 2 != 0) {
                    return false;
                }
                int embedding_count  = range_tensor.shape[1];
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
                        || embedding_length * model_->hidden_units_ * elem_size > emb_tensor.shape[1]) {
                        return false;
                    }
                    pre_end              = end;
                    num_valid_embeddings = i + 1;
                }
                return true;
            };

            int num_valid_embeddings = 0;
            if (!check_embeddings(num_valid_embeddings)) {
                TM_LOG_WARNING("[ImageFeature] Skip invalid input embeddings, id = %ld, input_length = %d, "
                               "input embeddings = %s, range_tensor = %s",
                               (long)seq.id,
                               input_length,
                               emb_tensor.toString().c_str(),
                               range_tensor.toString().c_str());
            }
            else {
                const char* emb_tensor_ptr = emb_tensor.getPtr<char>();
                for (size_t i = 0; i < num_valid_embeddings; i++) {
                    int    begin = ranges[i * 2];
                    int    end   = ranges[i * 2 + 1];
                    size_t count = (end - begin) * model_->hidden_units_ * elem_size;
                    seq.input_embeddings.emplace_back((std::byte*)emb_tensor_ptr, (std::byte*)(emb_tensor_ptr + count));
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
    Copy_(state_->h_context_length, batch_size, context_length_buf_);
    Copy_(state_->h_finished, batch_size, finished_buf_);
    Copy_(state_->h_rope_theta, batch_size, rope_theta_);

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
        g.max_init_ctx_len = max_context_len;
        g.step             = max_context_len;
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

    context_decoder_input_buf_ = {{max_forward_token_num_, hidden_units}, data_type_, MEMORY_GPU};
    context_decoder_ids_buf_   = {max_forward_token_num_, MEMORY_GPU};

    decoder_output_buf_ = {{batchxbeam, hidden_units}, data_type_, MEMORY_GPU};

    input_length_buf_    = {batchxbeam, MEMORY_GPU};
    context_length_buf_  = {batchxbeam, MEMORY_GPU};
    init_context_length_ = {batchxbeam, MEMORY_GPU};

    sequence_lengths_ = {batchxbeam, MEMORY_GPU};

    cu_block_counts_ = {batch_size, MEMORY_GPU};
    block_ptrs_      = {max_batch_block_count, MEMORY_GPU};

    if (!logits_buf_) {  // may be alias of local_logits_buf_
        logits_buf_ = {{batchxbeam, vocab_size}, TYPE_FP32, MEMORY_GPU};
    }

    sampled_logprobs_ = {batchxbeam * kMaxLogProb, MEMORY_GPU};
    sampled_indexes_  = {batchxbeam * kMaxLogProb, MEMORY_GPU};
    sampled_nums_     = {batchxbeam, MEMORY_GPU};

    token_ids_buf_ = {ssize_t(session_len * 2 * batchxbeam), MEMORY_GPU};

    finished_buf_  = {(int)batchxbeam, MEMORY_GPU};
    seq_limit_len_ = {batch_size, MEMORY_GPU};

    rope_theta_ = {batch_size, MEMORY_GPU};

    is_allocate_buffer_ = true;
}

void LlamaBatch::AllocatePersistantBuffer(ssize_t max_batch_size, int cache_block_seq_len)
{
    d_stop_words_ =
        (int*)allocator_->reMalloc(d_stop_words_, sizeof(int) * max_batch_size * 2 * kMaxStopBadWordsLen, true);
    d_bad_words_ =
        (int*)allocator_->reMalloc(d_bad_words_, sizeof(int) * max_batch_size * 2 * kMaxStopBadWordsLen, true);
    h_stop_words_ =
        (int*)allocator_->reMalloc(h_stop_words_, sizeof(int) * max_batch_size * 2 * kMaxStopBadWordsLen, true, true);
    h_bad_words_ =
        (int*)allocator_->reMalloc(h_bad_words_, sizeof(int) * max_batch_size * 2 * kMaxStopBadWordsLen, true, true);

    h_min_length_    = (int*)allocator_->reMalloc(h_min_length_, sizeof(int) * max_batch_size, true, true);
    h_runtime_top_k_ = (int*)allocator_->reMalloc(h_runtime_top_k_, sizeof(int) * max_batch_size, true, true);
    h_runtime_top_p_ = (float*)allocator_->reMalloc(h_runtime_top_p_, sizeof(float) * max_batch_size, true, true);
    h_runtime_min_p_ = (float*)allocator_->reMalloc(h_runtime_min_p_, sizeof(float) * max_batch_size, true, true);
    h_temperature_   = (float*)allocator_->reMalloc(h_temperature_, sizeof(float) * max_batch_size, true, true);
    h_repetition_penalty_ =
        (float*)allocator_->reMalloc(h_repetition_penalty_, sizeof(float) * max_batch_size, true, true);

    h_random_seed_ = {max_batch_size, MEMORY_CPU_PINNED};
    Clear(h_random_seed_);

    d_random_seed_ = {max_batch_size, MEMORY_GPU};
    Clear(d_random_seed_);

    h_curand_state_ = {{max_batch_size, sizeof(curandState_t)}, MEMORY_CPU_PINNED};
    Clear(h_curand_state_.buffer());

    d_curand_state_ = {{max_batch_size, sizeof(curandState_t)}, MEMORY_GPU};
    Clear(d_curand_state_.buffer());

    d_end_ids_buf_ = {max_batch_size * kMaxEndIdsSize, MEMORY_GPU};
    h_end_ids_buf_ = {max_batch_size * kMaxEndIdsSize, MEMORY_CPU_PINNED};

    for (auto& s : states_) {
        s.output_ids = {{max_batch_size, session_len_}, MEMORY_GPU};
        Clear(s.output_ids.buffer());

        s.curand_state = {{max_batch_size, sizeof(curandState_t)}, MEMORY_GPU};
        Clear(s.curand_state.buffer());
    }

    const size_t max_batch_block_count =
        max_batch_size * ((session_len_ + cache_block_seq_len - 1) / cache_block_seq_len);

    h_input_length_buf_ = {max_batch_size, MEMORY_CPU_PINNED};
    h_cu_block_counts_  = {max_batch_size + 1, MEMORY_CPU_PINNED};
    h_block_ptrs_       = {(ssize_t)max_batch_block_count, MEMORY_CPU_PINNED};

    for (auto& s : states_) {
        s.h_prompt_length  = {max_batch_size, MEMORY_CPU_PINNED};
        s.h_context_length = {max_batch_size, MEMORY_CPU_PINNED};
        s.h_finished       = {max_batch_size * 2, MEMORY_CPU_PINNED};
        s.h_rope_theta     = {max_batch_size, MEMORY_CPU_PINNED};
    }

    h_seq_limit_len_ = {max_batch_size, MEMORY_CPU_PINNED};
    std::fill_n(h_seq_limit_len_.data(), max_batch_size, 0);

    h_output_ids_ = {max_batch_size * session_len_, MEMORY_CPU_PINNED};

    sampled_logprobs_ = {max_batch_size * kMaxLogProb, MEMORY_CPU_PINNED};
    sampled_indexes_  = {max_batch_size * kMaxLogProb, MEMORY_CPU_PINNED};
    sampled_nums_     = {max_batch_size, MEMORY_CPU_PINNED};

    is_allocate_persistant_buffer_ = true;
}

void LlamaBatch::AllocSymmBuffers()
{
    core::ContextGuard guard{symm_alloc_};

    const ssize_t hidden_units      = model_->hidden_units_;
    const ssize_t vocab_size_padded = model_->vocab_size_padded_;

    // Native comm fuses allreduce & rmsnorm in token granularity
    const ssize_t max_fwd_token_num = ((size_t)max_forward_token_num_ + tp_size_ - 1) / tp_size_ * tp_size_;

    /// TODO: rename this to hidden_states
    context_decoder_output_buf_ =
        core::Tensor{{max_fwd_token_num, param_.attn_dp_size, hidden_units}, data_type_, MEMORY_GPU};

    local_logits_buf_ = core::Tensor{{max_batch_size_, vocab_size_padded}, TYPE_FP32, MEMORY_GPU};

    if (model_->use_allgather_2d_) {
        // Reference into `local_logits_buf_`
        logits_buf_ = core::Tensor{
            local_logits_buf_.raw_data(), local_logits_buf_.layout(), TYPE_FP32, local_logits_buf_.device()};
    }
}

void LlamaBatch::FreeSymmBuffers()
{
    context_decoder_output_buf_ = {};
    local_logits_buf_           = {};
    local_context_logits_buf_   = {};
}

void LlamaBatch::FreeBuffer()
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (is_allocate_persistant_buffer_) {
        allocator_->free((void**)&d_stop_words_);
        allocator_->free((void**)&h_stop_words_, true);
        allocator_->free((void**)&d_bad_words_);
        allocator_->free((void**)&h_bad_words_, true);
        is_allocate_persistant_buffer_ = false;
    }
}

LlamaBatch::~LlamaBatch()
{
    TM_LOG_DEBUG("~LlamaBatch()");

    internal_thread_.join();

    // The dtor maybe called from unknown thread, set device id before CUDA calls
    cudaSetDevice(device_id_);
    cudaStreamSynchronize(stream_);

    FreeBuffer();

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
    max_forward_token_num_(param.max_prefill_token_num + param.max_batch_size),
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
    allocator_(ctx->allocator.get()),
    cublas_wrapper_(ctx->cublas_wrapper.get()),
    context_(std::move(ctx)),
    model_(std::move(model)),
    comm_(context_->comm),
    session_len_(param.session_len)
{
    const auto cache_block_seq_len = model_->attn_param_.cache_block_seq_len;

    const int dbits = core::get_byte_size(data_type, 8);

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
        return AllReduce(model_->comm_->h_tp_group, free, comm::RedOp::kMin);
    };

    sequence_manager_.reset(new SequenceManager{model_->layer_num_,
                                                block_config,
                                                param.cache_max_block_count,
                                                param.cache_chunk_size,
                                                param.enable_prefix_caching,
                                                tp_rank_,
                                                allocator_,
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

    for (auto& s : states_) {
        s.requests.resize(max_batch_size_);
        s.sequences.resize(max_batch_size_);
        s.seq_len_limit.resize(max_batch_size_);
        s.errors.resize(max_batch_size_);
    }

    state_    = &states_[0];
    back_     = &states_[1];
    incoming_ = &states_[2];

    symm_alloc_ = core::SimpleAllocator::Create([this](ssize_t size) { return SymmAlloc(size, true); },
                                                [this](void* p, ssize_t size) { return SymmFree(p, size, true); },
                                                MEMORY_GPU);

    AllocSymmBuffers();

    AllocateBuffer(max_batch_size_, session_len_, cache_block_seq_len);
    AllocatePersistantBuffer(max_batch_size_, cache_block_seq_len);

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

    // Context length at initialization, will stay constant until re-initialziation
    Copy(context_length_buf_, batch_size, init_context_length_);

    Copy(context_length_buf_, batch_size, sequence_lengths_);
    // `sequence_lengths_` will be increased by dynamic decode
    // note that in decoder and in output "sequence length" has different semantic
    // - in decoder it means length of sequence that has kv cache already computed
    // - in output it means length of all tokens (the last generated token does not have k/v cache computed yet)
    invokePlusScalar(sequence_lengths_.data(), -1, batch_size, stream_);
    sync_check_cuda_error();

    Clear(token_ids_buf_.slice(0, batch_size * session_len_));
    invokeTranspose2D(token_ids_buf_.data(), state_->output_ids.data(), batch_size, session_len_, stream_);
    sync_check_cuda_error();

    // token_ids_buf_[s, b]
    // ABCDe            ABCDe     e
    // ABCDEFGHIJk      ABCDEFGHIJk
    // ABCDEFGHi    ->  ABCDEFGHi i
    // ABCDEFGh         ABCDEFGh  h
    // ABCd             ABCd      d
    invokePadLastTokenIds(token_ids_buf_.data(), init_context_length_.data(), g.max_init_ctx_len, batch_size, stream_);
    sync_check_cuda_error();

    // seq_limit_len_, will be compared to `step` instead of `sequence_length`, so padding len should be accounted for
    for (int i = 0; i < batch_size; ++i) {
        h_seq_limit_len_[i] = state_->seq_len_limit[i] + (g.max_init_ctx_len - state_->h_context_length[i]);
    }
    Copy(h_seq_limit_len_, batch_size, seq_limit_len_);

    TensorMap inputs;

    auto member_to_tensor = [&](auto getter, auto key, auto dest, auto init) {
        int count = 0;
        for (int i = 0; i < batch_size; ++i) {
            // `std::invoke`
            dest[i] = state_->requests[i]->gen_cfg.*getter;
            count += dest[i] != init;
        }
        if (count) {
            inputs.insert(key, {MEMORY_CPU, getTensorType<decltype(init)>(), {(size_t)batch_size}, dest});
        }
    };

    using G = GenerationConfig;
    member_to_tensor(&G::top_k, "runtime_top_k", h_runtime_top_k_, 0);
    member_to_tensor(&G::top_p, "runtime_top_p", h_runtime_top_p_, 0);
    member_to_tensor(&G::min_p, "runtime_min_p", h_runtime_min_p_, 0);
    member_to_tensor(&G::temperature, "temperature", h_temperature_, 1.f);
    member_to_tensor(&G::repetition_penalty, "repetition_penalty", h_repetition_penalty_, 1.f);
    member_to_tensor(&G::min_new_tokens, "min_length", h_min_length_, 0);

    auto init_stop_bad_words = [&](auto getter, auto key, auto h_buf, auto d_buf) {
        int                                     max_length = 0;
        std::vector<std::pair<const int*, int>> copy_tokens(batch_size);
        std::vector<std::pair<const int*, int>> copy_offsets(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            const auto& [token_ids, offsets] = std::invoke(getter, state_->requests[i]->gen_cfg);
            if (offsets.size() == 0 || token_ids.size() == 0) {
                continue;
            }
            FT_CHECK(offsets.back() == token_ids.size());
            if (offsets.back() <= kMaxStopBadWordsLen) {
                copy_tokens[i]  = std::make_pair(token_ids.data(), (int)token_ids.size());
                copy_offsets[i] = std::make_pair(offsets.data(), (int)offsets.size());
                max_length      = std::max(max_length, (int)token_ids.size());
            }
            else {
                auto trunc_offset_size =
                    std::upper_bound(offsets.begin(),
                                     offsets.begin() + std::min(kMaxStopBadWordsLen, (int)offsets.size()),
                                     kMaxStopBadWordsLen)
                    - offsets.begin();
                TM_LOG_WARNING("[InitializeSampling] [%ld] %s length (%d) exceeds %d, truncated to %d",
                               state_->requests[i]->id,
                               key,
                               offsets.back(),
                               kMaxStopBadWordsLen,
                               trunc_offset_size);
                if (trunc_offset_size > 0) {
                    int trunc_token_size = offsets[trunc_token_size - 1];
                    copy_tokens[i]       = std::make_pair(token_ids.data(), trunc_token_size);
                    copy_offsets[i]      = std::make_pair(offsets.data(), trunc_offset_size);
                    max_length           = std::max(max_length, trunc_token_size);
                }
            }
        }
        if (!max_length) {
            return;
        }
        std::fill_n(h_buf, batch_size * 2 * max_length, -1);
        for (int i = 0; i < batch_size; ++i) {
            if (copy_tokens[i].first != nullptr) {
                std::copy_n(copy_tokens[i].first, copy_tokens[i].second, h_buf + i * 2 * max_length);
            }
            if (copy_offsets[i].first != nullptr) {
                std::copy_n(copy_offsets[i].first, copy_offsets[i].second, h_buf + i * 2 * max_length + max_length);
            }
        }
        core::Copy(h_buf, batch_size * 2 * max_length, d_buf);
        inputs.insert(key, {MEMORY_GPU, TYPE_INT32, {(size_t)batch_size, (size_t)2, (size_t)max_length}, d_buf});
    };
    init_stop_bad_words(&G::stop_ids, "stop_words_list", h_stop_words_, d_stop_words_);
    init_stop_bad_words(&G::bad_ids, "bad_words_list", h_bad_words_, d_bad_words_);

    // MinLengthPenalty
    if (inputs.isExist("min_length")) {
        inputs.insert(
            {"prompt_length", {MEMORY_CPU, TYPE_INT32, {(size_t)batch_size}, state_->h_prompt_length.data()}});
        inputs.insert(
            {"context_length", {MEMORY_CPU, TYPE_INT32, {(size_t)batch_size}, state_->h_context_length.data()}});
    }

    // init for eos
    auto init_for_eos = [&] {
        int max_length = 0;
        for (int i = 0; i < batch_size; ++i) {
            max_length = std::max(max_length, (int)state_->requests[i]->gen_cfg.eos_ids.size());
        }
        if (max_length) {
            max_length     = std::min(max_length, kMaxEndIdsSize);
            int* h_end_ids = h_end_ids_buf_.data();
            std::fill(h_end_ids, h_end_ids + std::min(kMaxEndIdsSize, max_length) * batch_size, -1);
            for (int i = 0; i < batch_size; ++i) {
                const auto& eos_ids = state_->requests[i]->gen_cfg.eos_ids;
                if (eos_ids.size() == 0) {
                    continue;
                }
                if (eos_ids.size() > kMaxEndIdsSize) {
                    TM_LOG_WARNING("[InitializeSampling] [%ld] eos length (%d) exceeds %d, truncated to %d",
                                   (long)state_->requests[i]->id,
                                   (int)eos_ids.size(),
                                   kMaxEndIdsSize,
                                   kMaxEndIdsSize);
                }
                std::copy_n(eos_ids.begin(), std::min((int)eos_ids.size(), kMaxEndIdsSize), h_end_ids);
                h_end_ids += max_length;
            }
            Copy(h_end_ids_buf_, batch_size * max_length, d_end_ids_buf_);
            inputs.insert("end_ids",
                          {MEMORY_GPU, TYPE_INT32, {(size_t)batch_size, (size_t)max_length}, d_end_ids_buf_.data()});
        }
    };
    init_for_eos();

    inputs_ = std::move(inputs);

    {
        NvtxScope setup("DynamicDecodeLayer.setup");
        model_->dynamic_decode_layer_->setup(batch_size, 1, &inputs_);
    }

    TensorMap outputs;
    for (int i = 0; i < batch_size; i++) {
        if (state_->requests[i]->gen_cfg.output_logprobs) {
            outputs.insert({"sampled_logprobs",
                            {MEMORY_GPU, TYPE_FP32, {(size_t)batch_size, 1, kMaxLogProb}, sampled_logprobs_.data()}});
            outputs.insert({"sampled_indexes",
                            {MEMORY_GPU, TYPE_UINT32, {(size_t)batch_size, 1, kMaxLogProb}, sampled_indexes_.data()}});
            outputs.insert({"sampled_nums", {MEMORY_GPU, TYPE_UINT32, {(size_t)batch_size, 1}, sampled_nums_.data()}});

            break;
        }
    }
    outputs_ = std::move(outputs);
    sync_check_cuda_error();
}

void LlamaBatch::ComputeAndOutputLogits(const core::Tensor& hidden_states, int first, int last)
{
    int  token_num = 0;
    bool found     = false;
    for (int i = first; i < last; ++i) {
        if (state_->requests[i]->gen_cfg.output_logits == GenerationConfig::kAll) {
            const auto& s = *state_->sequences[i];
            // Skip when the seq is filling missed cache only
            if (s.cache_len + h_input_length_buf_[i] > s.tokens.size()) {
                found = true;
            }
        }
        token_num += h_input_length_buf_[i];
    }

    if (!found) {
        return;
    }

    const ssize_t vocab_size_padded = model_->vocab_size_padded_;

    if (tp_size_ > 1) {
        FT_CHECK(vocab_size_padded % tp_size_ == 0);
        if (local_context_logits_buf_.size() < token_num * vocab_size_padded) {
            check_cuda_error(cudaStreamSynchronize(stream_));
            comm_.h_tp_group->Sync();
            local_context_logits_buf_ = core::Tensor_<float>{{token_num, vocab_size_padded}, symm_alloc_};
            check_cuda_error(cudaStreamSynchronize(stream_));
            comm_.h_tp_group->Sync();
        }
    }

    if (model_->use_allgather_2d_) {
        // No intermediate transpose needed
        // Reference into `local_context_logits_buf_`
        context_logits_buf_ = core::Tensor{local_context_logits_buf_.raw_data(),
                                           local_context_logits_buf_.layout(),
                                           TYPE_FP32,
                                           local_context_logits_buf_.device()};
    }
    else {
        context_logits_buf_ = core::Tensor{{token_num, vocab_size_padded}, TYPE_FP32, MEMORY_GPU};
    }

    model_->postDecodeEmbedding(context_logits_buf_.data<float>(),
                                local_context_logits_buf_.data<float>(),
                                hidden_states.raw_data(),
                                token_num);

    if (tp_rank_ != 0) {
        return;
    }

    OutputLogits(context_logits_buf_.data<float>(), first, last, GenerationConfig::kAll);
}

void LlamaBatch::OutputLogits(const float* logits, int first, int last, GenerationConfig::OutType out_type)
{
    // when `is_all` is true, logits only contains last token of the sequences
    const bool is_all = out_type == GenerationConfig::kAll;

    for (int i = first; i < last; ++i) {

        const int    input_len = h_input_length_buf_[i];  // input lenght for this iter
        const float* src_ptr   = logits;

        logits += (is_all ? input_len : 1) * model_->vocab_size_padded_;

        if (state_->requests[i]->gen_cfg.output_logits == out_type) {

            auto dst_ptr = state_->requests[i]->outputs.getPtr<float>("logits");

            const int cache_len   = state_->sequences[i]->cache_len;
            const int history_len = state_->sequences[i]->tokens.size();

            // ----------H------I-------P-----------
            //      C        C      C         C

            // offset to the last token prompt
            const int offset = is_all ? 0 : state_->requests[i]->inputs.at("input_ids").shape[0] - 1;

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

            if (is_all) {
                // Skip invalid tokens caused by cache miss
                src_ptr += std::max(0, (history_len + offset) - cache_len) * model_->vocab_size_padded_;
            }
            // Skip previous chunks
            dst_ptr += std::max(0, cache_len - (history_len + offset)) * model_->vocab_size_;

            check_cuda_error(cudaMemcpy2DAsync(dst_ptr,
                                               sizeof(float) * model_->vocab_size_,
                                               src_ptr,
                                               sizeof(float) * model_->vocab_size_padded_,
                                               sizeof(float) * model_->vocab_size_,
                                               valid_len,
                                               cudaMemcpyDefault,
                                               stream_));
        }
    }
}

void LlamaBatch::OutputLastHiddenState(const core::Tensor& hidden_states, int first, int last)
{
    const auto& src_buf = hidden_states.buffer();
    int         base    = 0;

    for (int i = first; i < last; ++i) {
        const int input_len = h_input_length_buf_[i];  // input lenght for this iter

        if (auto out_type = state_->requests[i]->gen_cfg.output_last_hidden_state) {

            const bool is_all = out_type == GenerationConfig::kAll;

            auto&        dst = state_->requests[i]->outputs.at("last_hidden_state");
            core::Buffer dst_buf{dst.getPtr<void>(), (core::ssize_t)dst.size(), dst.where};

            const int cache_len   = state_->sequences[i]->cache_len;
            const int history_len = state_->sequences[i]->tokens.size();

            // offset to the last prompt token
            const int offset = is_all ? 0 : state_->requests[i]->inputs.at("input_ids").shape[0] - 1;

            const int valid_len = input_len - std::max(0, (history_len + offset) - cache_len);

            // TM_LOG_ERROR("%d %d %d %d %d", history_len, offset, cache_len, input_len, valid_len);

            if (valid_len > 0) {
                // Skip invalid tokens caused by cache miss
                int src_base = std::max(0, (history_len + offset) - cache_len) + base;
                // Skip previous chunks
                int dst_base = std::max(0, cache_len - (history_len + offset));

                core::Copy(src_buf.raw_data(src_base * model_->hidden_units_),
                           valid_len * model_->hidden_units_,
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
                           token_ids_buf_.data(),
                           init_context_length_.data(),
                           g.max_init_ctx_len,
                           g.step,
                           session_len_,
                           batch_size - g.partial,
                           stream_);
        sync_check_cuda_error();
    }

    Copy(token_ids_buf_.slice((g.step - 1) * (batch_size - g.partial), -1), batch_size - g.partial, h_output_ids_);
    Copy(finished_buf_, batch_size, state_->h_finished);
    Copy(sequence_lengths_, batch_size, state_->h_context_length);

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
        // output logprobs, should be set before sequence_length
        float*    sampled_logprobs_ptr = h_sampled_logprobs_.data();
        uint32_t* sampled_indexes_ptr  = h_sampled_indexes_.data();
        uint32_t* sampled_nums_ptr     = h_sampled_nums_.data();
        for (int i = 0; i < batch_size - g.partial; ++i) {
            if (state_->requests[i] && state_->requests[i]->gen_cfg.output_logprobs) {
                auto logprob_vals    = state_->requests[i]->outputs.getPtr<float>("logprob_vals");
                auto logprob_indexes = state_->requests[i]->outputs.getPtr<uint32_t>("logprob_indexes");
                auto logprob_nums    = state_->requests[i]->outputs.getPtr<uint32_t>("logprob_nums");

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
                auto      output_ids  = static_cast<int*>(r->output_ids.data);
                auto      output_len  = static_cast<int*>(r->sequence_length.data);
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
                const auto seq_len = r->sequence_length.getVal<int>();
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
        const auto output_ids = state_->requests[index]->output_ids.getPtr<int>();
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

    const auto len = state_->requests[index]->sequence_length.getVal<int>();
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

void LlamaBatch::InternalThreadEntry()
{
    // TM_LOG_INFO("[InternalThreadEntry] %d", (int)rank_);
    check_cuda_error(cudaSetDevice(device_id_));

    core::ContextGuard guard{context_->core_stream, context_->core_allocator};

    // Initialize `AnomalyHandler`
    AnomalyHandler::instance().Init(tp_rank_, model_->vocab_size_padded_, 0, max_batch_size_, stream_);

    GenerationState g{};

    while (1) {

        std::shared_ptr<RequestData> req;

        if (tp_rank_ == 0) {
            req = std::make_shared<RequestData>();
            {
                NvtxScope  _("pop");
                const int  free_slot_count = max_batch_size_ - state_->size + g.finished_count;
                const bool is_empty        = (free_slot_count == max_batch_size_);
                // Block if batch is empty AND no silbings are ready
                gateway_->pop(req->infer, req->kill, free_slot_count, is_empty, req->abort, dp_rank_);
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

        Broadcast(comm_.h_tp_group, req, 0);

        if (req->abort) {
            TM_LOG_INFO("[InternalThreadEntry] stop requested.");
            break;
        }

        std::vector<Signal> signals;

        ProcessKillRequests(req->kill, signals);

        // Shared `priority` field will be assigned by rank-0
        ProcessInferRequests(req->infer, signals);

        // 1. Wait while shared `requests` is being used
        // 2. Broadcast modifcations from rank-0
        // comm_.h_comm->Sync(comm_.h_comm_tp_group);

        ProcessCancelRequests(req->cancel, signals);

        if (tp_rank_ == 0) {
            gateway_->notify(std::move(signals));
        }

        Initialize(g);

        const int n_active = AllReduce(comm_.h_dp_group, state_->active_size, comm::RedOp::kSum);

        if (n_active) {
            //
            Forward(g);

            Finish(g, signals);

            if (g.finished_count) {
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

bool LlamaBatch::Forward(GenerationState& g)
{
    NvtxScope _("Forward");

    FT_CHECK(max_context_token_num_ >= max_batch_size_);

    const int active_size = state_->active_size;

    constexpr int kLogInterval = 10;
    if (tp_rank_ == 0 && (g.step - 1) % kLogInterval == 0) {
        TM_LOG_INFO("------------------------- step = %d -------------------------", g.step - 1);
    }

    int               pf_offset = -1;
    std::vector<int*> input_d_ptrs(active_size);

    for (int i = 0; i < active_size; ++i) {
        const auto& seq = *state_->sequences[i];
        // const int   missing = state_->h_context_length[i] - seq.cache_len;
        FT_CHECK(seq.input_length >= 1);
        h_input_length_buf_[i] = seq.input_length;
        input_d_ptrs[i]        = state_->output_ids.data() + i * session_len_ + seq.cache_len;
        if (seq.input_length > 1 && pf_offset < 0) {
            pf_offset = i;
        }
    }
    if (pf_offset < 0) {
        pf_offset = active_size;
    }

    // These buffers are only accessed when there are prefill workloads
    if (pf_offset != active_size) {
        Copy(state_->h_context_length, active_size, context_length_buf_);
        Copy(h_input_length_buf_, active_size, input_length_buf_);
    }

    // Find mini-batch offsets: input length > 1 ? prefill() : decode()
    // Constraints on mini-batches
    //   sum(Q) <= `max_forward_token_num` && sum(K) <= `max_context_token_num`
    std::vector<int> offsets{0};
    // initialize first mini-batch with decode tokens
    int sum_q = pf_offset;
    int sum_k = 0;  // only for prefill
    for (int i = pf_offset; i < active_size; ++i) {
        FT_CHECK(h_input_length_buf_[i] <= max_forward_token_num_);
        const int q = sum_q + h_input_length_buf_[i];
        const int k = sum_k + state_->h_context_length[i];
        if (q <= max_forward_token_num_ && k <= max_context_token_num_) {
            sum_q = q;
            sum_k = k;
        }
        else {
            offsets.push_back(i);
            sum_q = h_input_length_buf_[i];
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

    // forward on mini-batches
    for (int p = 0; p < (int)offsets.size() - 1; ++p) {
        const int first           = offsets[p];
        const int last            = offsets[p + 1];
        const int mini_batch_size = last - first;
        int*      input_ids       = context_decoder_ids_buf_.data();

        BatchedCopy batched_copy;
        int         sum_k = 0;
        for (int i = first; i < last; ++i) {
            input_ids = batched_copy.Add(input_d_ptrs[i], h_input_length_buf_[i], input_ids);
            if (h_input_length_buf_[i] > 1) {
                sum_k += state_->h_context_length[i];
            }
        }
        int sum_q = input_ids - context_decoder_ids_buf_.data();

        batched_copy.Submit(stream_);

        const int dc_batch_size = p ? 0 : pf_offset;
        const int pf_batch_size = mini_batch_size - dc_batch_size;

        if (tp_rank_ == 0) {
            if (pf_batch_size) {
                const auto max_q =
                    *std::max_element(h_input_length_buf_.data() + first, h_input_length_buf_.data() + last);
                const auto max_k =
                    *std::max_element(state_->h_context_length.data() + first, state_->h_context_length.data() + last);
                TM_LOG_INFO("[Forward] [%d, %d), dc=%d, pf=%d, sum_q=%d, sum_k=%d, max_q=%d, max_k=%d",
                            first,
                            last,
                            dc_batch_size,
                            pf_batch_size,
                            sum_q,
                            sum_k,
                            max_q,
                            max_k);
            }
        }

        // Synchronize batch token num with sync DP ranks
        auto local_token_nums = AllGather(comm_.h_dp_group, sum_q);
        auto global_token_num = std::accumulate(local_token_nums.begin(), local_token_nums.end(), 0);

        auto hidden_states = context_decoder_output_buf_.slice(0, global_token_num);

        model_->Forward(context_decoder_ids_buf_.slice(0, sum_q),  // temp
                        hidden_states,                             // temp
                        decoder_output_buf_.slice(first, mini_batch_size),
                        block_ptrs_,
                        cu_block_counts_.slice(first, mini_batch_size + 1),
                        h_input_length_buf_.slice(first, mini_batch_size),
                        state_->h_context_length.slice(first, mini_batch_size),
                        rope_theta_.slice(first, mini_batch_size),
                        finished_buf_.slice(first, mini_batch_size),
                        Buffer(local_token_nums.data(), local_token_nums.size(), MEMORY_CPU),
                        lora_mask_buf_,
                        dc_batch_size,
                        pf_batch_size,
                        state_->sequences.data() + first);

        ComputeAndOutputLogits(context_decoder_output_buf_, first, last);
        OutputLastHiddenState(context_decoder_output_buf_, first, last);
    }

    if (active_size > g.partial) {
        model_->postDecodeEmbedding(logits_buf_.data<float>(),
                                    local_logits_buf_.data<float>(),
                                    decoder_output_buf_.raw_data(),
                                    active_size - g.partial);

        AnomalyHandler::instance().FixLogits(logits_buf_.data<float>(), active_size - g.partial, 1);

        OutputLogits(logits_buf_.data<float>(), 0, active_size - g.partial, GenerationConfig::kGeneration);

        FT_CHECK(g.step >= 0);

        if (!g.skip_init_sampling) {
            InitializeSampling(g);
        }
        // stop-words & bad-words require the matched tokens to be contiguous, so item size > 1 is
        // not supported yet.
        model_->dynamicDecode(token_ids_buf_.data(),
                              finished_buf_.data(),
                              sequence_lengths_.data(),
                              nullptr,
                              (curandState_t*)state_->curand_state.data(),
                              &inputs_,
                              &outputs_,
                              logits_buf_.data<float>(),
                              seq_limit_len_.data(),
                              init_context_length_.data(),
                              g.step,
                              0,
                              g.max_init_ctx_len,
                              session_len_ * 2,
                              active_size - g.partial);
    }

    std::fill(h_input_length_buf_.data(), h_input_length_buf_.data() + active_size, 0);

    // `SequenceManager` needs real-time value of cache length
    for (int i = 0; i < active_size; ++i) {
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
        std::vector<int> curr(active_size);
        core::Copy(token_ids_buf_.data() + g.step * active_size, active_size, curr.data());
        cudaStreamSynchronize(stream_);
        std::stringstream scurr;
        for (int k = 0; k < curr.size(); ++k) {
            scurr << std::setw(6) << curr[k];
        }
        TM_LOG_INFO("[Forward] step = %d, [%s]", g.step - 1, scurr.str().c_str());
    }

    // check_cuda_error(cudaStreamSynchronize(stream_));

    ////////////////////////////////////////////////
    /// ! increase the counters
    g.step += 1;

    // PrintDecodeTokens(token_ids_buf_, g.step, active_size, stream_, "Forward");

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

    if (tp_rank_ == 0) {
        auto str = Join(bss.begin(), bss.end(), ", ");
        TM_LOG_INFO("[Gemm2] Tuning sequence: %s", str.c_str());
    }

    if (!bss.empty()) {
        const auto                         max_bs = *std::max_element(bss.begin(), bss.end());
        std::vector<int>                   input_ids(max_bs);
        std::mt19937                       g{};
        std::uniform_int_distribution<int> d{0, (int)model_->vocab_size_ - 1};
        for (auto& x : input_ids) {
            x = d(g);
        }
        core::Copy(input_ids.data(), max_bs, context_decoder_ids_buf_.data());
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
            model_->Forward(context_decoder_ids_buf_.slice(0, token_num),
                            context_decoder_output_buf_.slice(0, token_num * param_.attn_dp_size),
                            decoder_output_buf_.slice(0, bsz),
                            block_ptrs_,
                            cu_block_counts_.slice(0, bsz + 1),
                            Buffer{&input_length, 1, MEMORY_CPU},
                            Buffer{&input_length, 1, MEMORY_CPU},
                            rope_theta_.slice(0, bsz),
                            finished_buf_.slice(0, bsz),
                            Buffer{local_token_nums.data(), (int)local_token_nums.size(), MEMORY_CPU},
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
        return context_->core_allocator->allocate(size);
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
        context_->core_allocator->deallocate(ptr, size);
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

    cudaStreamSynchronize(stream_);
    comm_.h_comm->Sync();
}

}  // namespace turbomind
