// Copyright (c) OpenMMLab. All rights reserved.

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <numeric>

#include "src/turbomind/engine/request.h"
#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/utils/logger.h"

// #include "dbg.h"

namespace turbomind {

namespace {

// Lightweight counters for prefix-cache diagnostics (single scheduler thread; relaxed atomics are sufficient).
std::atomic<uint64_t> g_linear_snapshot_publish_ok{0};
std::atomic<uint64_t> g_linear_snapshot_publish_miss{0};
std::atomic<uint64_t> g_linear_snapshot_publish_pool_exhausted{0};
std::atomic<uint64_t> g_prefix_match_skipped_alpha{0};
std::atomic<uint64_t> g_prefix_match_restored_linear{0};

}  // namespace

template<typename T>
std::string vector2string(const std::vector<T>& data)
{
    if (data.empty()) {
        return "nil";
    }
    std::stringstream ss;

    auto it = data.begin();
    ss << *it;

    for (++it; it != data.end(); ++it) {
        ss << ", " << *it;
    }
    return ss.str();
}

SequenceManager::SequenceManager(const ModelParam& model_param,
                                 DataType          runtime_dtype,
                                 int               cache_block_seq_len,
                                 int               attn_tp_size,
                                 int               max_batch_size,
                                 double            block_count,
                                 int               chunk_size,
                                 bool              enable_prefix_caching,
                                 int               rank,
                                 int               attn_cp_size,
                                 core::Allocator   allocator,
                                 GetFreeMemSize    get_free_size):
    block_seq_len_(cache_block_seq_len), rank_(rank), attn_cp_size_(attn_cp_size)
{
    TM_CHECK_GT(attn_tp_size, 0);
    TM_CHECK_GT(cache_block_seq_len, 0);

    int cache_layer_num = model_param.layer_num;
    for (const auto& type : model_param.layer_types) {
        if (type == 1) {
            --cache_layer_num;
            ++num_linear_layers_;
        }
    }

    const bool   need_free_mem_probe = num_linear_layers_ > 0 && (block_count < 1. || enable_prefix_caching);
    const size_t free_before         = need_free_mem_probe ? get_free_size() : 0;

    if (num_linear_layers_ > 0) {

        key_head_dim_ = model_param.linear_key_head_dim > 0 ? model_param.linear_key_head_dim : model_param.head_dim;
        value_head_dim_ =
            model_param.linear_value_head_dim > 0 ? model_param.linear_value_head_dim : model_param.head_dim;
        d_conv_               = model_param.linear_conv_kernel_dim > 0 ? model_param.linear_conv_kernel_dim : 4;
        const int num_k_heads = model_param.linear_num_key_heads / attn_tp_size;
        num_v_heads_          = model_param.linear_num_value_heads / attn_tp_size;
        const int key_dim     = num_k_heads * key_head_dim_;
        const int value_dim   = num_v_heads_ * value_head_dim_;
        conv_dim_             = key_dim * 2 + value_dim;
        linear_conv_dtype_    = model_param.data_type;
        linear_state_dtype_   = model_param.linear_state_dtype;

        TM_CHECK_GT(max_batch_size, 0);
        pooled_conv_states_ = {{max_batch_size, num_linear_layers_, d_conv_, conv_dim_}, linear_conv_dtype_, kDEVICE};
        pooled_recurrent_states_  = {{max_batch_size, num_linear_layers_, num_v_heads_, key_head_dim_, value_head_dim_},
                                    linear_state_dtype_,
                                    kDEVICE};
        linear_active_pool_bytes_ = pooled_conv_states_.byte_size() + pooled_recurrent_states_.byte_size();
        linear_snapshot_bytes_per_block_ = pooled_conv_states_.slice(0, 1).squeeze(0).byte_size()
                                           + pooled_recurrent_states_.slice(0, 1).squeeze(0).byte_size();

        free_linear_state_slots_.reserve(max_batch_size);
        for (int slot = max_batch_size - 1; slot >= 0; --slot) {
            free_linear_state_slots_.push_back(slot);
        }
        TM_LOG_INFO("[SeqMgr] linear-state slot pool initialized: %d slots", max_batch_size);
        const auto   conv_one      = pooled_conv_states_.slice(0, 1).squeeze(0);
        const auto   recurrent_one = pooled_recurrent_states_.slice(0, 1).squeeze(0);
        const double mb            = 1.0 / (1024.0 * 1024.0);
        TM_LOG_INFO("[SeqMgr] linear-state per slot: conv %.2f MB + recurrent %.2f MB = %.2f MB",
                    conv_one.byte_size() * mb,
                    recurrent_one.byte_size() * mb,
                    (conv_one.byte_size() + recurrent_one.byte_size()) * mb);
        TM_LOG_INFO("[SeqMgr] linear-state combined total: %.2f MB", linear_active_pool_bytes_ * mb);
        if (enable_prefix_caching) {
            TM_LOG_INFO("[SeqMgr] linear prefix snapshot per reusable block: %.2f MB",
                        linear_snapshot_bytes_per_block_ * mb);
        }
    }

    const int  dbits        = byte_size(runtime_dtype, 8);
    const auto quant_policy = model_param.quant_policy;
    const int  elem_bits    = quant_policy ? quant_policy : dbits;

    BlockConfig block_config{
        (int)model_param.head_dim,
        (int)model_param.kv_head_num / attn_tp_size,
        cache_block_seq_len,
        elem_bits == dbits ? 0 : dbits,
        elem_bits,
        model_param.head_dim == 576,  // share kv
    };

    block::Layout layout{block_config};
    // dump(layout);

    size_t block_size           = layout.block_size(cache_layer_num);
    size_t effective_block_size = block_size + (enable_prefix_caching ? linear_snapshot_bytes_per_block_ : 0);

    if (num_linear_layers_ > 0 && block_count < 1.) {
        const size_t target_bytes = static_cast<size_t>(free_before * block_count);
        TM_LOG_INFO("[SeqMgr] Adjusting block_count: free_before %.2f MB, linear %.2f MB, target %.2f MB",
                    free_before / (1024. * 1024.),
                    linear_active_pool_bytes_ / (1024. * 1024.),
                    target_bytes / (1024. * 1024.));
        if (target_bytes <= linear_active_pool_bytes_) {
            TM_LOG_ERROR("[SeqMgr] Linear-state memory (%.2f MB) >= cache budget (%.2f MB). ",
                         linear_active_pool_bytes_ / (1024. * 1024.),
                         target_bytes / (1024. * 1024.));
            TM_CHECK(0)
                << "Please decrease max_batch_size to reduce total linear state size or increase cache_max_entry_count.";
        }
        const size_t cache_bytes          = target_bytes - linear_active_pool_bytes_;
        const size_t reusable_block_count = cache_bytes / std::max<size_t>(effective_block_size, 1);
        TM_CHECK_GT(reusable_block_count, 0) << "Insufficient GPU memory for reusable linear-attention cache blocks";
        block_count = static_cast<double>(reusable_block_count);
        TM_LOG_INFO("[SeqMgr] Adjusted reusable block_count to %.0f (effective block %.3f MB)",
                    block_count,
                    (double)effective_block_size / (1024.0 * 1024.0));
    }
    else if (num_linear_layers_ > 0 && enable_prefix_caching && block_count >= 1.) {
        TM_CHECK_GT(free_before, linear_active_pool_bytes_)
            << "Insufficient GPU memory for linear state pool before prefix-cache allocation";
        const size_t free_after_active = free_before - linear_active_pool_bytes_;
        const size_t max_blocks_by_mem = free_after_active / std::max<size_t>(effective_block_size, 1);
        const size_t requested_blocks  = static_cast<size_t>(block_count);
        const size_t clamped_blocks    = std::min(requested_blocks, max_blocks_by_mem);
        block_count                    = static_cast<double>(clamped_blocks);
        TM_CHECK_GT(block_count, 0) << "Insufficient GPU memory for reusable linear-attention prefix cache blocks";
        if (clamped_blocks < requested_blocks) {
            TM_LOG_WARNING("[SeqMgr] Reducing reusable block_count from %zu to %zu for linear prefix snapshots",
                           requested_blocks,
                           clamped_blocks);
        }
    }

    block_manager_ = std::make_shared<BlockManager>(block_size, block_count, chunk_size, allocator, get_free_size);

    if (enable_prefix_caching) {
        BlockTrie::SnapshotPublisher snapshot_publisher;
        BlockTrie::SnapshotValidator snapshot_validator;
        BlockTrie::SnapshotReleaser  snapshot_releaser;
        if (num_linear_layers_ > 0) {
            const int snapshot_slots      = block_manager_->max_block_count();
            pooled_prefix_conv_snapshots_ = {
                {snapshot_slots, num_linear_layers_, d_conv_, conv_dim_}, linear_conv_dtype_, kDEVICE};
            pooled_prefix_recurrent_snapshots_ = {
                {snapshot_slots, num_linear_layers_, num_v_heads_, key_head_dim_, value_head_dim_},
                linear_state_dtype_,
                kDEVICE};
            free_linear_snapshot_slots_.reserve(snapshot_slots);
            linear_snapshot_unique_ids_.assign(snapshot_slots, 0);
            for (int slot = snapshot_slots - 1; slot >= 0; --slot) {
                free_linear_snapshot_slots_.push_back(slot);
            }
            TM_LOG_INFO("[SeqMgr] linear prefix snapshot pool initialized: %d slots (%.2f MB total)",
                        snapshot_slots,
                        (pooled_prefix_conv_snapshots_.byte_size() + pooled_prefix_recurrent_snapshots_.byte_size())
                            / (1024.0 * 1024.0));
            snapshot_publisher = [this](const Sequence& seq, int block_idx, int slot_hint) {
                return PublishLinearSnapshot(seq, block_idx, slot_hint);
            };
            snapshot_validator = [this](int slot, uint64_t unique_id) {
                return IsLinearSnapshotValid(slot, unique_id);
            };
            snapshot_releaser = [this](int slot, uint64_t unique_id) { ReleaseLinearSnapshot(slot, unique_id); };
        }
        block_trie_ = std::make_shared<BlockTrie>(
            block_config.block_len_, block_manager_, snapshot_publisher, snapshot_validator, snapshot_releaser);
    }
    TM_LOG_WARNING("[SegMgr] prefix caching is %s", enable_prefix_caching ? "enabled" : "disabled");
    if (enable_prefix_caching && num_linear_layers_ > 0) {
        TM_LOG_INFO(
            "[SeqMgr] linear prefix caching enabled: active pool %.2f MB, snapshot/block %.2f MB, max blocks %d",
            linear_active_pool_bytes_ / (1024.0 * 1024.0),
            linear_snapshot_bytes_per_block_ / (1024.0 * 1024.0),
            block_manager_->max_block_count());
    }
}

const Sequence* SequenceManager::Create(uint64_t id)
{
    Sequence sequence{id};
    auto     it = sequences_.find(id);
    if (it != sequences_.end()) {
        if (rank_ == 0) {
            TM_LOG_WARNING("[SeqMgr][Create] Removing conflicting ID %llu", id);
        }
        Erase(it);
    }
    it = sequences_.emplace_hint(it, id, std::move(sequence));
    if (rank_ == 0) {
        TM_LOG_DEBUG("[SeqMgr][Create] ID %llu", id);
    }
    return &it->second;
}

const Sequence* SequenceManager::Get(uint64_t id)
{
    if (auto it = sequences_.find(id); it != sequences_.end()) {
        return &it->second;
    }
    return nullptr;
}

bool SequenceManager::Contains(uint64_t id)
{
    return sequences_.find(id) != sequences_.end();
}

void SequenceManager::Erase(std::map<uint64_t, Sequence>::iterator& it)
{
    auto& seq = it->second;
    if (seq.status == Sequence::kCached) {
        const int count = block_manager_->Verify(seq.blocks, seq.block_unique_ids);
        seq.blocks.resize(count);
    }
    else {
        UpdateAndSetUnlock(seq);
    }
    // if prefix cache enabled, blocks will be shared by sequences, cannot be freed immediately
    if (!block_trie_) {
        freed_.insert(freed_.end(), seq.blocks.begin(), seq.blocks.end());
    }
    ClearLinearSnapshotStaging(seq);
    ReleaseLinearStateSlot(seq);
    it = sequences_.erase(it);
}

bool SequenceManager::Erase(uint64_t id)
{
    if (auto it = sequences_.find(id); it != sequences_.end()) {
        Erase(it);
        return true;
    }
    return false;
}

void SequenceManager::PrepareLinearCheckpointStaging(RequestCache& cache)
{
    auto& seq = *TM_CHECK_NOTNULL(cache.seq);

    // If linear prefix staging is inactive, clear bookkeeping so GatedDeltaNetLayer never
    // combines a stale staged_linear_block_count with missing or empty snapshot tensors.
    if (!block_trie_ || num_linear_layers_ == 0) {
        auto& s                     = const_cast<Sequence&>(seq);
        s.staged_linear_block_begin = 0;
        s.staged_linear_block_count = 0;
        return;
    }

    const int effective_history = cache.history_len + cache.alpha;
    const int first_block       = effective_history / block_seq_len_;
    const int last_block        = (effective_history + cache.input_len) / block_seq_len_;
    const int block_count       = std::max(0, last_block - first_block);

    auto& s = const_cast<Sequence&>(seq);
    // Previous compact staging window [prev_begin, prev_begin + prev_count) — capture before mutating.
    const int prev_begin = s.staged_linear_block_begin;
    const int prev_count = s.staged_conv_snapshots ? static_cast<int>(s.staged_conv_snapshots.shape(0)) : 0;

    s.staged_linear_block_begin = first_block;
    s.staged_linear_block_count = block_count;

    if (block_count == 0) {
        return;
    }

    if ((int)s.staged_linear_block_valid.size() < last_block) {
        s.staged_linear_block_valid.resize(last_block, 0);
    }
    std::fill(s.staged_linear_block_valid.begin() + first_block, s.staged_linear_block_valid.begin() + last_block, 0);

    const std::vector<ssize_t> conv_shape{block_count, num_linear_layers_, d_conv_, conv_dim_};
    const std::vector<ssize_t> recurrent_shape{
        block_count, num_linear_layers_, num_v_heads_, key_head_dim_, value_head_dim_};

    const bool need_conv_resize =
        !s.staged_conv_snapshots || first_block != prev_begin || s.staged_conv_snapshots.shape(0) != block_count
        || s.staged_conv_snapshots.shape(1) != num_linear_layers_ || s.staged_conv_snapshots.shape(2) != d_conv_
        || s.staged_conv_snapshots.shape(3) != conv_dim_;
    if (need_conv_resize) {
        Tensor new_conv{conv_shape, linear_conv_dtype_, kDEVICE};
        if (s.staged_conv_snapshots && prev_count > 0) {
            const int overlap_lo = std::max(first_block, prev_begin);
            const int overlap_hi = std::min(last_block, prev_begin + prev_count);
            for (int b = overlap_lo; b < overlap_hi; ++b) {
                const int old_row = b - prev_begin;
                const int new_row = b - first_block;
                Copy(s.staged_conv_snapshots.slice(old_row, 1).squeeze(0), new_conv.slice(new_row, 1).squeeze(0));
            }
        }
        s.staged_conv_snapshots = std::move(new_conv);
    }
    const bool need_recurrent_resize = !s.staged_recurrent_snapshots || first_block != prev_begin
                                       || s.staged_recurrent_snapshots.shape(0) != block_count
                                       || s.staged_recurrent_snapshots.shape(1) != num_linear_layers_
                                       || s.staged_recurrent_snapshots.shape(2) != num_v_heads_
                                       || s.staged_recurrent_snapshots.shape(3) != key_head_dim_
                                       || s.staged_recurrent_snapshots.shape(4) != value_head_dim_;
    if (need_recurrent_resize) {
        Tensor new_recurrent{recurrent_shape, linear_state_dtype_, kDEVICE};
        if (s.staged_recurrent_snapshots && prev_count > 0) {
            const int overlap_lo = std::max(first_block, prev_begin);
            const int overlap_hi = std::min(last_block, prev_begin + prev_count);
            for (int b = overlap_lo; b < overlap_hi; ++b) {
                const int old_row = b - prev_begin;
                const int new_row = b - first_block;
                Copy(s.staged_recurrent_snapshots.slice(old_row, 1).squeeze(0),
                     new_recurrent.slice(new_row, 1).squeeze(0));
            }
        }
        s.staged_recurrent_snapshots = std::move(new_recurrent);
    }
}

void SequenceManager::AcquireLinearStateSlot(const Sequence& sequence)
{
    if (!pooled_recurrent_states_) {
        return;
    }

    auto& seq = const_cast<Sequence&>(sequence);

    auto slot_it = seq_to_linear_state_slot_.find(seq.id);
    if (slot_it != seq_to_linear_state_slot_.end()) {
        const int slot       = slot_it->second;
        seq.conv_states      = pooled_conv_states_.slice(slot).squeeze(0);
        seq.recurrent_states = pooled_recurrent_states_.slice(slot).squeeze(0);
        return;
    }

    TM_CHECK(!free_linear_state_slots_.empty()) << "No free linear-state slot for sequence " << seq.id
                                                << ", max_batch_size=" << pooled_recurrent_states_.shape(0);

    const int slot = free_linear_state_slots_.back();
    free_linear_state_slots_.pop_back();
    seq_to_linear_state_slot_.emplace(seq.id, slot);

    seq.conv_states              = pooled_conv_states_.slice(slot).squeeze(0);
    seq.recurrent_states         = pooled_recurrent_states_.slice(slot).squeeze(0);
    seq.linear_states_need_reset = true;
}

void SequenceManager::ReleaseLinearStateSlot(const Sequence& sequence)
{
    if (!pooled_recurrent_states_) {
        return;
    }

    auto& seq = const_cast<Sequence&>(sequence);

    if (auto slot_it = seq_to_linear_state_slot_.find(seq.id); slot_it != seq_to_linear_state_slot_.end()) {
        free_linear_state_slots_.push_back(slot_it->second);
        seq_to_linear_state_slot_.erase(slot_it);
    }
    seq.conv_states                       = {};
    seq.recurrent_states                  = {};
    seq.linear_states_need_reset          = false;
    seq.linear_restore_snapshot_slot      = -1;
    seq.linear_restore_snapshot_unique_id = 0;
}

void SequenceManager::ClearLinearSnapshotStaging(const Sequence& sequence)
{
    auto& seq                     = const_cast<Sequence&>(sequence);
    seq.staged_linear_block_begin = 0;
    seq.staged_linear_block_count = 0;
}

bool SequenceManager::IsLinearSnapshotValid(int slot, uint64_t unique_id) const
{
    if (slot < 0 || slot >= (int)linear_snapshot_unique_ids_.size() || unique_id == 0) {
        return false;
    }
    return linear_snapshot_unique_ids_[slot] == unique_id;
}

std::pair<int, uint64_t> SequenceManager::PublishLinearSnapshot(const Sequence& sequence, int block_idx, int slot_hint)
{
    if (num_linear_layers_ == 0) {
        return {slot_hint, 0};
    }

    const auto& seq       = sequence;
    const bool  staged_ok = block_idx >= 0 && block_idx < (int)seq.staged_linear_block_valid.size()
                           && seq.staged_linear_block_valid[block_idx];

    if (!staged_ok) {
        if (rank_ == 0) {
            TM_LOG_WARNING("[SeqMgr][publish] ID %llu missing staged snapshot for block %d, staged range [%d, %d)",
                           seq.id,
                           block_idx,
                           seq.staged_linear_block_begin,
                           seq.staged_linear_block_begin + seq.staged_linear_block_count);
        }
        // Never associate a live unique_id with a stale GPU slot: release the trie's previous slot so KV + linear
        // metadata cannot diverge.
        if (slot_hint >= 0 && slot_hint < (int)linear_snapshot_unique_ids_.size()) {
            const uint64_t old_uid = linear_snapshot_unique_ids_[slot_hint];
            if (old_uid != 0) {
                ReleaseLinearSnapshot(slot_hint, old_uid);
            }
        }
        g_linear_snapshot_publish_miss.fetch_add(1, std::memory_order_relaxed);
        return {-1, 0};
    }

    int slot = slot_hint;
    if (slot < 0) {
        if (free_linear_snapshot_slots_.empty()) {
            if (rank_ == 0) {
                TM_LOG_WARNING("[SeqMgr][publish] ID %llu no free linear prefix snapshot slot for block %d "
                               "(pool exhausted); trie will omit linear snapshot for this block",
                               seq.id,
                               block_idx);
            }
            g_linear_snapshot_publish_pool_exhausted.fetch_add(1, std::memory_order_relaxed);
            return {-1, 0};
        }
        slot = free_linear_snapshot_slots_.back();
        free_linear_snapshot_slots_.pop_back();
    }

    const int stage_row = block_idx - seq.staged_linear_block_begin;
    TM_CHECK_GE(stage_row, 0);
    TM_CHECK_LT(stage_row, seq.staged_linear_block_count);

    Copy(seq.staged_conv_snapshots.slice(stage_row, 1).squeeze(0),
         pooled_prefix_conv_snapshots_.slice(slot, 1).squeeze(0));
    Copy(seq.staged_recurrent_snapshots.slice(stage_row, 1).squeeze(0),
         pooled_prefix_recurrent_snapshots_.slice(slot, 1).squeeze(0));

    const uint64_t unique_id          = next_linear_snapshot_unique_id_++;
    linear_snapshot_unique_ids_[slot] = unique_id;
    g_linear_snapshot_publish_ok.fetch_add(1, std::memory_order_relaxed);
    return {slot, unique_id};
}

void SequenceManager::ReleaseLinearSnapshot(int slot, uint64_t unique_id)
{
    if (!IsLinearSnapshotValid(slot, unique_id)) {
        return;
    }
    linear_snapshot_unique_ids_[slot] = 0;
    free_linear_snapshot_slots_.push_back(slot);
}

bool SequenceManager::RestoreLinearSnapshot(const Sequence& sequence, int slot, uint64_t unique_id)
{
    if (!IsLinearSnapshotValid(slot, unique_id)) {
        if (rank_ == 0) {
            TM_LOG_WARNING("[SeqMgr][restore] stale or invalid linear snapshot (slot=%d uid=%llu); skipping restore",
                           slot,
                           (unsigned long long)unique_id);
        }
        return false;
    }

    auto& seq = const_cast<Sequence&>(sequence);
    if (!seq.conv_states || !seq.recurrent_states) {
        if (rank_ == 0) {
            TM_LOG_WARNING("[SeqMgr][restore] ID %llu missing linear state tensors; skipping restore", seq.id);
        }
        return false;
    }

    Copy(pooled_prefix_conv_snapshots_.slice(slot, 1).squeeze(0), seq.conv_states);
    Copy(pooled_prefix_recurrent_snapshots_.slice(slot, 1).squeeze(0), seq.recurrent_states);
    seq.linear_states_need_reset          = false;
    seq.linear_restore_snapshot_slot      = slot;
    seq.linear_restore_snapshot_unique_id = unique_id;
    g_prefix_match_restored_linear.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void SequenceManager::InvalidateStatesAndCache(const Sequence& sequence)
{
    InvalidateStatesAndCache(sequence, freed_);
}

void SequenceManager::InvalidateStatesAndCache(const Sequence& sequence, BlockIds& freed_blocks)
{
    auto& seq = const_cast<Sequence&>(sequence);
    if (seq.status != Sequence::kCached) {
        UpdateAndSetUnlock(seq);
    }
    freed_blocks.insert(freed_blocks.end(), seq.blocks.begin(), seq.blocks.end());

    seq.blocks.clear();
    seq.block_unique_ids.clear();
    seq.input_length = 0;
    seq.cache_len    = 0;
    ClearLinearSnapshotStaging(seq);
    ReleaseLinearStateSlot(seq);
}

void SequenceManager::CachePrompt(const Sequences& sequences, int active_size)
{
    if (!block_trie_) {
        return;
    }

    for (int i = 0; i < active_size; ++i) {
        if (auto& seq = *sequences[i]; !seq.prompt.empty()) {
            const auto  result     = block_trie_->Cache(seq, seq.prompt);
            const auto& block_ids  = result.block_ids;
            const auto& unique_ids = result.unique_ids;
            if (rank_ == 0) {
                // clang-format off
                TM_LOG_DEBUG("[SeqMgr][CachePrompt] ID %llu, cached blocks %d, tokens %d", seq.id,
                             (int)block_ids.size(), (int)seq.prompt.size());
                TM_LOG_DEBUG("[SeqMgr][CachePrompt] ID %llu, cached block_ids %s, unique_ids %s", seq.id,
                             vector2string(block_ids).c_str(), vector2string(unique_ids).c_str());
                // clang-format on
            }
            if (seq.cache_len >= seq.prompt.size()) {
                seq.prompt.clear();
            }
            ClearLinearSnapshotStaging(seq);
        }
    }
}

void SequenceManager::CacheGeneration(const Sequence& seq)
{
    if (!block_trie_) {
        return;
    }

    const auto  result     = block_trie_->Cache(seq, seq.tokens);
    const auto& block_ids  = result.block_ids;
    const auto& unique_ids = result.unique_ids;

    if (rank_ == 0) {
        // clang-format off
        TM_LOG_DEBUG("[SeqMgr][CacheGeneration] ID %llu, cached blocks %d, tokens %d",
                     seq.id, (int)block_ids.size(), (int)seq.tokens.size());
        TM_LOG_DEBUG("[SeqMgr][CacheGeneration] ID %llu, cached block_ids %s, unique_ids %s", seq.id,
                     vector2string(block_ids).c_str(), vector2string(unique_ids).c_str());
        // clang-format on
    }
    ClearLinearSnapshotStaging(seq);
}

void SequenceManager::VerifyAndLockCached(const Sequences& sequences)
{
    BlockIds valid_blocks;
    BlockIds freed_blocks;
    for (const auto& p : sequences) {
        auto& seq = const_cast<Sequence&>(*p);
        if (seq.status != Sequence::kCached) {
            continue;
        }
        TM_CHECK_EQ(seq.blocks.size(), seq.block_unique_ids.size());
        // Verify cache blocks that may be invalidated
        const int original_count = seq.blocks.size();
        const int count          = block_manager_->Verify(seq.blocks, seq.block_unique_ids);
        seq.blocks.resize(count);
        seq.block_unique_ids.resize(count);

        const bool has_linear_states = static_cast<bool>(seq.recurrent_states);
        if (has_linear_states && count < original_count) {
            InvalidateStatesAndCache(seq, freed_blocks);
            // This request can still continue in the current scheduling round.
            // Rebind a slot immediately so GatedDeltaNetLayer::Setup always sees
            // valid linear-state views.
            AcquireLinearStateSlot(seq);
            continue;
        }

        valid_blocks.insert(valid_blocks.end(), seq.blocks.begin(), seq.blocks.end());
        seq.cache_len = std::min<int>(seq.cache_len, seq.blocks.size() * block_seq_len_);
        seq.status    = Sequence::kLocked;
    }
    if (!freed_blocks.empty()) {
        block_manager_->Free(freed_blocks);
    }
    block_manager_->Lock(valid_blocks);
}

void SequenceManager::CommitUnlockAndFree()
{
    if (!unlocked_.empty()) {
        block_manager_->Unlock(unlocked_);
        unlocked_.clear();
    }

    if (!freed_.empty()) {
        block_manager_->Free(freed_);
        freed_.clear();
    }
}

void SequenceManager::UpdateAndSetUnlock(const Sequence& sequence)
{
    TM_CHECK_NE(sequence.status, Sequence::kCached);
    auto& seq = const_cast<Sequence&>(sequence);
    block_manager_->Touch(seq.blocks);
    unlocked_.insert(unlocked_.end(), seq.blocks.begin(), seq.blocks.end());
    seq.status = Sequence::kCached;
}

namespace {

struct Schedule {
    int free;
    int cached;

    int allocate{};
    int evict{};
    int preempt{};

    int last;

    int max_fwd_tokens;
    int max_tmp_tokens;

    Sequences        active;
    std::vector<int> block_counts;
    Sequences        inactive;
    Sequences        victims;

    Schedule(Snapshot snapshot, int size, int max_fwd_tokens, int max_tmp_tokens):
        free{snapshot.free},
        cached{snapshot.cached},
        last{size},
        max_fwd_tokens{max_fwd_tokens},
        max_tmp_tokens{max_tmp_tokens},
        use_count_{std::move(snapshot.use_count)},
        unlocked_(size),  // ! This is a vector, DO NOT brace initialize it
        it_{size}
    {
    }

    int Unlock(const Sequences& seqs, int vidx)
    {
        while (vidx < it_) {
            const auto& blocks = seqs[--it_]->blocks;
            int         count  = 0;
            for (const auto& bid : blocks) {
                count += static_cast<int>(--use_count_[bid] == 0);
            }
            unlocked_[it_] = count;
        }
        return unlocked_[vidx];
    }

private:
    std::vector<int> use_count_;
    std::vector<int> unlocked_;
    int              it_;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << (i ? "," : "") << v[i];
    }
    os << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Schedule& s)
{
    os << "free=" << s.free << ", cached=" << s.cached << ", allocate=" << s.allocate << ", evict=" << s.evict
       << ", preempt=" << s.preempt << ", active=" << s.active << ", victims=" << s.victims
       << ", block_counts=" << s.block_counts << ", inactive=" << s.inactive;
    return os;
}

struct Transaction {
    int index_;
    int block_count_;
    int input_len_;
    int temp_len_;

    int allocate_{};
    int evict_{};
    int preempt_{};

    Sequences victims_;

    const Sequences& sequences_;
    Schedule&        schedule_;

    explicit Transaction(
        const Sequences& sequences, int index, int block_count, int input_len, int temp_len, Schedule& sched):
        index_{index},
        block_count_{block_count},
        input_len_{input_len},
        temp_len_{temp_len},
        sequences_{sequences},
        schedule_{sched}
    {
    }

    void Process()
    {
        if (schedule_.max_fwd_tokens > 0 && schedule_.max_tmp_tokens >= temp_len_) {
            int count = block_count_;

            int tmp = std::min(schedule_.free, count);
            count -= tmp;
            allocate_ += tmp;

            tmp = std::min(schedule_.cached, count);
            count -= tmp;
            evict_ += tmp;

            for (int vidx = schedule_.last - 1; count && vidx > index_; --vidx) {
                if (sequences_[vidx]->status == Sequence::kCached) {
                    continue;
                }
                victims_.push_back(sequences_[vidx]);
                preempt_ += schedule_.Unlock(sequences_, vidx);

                if (count <= preempt_) {
                    evict_ += count;
                    count -= count;
                    schedule_.last = vidx;  // ! modifiying `sched_.last` is part of commit
                    break;
                }
            }
            if (count == 0) {
                return Commit();
            }
        }

        const_cast<Sequence*>(sequences_[index_])->input_length = 0;
        schedule_.inactive.push_back(sequences_[index_]);
    }

    void Commit()
    {
        // update available resources
        schedule_.free -= allocate_;
        TM_CHECK_GE(schedule_.free, 0);
        schedule_.cached += preempt_;
        schedule_.cached -= evict_;
        TM_CHECK_GE(schedule_.cached, 0);

        // update scheduled operations
        schedule_.allocate += allocate_;
        schedule_.evict += evict_;
        schedule_.preempt += preempt_;
        schedule_.victims.insert(schedule_.victims.end(), victims_.begin(), victims_.end());

        // update active sequences
        schedule_.active.push_back(sequences_[index_]);
        schedule_.block_counts.push_back(block_count_);

        input_len_ = std::min(input_len_, schedule_.max_fwd_tokens);
        schedule_.max_fwd_tokens -= input_len_;
        const_cast<Sequence*>(sequences_[index_])->input_length = input_len_;

        schedule_.max_tmp_tokens -= temp_len_;
    }
};

std::ostream& operator<<(std::ostream& os, const Transaction& trans)
{
    os << "index=" << trans.index_ << ", block_count=" << trans.block_count_ << ", allocate=" << trans.allocate_
       << ", evict=" << trans.evict_ << ", preempt=" << trans.preempt_ << ", victims=" << trans.victims_;
    return os;
}

}  // namespace

template<class Key, class... Ts>
static void SortByKey(const std::vector<Key>& keys, std::vector<Ts>&... vals)
{
    std::vector<int> idxs(keys.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) { return keys[i] < keys[j]; });
    auto reorder = [&](auto& xs) {
        std::remove_reference_t<decltype(xs)> ys(xs.size());
        for (size_t i = 0; i < xs.size(); ++i) {
            ys[i] = xs[idxs[i]];
        }
        xs.swap(ys);
    };
    (reorder(vals), ...);
}

std::vector<int> SequenceManager::CountRequiredBlocks(const Sequences&        sequences,
                                                      const std::vector<int>& context_length)
{
    std::vector<int> required(sequences.size());
    for (int i = 0; i < sequences.size(); ++i) {
        int length  = (context_length[i] + attn_cp_size_ - 1) / attn_cp_size_;
        int count   = (length + block_seq_len_ - 1) / block_seq_len_ - static_cast<int>(sequences[i]->blocks.size());
        required[i] = std::max(0, count);
    }
    return required;
}

void SequenceManager::AssignAndActivate(const Sequences&        sequences,  //
                                        const std::vector<int>& counts,
                                        const BlockIds&         blocks,
                                        const UniqueIds&        unique_ids)
{
    TM_CHECK_EQ(sequences.size(), counts.size());
    int first = 0;
    for (int i = 0; i < sequences.size(); ++i) {
        auto& s     = const_cast<Sequence&>(*sequences[i]);
        auto  count = counts[i];
        int   last  = first + count;
        TM_CHECK_LE(last, blocks.size());
        s.blocks.insert(s.blocks.end(), blocks.begin() + first, blocks.begin() + last);
        s.block_unique_ids.insert(s.block_unique_ids.end(), unique_ids.begin() + first, unique_ids.begin() + last);
        s.status = Sequence::kActive;
        first    = last;
    }
}

void SequenceManager::PrefixMatch(Sequences& sequences, const std::vector<int>& alpha)
{
    if (!block_trie_) {
        return;
    }

    for (int i = 0; i < sequences.size(); i++) {

        auto& seq = const_cast<Sequence&>(*sequences[i]);

        // alpha counts tokens not yet committed into cache_len (e.g. draft / overlap). Prefix matching assumes the
        // trie-aligned prompt prefix matches the committed KV + linear state; skip until alpha==0 so we do not bind
        // cached blocks to a sequence that still has uncommitted prefix tokens.
        if (alpha[i] != 0) {
            g_prefix_match_skipped_alpha.fetch_add(1, std::memory_order_relaxed);
            if (rank_ == 0) {
                TM_LOG_DEBUG(
                    "[SeqMgr][match] ID %llu skip prefix match: alpha=%d (linear prefix cache requires alpha=0)",
                    seq.id,
                    alpha[i]);
            }
            continue;
        }
        if (seq.cache_len >= seq.prompt.size()) {
            continue;
        }

        const auto  match      = block_trie_->Match(seq);
        const auto& block_ids  = match.block_ids;
        const auto& unique_ids = match.unique_ids;

        if (rank_ == 0) {
            // clang-format off
            TM_LOG_DEBUG("[SeqMgr][match] ID %llu, hit blocks %d, cache_len %d, linear_snapshot=%s",
                         seq.id,
                         (int)block_ids.size(),
                         seq.cache_len,
                         match.snapshot_slot >= 0 ? "yes" : "no");
            TM_LOG_DEBUG("[SeqMgr][match] ID %llu, hit block_ids %s, unique_ids %s", seq.id,
                         vector2string(block_ids).c_str(), vector2string(unique_ids).c_str());
            // clang-format on
        }

        /// TODO: `Unlock` and `Lock` can't be batched because there may be repeated blocks between sequences
        if (const int offset = seq.cache_len / block_seq_len_; offset < block_ids.size()) {
            if (BlockIds tail{seq.blocks.begin() + offset, seq.blocks.end()}; !tail.empty()) {
                block_manager_->Unlock(tail);
                seq.blocks.resize(offset);
                seq.block_unique_ids.resize(offset);
            }
            seq.blocks.insert(seq.blocks.end(), block_ids.begin() + offset, block_ids.end());
            seq.block_unique_ids.insert(seq.block_unique_ids.end(), unique_ids.begin() + offset, unique_ids.end());
            seq.cache_len = seq.blocks.size() * block_seq_len_;
            block_manager_->Lock({block_ids.begin() + offset, block_ids.end()});
        }
        if (match.snapshot_slot >= 0 && seq.recurrent_states) {
            if (RestoreLinearSnapshot(seq, match.snapshot_slot, match.snapshot_unique_id)) {
                if (rank_ == 0) {
                    TM_LOG_DEBUG(
                        "[SeqMgr][match] ID %llu restored linear snapshot from slot %d", seq.id, match.snapshot_slot);
                }
            }
            else {
                seq.linear_restore_snapshot_slot      = -1;
                seq.linear_restore_snapshot_unique_id = 0;
            }
        }
        else if (seq.recurrent_states) {
            seq.linear_restore_snapshot_slot      = -1;
            seq.linear_restore_snapshot_unique_id = 0;
        }

        if (rank_ == 0) {
            // clang-format off
            TM_LOG_DEBUG("[SeqMgr][match] ID %llu, after matching, blocks %d, cache_len %d",
                         seq.id, seq.blocks.size(), seq.cache_len);
            TM_LOG_DEBUG("[SeqMgr][match] ID %llu, after matching, block_ids %s, unique_ids %s", seq.id,
                         vector2string(seq.blocks).c_str(), vector2string(seq.block_unique_ids).c_str());
            // clang-format on
        }
    }
}

auto SequenceManager::Materialize(Sequences             sequences,
                                  std::vector<int>      context_length,
                                  std::vector<int>      alpha,
                                  std::vector<uint64_t> priorities,
                                  int                   max_fwd_tokens,
                                  int                   max_tmp_tokens) -> Outcome
{
    ////////////////////////////////////////////////////////////////////////////////
    /// Schedule the assignment of blocks to sequences

    // process deferred unlock and free operations
    CommitUnlockAndFree();

    SortByKey(priorities, sequences, context_length, alpha);

    // Verify and lock cache sequences to avoid their blocks being evicted unnoticed
    // the blocks can still be preempted later
    VerifyAndLockCached(sequences);

    PrefixMatch(sequences, alpha);

    std::vector required = CountRequiredBlocks(sequences, context_length);

    Schedule schedule(block_manager_->TakeSnapshot(), sequences.size(), max_fwd_tokens, max_tmp_tokens);

    // `schedule.last` is decreasing in the loop
    for (int i = 0; i < schedule.last; ++i) {
        auto&     s         = *sequences[i];
        const int input_len = context_length[i] - alpha[i] - s.cache_len;
        // sanity check
        TM_CHECK_GT(input_len, 0) << "Logical error: " << context_length[i] << " " << alpha[i] << " " << s.cache_len
                                  << " " << s.status;
        // temp buffer for flatten KV cache
        const int temp_len = (input_len > 1 || s.status != Sequence::kActive) ? context_length[i] : 0;
        Transaction{sequences, i, required[i], input_len, temp_len, schedule}.Process();
    }

    // mark remaining sequences invalid
    for (int i = schedule.last; i < sequences.size(); ++i) {
        schedule.inactive.push_back(sequences[i]);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Schedule is ready, time to execute it. (locked -> cached -> free -> locked)

    // combine allocate and evict since evicted blocks are reused by allocation
    schedule.allocate += schedule.evict;

    // if (schedule.allocate) {
    //     dbg(*block_manager_);
    // }

    Outcome outcome{};
    outcome.allocation = schedule.allocate;
    outcome.swap_in    = std::count_if(schedule.active.begin(), schedule.active.end(), [](auto p) {
        // if (p->status != Sequence::kActive) {
        //     dbg(*p);
        // }
        return p->status != Sequence::kActive;
    });
    outcome.swap_out = std::count_if(schedule.inactive.begin(), schedule.inactive.end(), [](auto p) {
        // if (p->status == Sequence::kActive) {
        //     dbg(*p);
        // }
        return p->status == Sequence::kActive;
    });

    // release preempted blocks -> cached
    if (!schedule.victims.empty()) {
        TM_LOG_INFO("[SeqMgr] #victim: %d", (int)schedule.victims.size());
        for (const auto& p : schedule.victims) {
            UpdateAndSetUnlock(*p);
        }
        CommitUnlockAndFree();
    }

    // evict cached blocks -> free
    if (schedule.evict) {
        block_manager_->Evict(schedule.evict);
    }

    // allocate & assign blocks
    {
        BlockIds  block_ids;
        UniqueIds unique_ids;
        if (schedule.allocate) {
            std::tie(block_ids, unique_ids) = block_manager_->Allocate(schedule.allocate);
        }
        AssignAndActivate(schedule.active, schedule.block_counts, block_ids, unique_ids);
    }

    // active -> locked
    for (const auto& p : schedule.inactive) {
        if (p->status == Sequence::kActive) {
            const_cast<Sequence*>(p)->status = Sequence::kLocked;
        }
    }

    // TM_LOG_ERROR("active: %4d, cached: %4d, free: %4d",
    //              block_manager_->active_count(),
    //              block_manager_->cached_count(),
    //              block_manager_->free_count());
    if (block_trie_) {
        block_trie_->Verify();
    }

    return outcome;
}

std::tuple<int, int, int> SequenceManager::seq_stats() const noexcept
{
    int total  = static_cast<int>(sequences_.size());
    int active = 0;
    int cached = 0;
    for (const auto& p : sequences_) {
        if (p.second.status == Sequence::kActive) {
            ++active;
        }
        else if (p.second.status == Sequence::kCached) {
            ++cached;
        }
    }
    return std::make_tuple(total, active, cached);
}

std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> SequenceManager::LinearPrefixCacheStats() noexcept
{
    return {g_linear_snapshot_publish_ok.load(std::memory_order_relaxed),
            g_linear_snapshot_publish_miss.load(std::memory_order_relaxed),
            g_linear_snapshot_publish_pool_exhausted.load(std::memory_order_relaxed),
            g_prefix_match_skipped_alpha.load(std::memory_order_relaxed),
            g_prefix_match_restored_linear.load(std::memory_order_relaxed)};
}

}  // namespace turbomind
