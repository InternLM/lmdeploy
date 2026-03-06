// Copyright (c) OpenMMLab. All rights reserved.

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <numeric>

#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/core/logger.h"

// #include "dbg.h"

namespace turbomind {

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

    int cache_layer_num   = model_param.layer_num;
    int num_linear_layers = 0;
    for (const auto& type : model_param.layer_types) {
        if (type == 1) {
            --cache_layer_num;
            ++num_linear_layers;
        }
    }

    const size_t free_before = (block_count < 1. && num_linear_layers > 0) ? get_free_size() : 0;

    if (num_linear_layers > 0) {

        const int key_head_dim =
            model_param.linear_key_head_dim > 0 ? model_param.linear_key_head_dim : model_param.head_dim;
        const int value_head_dim =
            model_param.linear_value_head_dim > 0 ? model_param.linear_value_head_dim : model_param.head_dim;
        const int d_conv      = model_param.linear_conv_kernel_dim > 0 ? model_param.linear_conv_kernel_dim : 4;
        const int num_k_heads = model_param.linear_num_key_heads / attn_tp_size;
        const int num_v_heads = model_param.linear_num_value_heads / attn_tp_size;
        const int key_dim     = num_k_heads * key_head_dim;
        const int value_dim   = num_v_heads * value_head_dim;
        const int conv_dim    = key_dim * 2 + value_dim;

        TM_CHECK_GT(max_batch_size, 0);
        pooled_conv_states_ = {{max_batch_size, num_linear_layers, d_conv, conv_dim}, model_param.data_type, kDEVICE};
        pooled_recurrent_states_ = {{max_batch_size, num_linear_layers, num_v_heads, key_head_dim, value_head_dim},
                                    model_param.linear_state_dtype,
                                    kDEVICE};

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
        TM_LOG_INFO("[SeqMgr] linear-state combined total: %.2f MB",
                    (pooled_conv_states_.byte_size() + pooled_recurrent_states_.byte_size()) * mb);
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

    size_t block_size = layout.block_size(cache_layer_num);

    if (num_linear_layers > 0 && block_count < 1.) {
        const size_t linear_bytes = pooled_conv_states_.byte_size() + pooled_recurrent_states_.byte_size();
        const size_t target_bytes = static_cast<size_t>(free_before * block_count);
        TM_LOG_INFO("[SeqMgr] Adjusting block_count: free_before %.2f MB, linear %.2f MB, target %.2f MB",
                    free_before / (1024. * 1024.),
                    linear_bytes / (1024. * 1024.),
                    target_bytes / (1024. * 1024.));
        if (target_bytes <= linear_bytes) {
            TM_LOG_ERROR("[SeqMgr] Linear-state memory (%.2f MB) >= cache budget (%.2f MB). ",
                         linear_bytes / (1024. * 1024.),
                         target_bytes / (1024. * 1024.));
            TM_CHECK(0)
                << "Please decrease max_batch_size to reduce total linear state size or increase cache_max_entry_count.";
        }
        const size_t cache_bytes = target_bytes - linear_bytes;
        block_count              = static_cast<double>(cache_bytes) / static_cast<double>(block_size);
        TM_LOG_INFO("[SeqMgr] Adjusted block_count to %.0f", block_count);
    }

    block_manager_ = std::make_shared<BlockManager>(block_size, block_count, chunk_size, allocator, get_free_size);

    if (enable_prefix_caching) {
        block_trie_ = std::make_shared<BlockTrie>(block_config.block_len_, block_manager_);
    }
    TM_LOG_WARN("[SegMgr] prefix caching is {}", enable_prefix_caching ? "enabled" : "disabled");
}

const Sequence* SequenceManager::Create(uint64_t id)
{
    Sequence sequence{id};
    auto     it = sequences_.find(id);
    if (it != sequences_.end()) {
        if (rank_ == 0) {
            TM_LOG_WARN("[SeqMgr][Create] Removing conflicting ID {}", id);
        }
        Erase(it);
    }
    it = sequences_.emplace_hint(it, id, std::move(sequence));
    if (rank_ == 0) {
        TM_LOG_INFO("[SeqMgr][Create] ID {}", id);
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
    seq.conv_states              = {};
    seq.recurrent_states         = {};
    seq.linear_states_need_reset = false;
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
    ReleaseLinearStateSlot(seq);
}

void SequenceManager::CachePrompt(const Sequences& sequences, int active_size)
{
    if (!block_trie_) {
        return;
    }

    for (int i = 0; i < active_size; ++i) {
        if (auto& seq = *sequences[i]; !seq.prompt.empty()) {
            const auto& [block_ids, unique_ids] = block_trie_->Cache(seq, seq.prompt);
            if (rank_ == 0) {
                // clang-format off
                TM_LOG_INFO("[SeqMgr][CachePrompt] ID {}, cached blocks {}, tokens {}", seq.id,
                            (int)block_ids.size(), (int)seq.prompt.size());
                TM_LOG_DEBUG("[SeqMgr][CachePrompt] ID {}, cached block_ids {}, unique_ids {}", seq.id,
                             vector2string(block_ids), vector2string(unique_ids));
                // clang-format on
            }
            if (seq.cache_len >= seq.prompt.size()) {
                seq.prompt.clear();
            }
        }
    }
}

void SequenceManager::CacheGeneration(const Sequence& seq)
{
    if (!block_trie_) {
        return;
    }

    const auto& [block_ids, unique_ids] = block_trie_->Cache(seq, seq.tokens);

    if (rank_ == 0) {
        // clang-format off
        TM_LOG_INFO("[SeqMgr][CacheGeneration] ID {}, cached blocks {}, tokens {}",
                    seq.id, (int)block_ids.size(), (int)seq.tokens.size());
        TM_LOG_DEBUG("[SeqMgr][CacheGeneration] ID {}, cached block_ids {}, unique_ids {}", seq.id,
                     vector2string(block_ids), vector2string(unique_ids));
        // clang-format on
    }
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

        /// TODO: Is there a way to exploit the alpha[i] != 0 case?
        if (alpha[i] != 0 || seq.cache_len >= seq.prompt.size()) {
            continue;
        }

        const auto& [block_ids, unique_ids] = block_trie_->Match(seq);

        if (rank_ == 0) {
            // clang-format off
            TM_LOG_INFO("[SeqMgr][match] ID {}, hit blocks {}, cache_len {}", seq.id, (int)block_ids.size(), seq.cache_len);
            TM_LOG_DEBUG("[SeqMgr][match] ID {}, hit block_ids {}, unique_ids {}", seq.id,
                         vector2string(block_ids), vector2string(unique_ids));
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

        if (rank_ == 0) {
            // clang-format off
            TM_LOG_INFO("[SeqMgr][match] ID {}, after matching, blocks {}, cache_len {}",
                        seq.id, seq.blocks.size(), seq.cache_len);
            TM_LOG_DEBUG("[SeqMgr][match] ID {}, after matching, block_ids {}, unique_ids {}", seq.id,
                         vector2string(seq.blocks), vector2string(seq.block_unique_ids));
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
        TM_LOG_WARN("[SeqMgr] #victim: {}", (int)schedule.victims.size());
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

}  // namespace turbomind
