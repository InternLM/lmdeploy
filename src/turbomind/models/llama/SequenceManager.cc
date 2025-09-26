// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/utils/debug_utils.h"
#include "src/turbomind/utils/logger.h"
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <numeric>
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

SequenceManager::SequenceManager(size_t             layer_num,
                                 const BlockConfig& block_config,
                                 double             block_count,
                                 int                chunk_size,
                                 bool               enable_prefix_caching,
                                 int                rank,
                                 core::Allocator    allocator,
                                 GetFreeMemSize     get_free_size):
    block_seq_len_(block_config.block_len_), rank_(rank)
{
    block::Layout layout{block_config};
    // dump(layout);

    size_t block_size = layout.block_size(layer_num);

    block_manager_ = std::make_shared<BlockManager>(block_size, block_count, chunk_size, allocator, get_free_size);
    if (enable_prefix_caching) {
        block_trie_ = std::make_shared<BlockTrie>(block_config.block_len_, block_manager_);
    }
    TM_LOG_WARNING("[SegMgr] prefix caching is %s", enable_prefix_caching ? "enabled" : "disabled");
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
        TM_LOG_INFO("[SeqMgr][Create] ID %llu", id);
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

void SequenceManager::CachePrompt(const Sequences& sequences, int active_size)
{
    if (!block_trie_) {
        return;
    }

    for (int i = 0; i < active_size; ++i) {
        auto& seq = *sequences[i];
        if (seq.cache_len > seq.prompt.size()) {
            // seq prefill finished. We don't cache the prompt any longer
            seq.prompt.clear();
            continue;
        }
        BlockIds  block_ids;
        UniqueIds block_unique_ids;
        std::tie(block_ids, block_unique_ids) = block_trie_->Cache(seq, seq.prompt);
        if (rank_ == 0) {
            TM_LOG_INFO("[SeqMgr][CachePrompt] ID %llu, cached blocks %d, tokens %d",
                        seq.id,
                        block_ids.size(),
                        seq.prompt.size());
            TM_LOG_DEBUG("[SeqMgr][CachePrompt] ID %llu, cached block_ids %s, unique_ids %s",
                         seq.id,
                         vector2string(block_ids).c_str(),
                         vector2string(block_unique_ids).c_str());
        }
    }
}

void SequenceManager::CacheGeneration(const Sequence& seq)
{
    if (!block_trie_) {
        return;
    }

    BlockIds  block_ids;
    UniqueIds block_unique_ids;

    std::tie(block_ids, block_unique_ids) = block_trie_->Cache(seq, seq.tokens);
    if (rank_ == 0) {
        TM_LOG_INFO("[SeqMgr][CacheGeneration] ID %llu, cached blocks %d, tokens %d",
                    seq.id,
                    block_ids.size(),
                    seq.tokens.size());
        TM_LOG_DEBUG("[SeqMgr][CacheGeneration] ID %llu, cached block_ids %s, unique_ids %s",
                     seq.id,
                     vector2string(block_ids).c_str(),
                     vector2string(block_unique_ids).c_str());
    }
}

void SequenceManager::VerifyAndLockCached(const Sequences& sequences)
{
    BlockIds blocks;
    for (const auto& p : sequences) {
        auto& seq = const_cast<Sequence&>(*p);
        if (seq.status != Sequence::kCached) {
            continue;
        }
        FT_CHECK(seq.blocks.size() == seq.block_unique_ids.size());
        // Verify cache blocks that may be invalidated
        const int count = block_manager_->Verify(seq.blocks, seq.block_unique_ids);
        seq.blocks.resize(count);
        seq.block_unique_ids.resize(count);

        blocks.insert(blocks.end(), seq.blocks.begin(), seq.blocks.end());
        seq.cache_len = std::min<int>(seq.cache_len, seq.blocks.size() * block_seq_len_);
        seq.status    = Sequence::kLocked;
    }
    block_manager_->Lock(blocks);
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
    FT_CHECK(sequence.status != Sequence::kCached);
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

    int remaining_input_count;

    Sequences        active;
    std::vector<int> block_counts;
    Sequences        inactive;
    Sequences        victims;

    Schedule(Snapshot snapshot, int size, int max_input_count):
        free(snapshot.free),
        cached(snapshot.cached),
        last(size),
        use_count_(std::move(snapshot.use_count)),
        unlocked_(size),
        it_(size),
        remaining_input_count(max_input_count)
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
    int input_count_;

    int allocate_{};
    int evict_{};
    int preempt_{};

    Sequences victims_;

    const Sequences& sequences_;
    Schedule&        schedule_;

    std::shared_ptr<BlockTrie> block_trie_;

    explicit Transaction(const Sequences& sequences, int index, int block_count, int input_count, Schedule& sched):
        sequences_(sequences), schedule_(sched), index_(index), block_count_(block_count), input_count_(input_count)
    {
    }

    void Process()
    {
        if (schedule_.remaining_input_count > 0) {
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
        FT_CHECK(schedule_.free >= 0);
        schedule_.cached += preempt_;
        schedule_.cached -= evict_;
        FT_CHECK(schedule_.cached >= 0);

        // update scheduled operations
        schedule_.allocate += allocate_;
        schedule_.evict += evict_;
        schedule_.preempt += preempt_;
        schedule_.victims.insert(schedule_.victims.end(), victims_.begin(), victims_.end());

        // update active sequences
        schedule_.active.push_back(sequences_[index_]);
        schedule_.block_counts.push_back(block_count_);

        input_count_ = std::min(input_count_, schedule_.remaining_input_count);
        schedule_.remaining_input_count -= input_count_;
        const_cast<Sequence*>(sequences_[index_])->input_length = input_count_;
    }
};

std::ostream& operator<<(std::ostream& os, const Transaction& trans)
{
    os << "index=" << trans.index_ << ", block_count=" << trans.block_count_ << ", allocate=" << trans.allocate_
       << ", evict=" << trans.evict_ << ", preempt=" << trans.preempt_ << ", victims=" << trans.victims_;
    return os;
}

}  // namespace

void SequenceManager::SortByPriority(Sequences&                   sequences,
                                     std::vector<int>&            context_lengths,
                                     const std::vector<uint64_t>& priorities)
{
    // sort according to priority
    std::vector<int> idxs(sequences.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
        return priorities[i] < priorities[j];  //
    });
    Sequences        tmp_sequences(sequences.size());
    std::vector<int> tmp_lengths(context_lengths.size());
    for (int i = 0; i < sequences.size(); ++i) {
        tmp_sequences[i] = sequences[idxs[i]];
        tmp_lengths[i]   = context_lengths[idxs[i]];
    }
    sequences.swap(tmp_sequences);
    context_lengths.swap(tmp_lengths);
}

std::vector<int> SequenceManager::CountRequiredBlocks(const Sequences&        sequences,
                                                      const std::vector<int>& context_lengths,
                                                      int                     step_length)
{
    std::vector<int> required(sequences.size());
    for (int i = 0; i < sequences.size(); ++i) {
        int seq_len = context_lengths[i] + step_length;
        int count   = (seq_len + block_seq_len_ - 1) / block_seq_len_ - static_cast<int>(sequences[i]->blocks.size());
        required[i] = std::max(0, count);
    }
    return required;
}

void SequenceManager::AssignAndActivate(const Sequences&        sequences,  //
                                        const std::vector<int>& counts,
                                        const BlockIds&         blocks,
                                        const UniqueIds&        unique_ids)
{
    FT_CHECK(sequences.size() == counts.size());
    int first = 0;
    for (int i = 0; i < sequences.size(); ++i) {
        auto& s     = const_cast<Sequence&>(*sequences[i]);
        auto  count = counts[i];
        int   last  = first + count;
        FT_CHECK(last <= blocks.size());
        s.blocks.insert(s.blocks.end(), blocks.begin() + first, blocks.begin() + last);
        s.block_unique_ids.insert(s.block_unique_ids.end(), unique_ids.begin() + first, unique_ids.begin() + last);
        s.status = Sequence::kActive;
        first    = last;
    }
}

void SequenceManager::PrefixMatch(Sequences& sequences)
{
    if (!block_trie_) {
        return;
    }

    for (int i = 0; i < sequences.size(); i++) {
        BlockIds  block_ids;
        UniqueIds unique_ids;
        auto&     seq = const_cast<Sequence&>(*sequences[i]);
        if (seq.cache_len != 0) {
            // We only apply prefix-cache matching when seq.cache_len is 0,
            // which means this seq is a brand-new sequence.
            // seq.cache_len is updated after every forward iter. Refer to `LlamaBatch::Forward`
            continue;
        }
        std::tie(block_ids, unique_ids) = block_trie_->Match(seq);

        block_manager_->Lock(block_ids);
        int valid = block_ids.size();
        if (rank_ == 0) {
            TM_LOG_INFO("[SeqMgr][match] ID %llu, hit blocks %d, cache_len %d", seq.id, valid, seq.cache_len);
            TM_LOG_DEBUG("[SeqMgr][match] ID %llu, hit block_ids %s, unique_ids %s",
                         seq.id,
                         vector2string(block_ids).c_str(),
                         vector2string(unique_ids).c_str());
        }

        FT_CHECK(seq.blocks.empty());
        seq.cache_len = valid * block_seq_len_;
        seq.blocks.insert(seq.blocks.end(), block_ids.begin(), block_ids.end());
        seq.block_unique_ids.insert(seq.block_unique_ids.end(), unique_ids.begin(), unique_ids.end());
        if (rank_ == 0) {
            TM_LOG_INFO("[SeqMgr][match] ID %llu, after matching, blocks %d, cache_len %d",
                        seq.id,
                        seq.blocks.size(),
                        seq.cache_len);
            TM_LOG_DEBUG("[SeqMgr][match] ID %llu, after matching, block_ids %s, unique_ids %s",
                         seq.id,
                         vector2string(seq.blocks).c_str(),
                         vector2string(seq.block_unique_ids).c_str());
        }
    }
}

auto SequenceManager::Materialize(Sequences                    sequences,
                                  std::vector<int>             context_lengths,
                                  const std::vector<uint64_t>& priorities,
                                  int                          step_length,
                                  AdjustInputCount             adjust) -> Outcome
{
    ////////////////////////////////////////////////////////////////////////////////
    /// Schedule the assignment of blocks to sequences

    // process deferred unlock and free operations
    CommitUnlockAndFree();

    SortByPriority(sequences, context_lengths, priorities);

    // Verify and lock cache sequences to avoid their blocks being evicted unnoticed
    // the blocks can still be preempted later
    VerifyAndLockCached(sequences);

    PrefixMatch(sequences);

    const int max_input_count = adjust(sequences, context_lengths);

    std::vector<int> required = CountRequiredBlocks(sequences, context_lengths, step_length);
    // dbg(required);

    Schedule schedule(block_manager_->TakeSnapshot(), sequences.size(), max_input_count);

    // `schedule.last` is decreasing in the loop
    for (int i = 0; i < schedule.last; ++i) {
        const int input_length = context_lengths[i] - sequences[i]->cache_len;
        Transaction{sequences, i, required[i], input_length, schedule}.Process();
    }

    // mark remaining sequences invalid
    for (int i = schedule.last; i < sequences.size(); ++i) {
        schedule.inactive.push_back(sequences[i]);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Schedule is ready, time to execute it. (locked -> cached -> free -> locked)

    // combine allocate and evict since evicted blocks are reused by allocation
    schedule.allocate += schedule.evict;

    if (schedule.allocate) {
        dbg(*block_manager_);
    }

    Outcome outcome{};
    outcome.allocation = schedule.allocate;
    outcome.swap_in    = std::count_if(schedule.active.begin(), schedule.active.end(), [](auto p) {
        if (p->status != Sequence::kActive) {
            dbg(*p);
        }
        return p->status != Sequence::kActive;  //
    });
    outcome.swap_out   = std::count_if(schedule.inactive.begin(), schedule.inactive.end(), [](auto p) {
        if (p->status == Sequence::kActive) {
            dbg(*p);
        }
        return p->status == Sequence::kActive;  //
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

}  // namespace turbomind
