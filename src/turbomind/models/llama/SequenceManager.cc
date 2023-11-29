// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/BlockManager.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/debug_utils.h"
#include "src/turbomind/utils/logger.h"
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <stdexcept>

namespace turbomind {

SequenceManager::SequenceManager(size_t      layer_num,
                                 size_t      head_num,
                                 size_t      head_dim,
                                 size_t      block_seq_len,
                                 double      block_count,
                                 int         chunk_size,
                                 size_t      elem_bits,
                                 int         rank,
                                 IAllocator* allocator):
    block_seq_len_(block_seq_len)
{
    constexpr int kBitsPerByte = 8;

    // [2, L, H, block_seq_len, D]
    size_t block_size = 2UL * layer_num * head_num * block_seq_len * head_dim * elem_bits / kBitsPerByte;

    block_manager_ = std::make_unique<BlockManager>(block_size, block_count, chunk_size, allocator);

    val_offset_ = block_size / 2;
}

const Sequence* SequenceManager::Create(uint64_t id)
{
    Sequence sequence{id};

    auto it = sequences_.find(id);
    if (it != sequences_.end()) {
        if (rank_ == 0) {
            TM_LOG_WARNING("[SequenceManager][Create] Removing conflicting ID %ld", (long)id);
        }
        auto& seq = it->second;
        if (seq.status != Sequence::kCached) {
            unlocked_.insert(unlocked_.end(), seq.blocks.begin(), seq.blocks.end());
        }
        seq = std::move(sequence);
    }
    else {
        it = sequences_.emplace_hint(it, id, std::move(sequence));
    }

    return &it->second;
}

const Sequence* SequenceManager::Get(uint64_t id)
{
    if (auto it = sequences_.find(id); it != sequences_.end()) {
        auto& sequence = it->second;
        return &it->second;
    }
    return nullptr;
}

bool SequenceManager::Contains(uint64_t id)
{
    return sequences_.find(id) != sequences_.end();
}

bool SequenceManager::Erase(uint64_t id)
{
    if (auto it = sequences_.find(id); it != sequences_.end()) {
        auto& seq = it->second;
        if (seq.status != Sequence::kCached) {
            unlocked_.insert(unlocked_.end(), seq.blocks.begin(), seq.blocks.end());
            freed_.insert(freed_.end(), seq.blocks.begin(), seq.blocks.end());
        }
        else {
            for (int i = 0; i < seq.blocks.size(); ++i) {
                // filter invalidated blocks
                if (seq.blocks[i]->unique_id == seq.block_unique_ids[i]) {
                    freed_.push_back(seq.blocks[i]);
                }
            }
        }
        sequences_.erase(it);
        return true;
    }
    return false;
}

void SequenceManager::VerifyAndLockCached(const Sequences& sequences)
{
    std::vector<const Block*> blocks;
    for (const auto& p : sequences) {
        auto& seq = const_cast<Sequence&>(*p);
        if (seq.status != Sequence::kCached) {
            continue;
        }
        FT_CHECK(seq.blocks.size() == seq.block_unique_ids.size());
        if (need_verify_) {
            for (int i = 0; i < seq.blocks.size(); ++i) {
                if (seq.blocks[i]->unique_id != seq.block_unique_ids[i]) {
                    seq.blocks.resize(i);
                    seq.block_unique_ids.resize(i);
                    break;
                }
            }
        }
        blocks.insert(blocks.end(), seq.blocks.begin(), seq.blocks.end());
        seq.cache_len = std::min<int>(seq.cache_len, seq.blocks.size() * block_seq_len_);
        seq.status    = Sequence::kLocked;
    }
    block_manager_->Lock(blocks);
    need_verify_ = false;
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

    int input_count1;
    int input_count2;

    Sequences        active;
    std::vector<int> block_counts;
    Sequences        inactive;
    Sequences        victims;

    Schedule(Snapshot snapshot, int size, int _input_count1, int _input_count2):
        free(snapshot.free),
        cached(snapshot.cached),
        last(size),
        use_count_(std::move(snapshot.use_count)),
        unlocked_(size),
        it_(size),
        input_count1(_input_count1),
        input_count2(_input_count2)
    {
    }

    int Unlock(const Sequences& seqs, int vidx)
    {
        while (vidx < it_) {
            const auto& blocks = seqs[--it_]->blocks;
            int         count  = 0;
            for (const auto& p : blocks) {
                count += static_cast<int>(--use_count_[p->id] == 0);
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

    explicit Transaction(const Sequences& sequences, int index, int block_count, int input_count, Schedule& sched):
        sequences_(sequences), schedule_(sched), index_(index), block_count_(block_count), input_count_(input_count)
    {
    }

    void Process()
    {
        if (schedule_.input_count1 > 0) {
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

        if (input_count_ > schedule_.input_count2) {
            input_count_ = schedule_.input_count1;
        }
        schedule_.input_count1 -= input_count_;
        schedule_.input_count2 -= input_count_;
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

// template<class P, class... Ts>
// void SortByPriority(const std::vector<P>& priorities, Ts&... ranges)
// {
//     // sort according to priority
//     std::vector<int> idxs(priorities.size());
//     std::iota(idxs.begin(), idxs.end(), 0);
//     std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
//         return priorities[i] < priorities[j];  //
//     });
//     auto reorder = [&](auto& src) {
//         auto dst = src;
//         for (size_t i = 0; i < idxs.size(); ++i) {
//             dst[i] = src[idxs[i]];
//         }
//         src.swap(dst);
//     };
//     (reorder(ranges), ...);
// }

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

void SequenceManager::AssignAndActivate(const Sequences&                 sequences,  //
                                        const std::vector<int>&          counts,
                                        const std::vector<const Block*>& blocks)
{
    FT_CHECK(sequences.size() == counts.size());
    auto first = blocks.begin();
    for (int i = 0; i < sequences.size(); ++i) {
        auto& s     = const_cast<Sequence&>(*sequences[i]);
        auto  count = counts[i];
        // dbg(count);
        auto last = first + count;
        std::for_each(first, last, [&](const Block* b) {
            s.blocks.push_back(b);
            s.block_unique_ids.push_back(b->unique_id);
        });
        s.status = Sequence::kActive;
        first    = last;
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

    // SortByPriority(priorities, sequences, context_lengths);

    // Verify and lock cache sequences to avoid their blocks being evicted unnoticed
    // the blocks can still be preempted later
    VerifyAndLockCached(sequences);

    auto [input_count1, input_count2] = adjust(sequences, context_lengths);

    std::vector<int> required = CountRequiredBlocks(sequences, context_lengths, step_length);
    // dbg(required);

    Schedule schedule(block_manager_->TakeSnapshot(), sequences.size(), input_count1, input_count2);

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
        for (const auto& p : schedule.victims) {
            UpdateAndSetUnlock(*p);
        }
        CommitUnlockAndFree();
    }

    // evict cached blocks -> free
    if (schedule.evict) {
        block_manager_->Evict(schedule.evict);
        need_verify_ = true;
    }

    // allocate & assign blocks
    {
        std::vector<const Block*> blocks;
        if (schedule.allocate) {
            blocks = block_manager_->Allocate(schedule.allocate);
        }
        AssignAndActivate(schedule.active, schedule.block_counts, blocks);
    }

    // active -> locked
    for (const auto& p : schedule.inactive) {
        if (p->status == Sequence::kActive) {
            const_cast<Sequence*>(p)->status = Sequence::kLocked;
        }
    }

    return outcome;
}

}  // namespace turbomind
