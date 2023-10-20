// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/utils/allocator.h"
#include "src/turbomind/utils/debug_utils.h"
#include "src/turbomind/utils/logger.h"
#include <ctime>
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
    Sequence sequence{id, {}, {}, {}, {}, {}};

    auto it = sequences_.find(id);
    if (it != sequences_.end()) {
        if (rank_ == 0) {
            TM_LOG_WARNING("[SequenceManager][Create] Removing conflicting ID %ld", (long)id);
        }
        auto& seq = it->second;
        if (seq.status != Sequence::kCached) {
            released_.insert(released_.end(), seq.blocks.begin(), seq.blocks.end());
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
            released_.insert(released_.end(), seq.blocks.begin(), seq.blocks.end());
        }
        sequences_.erase(it);
    }
    else {
        throw std::out_of_range(std::to_string(id));
    }
    return false;
}

void SequenceManager::Verify(Sequence& seq, std::vector<const Block*>& retain)
{
    FT_CHECK(seq.blocks.size() == seq.block_unique_ids.size());
    for (int i = 0; i < seq.blocks.size(); ++i) {
        if (seq.blocks[i]->unique_id != seq.block_unique_ids[i]) {
            seq.blocks.resize(i);
            seq.block_unique_ids.resize(i);
            break;
        }
    }
    retain.insert(retain.end(), seq.blocks.begin(), seq.blocks.end());
    seq.status    = Sequence::kLocked;
    seq.cache_len = std::min<int>(seq.cache_len, seq.blocks.size() * block_seq_len_);
}

void SequenceManager::Release(const Sequence& sequence)
{
    auto& seq = const_cast<Sequence&>(sequence);
    if (seq.status == Sequence::kActive) {
        block_manager_->Touch(seq.blocks);
    }
    if (seq.status != Sequence::kCached) {
        released_.insert(released_.end(), seq.blocks.begin(), seq.blocks.end());
    }
    seq.status = Sequence::kCached;
}

namespace {

struct Schedule {
    int free;
    int cached;

    int allocate;
    int evict;
    int preempt;

    std::vector<int> victims;

    std::vector<int> active;
    std::vector<int> block_counts;

    std::vector<int> inactive;
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

class Simulator {
public:
    explicit Simulator(const std::vector<const Sequence*>& seqs,
                       const std::vector<int>&             idxs,
                       std::vector<int>&                   ref_count):
        seqs_(seqs), idxs_(idxs), ref_count_(ref_count)
    {
        // dbg(seqs.size());
        released_.resize(seqs.size());
        ptr_ = released_.size();
    }

    int Release(int order)
    {
        while (order < ptr_) {
            --ptr_;
            int count = 0;
            for (const auto& p : seqs_[idxs_[ptr_]]->blocks) {
                if (--ref_count_[p->id] == 0) {
                    ++count;
                }
            }
            released_[ptr_] = count;
        }

        return released_[order];
    }

private:
    const std::vector<const Sequence*>& seqs_;
    const std::vector<int>&             idxs_;

    std::vector<int>& ref_count_;

    std::vector<int> released_;
    int              ptr_;
};

struct Transaction {
    int index_;
    int block_count_;

    int allocate_{};
    int evict_{};
    int preempt_{};

    std::vector<int> victims_;

    Schedule&  sched_;
    Simulator& simulator_;

    explicit Transaction(Schedule& sched, int index, int block_count, Simulator& simulator):
        sched_(sched), index_(index), block_count_(block_count), simulator_(simulator)
    {
    }

    int Allocate(int count)
    {
        allocate_ += count;
        return count;
    }

    int Evict(int count)
    {
        evict_ += count;
        return count;
    }

    int Preempt(int order, int idx)
    {
        victims_.push_back(idx);
        preempt_ += simulator_.Release(order);
        return preempt_;
    }

    void Commit()
    {
        sched_.free -= allocate_;
        FT_CHECK(sched_.free >= 0);

        sched_.cached += preempt_;
        sched_.cached -= evict_;
        FT_CHECK(sched_.cached >= 0);

        sched_.allocate += allocate_;
        sched_.evict += evict_;
        sched_.preempt += preempt_;

        sched_.victims.insert(sched_.victims.end(), victims_.begin(), victims_.end());

        sched_.active.push_back(index_);
        sched_.block_counts.push_back(block_count_);
    }
};

std::ostream& operator<<(std::ostream& os, const Transaction& trans)
{
    os << "index=" << trans.index_ << ", block_count=" << trans.block_count_ << ", allocate=" << trans.allocate_
       << ", evict=" << trans.evict_ << ", preempt=" << trans.preempt_ << ", victims=" << trans.victims_;
    return os;
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const Sequence& seq)
{
    os << "id=" << seq.id << ", status=" << seq.status << ", size(blocks)=" << seq.blocks.size()
       << ", cache_len=" << seq.cache_len << ", size(random_state)=" << seq.random_state.size();
    return os;
}

auto SequenceManager::Materialize(const std::vector<const Sequence*>& sequences,
                                  const std::vector<int>&             context_lengths,
                                  const std::vector<uint64_t>&        priorities,
                                  int                                 step_length) -> Outcome
{
    dbg(__PRETTY_FUNCTION__);
    ////////////////////////////////////////////////////////////////////////////////
    /// Schedule the assignment of blocks to sequences
    auto    seqs = const_cast<Sequence* const*>(sequences.data());
    Outcome outcome{};

    if (!released_.empty()) {
        block_manager_->Release(released_);
        released_.clear();
    }

    // check validity of of cached blocks (blocks of active & locked seqs are always valid)
    if (need_verification_) {
        need_verification_ = false;
        std::vector<const Block*> retain;
        for (int i = 0; i < sequences.size(); ++i) {
            if (seqs[i]->status == Sequence::kCached) {
                Verify(*seqs[i], retain);
            }
        }
        block_manager_->Retain(retain);
    }

    // count required blocks based on block validity
    std::vector<int> required(sequences.size());
    // int              total_required{};
    for (int i = 0; i < sequences.size(); ++i) {
        int seq_len = context_lengths[i] + step_length;
        int count   = (seq_len + block_seq_len_ - 1) / block_seq_len_ - static_cast<int>(seqs[i]->blocks.size());
        required[i] = std::max(0, count);
        // total_required += required[i];
    }

    // dbg(required);

    // no new blocks required, exit early
    // if (total_required == 0) {
    //     dbg("early exit");
    //     return outcome;
    // }

    /// TODO: more early exit heuristics

    // sort according to priority
    std::vector<int> idxs(sequences.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) { return priorities[i] < priorities[j]; });

    Snapshot snapshot = block_manager_->TakeSnapshot();

    Schedule schedule{snapshot.free, snapshot.cached};
    schedule.cached += released_.size();

    Simulator simulator(sequences, idxs, snapshot.ref_count);

    std::vector<int> active(idxs.size());
    std::vector<int> victim(idxs.size());

    for (int i = 0, j = idxs.size(); i < j; ++i) {
        const int idx = idxs[i];

        const auto& seq         = *sequences[idx];
        auto        block_count = required[idx];

        Transaction trans{schedule, idx, block_count, simulator};

        // allocate from free blocks
        if (block_count) {
            block_count -= trans.Allocate(std::min(block_count, schedule.free));
        }
        // evict cached blocks
        if (block_count) {
            block_count -= trans.Evict(std::min(block_count, schedule.cached));
        }

        for (int v = j - 1; block_count && v > i; --v) {
            if (sequences[idxs[v]]->status == Sequence::kCached) {
                continue;
            }
            // dbg(v, idxs[v]);
            int preempt = trans.Preempt(v, idxs[v]);
            // dbg(preempt);
            // Commit only when preemption actually free enough blocks for the sequence to run
            if (block_count <= preempt) {
                // preempted blocks are in cached state
                block_count -= trans.Evict(block_count);
                j = v;
                break;
            }
        }

        // dbg(block_count, trans);

        if (block_count == 0) {
            trans.Commit();
            active[idx] = 1;
            if (seq.status != Sequence::kActive) {
                ++outcome.swap_in;
            }
        }
    }

    for (const auto& i : idxs) {
        if (!active[i]) {
            schedule.inactive.push_back(i);
            if (seqs[i]->status == Sequence::kActive) {
                ++outcome.swap_out;
            }
        }
    }

    // dbg(schedule);

    ////////////////////////////////////////////////////////////////////////////////
    /// Schedule is ready, time to execute it. (locked -> cached -> free -> locked)
    schedule.allocate += schedule.evict;

    outcome.allocation = schedule.allocate;

    // release preempted blocks -> cached
    for (const auto& v : schedule.victims) {
        Release(*sequences[v]);
    }
    block_manager_->Release(released_);
    released_.clear();

    // evict cached blocks -> free
    if (schedule.evict) {
        block_manager_->Evict(schedule.evict);
        need_verification_ = true;
    }

    // allocate & assign blocks
    auto blocks = block_manager_->Allocate(schedule.allocate);
    auto first  = blocks.begin();

    for (const auto& idx : schedule.active) {
        auto& sequence = *seqs[idx];

        // retain blocks for swap-in sequences
        if (sequence.status == Sequence::kCached) {
            block_manager_->Retain(sequence.blocks);
        }

        sequence.status = Sequence::kActive;

        auto last = first + required[idx];
        std::for_each(first, last, [&](const Block* b) {
            sequence.blocks.push_back(b);
            sequence.block_unique_ids.push_back(b->unique_id);
        });

        first = last;
    }

    for (const auto& idx : schedule.inactive) {
        if (seqs[idx]->status == Sequence::kActive) {
            seqs[idx]->status = Sequence::kLocked;
        }
    }

    for (const auto& idx : schedule.victims) {
        seqs[idx]->status = Sequence::kCached;
    }

    return outcome;
}

}  // namespace turbomind