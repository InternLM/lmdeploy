#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/utils/logger.h"
#include <ctime>

namespace turbomind {

SequenceManager::SequenceManager(size_t      layer_num,
                                 size_t      head_num,
                                 size_t      head_dim,
                                 size_t      block_len,
                                 double      block_count,
                                 int         chunk_size,
                                 size_t      elem_bits,
                                 int         rank,
                                 IAllocator* allocator):
    block_len_(block_len), rank_(rank)
{
    constexpr int kBitsPerByte = 8;

    size_t block_size = layer_num * head_num * block_len * head_dim * elem_bits / kBitsPerByte * 2;

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
        block_manager_->Release(it->second.blocks);
        it->second = std::move(sequence);
    }
    else {
        it = sequences_.emplace_hint(it, id, std::move(sequence));
    }

    return &it->second;
}

void SequenceManager::VerifyBlocks(Sequence& seq)
{
    FT_CHECK(seq.blocks.size() == seq.block_unique_ids.size());
    for (int i = 0; i < seq.blocks.size(); ++i) {
        if (seq.blocks[i]->unique_id != seq.block_unique_ids[i]) {
            seq.blocks.resize(i);
            seq.block_unique_ids.resize(i);
            break;
        }
    }
    seq.cache_len = std::min<int>(seq.cache_len, seq.blocks.size() * block_len_);
}

const Sequence* SequenceManager::Fetch(uint64_t id)
{
    if (auto it = sequences_.find(id); it != sequences_.end()) {
        auto& sequence = it->second;
        return &it->second;
    }

    return nullptr;
}

bool SequenceManager::Erase(uint64_t id)
{
    if (auto it = sequences_.find(id); it != sequences_.end()) {
        auto& seq = it->second;
        if (seq.status != Sequence::kCached) {
            if (released_.empty()) {
                released_ = std::move(seq.blocks);
            }
            else {
                released_.insert(released_.end(), seq.blocks.begin(), seq.blocks.end());
            }
        }
        sequences_.erase(it);
    }

    return false;
}

void SequenceManager::Update(const Sequence& sequence)
{
    block_manager_->Touch(sequence.blocks);
}

bool SequenceManager::Contains(uint64_t id)
{
    return sequences_.find(id) != sequences_.end();
}

namespace {

struct Schedule {
    int free;
    int cached;

    int allocate;
    int evict;

    std::vector<int> victims;

    std::vector<int> active;
    std::vector<int> block_counts;

    std::vector<int> inactive;
};

class Simulator {
public:
    explicit Simulator(const std::vector<const Sequence*>& seqs,
                       const std::vector<int>&             idxs,
                       std::vector<int>&                   ref_count):
        seqs_(seqs), idxs_(idxs), ref_count_(ref_count)
    {
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
        sched_.cached += preempt_ - evict_;

        sched_.allocate += allocate_;
        sched_.evict += evict_;

        sched_.victims.insert(sched_.victims.end(), victims_.begin(), victims_.end());

        sched_.active.push_back(index_);
        sched_.block_counts.push_back(block_count_);
    }
};

}  // namespace

std::ostream& operator<<(std::ostream& os, const Sequence& seq)
{
    os << "Sequence[id=" << seq.id << ",status=" << seq.status << ",size(blocks)=" << seq.blocks.size()
       << ",cache_len=" << seq.cache_len << ",size(random_state)=" << seq.random_state.size() << "]";
    return os;
}

bool SequenceManager::Materialize(const std::vector<const Sequence*>& sequences,
                                  const std::vector<int>&             context_lengths,
                                  const std::vector<uint64_t>&        priorities,
                                  int                                 step_length)
{
    ////////////////////////////////////////////////////////////////////////////////
    /// Schedule the assignment of blocks to sequences
    auto seqs = const_cast<Sequence* const*>(sequences.data());

    // check validity of of cached blocks (blocks of active & locked seqs are always valid)
    if (need_verification_) {
        for (int i = 0; i < sequences.size(); ++i) {
            if (seqs[i]->status == Sequence::kCached) {
                VerifyBlocks(*seqs[i]);
            }
        }
        need_verification_ = false;
    }

    // count required blocks based on block validity
    std::vector<int> required(sequences.size());
    int              total_required{};
    for (int i = 0; i < sequences.size(); ++i) {
        int seq_len = context_lengths[i] + step_length;
        int count   = (seq_len + block_len_ - 1) / block_len_ - static_cast<int>(seqs[i]->blocks.size());
        required.push_back(std::max(0, count));
        total_required += required.back();
    }

    // no new blocks required, exit early
    if (total_required == 0) {
        return false;
    }

    /// TODO: more early exit heuristics

    // sort according to priority
    std::vector<int> idxs(sequences.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&](int i, int j) { return priorities[i] < priorities[j]; });

    Snapshot snapshot = block_manager_->TakeSnapshot();

    Schedule schedule{snapshot.free, snapshot.cached};
    schedule.cached += released_.size();

    Simulator simulator(sequences, idxs, snapshot.ref_count);

    bool modified = false;

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
            block_count -= trans.Evict(std::min(block_count, schedule.free));
        }

        for (int v = j - 1; block_count && v > i; --v) {
            if (sequences[idxs[v]]->status == Sequence::kCached) {
                continue;
            }
            int preempt = trans.Preempt(v, idxs[v]);
            // Commit only when preemption actually free enough blocks for the sequence to run
            if (block_count <= preempt) {
                // preempted blocks are in cached state
                block_count -= trans.Evict(block_count);
                j = v + 1;
                break;
            }
        }

        if (block_count == 0) {
            trans.Commit();
            if (seq.status != Sequence::kActive) {
                modified = true;
            }
        }
        else {
            // failed to collect enough block for the sequence, transaction aborted. Active sequence will be kept
            // locked if not preempted by seq with higher priority
            schedule.inactive.push_back(idx);
            if (seq.status == Sequence::kActive) {
                modified = true;
            }
        }
    }

    // Verify the schedule
    FT_CHECK(schedule.allocate <= snapshot.free);
    FT_CHECK(schedule.evict <= snapshot.cached);
    // FT_CHECK(schedule.allocate + schedule.evict + schedule.preempt == total_block_count);

    ////////////////////////////////////////////////////////////////////////////////
    /// Schedule is ready, time to execute it. (locked -> cached -> free -> locked)
    schedule.allocate += schedule.evict;

    // release preempted blocks -> cached
    {
        std::vector<const Block*> blocks;
        for (const auto& v : schedule.victims) {
            auto& seq = *seqs[v];
            block_manager_->Touch(seq.blocks);
            seq.status = Sequence::kCached;
            blocks.insert(blocks.end(), seq.blocks.begin(), seq.blocks.end());
        }
        block_manager_->Release(blocks);
    }

    // evict cached blocks -> free
    if (schedule.evict) {
        need_verification_ = true;
        block_manager_->Evict(schedule.evict);
    }

    // allocate & assign blocks
    auto blocks = block_manager_->Allocate(schedule.allocate + schedule.evict);
    auto first  = blocks.begin();

    for (const auto& idx : schedule.active) {
        auto& sequence  = *seqs[idx];
        sequence.status = Sequence::kActive;

        auto last = first + required[idx];
        std::for_each(first, last, [&sequence](const Block* b) {
            sequence.blocks.push_back(b);
            sequence.block_unique_ids.push_back(b->unique_id);
        });

        first = last;
    }

    block_manager_->Touch(blocks);

    for (const auto& idx : schedule.inactive) {
        if (seqs[idx]->status == Sequence::kActive) {
            seqs[idx]->status = Sequence::kLocked;
        }
    }

    for (const auto& idx : schedule.victims) {
        seqs[idx]->status = Sequence::kCached;
    }

    return modified;
}

}  // namespace turbomind