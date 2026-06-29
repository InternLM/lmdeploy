// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/memory/common.h"
#include "src/turbomind/memory/page.h"
#include "src/turbomind/memory/stats.h"
#include "src/turbomind/core/logger.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace turbomind {

class Slab {
public:
    Slab() = default;

    Slab(int slab_id, char* base, size_t size, int object_size, int object_count):
        slab_id_{slab_id},
        base_{base},
        size_{size},
        object_size_{object_size},
        object_count_{object_count},
        slot_owner_(object_count)  // default object_alloc_t{} == {nullptr}
    {
        free_list_.reserve(object_count_);
        for (int i = 0; i < object_count_; ++i) {
            free_list_.push_back(i);
        }
    }

    // Compiler-generated copy/move/dtor.

    int allocate(SlabSlot* objects, size_t count)
    {
        count = std::min(count, free_list_.size());
        for (size_t i = 0; i < count; ++i) {
            const int slot_id = free_list_.back();
            free_list_.pop_back();
            objects[i] = {slab_id_, slot_id};
        }
        return static_cast<int>(count);
    }

    void deallocate(const SlabSlot* objects, size_t count)
    {
        for (size_t i = 0; i < count; ++i) {
            const int slot_id      = objects[i].slot_id;
            slot_owner_[slot_id]   = {};  // clear reverse link
            free_list_.push_back(slot_id);
        }
    }

    void           set_owner(int slot_id, object_alloc_t o) noexcept { slot_owner_[slot_id] = o; }
    object_alloc_t owner(int slot_id) const noexcept { return slot_owner_[slot_id]; }

    int slab_id() const noexcept { return slab_id_; }
    bool is_full() const noexcept { return free_list_.empty(); }
    bool is_empty() const noexcept { return free_list_.size() == static_cast<size_t>(object_count_); }
    bool is_partial() const noexcept { return !is_full() && !is_empty(); }
    size_t n_free() const noexcept { return free_list_.size(); }
    char* base() const noexcept { return base_; }
    int object_size() const noexcept { return object_size_; }
    int object_count() const noexcept { return object_count_; }

    // Intrusive list links -- directly read/written by SlabAllocator's list helpers.
    int prev_id_ = -1;
    int next_id_ = -1;

private:
    int                         slab_id_      = -1;
    char*                       base_         = nullptr;
    size_t                      size_         = 0;
    int                         object_size_  = 0;
    int                         object_count_ = 0;
    std::vector<int>            free_list_;
    std::vector<object_alloc_t> slot_owner_;  // slot_id -> owning token (reverse link)
};

class SlabAllocator {
public:
    SlabAllocator(size_t object_size, size_t min_size, size_t max_size, float threshold, size_t max_empty_slabs):
        object_size_{object_size}, max_empty_slabs_{max_empty_slabs}
    {
        auto get_ratio = [&](size_t size) -> float {
            auto quantized = size / object_size * object_size;
            return quantized / static_cast<float>(size);
        };

        // smallest power-of-2 size that satisfies both min_size and object_size
        size_t size = ceil_pow2(std::max(min_size, object_size));

        // double slab size until utilization >= threshold
        float ratio = get_ratio(size);
        while (ratio < threshold && size < max_size) {
            size <<= 1;
            ratio = get_ratio(size);
        }

        slab_size_    = size;
        object_count_ = size / object_size;

        TM_LOG_WARN("slab_size {}, object_size {}, object_count {}, ratio {}", slab_size_, object_size_, object_count_, ratio);

        slabs_.resize(kFirstSlabId);  // three default-constructed sentinel Slabs
        list_init(kFull);
        list_init(kPartial);
        list_init(kEmpty);
    }

    // Compiler-generated copy/move/dtor -- pure value type.

    int allocate(SlabSlot* objects, size_t count, PageAllocator& page_alloc)
    {
        ptrdiff_t remain = count;

        for (int id = head(kPartial); id != kPartial && remain > 0;) {
            Slab& slab = slabs_[id];
            int   next = slab.next_id_;
            remain -= slab.allocate(objects + (count - remain), remain);
            if (slab.is_full())
                splice(kFull, kPartial, id);
            id = next;
        }

        while (remain > static_cast<ptrdiff_t>(sizes_[kEmpty] * object_count_)) {
            if (!create_empty(page_alloc))
                break;
        }

        for (int id = head(kEmpty); id != kEmpty && remain > 0;) {
            Slab& slab = slabs_[id];
            int   next = slab.next_id_;
            remain -= slab.allocate(objects + (count - remain), remain);
            if (slab.is_full())
                splice(kFull, kEmpty, id);
            else
                splice(kPartial, kEmpty, id);
            id = next;
        }

        return static_cast<int>(count - remain);
    }

    int deallocate(const SlabSlot* objects, size_t count, PageAllocator& page_alloc)
    {
        for (size_t i = 0; i < count;) {
            const int slab_id = objects[i].slab_id;

            size_t j = i + 1;
            while (j < count && objects[j].slab_id == slab_id) {
                ++j;
            }

            Slab&      slab     = slabs_[slab_id];
            const bool was_full = slab.is_full();
            slab.deallocate(objects + i, j - i);
            if (was_full)
                splice_front(kPartial, kFull, slab_id);
            if (slab.is_empty())
                splice(kEmpty, kPartial, slab_id);

            i = j;
        }

        while (static_cast<size_t>(sizes_[kEmpty]) > max_empty_slabs_) {
            int slab_id = head(kEmpty);
            erase(kEmpty, slab_id);
            page_alloc.deallocate(slabs_[slab_id].base(), slab_size_);
            free_ids_.push_back(slab_id);
        }

        return static_cast<int>(count);
    }

    char* AddressOf(SlabSlot h) const noexcept
    {
        const Slab& slab = slabs_[h.slab_id];
        return slab.base() + static_cast<size_t>(h.slot_id) * slab.object_size();
    }

    void           set_owner(SlabSlot h, object_alloc_t o) noexcept { slabs_[h.slab_id].set_owner(h.slot_id, o); }
    object_alloc_t owner(SlabSlot h) const noexcept { return slabs_[h.slab_id].owner(h.slot_id); }

    size_t object_size() const noexcept
    {
        return object_size_;
    }

    SlabStats stats() const
    {
        SlabStats s{};
        s.object_size   = object_size_;
        s.slab_size     = slab_size_;
        s.object_count  = object_count_;
        s.n_full        = sizes_[kFull];
        s.n_partial     = sizes_[kPartial];
        s.n_empty       = sizes_[kEmpty];
        s.n_slabs       = s.n_full + s.n_partial + s.n_empty;
        s.total_objects = static_cast<size_t>(s.n_slabs) * object_count_;

        size_t free_objects = static_cast<size_t>(s.n_empty) * object_count_;
        for (int id = head(kPartial); id != kPartial; id = slabs_[id].next_id_) {
            free_objects += slabs_[id].n_free();
        }
        s.free_objects = free_objects;
        s.used_objects = s.total_objects - free_objects;
        return s;
    }

private:
    static constexpr int kFull        = 0;
    static constexpr int kPartial     = 1;
    static constexpr int kEmpty       = 2;
    static constexpr int kFirstSlabId = 3;

    int head(int sentinel) const noexcept
    {
        return slabs_[sentinel].next_id_;
    }
    int tail(int sentinel) const noexcept
    {
        return slabs_[sentinel].prev_id_;
    }

    void list_init(int sentinel) noexcept
    {
        slabs_[sentinel].prev_id_ = sentinel;
        slabs_[sentinel].next_id_ = sentinel;
    }

    void push_back(int sentinel, int id) noexcept
    {
        const int prev            = slabs_[sentinel].prev_id_;
        slabs_[prev].next_id_     = id;
        slabs_[id].prev_id_       = prev;
        slabs_[id].next_id_       = sentinel;
        slabs_[sentinel].prev_id_ = id;
        ++sizes_[sentinel];
    }

    void push_front(int sentinel, int id) noexcept
    {
        const int next            = slabs_[sentinel].next_id_;
        slabs_[next].prev_id_     = id;
        slabs_[id].prev_id_       = sentinel;
        slabs_[id].next_id_       = next;
        slabs_[sentinel].next_id_ = id;
        ++sizes_[sentinel];
    }

    void erase(int sentinel, int id) noexcept
    {
        const int prev        = slabs_[id].prev_id_;
        const int next        = slabs_[id].next_id_;
        slabs_[prev].next_id_ = next;
        slabs_[next].prev_id_ = prev;
        --sizes_[sentinel];
    }

    void splice(int to, int from, int id) noexcept
    {
        erase(from, id);
        push_back(to, id);
    }

    void splice_front(int to, int from, int id) noexcept
    {
        erase(from, id);
        push_front(to, id);
    }

    // INVARIANT: callers (allocate, deallocate) MUST NOT hold a Slab& across
    // a call to create_empty -- slabs_.emplace_back may reallocate the vector.
    bool create_empty(PageAllocator& page_alloc)
    {
        auto memory = page_alloc.allocate(slab_size_);
        if (!memory)
            return false;

        int slab_id;
        if (!free_ids_.empty()) {
            slab_id = free_ids_.back();
            free_ids_.pop_back();
            // Move-assign over the ghost entry. Safe: Slab's move-assign only
            // copies scalars and swaps containers -- it never dereferences base_.
            slabs_[slab_id] = Slab{
                slab_id, (char*)memory, slab_size_, static_cast<int>(object_size_), static_cast<int>(object_count_)};
        }
        else {
            slab_id = static_cast<int>(slabs_.size());
            slabs_.emplace_back(
                slab_id, (char*)memory, slab_size_, static_cast<int>(object_size_), static_cast<int>(object_count_));
        }
        push_back(kEmpty, slab_id);
        return true;
    }

    size_t object_size_     = 0;
    size_t object_count_    = 0;
    size_t slab_size_       = 0;
    size_t max_empty_slabs_ = 0;

    std::vector<Slab>  slabs_;               // [0..2] sentinels; [3..] real slabs (some may be ghosts)
    std::vector<int>   free_ids_;            // recycled real slab_ids (always >= 3)
    std::array<int, 3> sizes_{};             // sizes_[kFull|kPartial|kEmpty]
};

}  // namespace turbomind
