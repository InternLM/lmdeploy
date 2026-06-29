#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/memory/common.h"
#include "src/turbomind/memory/stats.h"

#include <array>
#include <memory>
#include <vector>

namespace turbomind {

// The durable per-allocation entry. One per allocation (simple = 1 part). The
// handle (object_alloc_t) is a pointer to this. A compactor rewrites base0 /
// bases[] in place; every cached handle then sees the new address.
struct Allocation {
    uint64_t key{};   // unique, monotonic; set on Acquire, 0 while free (ABA stale check)
    int      n{0};    // part count; 0 == free

    // Inline storage for the dominant single-part (simple) object: no heap.
    char*    base0{};
    SlabSlot slot0{};

    // Used ONLY for composites (n > 1), holding all n parts. Empty for simple.
    std::vector<char*>    bases;
    std::vector<SlabSlot> slots;

    char* base(int p) const noexcept { return n == 1 ? base0 : bases[p]; }
    int   part_count() const noexcept { return n; }
};

class ObjectAllocator {
public:
    ~ObjectAllocator();
    ObjectAllocator();

    explicit ObjectAllocator(Buffer region);

    // No longer copyable: trials use ScratchAllocator (copies only capacity).
    ObjectAllocator(const ObjectAllocator&)            = delete;
    ObjectAllocator& operator=(const ObjectAllocator&) = delete;
    ObjectAllocator(ObjectAllocator&&) noexcept;
    ObjectAllocator& operator=(ObjectAllocator&&) noexcept;

    int Register(size_t size, size_t alignment);
    int Register(const std::vector<std::array<size_t, 3>>& parts);

    // Single-object fast path (primary; production count is always 1).
    [[nodiscard]] object_alloc_t Allocate(int index);  // {nullptr} on OOM
    void                         Deallocate(int index, object_alloc_t handle);

    // Batch forms (thin loops over the single-object core; used by tests).
    [[nodiscard]] size_t Allocate(int index, object_alloc_t* objects, size_t count);
    void                 Deallocate(int index, const object_alloc_t* objects, size_t count);

    // Registry queries (need the index, not the handle).
    int    PartCount(int index) const;
    size_t PartBytes(int index, int part) const;

    // ABA-safe stale check: handle.a && handle->key == saved_key.
    [[nodiscard]] bool IsValid(object_alloc_t handle, uint64_t saved_key) const;

    MemoryStats Stats() const;

private:
    friend class ScratchAllocator;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Admission probe: owns a copy of the committed allocator's capacity
// (MemoryState) and borrows the live allocator for object layout and to read a
// committed Allocation's slot list. Never touches an AllocationTable.
//
// LIFETIME: a ScratchAllocator borrows its source ObjectAllocator by pointer.
// It MUST NOT outlive that source, and the source MUST NOT be moved-from while
// the scratch is alive.
class ScratchAllocator {
public:
    ScratchAllocator();
    explicit ScratchAllocator(const ObjectAllocator& src);
    ~ScratchAllocator();

    ScratchAllocator(const ScratchAllocator&)            = delete;
    ScratchAllocator& operator=(const ScratchAllocator&) = delete;
    ScratchAllocator(ScratchAllocator&&) noexcept;
    ScratchAllocator& operator=(ScratchAllocator&&) noexcept;

    // Reserve one object's slots in the capacity copy; false on OOM.
    [[nodiscard]] bool Allocate(int object_id);
    // Free a committed Allocation's slots in the capacity copy.
    void Evict(int object_id, const Allocation* committed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace turbomind
