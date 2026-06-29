// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cstdint>
#include <functional>

namespace turbomind {

struct Allocation;  // defined in object.h; the handle is just a pointer to it

// Opaque 8-byte allocation handle: a pointer to the durable Allocation that
// owns the data pointer(s). {nullptr} == null / never-allocated. Staleness is
// detected by snapshotting Allocation::key (see ObjectAllocator::IsValid), not
// by this pointer.
struct object_alloc_t {
    const Allocation* a{};
    const Allocation* operator->() const noexcept
    {
        return a;
    }
};

inline bool operator==(object_alloc_t x, object_alloc_t y) noexcept
{
    return x.a == y.a;
}
inline bool operator!=(object_alloc_t x, object_alloc_t y) noexcept
{
    return x.a != y.a;
}
inline bool operator<(object_alloc_t x, object_alloc_t y) noexcept
{
    return std::less<const Allocation*>{}(x.a, y.a);
}

// A pure slab coordinate: which Slab (within a SlabAllocator) and which slot.
// No identity/stamp -- staleness is owned by the Allocation key.
struct SlabSlot {
    int32_t slab_id;
    int32_t slot_id;
};

static_assert(sizeof(object_alloc_t) == 8, "object_alloc_t must be 8 bytes");
static_assert(alignof(object_alloc_t) == 8, "object_alloc_t must be 8-byte aligned");

}  // namespace turbomind
