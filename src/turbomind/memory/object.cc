#include "src/turbomind/memory/object.h"
#include "src/turbomind/memory/slab.h"

#include <deque>
#include <unordered_map>

namespace turbomind {

// ---- AllocationTable: the handle store (one backend) ----
// The handle (object_alloc_t) wraps a const Allocation* into an address-stable
// std::deque; resolving is a field read. Staleness is a monotonic per-Allocation
// key: Acquire assigns the next key, Release zeroes it; a consumer that snapshots
// the key detects reuse/free via ObjectAllocator::IsValid.
struct AllocationTable {
    std::deque<Allocation>   pool_;
    std::vector<Allocation*> free_;
    uint64_t                 next_key_{1};  // never 0, never reused -> ABA-proof

    Allocation* Acquire()
    {
        Allocation* a;
        if (free_.empty()) {
            a = &pool_.emplace_back();
        }
        else {
            a = free_.back();
            free_.pop_back();
        }
        a->key = next_key_++;
        return a;
    }

    void Release(Allocation* a)
    {
        a->key   = 0;
        a->n     = 0;
        a->base0 = nullptr;
        a->bases.clear();  // keep capacity for reuse
        a->slots.clear();
        free_.push_back(a);
    }
};

// ---- Static registration descriptors ----

struct MemberSpec {
    int slab_index;
    int count;
};

struct ObjectSpec {
    std::vector<MemberSpec> members;
    int                     total_parts{0};
    std::vector<int>        part_slab;  // positional part -> SlabAllocator index
};

// ---- MemoryState: capacity layer (the only thing a trial copies) ----

struct MemoryState {
    static constexpr size_t kPageSize      = 32 << 20UL;
    static constexpr size_t kMinSlabSize   = 32 << 20UL;
    static constexpr size_t kMaxSlabSize   = 1 << 30UL;
    static constexpr size_t kMaxEmptySlabs = 0;
    static constexpr float  kUtilThresh    = .95f;

    Buffer                     mem_;
    PageAllocator              pages_;
    std::vector<SlabAllocator> slabs_;

    explicit MemoryState(Buffer memory):
        mem_{std::move(memory)}, pages_{mem_.raw_data(), (size_t)mem_.byte_size(), kPageSize}
    {
    }

    int add_slab_class(size_t aligned)
    {
        const int slab = static_cast<int>(slabs_.size());
        slabs_.emplace_back(aligned, kMinSlabSize, kMaxSlabSize, kUtilThresh, kMaxEmptySlabs);
        return slab;
    }

    // Reserve spec.total_parts slots into `out`, atomic over members.
    bool reserve(const ObjectSpec& spec, SlabSlot* out)
    {
        int filled = 0;
        for (const MemberSpec& m : spec.members) {
            const int got = slabs_[m.slab_index].allocate(out + filled, m.count, pages_);
            if (got < m.count) {
                slabs_[m.slab_index].deallocate(out + filled, got, pages_);
                int back = 0;
                for (const MemberSpec& mm : spec.members) {
                    if (&mm == &m) {
                        break;
                    }
                    slabs_[mm.slab_index].deallocate(out + back, mm.count, pages_);
                    back += mm.count;
                }
                return false;
            }
            filled += m.count;
        }
        return true;
    }

    void free(const ObjectSpec& spec, const SlabSlot* slots)
    {
        int off = 0;
        for (const MemberSpec& m : spec.members) {
            slabs_[m.slab_index].deallocate(slots + off, m.count, pages_);
            off += m.count;
        }
    }

    char* address_of(int part_slab, SlabSlot s) const
    {
        return slabs_[part_slab].AddressOf(s);
    }
    void set_owner(int part_slab, SlabSlot s, object_alloc_t o)
    {
        slabs_[part_slab].set_owner(s, o);
    }
};

// ---- ObjectAllocator::Impl ----

struct ObjectAllocator::Impl {
    MemoryState                     space_;
    AllocationTable                 table_;
    std::vector<ObjectSpec>         objects_;
    std::unordered_map<size_t, int> slab_of_size_;
    std::unordered_map<size_t, int> simple_id_;

    explicit Impl(Buffer memory): space_{std::move(memory)} {}

    static size_t align_up(size_t size, size_t align)
    {
        return (size + align - 1) / align * align;
    }

    int slab_for_aligned_size(size_t aligned)
    {
        if (auto it = slab_of_size_.find(aligned); it != slab_of_size_.end()) {
            return it->second;
        }
        const int slab         = space_.add_slab_class(aligned);
        slab_of_size_[aligned] = slab;
        return slab;
    }

    int register_simple(size_t size, size_t align)
    {
        const size_t aligned = align_up(size, align);
        if (auto it = simple_id_.find(aligned); it != simple_id_.end()) {
            return it->second;
        }
        const int  slab = slab_for_aligned_size(aligned);
        const int  id   = static_cast<int>(objects_.size());
        ObjectSpec spec;
        spec.members     = {{slab, 1}};
        spec.total_parts = 1;
        spec.part_slab   = {slab};
        objects_.push_back(std::move(spec));
        simple_id_[aligned] = id;
        return id;
    }

    int register_composite(const std::vector<std::array<size_t, 3>>& parts)
    {
        TM_CHECK(!parts.empty()) << "composite must have at least one member";
        if (parts.size() == 1 && parts[0][2] == 1) {
            return register_simple(parts[0][0], parts[0][1]);
        }
        ObjectSpec spec;
        for (const auto& p : parts) {
            const size_t size = p[0], align = p[1], count = p[2];
            TM_CHECK_GT(count, 0u);
            const size_t aligned = align_up(size, align);
            const int    slab    = slab_for_aligned_size(aligned);
            spec.members.push_back({slab, static_cast<int>(count)});
            for (size_t k = 0; k < count; ++k) {
                spec.part_slab.push_back(slab);
            }
            spec.total_parts += static_cast<int>(count);
        }
        const int id = static_cast<int>(objects_.size());
        objects_.push_back(std::move(spec));
        return id;
    }

    // Single-object fast path. `index` is trusted (it originates from a
    // registered CacheBlock::object_id) -- no check_index on the hot path.
    object_alloc_t allocate(int index)
    {
        const ObjectSpec& spec = objects_[index];
        Allocation*       a    = table_.Acquire();

        if (spec.total_parts == 1) {  // single-part fast path: no loops, no heap vectors
            const int slab = spec.part_slab[0];
            if (space_.slabs_[slab].allocate(&a->slot0, 1, space_.pages_) != 1) {
                table_.Release(a);
                return {};
            }
            a->n     = 1;
            a->base0 = space_.slabs_[slab].AddressOf(a->slot0);
            space_.slabs_[slab].set_owner(a->slot0, object_alloc_t{a});
            return {a};
        }

        a->n = spec.total_parts;  // composite
        a->bases.resize(a->n);
        a->slots.resize(a->n);
        if (!space_.reserve(spec, a->slots.data())) {
            table_.Release(a);
            return {};
        }
        for (int p = 0; p < a->n; ++p) {
            a->bases[p] = space_.address_of(spec.part_slab[p], a->slots[p]);
            space_.set_owner(spec.part_slab[p], a->slots[p], object_alloc_t{a});
        }
        return {a};
    }

    void deallocate(int index, object_alloc_t h)
    {
        const ObjectSpec& spec = objects_[index];
        Allocation*       a    = const_cast<Allocation*>(h.a);  // we own it in pool_
        if (spec.total_parts == 1) {
            space_.slabs_[spec.part_slab[0]].deallocate(&a->slot0, 1, space_.pages_);
        }
        else {
            space_.free(spec, a->slots.data());
        }
        table_.Release(a);
    }

    // Batch forms (used by the unit tests): thin loops over the single-object core.
    size_t allocate(int index, object_alloc_t* out, size_t count)
    {
        for (size_t k = 0; k < count; ++k) {
            out[k] = allocate(index);
            if (!out[k].a) {
                return k;  // [0,k) placed; matches the partial-OOM contract
            }
        }
        return count;
    }

    void deallocate(int index, const object_alloc_t* objs, size_t count)
    {
        for (size_t k = 0; k < count; ++k) {
            deallocate(index, objs[k]);
        }
    }

    int part_count(int index) const
    {
        return objects_[index].total_parts;
    }

    size_t part_bytes(int index, int part) const
    {
        return space_.slabs_[objects_[index].part_slab[part]].object_size();
    }

    MemoryStats stats() const
    {
        MemoryStats s;
        s.page             = space_.pages_.stats();
        s.region_bytes     = static_cast<size_t>(space_.mem_.byte_size());
        s.live_allocations = table_.pool_.size() - table_.free_.size();
        s.slabs.reserve(space_.slabs_.size());
        for (const SlabAllocator& slab : space_.slabs_) {
            s.slabs.push_back(slab.stats());
        }
        return s;
    }
};

// ---- ScratchAllocator::Impl ----

struct ScratchAllocator::Impl {
    MemoryState            space_;  // deep copy of source capacity (shares the backing Buffer)
    const ObjectAllocator* src_;    // borrow: the registry (objects_) lives in src's Impl
    std::vector<SlabSlot>  scratch_;

    Impl(MemoryState space, const ObjectAllocator* src): space_{std::move(space)}, src_{src} {}
};

// ---- ObjectAllocator public surface ----

ObjectAllocator::~ObjectAllocator() = default;
ObjectAllocator::ObjectAllocator()  = default;

ObjectAllocator::ObjectAllocator(Buffer region): impl_{std::make_unique<Impl>(std::move(region))} {}

ObjectAllocator::ObjectAllocator(ObjectAllocator&&) noexcept = default;
ObjectAllocator& ObjectAllocator::operator=(ObjectAllocator&&) noexcept = default;

int ObjectAllocator::Register(size_t size, size_t alignment)
{
    return impl_->register_simple(size, alignment);
}
int ObjectAllocator::Register(const std::vector<std::array<size_t, 3>>& parts)
{
    return impl_->register_composite(parts);
}
object_alloc_t ObjectAllocator::Allocate(int index)
{
    return impl_->allocate(index);
}
void ObjectAllocator::Deallocate(int index, object_alloc_t handle)
{
    impl_->deallocate(index, handle);
}
size_t ObjectAllocator::Allocate(int index, object_alloc_t* objects, size_t count)
{
    return impl_->allocate(index, objects, count);
}
void ObjectAllocator::Deallocate(int index, const object_alloc_t* objects, size_t count)
{
    impl_->deallocate(index, objects, count);
}
int ObjectAllocator::PartCount(int index) const
{
    return impl_->part_count(index);
}
size_t ObjectAllocator::PartBytes(int index, int part) const
{
    return impl_->part_bytes(index, part);
}
bool ObjectAllocator::IsValid(object_alloc_t handle, uint64_t saved_key) const
{
    return handle.a != nullptr && handle->key == saved_key;
}

MemoryStats ObjectAllocator::Stats() const
{
    return impl_->stats();
}

// ---- ScratchAllocator public surface ----

ScratchAllocator::ScratchAllocator(const ObjectAllocator& src): impl_{std::make_unique<Impl>(src.impl_->space_, &src)}
{
}

ScratchAllocator::ScratchAllocator(ScratchAllocator&&) noexcept = default;
ScratchAllocator& ScratchAllocator::operator=(ScratchAllocator&&) noexcept = default;
ScratchAllocator::ScratchAllocator()                                       = default;
ScratchAllocator::~ScratchAllocator()                                      = default;

bool ScratchAllocator::Allocate(int object_id)
{
    const ObjectSpec& spec = impl_->src_->impl_->objects_[object_id];  // friend reaches registry
    if (spec.total_parts == 1) {
        SlabSlot s;
        return impl_->space_.slabs_[spec.part_slab[0]].allocate(&s, 1, impl_->space_.pages_) == 1;
    }
    impl_->scratch_.resize(spec.total_parts);
    return impl_->space_.reserve(spec, impl_->scratch_.data());
}
void ScratchAllocator::Evict(int object_id, const Allocation* committed)
{
    const ObjectSpec& spec = impl_->src_->impl_->objects_[object_id];
    if (committed->n == 1) {
        impl_->space_.slabs_[spec.part_slab[0]].deallocate(&committed->slot0, 1, impl_->space_.pages_);
    }
    else {
        impl_->space_.free(spec, committed->slots.data());
    }
}

}  // namespace turbomind
