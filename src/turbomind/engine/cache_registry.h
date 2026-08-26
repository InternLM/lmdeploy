#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>

#include "src/turbomind/core/check.h"
#include "src/turbomind/memory/object.h"

namespace turbomind {

// Per-category composite object layout and ObjectAllocator registration.
// Modules claim [offset, offset + bytes) during model construction and keep
// only the returned byte offset. All cache policy lives in the scheduler.
class CacheCategory {
public:
    // Accumulation channel: claims [offset, offset + bytes) in part 0 and returns
    // the within-part byte offset. Modules keep only the returned offset.
    size_t Register(size_t bytes, size_t alignment)
    {
        TM_CHECK_LT(object_id_, 0);  // registration closes once the object id exists
        TM_CHECK_GT(bytes, 0);
        TM_CHECK_GT(alignment, 0);

        auto&            acc      = parts_[0];  // {bytes, alignment, count=1}
        constexpr size_t kMaxSize = std::numeric_limits<size_t>::max();
        TM_CHECK_LE(alignment - 1, kMaxSize - acc[0]);

        const size_t offset = (acc[0] + alignment - 1) / alignment * alignment;
        TM_CHECK_LE(bytes, kMaxSize - offset);

        acc[0] = offset + bytes;
        acc[1] = std::max(acc[1], alignment);
        return offset;
    }

    // Composite channel: appends sum(count) parts of the given aligned sizes and
    // returns the base part id (>= 1). Part ids count expanded parts.
    int Register(const std::vector<std::array<size_t, 3>>& parts)
    {
        TM_CHECK_LT(object_id_, 0);
        const int base = next_part_id_;
        for (const auto& p : parts) {
            TM_CHECK_GT(p[0], 0);
            TM_CHECK_GT(p[1], 0);
            TM_CHECK_GT(p[2], 0);
            parts_.push_back(p);
            next_part_id_ += static_cast<int>(p[2]);
        }
        return base;
    }

    void RegisterObjectId(ObjectAllocator& allocator);

    int object_id() const
    {
        TM_CHECK_GE(object_id_, 0);
        return object_id_;
    }

    int object_id_or_negative() const noexcept
    {
        return object_id_;
    }

    // Bytes accumulated into part 0 (accumulation channel). For the prefix
    // category this is the simple object's byte size; 0 before any Register.
    size_t accumulation_bytes() const noexcept
    {
        return parts_[0][0];
    }

    bool used() const noexcept
    {
        return parts_[0][0] != 0 || next_part_id_ > 1;  // accumulation non-empty or any composite part
    }

    // Parts list for ObjectAllocator::Register: accumulation part 0 first (when
    // non-empty), then composite parts in registration order. Part ids start at
    // 1 for composites, so part 0 must be present whenever composites exist or
    // positional indices would shift and base(part_id) would be wrong.
    std::vector<std::array<size_t, 3>> effective_parts() const
    {
        const bool has_acc       = parts_[0][0] != 0;
        const bool has_composite = next_part_id_ > 1;
        TM_CHECK(!has_composite || has_acc) << "composite category requires a non-empty accumulation part 0";

        std::vector<std::array<size_t, 3>> out;
        if (has_acc) {
            out.push_back(parts_[0]);
        }
        for (size_t k = 1; k < parts_.size(); ++k) {
            out.push_back(parts_[k]);
        }
        return out;
    }

private:
    // parts_[0] is the accumulation part (id 0): {bytes, alignment, count=1}.
    // parts_[1..] are composite member specs in registration order.
    std::vector<std::array<size_t, 3>> parts_{{0, 1, 1}};
    int                                next_part_id_{1};  // next composite part id (counts expanded parts)
    int                                object_id_{-1};
};

class CacheRegistry {
public:
    CacheCategory& prefix() noexcept
    {
        return prefix_;
    }

    const CacheCategory& prefix() const noexcept
    {
        return prefix_;
    }

    CacheCategory& checkpoint() noexcept
    {
        return checkpoint_;
    }

    const CacheCategory& checkpoint() const noexcept
    {
        return checkpoint_;
    }

    // True once a module registered checkpoint bytes and the object id exists.
    bool has_checkpoint() const noexcept
    {
        return checkpoint_.object_id_or_negative() >= 0;
    }

    int checkpoint_min_interval() const noexcept
    {
        return checkpoint_min_interval_;
    }

    void set_checkpoint_min_interval(int interval)
    {
        TM_CHECK_GT(interval, 0);
        checkpoint_min_interval_ = interval;
    }

    void RegisterObjectIds(ObjectAllocator& allocator);

private:
    CacheCategory prefix_;
    CacheCategory checkpoint_;

    int checkpoint_min_interval_{1};
};

}  // namespace turbomind
