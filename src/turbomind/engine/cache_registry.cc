#include "src/turbomind/engine/cache_registry.h"

namespace turbomind {

void CacheCategory::RegisterObjectId(ObjectAllocator& allocator)
{
    TM_CHECK_LT(object_id_, 0);
    if (used()) {
        object_id_ = allocator.Register(effective_parts());
    }
}

void CacheRegistry::RegisterObjectIds(ObjectAllocator& allocator)
{
    prefix_.RegisterObjectId(allocator);
    checkpoint_.RegisterObjectId(allocator);
}

}  // namespace turbomind
