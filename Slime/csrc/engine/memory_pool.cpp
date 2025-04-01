#include "engine/memory_pool.h"
#include "utils/logging.h"

#include <cstdint>
#include <cstdlib>
#include <infiniband/verbs.h>
#include <sys/types.h>
#include <unordered_map>

namespace slime {
int MemoryPool::register_memory_region(std::string mr_key, uintptr_t data_ptr, uint64_t length)
{
    /* MemoryRegion Access Right = 777 */
    const static int access_rights = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    ibv_mr*          mr            = ibv_reg_mr(pd_, (void*)data_ptr, length, access_rights);

    SLIME_ASSERT(mr, " Failed to register memory " << data_ptr);

    SLIME_LOG_INFO("Memory region: " << (void*)data_ptr << " -- " << (void*)(data_ptr + length)
                                     << ", Device name: " << pd_->context->device->dev_name << ", Length: " << length
                                     << " (" << length / 1024 / 1024 << " MB)"
                                     << ", Permission: " << access_rights << ", LKey: " << mr->lkey
                                     << ", RKey: " << mr->rkey);

    mrs_[mr_key] = mr;
    return 0;
}

int MemoryPool::unregister_memory_region(std::string mr_key)
{
    mrs_.erase(mr_key);
    return 0;
}

int MemoryPool::register_remote_memory_region(std::string mr_key, json mr_info)
{
    remote_mrs_[mr_key] = mr_info;
    return 0;
}

int MemoryPool::unregister_remote_memory_region(std::string mr_key)
{
    remote_mrs_.erase(mr_key);
    return 0;
}

json MemoryPool::mr_info()
{
    json mr_info;
    for (auto& mr : mrs_) {
        mr_info[mr.first] = {
            {"addr", (uintptr_t)mr.second->addr},
            {"rkey", mr.second->rkey},
            {"length", mr.second->length},
        };
    }
    return mr_info;
}

json MemoryPool::remote_mr_info()
{
    json mr_info;
    for (auto& mr : remote_mrs_) {
        mr_info[mr.first] = remote_mrs_[mr.second];
    }
    return mr_info;
}

}  // namespace slime
