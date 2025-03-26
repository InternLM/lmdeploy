#pragma once

#include "engine/config.h"
#include "utils/json.hpp"

#include <cstdint>
#include <cstdlib>
#include <infiniband/verbs.h>
#include <string>
#include <sys/types.h>
#include <unordered_map>

namespace migration {

using json = nlohmann::json;

class MemoryPool {
public:
    MemoryPool() = default;
    MemoryPool(ibv_pd* pd): pd_(pd) {}

    int register_memory_region(std::string mr_key, uintptr_t data_ptr, uint64_t length);
    int unregister_memory_region(std::string mr_key);

    int register_remote_memory_region(std::string mr_key, json mr_info);
    int unregister_remote_memory_region(std::string mr_key);

    uint32_t get_lkey(std::string mr_key);
    uint32_t get_rkey(std::string mr_key);

    json mr_info();
    json remote_mr_info();

private:
    ibv_pd*                                         pd_;
    std::unordered_map<std::string, struct ibv_mr*> mrs_;
    std::unordered_map<std::string, json>           remote_mrs_;
};
}  // namespace migration
