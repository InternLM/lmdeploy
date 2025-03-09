#pragma once

#include <torch/extension.h>

#include <migration_manager.cuh>

#include <cstdint>

#include <map>
#include <string>
#include <vector>

using fptr_t = uint64_t;

void migrate(fptr_t manager_ptr, std::vector<torch::Tensor> &kv_caches,
             const torch::Tensor &block_mapping, int64_t decode_engine_rank);

fptr_t init_migration_manager(
    int64_t engine_id, int64_t num_bytes_per_elem, int64_t num_layers,
    int64_t total_num_heads, int64_t head_size,
    std::unordered_map<int64_t, KVCacheHandlerConfig> &kv_cache_ipc_handlers);
