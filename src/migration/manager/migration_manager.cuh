#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <vector>
#include <unordered_map>

#include "cuda_common.h"

std::vector<char> serialize_cuda_ipc_handle(const cudaIpcMemHandle_t& handle); 
cudaIpcMemHandle_t deserialize_from_vector(const std::vector<char>& buffer);

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

struct KVCacheHandlerConfig {
  KVCacheHandlerConfig() {}
  KVCacheHandlerConfig(int64_t tp_size, int64_t pp_size,
                       int64_t num_total_blocks, int64_t block_size)
      : tp_size(tp_size), pp_size(pp_size), num_total_blocks(num_total_blocks),
        block_size(block_size) {}
  KVCacheHandlerConfig(
      int64_t tp_size, int64_t pp_size, int64_t num_total_blocks,
      int64_t block_size,
      std::vector<std::vector<std::vector<std::vector<char>>>> &kv_cache_handlers,
      std::vector<std::vector<std::vector<int64_t>>> &kv_cache_offsets)
      : KVCacheHandlerConfig(tp_size, pp_size, num_total_blocks, block_size) {
    for (int i = 0; i < pp_size; i++) {
      std::vector<std::vector<std::vector<char *>>> pp_ipc_handlers;
      for (int j = 0; j < tp_size; j++) {
        std::vector<std::vector<char *>> tp_ipc_handlers;
        for (int k = 0; k < pp_size; k++) {
          std::vector<char *> virtual_engine_handlers;
          auto num_attention_layers =
              kv_cache_offsets[i * tp_size + j][k].size();
          for (int l = 0; l < num_attention_layers; l++) {
            int64_t offset = kv_cache_offsets[i * tp_size + j][k][l];
            cudaIpcMemHandle_t handler_ptr = deserialize_from_vector(kv_cache_handlers[i * tp_size + j][k][l]);
            char *ipc_ptr;
            CUDACHECK(cudaIpcOpenMemHandle((void **)&ipc_ptr, handler_ptr, cudaIpcMemLazyEnablePeerAccess));
            ipc_ptr += offset;
            virtual_engine_handlers.push_back(ipc_ptr);
          }
          tp_ipc_handlers.push_back(virtual_engine_handlers);
        }
        pp_ipc_handlers.push_back(tp_ipc_handlers);
      }
      engine_kv_cache_ipc_handles.push_back(pp_ipc_handlers);
    }
  }
  int64_t tp_size;
  int64_t pp_size;
  int64_t num_total_blocks;
  int64_t block_size;
  // num_pp_size, num_tp_size, virtual_engine, num_attention_layers
  std::vector<std::vector<std::vector<std::vector<char *>>>>
      engine_kv_cache_ipc_handles;
};

struct MigrationModelConfig {
  MigrationModelConfig(int64_t num_bytes_per_elem, int64_t num_layers,
                       int64_t total_num_heads, int64_t head_size)
      : num_bytes_per_elem(num_bytes_per_elem), num_layers(num_layers),
        total_num_heads(total_num_heads), head_size(head_size) {}
  // dtype config
  int64_t num_bytes_per_elem;
  // shape config
  int64_t num_layers;
  int64_t total_num_heads;
  int64_t head_size;
};

struct KVCacheMigrationManager {
  KVCacheMigrationManager(
      int64_t engine_id, MigrationModelConfig config,
      std::unordered_map<int64_t, KVCacheHandlerConfig> &kv_cache_ipc_handlers)
      : engine_id(engine_id), model_config(std::move(config)), kv_cache_ipc_handlers(std::move(kv_cache_ipc_handlers)) {
  }
  int64_t engine_id;
  MigrationModelConfig model_config;
  // num_pp_size, num_tp_size, virtual_engine, num_attention_layers
  std::unordered_map<int64_t, KVCacheHandlerConfig> kv_cache_ipc_handlers;
};
