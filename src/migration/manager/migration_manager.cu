#include <ATen/cuda/CUDAContext.h> // for at::cuda::getCurrentCUDAStream()
#include <torch/extension.h>

#include <vector>

#include <cuda.h>
// #include <nccl.h>

#include <ops.h>

#include <migration_manager.cuh>

void migrate(
    fptr_t manager_ptr,
    std::vector<torch::Tensor> &kv_caches,
    const torch::Tensor &block_mapping,
    int64_t decode_engine_rank
) {
  if (manager_ptr) {
    auto manager = reinterpret_cast<KVCacheMigrationManager *>(manager_ptr);
    auto model_config = manager->model_config;
    auto kv_cache_ipc_handlers = manager->kv_cache_ipc_handlers;

    int64_t decode_engine_id = manager->engine_id;
    auto engine_kv_cache_ipc_handler_decode =
        kv_cache_ipc_handlers[decode_engine_id];
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int64_t num_blocks = block_mapping.size(0);
    for (int i = 0; i < num_blocks; i++) {
      int64_t prefill_engine_id = block_mapping[i][0].item<int64_t>();
      int64_t prefill_virtual_engine_id = block_mapping[i][1].item<int64_t>();
      auto engine_kv_cache_ipc_handler_prefill =
          kv_cache_ipc_handlers[prefill_engine_id];
      const int64_t num_pp_size_prefill =
          engine_kv_cache_ipc_handler_prefill.pp_size;
      const int64_t num_pp_size_decode =
          engine_kv_cache_ipc_handler_decode.pp_size;

      const int64_t num_tp_size_prefill =
          engine_kv_cache_ipc_handler_prefill.tp_size;
      const int64_t num_tp_size_decode =
          engine_kv_cache_ipc_handler_decode.tp_size;

      const int64_t pp_rank = decode_engine_rank / num_tp_size_decode;
      const int64_t tp_rank = decode_engine_rank % num_tp_size_decode;

      const int64_t num_bytes_per_elem = model_config.num_bytes_per_elem;

      const int64_t num_layers = model_config.num_layers;

      const int64_t num_attention_layers_prefill =
          num_layers / num_pp_size_prefill;
      const int64_t num_attention_layers_decode =
          num_layers / num_pp_size_decode;

      const int64_t block_size = engine_kv_cache_ipc_handler_prefill.block_size;

      const int64_t total_num_heads = model_config.total_num_heads;
      const int64_t head_size = model_config.head_size;

      const int64_t num_total_blocks_decode =
          engine_kv_cache_ipc_handler_decode.num_total_blocks;

      const int64_t local_num_heads_prefill =
          total_num_heads / num_tp_size_prefill;
      const int64_t local_num_heads_decode =
          total_num_heads / num_tp_size_decode;

      const int64_t width_prefill = head_size * local_num_heads_prefill;

      const int64_t width_decode = head_size * local_num_heads_decode;
      const int64_t height_decode = num_total_blocks_decode * block_size;

      const int64_t num_head_pitch =
          local_num_heads_prefill > local_num_heads_decode
              ? local_num_heads_decode
              : local_num_heads_prefill;

      const int64_t pitch = num_head_pitch * head_size;
      const int64_t slice_pitch = block_size;

      const int64_t prefill_instance_idx_from =
          num_tp_size_prefill * tp_rank / num_tp_size_decode;
      const int64_t prefill_head_idx_from =
          (tp_rank * local_num_heads_decode) % local_num_heads_prefill;

      const int64_t layer_pitch =
          num_attention_layers_decode < num_attention_layers_prefill
              ? num_attention_layers_decode
              : num_attention_layers_prefill;

      const int64_t prefill_pp_idx_from =
          (pp_rank * num_attention_layers_decode) /
          num_attention_layers_prefill;
      const int64_t prefill_layer_idx_from =
          (pp_rank * num_attention_layers_decode) %
          num_attention_layers_prefill;

      for (int64_t transformed_num_layers = 0;
           transformed_num_layers < num_attention_layers_decode;
           transformed_num_layers += layer_pitch) {
        int64_t prefill_pp_idx =
            prefill_pp_idx_from +
            (prefill_layer_idx_from + transformed_num_layers) /
                num_attention_layers_prefill;
        int64_t prefill_layer_idx =
            (prefill_layer_idx_from + transformed_num_layers) %
            num_attention_layers_prefill;

        auto decode_instance_ptr = (char *)(kv_caches[0].data_ptr());

        for (int64_t transformed_num_heads = 0;
             transformed_num_heads < local_num_heads_decode;
             transformed_num_heads += num_head_pitch) {

          const int64_t num_total_blocks_prefill =
              engine_kv_cache_ipc_handler_prefill.num_total_blocks;
          const int64_t height_prefill = num_total_blocks_prefill * block_size;

          int64_t height_offset_prefill =
              block_mapping[i][3].item<int64_t>() * block_size;
          int64_t height_offset_decode =
              block_mapping[i][2].item<int64_t>() * block_size;
          int64_t prefill_instance_idx =
              prefill_instance_idx_from +
              (prefill_head_idx_from + transformed_num_heads) /
                  local_num_heads_prefill;
          int64_t prefill_head_idx =
              (prefill_head_idx_from + transformed_num_heads) %
              local_num_heads_prefill;

          auto prefill_instance_ptr =
              engine_kv_cache_ipc_handler_prefill
                  .engine_kv_cache_ipc_handles[prefill_pp_idx]
                                              [prefill_instance_idx]
                                              [prefill_virtual_engine_id][0];

          int64_t width_offset_prefill = prefill_head_idx * head_size;
          int64_t width_offset_decode = transformed_num_heads * head_size;

          cudaMemcpy3DParms copyParams = {0};

          copyParams.srcPtr = make_cudaPitchedPtr(
              prefill_instance_ptr, width_prefill * num_bytes_per_elem,
              width_prefill * num_bytes_per_elem, height_prefill);
          copyParams.srcPos =
              make_cudaPos(width_offset_prefill * num_bytes_per_elem,
                           height_offset_prefill, 2 * prefill_layer_idx);

          copyParams.dstPtr = make_cudaPitchedPtr(
              decode_instance_ptr, width_decode * num_bytes_per_elem,
              width_decode * num_bytes_per_elem, height_decode);
          copyParams.dstPos =
              make_cudaPos(width_offset_decode * num_bytes_per_elem,
                           height_offset_decode, 2 * transformed_num_layers);

          copyParams.extent = make_cudaExtent(pitch * num_bytes_per_elem,
                                              slice_pitch, layer_pitch);
          copyParams.kind = cudaMemcpyDeviceToDevice;

          cudaMemcpy3DAsync(&copyParams, stream);
        }
      }
    }
  }
}

fptr_t init_migration_manager(
    int64_t engine_id, int64_t num_bytes_per_elem, int64_t num_layers,
    int64_t total_num_heads, int64_t head_size,
    std::unordered_map<int64_t, KVCacheHandlerConfig> &kv_cache_ipc_handlers) {
  return (fptr_t) new KVCacheMigrationManager(
      engine_id,
      MigrationModelConfig(num_bytes_per_elem, num_layers, total_num_heads,
                           head_size),
      kv_cache_ipc_handlers);
}
