# Copyright (c) OpenMMLab. All rights reserved.
# dllm
DLLM_MASKED = 0
DLLM_UNMASKED = 1
DLLM_CACHED = 2

# DeepSeek-V4 FlashMLA sparse FP8 layout constants
V4_FLASHMLA_HEAD_DIM = 512
V4_FLASHMLA_D_NOPE = 448
V4_FLASHMLA_D_ROPE = 64
V4_FLASHMLA_TILE_SIZE = 64
V4_FLASHMLA_NUM_TILES = 7
V4_INDEX_SCALE_BYTES = 4
V4_PACKED_TOKEN_DIM = V4_FLASHMLA_D_NOPE + 2 * V4_FLASHMLA_D_ROPE + V4_FLASHMLA_NUM_TILES + 1
V4_COMPRESSED_KV_R4_CACHE_NAME = 'v4_compressed_kv_r4_fp8'
V4_COMPRESSED_KV_R128_CACHE_NAME = 'v4_compressed_kv_r128_fp8'
V4_INDEX_KV_R4_CACHE_NAME = 'v4_index_kv_r4'
V4_INDEX_KV_R4_SCALE_CACHE_NAME = 'v4_index_kv_r4_scale'
DSA_INDEXER_K_CACHE_NAME = 'dsa_indexer_k'
DSA_INDEX_SCALE_BYTES = 4


def v4_packed_index_cache_shape(entries_per_block: int, head_dim: int) -> tuple[int, int, int]:
    """Return the logical uint8 shape for the packed V4 index cache."""
    return (entries_per_block, 1, head_dim + V4_INDEX_SCALE_BYTES)


def dsa_packed_indexer_k_cache_shape(entries_per_block: int, head_dim: int) -> tuple[int, int, int]:
    """Return DeepGEMM's packed uint8 DSA block shape.

    Raw block layout: ``[all FP8 K][one FP32 scale per entry]``.
    """
    return (entries_per_block, 1, head_dim + DSA_INDEX_SCALE_BYTES)
