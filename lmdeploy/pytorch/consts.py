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


def v4_packed_index_cache_shape(entries_per_block: int, head_dim: int) -> tuple[int, int, int]:
    """Return the logical uint8 shape for the packed V4 index cache."""
    return (entries_per_block, 1, head_dim + V4_INDEX_SCALE_BYTES)
