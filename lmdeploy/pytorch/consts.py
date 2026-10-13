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

# GLM-5.3 hybrid sequence-state resources.  These names are shared by its
# config declaration and model consumer so they cannot silently drift back to
# anonymous positional state caches.
GLM5_KDA_CONV_STATE = 'glm5_kda_conv'
GLM5_KDA_RECURRENT_STATE = 'glm5_kda_recurrent'
GLM5_KPOOL_TAIL_K_STATE = 'glm5_kpool_tail_k'
GLM5_KPOOL_TAIL_SCORE_STATE = 'glm5_kpool_tail_score'
# Gated DeltaNet ReplaySSM sequence-state cache names.  These names are part
# of the model/cache-engine contract; keep them stable across implementations.
GDN_CONV_STATE_CACHE_NAME = 'gdn_conv_state'
GDN_REPLAY_CHECKPOINT_CACHE_NAME = 'gdn_replay_checkpoint'
GDN_REPLAY_D_CACHE_NAME = 'gdn_replay_d'
GDN_REPLAY_K_CACHE_NAME = 'gdn_replay_k'
GDN_REPLAY_G_CACHE_NAME = 'gdn_replay_g'
GDN_REPLAY_RAW_V_CACHE_NAME = 'gdn_replay_raw_v'
GDN_REPLAY_RAW_K_CACHE_NAME = 'gdn_replay_raw_k'
GDN_REPLAY_BETA_CACHE_NAME = 'gdn_replay_beta'
GDN_REPLAY_CURSOR_CACHE_NAME = 'gdn_replay_cursor'


def v4_packed_index_cache_shape(entries_per_block: int, head_dim: int) -> tuple[int, int, int]:
    """Return the logical uint8 shape for the packed V4 index cache."""
    return (entries_per_block, 1, head_dim + V4_INDEX_SCALE_BYTES)


def dsa_packed_indexer_k_cache_shape(entries_per_block: int, head_dim: int) -> tuple[int, int, int]:
    """Return DeepGEMM's packed uint8 DSA block shape.

    Raw block layout: ``[all FP8 K][one FP32 scale per entry]``.
    """
    return (entries_per_block, 1, head_dim + DSA_INDEX_SCALE_BYTES)
