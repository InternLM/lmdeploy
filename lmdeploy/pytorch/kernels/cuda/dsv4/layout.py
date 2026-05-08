# Copyright (c) OpenMMLab. All rights reserved.
"""DeepSeek-V4 FlashMLA sparse FP8 layout constants and helpers.

The V4 FlashMLA sparse layout packs a 512-dim K cache head as:
  [448 fp8 NoPE | 128 bytes (64 bf16) RoPE | 7 e8m0 scale bytes | 1 pad byte]
  = 584 bytes per token.

NoPE region: 7 tiles of 64 elements, each tile quantized to FP8 e4m3fn with
a per-tile e8m0fnu power-of-2 scale factor.
RoPE region: 64 BF16 values stored as raw bytes (128 bytes).
Scales: 7 e8m0fnu scale bytes + 1 padding byte = 8 bytes.
"""
import torch

V4_FLASHMLA_HEAD_DIM = 512
V4_FLASHMLA_D_NOPE = 448
V4_FLASHMLA_D_ROPE = 64
V4_FLASHMLA_TILE_SIZE = 64
V4_FLASHMLA_NUM_TILES = 7


def dequantize_v4_flashmla_sparse(quant_k_cache: torch.Tensor) -> torch.Tensor:
    """Dequantize V4 FlashMLA sparse FP8 KV cache to BF16.

    Args:
        quant_k_cache: ``[num_blocks, block_size, 1, packed_dim]`` FP8 cache.

    Returns:
        ``[num_blocks, block_size, 1, 512]`` BF16 tensor.
    """
    assert quant_k_cache.dim() == 4
    num_blocks, block_size, num_heads, _ = quant_k_cache.shape
    assert num_heads == 1

    result = torch.empty((num_blocks, block_size, V4_FLASHMLA_HEAD_DIM),
                         dtype=torch.bfloat16,
                         device=quant_k_cache.device)
    quant_k_cache = quant_k_cache.view(num_blocks, -1)
    input_nope_rope = quant_k_cache[:, :block_size * (V4_FLASHMLA_D_NOPE + 2 * V4_FLASHMLA_D_ROPE)].view(
        num_blocks, block_size, V4_FLASHMLA_D_NOPE + 2 * V4_FLASHMLA_D_ROPE)
    input_nope = input_nope_rope[:, :, :V4_FLASHMLA_D_NOPE]
    input_rope = input_nope_rope[:, :, V4_FLASHMLA_D_NOPE:].view(torch.bfloat16)
    input_scale = quant_k_cache[:, block_size * (V4_FLASHMLA_D_NOPE + 2 * V4_FLASHMLA_D_ROPE):].view(
        num_blocks, block_size, 8)[:, :, :V4_FLASHMLA_NUM_TILES].view(torch.float8_e8m0fnu)

    result[..., V4_FLASHMLA_D_NOPE:] = input_rope
    for tile_idx in range(V4_FLASHMLA_NUM_TILES):
        cur_nope = input_nope[..., tile_idx * V4_FLASHMLA_TILE_SIZE:(tile_idx + 1) * V4_FLASHMLA_TILE_SIZE].to(
            torch.bfloat16)
        cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
        result[..., tile_idx * V4_FLASHMLA_TILE_SIZE:(tile_idx + 1) * V4_FLASHMLA_TILE_SIZE] = cur_nope * cur_scales

    return result.view(num_blocks, block_size, 1, V4_FLASHMLA_HEAD_DIM)
