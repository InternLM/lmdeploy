"""Reference implementations for V4 FlashMLA sparse FP8 quantize/dequantize.

Used by kernel tests for correctness comparison only — the production path fuses these operations into Triton kernels.
"""

import torch

from lmdeploy.pytorch.consts import (
    V4_FLASHMLA_D_NOPE,
    V4_FLASHMLA_D_ROPE,
    V4_FLASHMLA_NUM_TILES,
    V4_FLASHMLA_TILE_SIZE,
)

D_NOPE = V4_FLASHMLA_D_NOPE      # 448
D_ROPE = V4_FLASHMLA_D_ROPE      # 64
TILE_SIZE = V4_FLASHMLA_TILE_SIZE  # 64
NUM_TILES = V4_FLASHMLA_NUM_TILES  # 7
NR_DIM = D_NOPE + 2 * D_ROPE    # 576 bytes per token (NoPE + RoPE in e4m3fn)
FP8_MAX = 448.0


def quantize_v4_flashmla_sparse(input_k_cache: torch.Tensor) -> torch.Tensor:
    """Pack BF16 ``[num_blocks, block_size, 1, 512]`` K cache into V4 FlashMLA
    sparse FP8 layout.

    Returns ``[num_blocks, block_size, 1, 584]`` e4m3fn tensor.
    """
    assert input_k_cache.dim() == 4
    num_blocks, block_size, _, head_dim = input_k_cache.shape
    assert head_dim == 512

    device = input_k_cache.device
    packed_dim = NR_DIM + 8  # 576 + 8 = 584
    output = torch.zeros(num_blocks, block_size, 1, packed_dim,
                         dtype=torch.float8_e4m3fn, device=device)

    # Flat view for layout construction (same pattern as v4_compressor.py / v4_flatten_kv.py)
    flat_out = output.view(num_blocks, -1)

    # NoPE+RoPE region: [num_blocks, block_size * NR_DIM] as e4m3fn
    nope_rope = flat_out[:, :block_size * NR_DIM].view(
        num_blocks, block_size, NR_DIM)
    nope_view = nope_rope[:, :, :D_NOPE]  # [num_blocks, block_size, 448] e4m3fn

    # RoPE region: view as bf16
    rope_e4 = nope_rope[:, :, D_NOPE:]  # [num_blocks, block_size, 128] e4m3fn
    rope_view = rope_e4.view(torch.bfloat16)  # [num_blocks, block_size, 64] bf16

    # Scale region: uint8
    scale_view = flat_out[:, block_size * NR_DIM:].view(
        num_blocks, block_size, 8).view(torch.uint8)

    # Per-block, per-token quantize
    for b in range(num_blocks):
        for t in range(block_size):
            token = input_k_cache[b, t, 0]  # [512] bf16

            # Quantize NoPE tiles
            for tile_idx in range(NUM_TILES):
                d_base = tile_idx * TILE_SIZE
                tile = token[d_base:d_base + TILE_SIZE].float()

                amax = tile.abs().max()
                scale_inv = max(amax.item() / FP8_MAX, 1e-4)
                ceil_log2 = torch.ceil(torch.log2(torch.tensor(scale_inv, dtype=torch.float32)))
                scale_inv_pow2 = torch.exp2(ceil_log2)

                quantized = (tile / scale_inv_pow2).to(torch.float8_e4m3fn)
                nope_view[b, t, d_base:d_base + TILE_SIZE] = quantized

                # e8m0fnu scale byte: raw byte = ceil_log2 + 127
                scale_byte = int(ceil_log2.item() + 127)
                scale_view[b, t, tile_idx] = scale_byte

            # RoPE: direct bf16 copy (128 e4m3fn bytes = 64 bf16 elements)
            rope_vals = token[D_NOPE:]  # [64] bf16
            rope_view[b, t] = rope_vals

    return output


def dequantize_v4_flashmla_sparse(quant_k_cache: torch.Tensor) -> torch.Tensor:
    """Dequantize V4 FlashMLA sparse FP8 K cache to BF16.

    Re-exports from the production module for test convenience.

    Args:
        quant_k_cache: [num_blocks, block_size, 1, 584] e4m3fn FP8 cache.

    Returns:
        [num_blocks, block_size, 1, 512] BF16 cache.
    """
    assert quant_k_cache.dim() == 4
    num_blocks, block_size, _, packed_dim = quant_k_cache.shape
    assert packed_dim == NR_DIM + 8

    device = quant_k_cache.device
    output = torch.zeros(num_blocks, block_size, 1, 512,
                         dtype=torch.bfloat16, device=device)

    # Build views (same layout as quantize)
    flat = quant_k_cache.view(num_blocks, -1)
    nope_rope = flat[:, :block_size * NR_DIM].view(
        num_blocks, block_size, NR_DIM)
    nope_view = nope_rope[:, :, :D_NOPE]  # [num_blocks, block_size, 448] e4m3fn

    rope_e4 = nope_rope[:, :, D_NOPE:]  # [num_blocks, block_size, 128] e4m3fn
    rope_view = rope_e4.view(torch.bfloat16)  # [num_blocks, block_size, 64] bf16

    scale_view = flat[:, block_size * NR_DIM:].view(
        num_blocks, block_size, 8).view(torch.uint8)

    # Per-block, per-token dequantize
    for b in range(num_blocks):
        for t in range(block_size):
            # Dequantize NoPE tiles
            for tile_idx in range(NUM_TILES):
                d_base = tile_idx * TILE_SIZE
                nope_fp8 = nope_view[b, t, d_base:d_base + TILE_SIZE].float()

                # Read scale byte and reconstruct float scale
                scale_byte = scale_view[b, t, tile_idx].item()
                # e8m0fnu: bits = scale_byte, float = 2^(scale_byte - 127)
                scale_bits = scale_byte << 23
                scale_f32 = torch.tensor(scale_bits, dtype=torch.int32).view(torch.float32)

                dequant = (nope_fp8 * scale_f32).to(torch.bfloat16)
                output[b, t, 0, d_base:d_base + TILE_SIZE] = dequant

            # RoPE: direct bf16 copy
            output[b, t, 0, D_NOPE:] = rope_view[b, t]

    return output
