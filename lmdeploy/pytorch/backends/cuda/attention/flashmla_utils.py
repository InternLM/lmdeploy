# Copyright (c) OpenMMLab. All rights reserved.
import torch


MODEL1_D = 512
MODEL1_D_NOPE = 448
MODEL1_D_ROPE = 64
MODEL1_TILE_SIZE = 64
MODEL1_NUM_TILES = 7


def model1_fp8_sparse_token_dim(block_size: int) -> int:
    """Return the per-token packed width for FlashMLA MODEL1 sparse KV."""
    bytes_per_token = MODEL1_D_NOPE + 2 * MODEL1_D_ROPE + MODEL1_NUM_TILES + 1
    return bytes_per_token


def quantize_model1_fp8_sparse(input_k_cache: torch.Tensor) -> torch.Tensor:
    """Pack BF16 `[num_blocks, block_size, 1, 512]` K cache into FlashMLA
    MODEL1 sparse FP8 layout."""
    assert input_k_cache.dim() == 4
    num_blocks, block_size, num_heads, head_dim = input_k_cache.shape
    assert num_heads == 1
    assert head_dim == MODEL1_D

    token_dim = model1_fp8_sparse_token_dim(block_size)
    input_k = input_k_cache.squeeze(2)
    result = torch.empty((num_blocks, block_size * token_dim),
                         dtype=torch.float8_e4m3fn,
                         device=input_k_cache.device)

    result_nope_rope = result[:, :block_size * (MODEL1_D_NOPE + 2 * MODEL1_D_ROPE)].view(
        num_blocks, block_size, MODEL1_D_NOPE + 2 * MODEL1_D_ROPE)
    result_nope = result_nope_rope[:, :, :MODEL1_D_NOPE]
    result_rope = result_nope_rope[:, :, MODEL1_D_NOPE:].view(input_k.dtype)
    result_scale = result[:, block_size * (MODEL1_D_NOPE + 2 * MODEL1_D_ROPE):].view(
        num_blocks, block_size, 8)[:, :, :MODEL1_NUM_TILES].view(torch.float8_e8m0fnu)

    result_rope.copy_(input_k[..., MODEL1_D_NOPE:])
    for tile_idx in range(MODEL1_NUM_TILES):
        tile = input_k[..., tile_idx * MODEL1_TILE_SIZE:(tile_idx + 1) * MODEL1_TILE_SIZE].float()
        scale_inv = tile.abs().amax(dim=-1) / 448.0
        scale_inv = torch.pow(2, torch.clamp_min(scale_inv, 1e-4).log2().ceil())
        result_scale[:, :, tile_idx].copy_(scale_inv.to(torch.float8_e8m0fnu))
        quantized = (tile / scale_inv.unsqueeze(-1)).to(torch.float8_e4m3fn)
        result_nope[:, :, tile_idx * MODEL1_TILE_SIZE:(tile_idx + 1) * MODEL1_TILE_SIZE].copy_(quantized)

    return result.view(num_blocks, block_size, 1, token_dim)


def quantize_model1_fp8_sparse_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Pack BF16 `[num_tokens, 512]` tokens into FlashMLA MODEL1 sparse
    per-token format `[num_tokens, packed_dim]`."""
    assert tokens.dim() == 2
    packed = quantize_model1_fp8_sparse(tokens.unsqueeze(1).unsqueeze(1))
    return packed.squeeze(2).squeeze(1)


def dequantize_model1_fp8_sparse(quant_k_cache: torch.Tensor) -> torch.Tensor:
    """Dequantize FlashMLA MODEL1 sparse K cache to BF16.

    Args:
        quant_k_cache: `[num_blocks, block_size, 1, packed_dim]`

    Returns:
        `[num_blocks, block_size, 1, 512]` BF16 tensor.
    """
    assert quant_k_cache.dim() == 4
    num_blocks, block_size, num_heads, _ = quant_k_cache.shape
    assert num_heads == 1

    result = torch.empty((num_blocks, block_size, MODEL1_D), dtype=torch.bfloat16, device=quant_k_cache.device)
    quant_k_cache = quant_k_cache.view(num_blocks, -1)
    input_nope_rope = quant_k_cache[:, :block_size * (MODEL1_D_NOPE + 2 * MODEL1_D_ROPE)].view(
        num_blocks, block_size, MODEL1_D_NOPE + 2 * MODEL1_D_ROPE)
    input_nope = input_nope_rope[:, :, :MODEL1_D_NOPE]
    input_rope = input_nope_rope[:, :, MODEL1_D_NOPE:].view(torch.bfloat16)
    input_scale = quant_k_cache[:, block_size * (MODEL1_D_NOPE + 2 * MODEL1_D_ROPE):].view(
        num_blocks, block_size, 8)[:, :, :MODEL1_NUM_TILES].view(torch.float8_e8m0fnu)

    result[..., MODEL1_D_NOPE:] = input_rope
    for tile_idx in range(MODEL1_NUM_TILES):
        cur_nope = input_nope[..., tile_idx * MODEL1_TILE_SIZE:(tile_idx + 1) * MODEL1_TILE_SIZE].to(torch.bfloat16)
        cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
        result[..., tile_idx * MODEL1_TILE_SIZE:(tile_idx + 1) * MODEL1_TILE_SIZE] = cur_nope * cur_scales

    return result.view(num_blocks, block_size, 1, MODEL1_D)


def abs_indices_to_phys_indices(abs_indices: torch.Tensor, block_table: torch.Tensor, block_size: int) -> torch.Tensor:
    """Convert logical token positions to FlashMLA sparse physical indices."""
    batch, seq_q, topk = abs_indices.shape
    _, max_blocks_per_seq = block_table.shape

    abs_indices = abs_indices.clone()
    invalid_mask = abs_indices < 0
    abs_indices[invalid_mask] = 0

    block_offsets = torch.arange(batch, device=abs_indices.device, dtype=abs_indices.dtype).view(batch, 1, 1)
    block_offsets = block_offsets * max_blocks_per_seq
    block_ids = (abs_indices // block_size + block_offsets).view(-1)
    phys_blocks = block_table.reshape(-1).index_select(0, block_ids)
    phys_indices = phys_blocks.view(batch, seq_q, topk) * block_size + abs_indices % block_size
    phys_indices[invalid_mask] = -1
    return phys_indices
