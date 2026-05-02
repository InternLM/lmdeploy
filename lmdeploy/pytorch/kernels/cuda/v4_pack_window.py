# Copyright (c) OpenMMLab. All rights reserved.
"""Triton kernel to pack BF16 tokens into FlashMLA MODEL1 sparse FP8 flat-layout
window cache, replacing the per-token Python loop in _pack_window_state_tokens.

FlashMLA MODEL1 flat layout per slot (viewed as flat bytes):
  [token_0 NoPE+RoPE | token_1 NoPE+RoPE | ... | token_0 scales | token_1 scales | ...]
  NoPE+RoPE per token = 576 bytes (448 e4m3fn + 128 bf16)
  Scales per token = 8 bytes (7 e8m0fnu + 1 padding)

We pass three view pointers of the same buffer with different dtypes (same pattern
as fill_compressed_kv in v4_compressor.py):
  - nope_ptr: e4m3fn view for NoPE region
  - rope_ptr: bf16 view for RoPE region (already at the RoPE offset)
  - scale_ptr: uint8 view for scales region
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_window_tokens_fp8_kernel(
    kv_ptr,
    nope_ptr,
    rope_ptr,
    scale_ptr,
    slot_ptr,
    pos_ptr,
    stride_kv_n,
    stride_kv_d,
    stride_nope_slot,
    stride_nope_pos,
    stride_rope_slot,
    stride_rope_pos,
    stride_scale_slot,
    stride_scale_pos,
    stride_slot,
    WINDOW_SIZE: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    token_id = tl.program_id(0)

    cur_slot = tl.load(slot_ptr + token_id * stride_slot)
    cur_pos = tl.load(pos_ptr + token_id * stride_slot)

    slot64 = cur_slot.to(tl.int64)
    pos64 = cur_pos.to(tl.int64)

    # ---- RoPE region: direct bf16 copy ----
    rope_offs = tl.arange(0, D_ROPE)  # 0..63
    rope_vals = tl.load(
        kv_ptr + token_id * stride_kv_n + (D_NOPE + rope_offs) * stride_kv_d)

    rope_base = rope_ptr + slot64 * stride_rope_slot + pos64 * stride_rope_pos
    tl.store(rope_base + rope_offs, rope_vals)

    # ---- NoPE region: per-tile quantize ----
    for tile_idx in range(NUM_TILES):
        tile_offs = tl.arange(0, TILE_SIZE)
        tile_vals = tl.load(
            kv_ptr + token_id * stride_kv_n + (tile_idx * TILE_SIZE + tile_offs) * stride_kv_d)

        amax = tl.max(tl.abs(tile_vals))
        scale_inv = amax / 448.0
        scale_inv = tl.where(scale_inv < 1e-4, 1e-4, scale_inv)
        ceil_log2 = tl.ceil(tl.log2(scale_inv))
        scale_inv = tl.exp2(ceil_log2)

        # e8m0fnu raw byte: ceil_log2 + 127
        scale_byte = (ceil_log2.to(tl.int32) + 127).to(tl.int32)

        # Quantize: tile / scale_inv → e4m3fn
        quantized = (tile_vals / scale_inv).to(tl.float8e4nv)

        # Write NoPE quantized values
        nope_base = nope_ptr + slot64 * stride_nope_slot + pos64 * stride_nope_pos + tile_idx * TILE_SIZE
        tl.store(nope_base + tile_offs, quantized)

        # Write scale byte (uint8 view)
        sc_base = scale_ptr + slot64 * stride_scale_slot + pos64 * stride_scale_pos + tile_idx
        tl.store(sc_base, scale_byte.to(tl.uint8))


def pack_window_tokens_fp8(
    kv_tokens: torch.Tensor,
    window_state_fp8_cache: torch.Tensor,
    slot: torch.Tensor,
    positions: torch.Tensor,
):
    """Pack BF16 tokens into FlashMLA MODEL1 sparse FP8 window cache.

    Args:
        kv_tokens: [num_tokens, 512] BF16 tokens to pack.
        window_state_fp8_cache: [num_total_slots, window_size, packed_dim] FP8 cache.
        slot: [num_tokens] slot indices (which cache row to write to).
        positions: [num_tokens] ring-buffer positions within the window.
    """
    from lmdeploy.pytorch.backends.cuda.attention.flashmla_utils import (
        MODEL1_D_NOPE, MODEL1_D_ROPE, MODEL1_TILE_SIZE, MODEL1_NUM_TILES,
    )

    assert kv_tokens.dim() == 2
    num_tokens = kv_tokens.size(0)
    if num_tokens == 0:
        return

    window_size = window_state_fp8_cache.size(1)
    nope_rope_stride = MODEL1_D_NOPE + 2 * MODEL1_D_ROPE  # 576 bytes per token in NoPE+RoPE region
    num_slots = window_state_fp8_cache.size(0)

    # Create three views of the same FP8 cache buffer (same pattern as fill_compressed_kv)
    flat = window_state_fp8_cache.view(num_slots, -1)

    # NoPE+RoPE region: [num_slots, window_size * 576] as e4m3fn
    nope_rope = flat[:, :window_size * nope_rope_stride].view(
        num_slots, window_size, nope_rope_stride)
    nope_view = nope_rope[:, :, :MODEL1_D_NOPE]  # [num_slots, window_size, 448] e4m3fn

    # RoPE region: slice the RoPE part first (128 e4m3fn bytes = 64 bf16 elements),
    # then view as bf16 — same pattern as quantize_model1_fp8_sparse
    rope_e4 = nope_rope[:, :, MODEL1_D_NOPE:]  # [num_slots, window_size, 128] e4m3fn
    rope_view = rope_e4.view(torch.bfloat16)    # [num_slots, window_size, 64] bf16

    # Scale region: uint8 view
    scale_view = flat[:, window_size * nope_rope_stride:].view(
        num_slots, window_size, 8)[:, :, :MODEL1_NUM_TILES].view(torch.uint8)

    grid = (num_tokens,)
    _pack_window_tokens_fp8_kernel[grid](
        kv_tokens,
        nope_view,
        rope_view,
        scale_view,
        slot.long(),
        positions.long(),
        stride_kv_n=kv_tokens.stride(0),
        stride_kv_d=kv_tokens.stride(1),
        stride_nope_slot=nope_view.stride(0),
        stride_nope_pos=nope_view.stride(1),
        stride_rope_slot=rope_view.stride(0),
        stride_rope_pos=rope_view.stride(1),
        stride_scale_slot=scale_view.stride(0),
        stride_scale_pos=scale_view.stride(1),
        stride_slot=1,
        WINDOW_SIZE=window_size,
        D_NOPE=MODEL1_D_NOPE,
        D_ROPE=MODEL1_D_ROPE,
        TILE_SIZE=MODEL1_TILE_SIZE,
        NUM_TILES=MODEL1_NUM_TILES,
    )
