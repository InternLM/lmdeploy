# Copyright (c) OpenMMLab. All rights reserved.
"""Write model routing rows into the same physical pages as KV."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_routes(routes, cache, starts, lengths, history, blocks,
                    TABLE_STRIDE: tl.constexpr, PAGE_SIZE: tl.constexpr,
                    FEATURES: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(1)
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    token = offsets // FEATURES
    feature = offsets % FEATURES
    mask = token < tl.load(lengths + row)
    position = tl.load(history + row) + token
    page = tl.load(blocks + row * TABLE_STRIDE + position // PAGE_SIZE, mask, other=0).to(tl.int64)
    src = (tl.load(starts + row).to(tl.int64) + token) * FEATURES + feature
    dst = (page * PAGE_SIZE + position % PAGE_SIZE) * FEATURES + feature
    values = tl.load(routes + src, mask)
    tl.store(cache + dst, values, mask)


def scatter_routed_experts(routes: torch.Tensor, cache: torch.Tensor,
                          starts: torch.Tensor, lengths: torch.Tensor, history: torch.Tensor,
                          blocks: torch.Tensor, max_q_seqlen: int, kernel_block_size: int) -> None:
    """Queue one scatter on the forward stream, including partial pages.

    The block table uses kernel pages; the owning pool is viewed as logical blocks by Store. Rejected speculative rows
    are overwritten by the next forward and never enter an accepted save frontier.
    """
    features = cache.shape[-2] * cache.shape[-1]
    grid = (triton.cdiv(max_q_seqlen * features, 512), lengths.numel())
    _scatter_routes[grid](routes, cache, starts, lengths, history, blocks,
                          blocks.stride(0), kernel_block_size, features, 512)
