# Copyright (c) OpenMMLab. All rights reserved.
"""Keep graph padding and idle DP ranks out of expert dispatch."""
import torch
import triton
import triton.language as tl


@triton.jit
def _mask_invalid_routes_kernel(Ids, ValidTokens, SIZE: tl.constexpr,
                                TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid_tokens = tl.load(ValidTokens)
    # Valid routes are untouched, including any existing -1 sentinel.
    tl.store(Ids + offsets, -1, (offsets < SIZE) & (offsets // TOP_K >= valid_tokens))


def mask_invalid_routes(topk_idx: torch.Tensor, valid_tokens: torch.Tensor) -> None:
    """Mask in place so dispatch and combine consume the same route IDs.

    ``valid_tokens`` is a graph-owned device scalar updated before every replay.
    It is zero for a dummy step, even when the captured graph has padded tokens.
    """
    if topk_idx.numel() == 0:
        return
    assert topk_idx.is_contiguous() and topk_idx.ndim == 2
    _mask_invalid_routes_kernel[(triton.cdiv(topk_idx.numel(), 256),)](
        topk_idx, valid_tokens, topk_idx.numel(), topk_idx.size(1), 256, num_warps=4)
