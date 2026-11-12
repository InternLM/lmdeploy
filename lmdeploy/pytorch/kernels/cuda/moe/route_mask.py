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


@triton.jit
def _prepare_routes_kernel(Ids, Weights, ScaledWeights, ValidTokens,
                           SIZE: tl.constexpr, TOP_K: tl.constexpr,
                           HAS_VALID_COUNT: tl.constexpr, IS_DUMMY: tl.constexpr,
                           SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    present = offsets < SIZE
    weights = tl.load(Weights + offsets, present, 0).to(tl.float32)
    tl.store(ScaledWeights + offsets, weights * SCALE, present)
    if HAS_VALID_COUNT:
        valid = tl.load(ValidTokens)
        tl.store(Ids + offsets, -1, present & (offsets // TOP_K >= valid))
    elif IS_DUMMY:
        tl.store(Ids + offsets, -1, present)


def prepare_low_latency_routes(topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                               valid_tokens: torch.Tensor | None, output_scale: float,
                               is_dummy: bool = False) -> torch.Tensor:
    """Mask dispatch IDs and scale FP32 combine weights in one launch.

    Routing itself remains normalized and unscaled. IDs retain their original dtype and existing -1 sentinels. The
    graph-owned valid count is read on device on every replay; no host synchronization or scalar extraction occurs.
    """
    assert topk_idx.ndim == 2 and topk_idx.is_contiguous()
    assert topk_weights.shape == topk_idx.shape and topk_weights.is_contiguous()
    scaled_weights = torch.empty_like(topk_weights, dtype=torch.float32)
    if topk_idx.numel() == 0:
        return scaled_weights
    _prepare_routes_kernel[(triton.cdiv(topk_idx.numel(), 256),)](
        topk_idx, topk_weights, scaled_weights, valid_tokens,
        topk_idx.numel(), topk_idx.size(1), valid_tokens is not None,
        is_dummy, output_scale, 256, num_warps=4)
    return scaled_weights
