# Copyright (c) OpenMMLab. All rights reserved.
"""FFN weight loading builder and w1+w3 fusion helpers.

Provides ``FfnBuilder`` for committing FFN weights (w1/w2/w3 with optional
w1+w3 fusion) and helper functions for determining whether SiLU fusion
(interleave vs chunk) should be used and whether w1+w3 fusion is safe for
the given TP configuration.
"""
from __future__ import annotations

import math

import torch

from ..linear import Linear, pad_in_dim, pad_out_dim
from ._base import Builder, SplitSide, transform_input_dim, transform_output_dim

__all__ = [
    'FfnBuilder',
    'fuse_w1w3',
]

# ---------------------------------------------------------------------------
# @transform_output_dim / @transform_input_dim helpers
# ---------------------------------------------------------------------------


@transform_output_dim
def _interleave_w1w3(w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """Interleave w1 and w3 along output dim for fused SiLU epilogue."""
    return torch.stack([w1, w3], dim=-1).reshape(w1.shape[:-1] + (-1,)).contiguous()


@transform_output_dim
def _chunk_w1w3(w1: torch.Tensor, w3: torch.Tensor, *,
                tp: int) -> torch.Tensor:
    """Concatenate w1 and w3 along output dim with TP interleaving."""
    if tp <= 1:
        return torch.cat([w1, w3], dim=-1).contiguous()
    d = w1.dim() - 1
    r1 = w1.reshape(w1.shape[:d] + (tp, w1.shape[d] // tp))
    r3 = w3.reshape(w3.shape[:d] + (tp, w3.shape[d] // tp))
    combined = torch.cat([r1, r3], dim=d + 1)
    return combined.reshape(w1.shape[:d] + (-1,)).contiguous()


@transform_output_dim
def _pad_out(tensor: torch.Tensor, *, target: int) -> torch.Tensor:
    """Pad output dimension to target size."""
    return pad_out_dim(tensor, target, dim=tensor.dim() - 1)


@transform_input_dim
def _pad_in(tensor: torch.Tensor, *, target: int) -> torch.Tensor:
    """Pad input dimension to target size (1-D tensors pass through)."""
    return pad_in_dim(tensor, target, dim=0)


# ---------------------------------------------------------------------------
# FFN fusion helpers
# ---------------------------------------------------------------------------


def _should_fuse_silu(w1_linear: Linear, act_type: str, is_moe: bool = False) -> bool:
    """Determine if fused SiLU (interleave) should be used for w1+w3 fusion.

    Gold standard condition (from GEMM kernel constraints — trust it):     act_type == SiLU && (int4 || mxfp4 || fp8 ||
    moe) && !(fp8 && SM90)
    """
    if act_type not in ('', 'silu', 'SiLU'):
        return False

    # Dense bf16/fp16 without MoE -> chunk, not interleave
    weight = w1_linear.tensors.get('weight')
    is_quantized = weight is not None and weight.element_size() < 2
    if not is_quantized and not is_moe:
        return False

    # FP8 on SM90 -> chunk
    fmt = w1_linear.weight_format
    if fmt is not None and fmt.name == 'fp8':
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap == (9, 0):
                return False

    return True


def _can_fuse_w1w3(w1: Linear, tp: int) -> bool:
    """Check whether w1+w3 fusion is safe for the given TP.

    Fusion (interleave or chunk) concatenates w1 and w3 along the output dim.
    For block-quantized formats (e.g. FP8 with block_out=128), the fused
    scale count ``2 * cdiv(N/tp, block_out)`` must equal
    ``cdiv(2*N/tp, block_out)``.  This holds iff ``(N/tp) % block_out == 0``.
    When it doesn't, the fused module's C++ allocation won't match the
    concatenated scales and we must commit w1/w3 separately.
    """
    if tp <= 1:
        return True
    fmt = w1.weight_format
    if fmt is None or fmt.block_out is None:
        return True
    w = w1.tensors.get('weight')
    if w is None:
        return True
    return (w.size(-1) // tp) % fmt.block_out == 0


def fuse_w1w3(
    w1: Linear,
    w3: Linear,
    tp: int,
    act_type: str,
    is_moe: bool = False,
) -> tuple[Linear | None, bool]:
    """Optionally fuse w1/w3 on full (unsharded) tensors for FFN.

    Returns (fused_w1w3_or_none, fused_silu).
    When fusion is possible, fused_w1w3 is set.
    When block-scale boundaries prevent fusion, returns (None, fused_silu).

    TP sharding is NOT done here — the caller's commit path handles it
    via split_side=SplitSide.OUTPUT.  ``tp`` is only used for the
    block-scale alignment check in ``_can_fuse_w1w3``.
    """
    fused_silu = _should_fuse_silu(w1, act_type, is_moe)
    can_fuse = _can_fuse_w1w3(w1, tp)

    if can_fuse:
        if fused_silu:
            w1w3 = _interleave_w1w3(w1, w3)
        else:
            w1w3 = _chunk_w1w3(w1, w3, tp=tp)
        return (w1w3, fused_silu)
    else:
        return (None, fused_silu)


# ---------------------------------------------------------------------------
# TP padding
# ---------------------------------------------------------------------------


def _pad_ffn_for_tp(w1: Linear, w2: Linear, w3: Linear,
                     tp: int) -> tuple[Linear, Linear, Linear]:
    """Pad w1/w3 output dim and w2 input dim for TP sharding."""
    raw_inter = w1.tensors['weight'].size(-1)

    if tp <= 1:
        return w1, w2, w3
    fmt = w1.weight_format
    block_out = (fmt.block_out or 1) if fmt else 1
    block_in = (fmt.block_in or 1) if fmt else 1
    effective_block = math.lcm(block_in, block_out) if block_in != block_out else block_out

    groups = (raw_inter + effective_block - 1) // effective_block
    groups_per_rank = (groups + tp - 1) // tp
    padded_inter = groups_per_rank * effective_block * tp
    if padded_inter == raw_inter:
        return w1, w2, w3

    w1 = _pad_out(w1, target=padded_inter)
    w3 = _pad_out(w3, target=padded_inter)
    w2 = _pad_in(w2, target=padded_inter)
    return w1, w2, w3


# ---------------------------------------------------------------------------
# FfnBuilder -- w1+w3 fusion, w2 commit
# ---------------------------------------------------------------------------


class FfnBuilder(Builder):
    """FFN weight loading builder with w1+w3 fusion."""

    def add_ffn(self, w1, w2, w3):
        """Pad weights for TP alignment, fuse w1+w3 if possible, then shard and
        commit.

        The fusion result determines ``fuse_silu`` on the C++ module config.
        Updating ``self.config.fuse_silu`` **before** any ``_add_linear``
        call ensures the C++ module is lazily created with the correct flag.
        """
        # Pad weights for TP alignment before any fusion or sharding.
        # After padding, push the padded-global inter_size onto config so
        # that C++ module creation sees the correct dimension.
        w1, w2, w3 = _pad_ffn_for_tp(w1, w2, w3, self._tp)
        self.config.inter_size = w1.tensors['weight'].size(-1)

        act_type = getattr(self.config, 'act_type', 0)
        if isinstance(act_type, int):
            act_type = {0: 'silu', 1: 'gpt-oss'}.get(act_type, 'silu')
        fused, fused_silu = fuse_w1w3(
            w1, w3, self._tp, act_type,
            is_moe=getattr(self.config, 'fused_moe', False))

        self.config.fuse_silu = fused_silu

        if fused is not None:
            self._add_linear('w1w3', fused, SplitSide.OUTPUT)
        else:
            self._add_linear('w1', w1, SplitSide.OUTPUT)
            self._add_linear('w3', w3, SplitSide.OUTPUT)
        self._add_linear('w2', w2, SplitSide.INPUT)
