# Copyright (c) OpenMMLab. All rights reserved.
"""Attention weight loading builder and QKV fusion pipeline.

Provides ``AttentionBuilder`` for committing attention weights (QKV fusion,
O-proj, QK-norm, direct params) and pipeline functions (``dequant_mixed``,
``repeat_kv_for_tp``, ``split_output_gate``, ``fuse_qkv``) for fusing Q/K/V
Linear bundles into a single interleaved w_qkv with KV head padding and
output-gate splitting.
"""
from __future__ import annotations

import torch

from ..linear import Linear
from ._base import Builder, SplitSide, _dequant_linear, transform_output_dim

# ---------------------------------------------------------------------------
# New pipeline functions (replacing merge_qkv_linear)
# ---------------------------------------------------------------------------


def dequant_mixed(*linears: Linear, data_type) -> tuple[Linear, ...]:
    """Dequantize to trivial if any arg is in trivial format.

    When any Linear has trivial weight format (e.g. from RoPE reordering), dequantize all non-trivial args so formats
    match for fusion. None args pass through unchanged.
    """
    has_trivial = any(
        l is not None
        and l.weight_format is not None
        and l.weight_format.name == 'trivial'
        for l in linears
    )
    if not has_trivial:
        return linears
    return tuple(_dequant_linear(l, data_type=data_type) if l is not None else l
                 for l in linears)


def _infer_heads(linear: Linear, head_dim: int) -> int:
    """Derive head count from the weight tensor's output dimension."""
    w = linear.tensors.get('weight')
    if w is None:
        return 0
    return w.size(-1) // head_dim


@transform_output_dim
def _repeat_kv_heads(tensor: torch.Tensor, *, tp: int,
                     heads: int) -> torch.Tensor:
    """Repeat KV heads to reach a TP-divisible count."""
    if heads % tp == 0:
        return tensor
    target_heads = ((heads + tp - 1) // tp) * tp
    assert target_heads % heads == 0, (
        f'target_heads={target_heads} must be divisible by heads={heads}')
    n_repeat = target_heads // heads
    per_head = tensor.size(-1) // heads
    t = tensor.view(tensor.size(0), heads, per_head)
    return t.repeat(1, n_repeat, 1).reshape(tensor.size(0), target_heads * per_head)


def repeat_kv_for_tp(k: Linear, v: Linear, *,
                     tp: int, head_dim: int) -> tuple[Linear, Linear]:
    """Repeat KV heads to reach a TP-divisible count."""
    k = _repeat_kv_heads(k, tp=tp, heads=_infer_heads(k, head_dim))
    v = _repeat_kv_heads(v, tp=tp, heads=_infer_heads(v, head_dim))
    return k, v


@transform_output_dim
def split_output_gate(tensor: torch.Tensor, *, head_dim: int
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Split output gate from Q projection (Qwen3.5).

    Q's output dim is 2 * head_num * head_dim. Reshape to [batch, head_num, 2, head_dim], split into q_real and gate.
    """
    head_num = tensor.size(-1) // (head_dim * 2)
    q, gate = tensor.view(-1, head_num, 2, head_dim).unbind(2)
    return q.reshape(-1, head_num * head_dim), gate.reshape(-1, head_num * head_dim)


@transform_output_dim
def fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             *, tp: int, gate: torch.Tensor | None = None) -> torch.Tensor:
    """Fuse Q, K, V (and optionally gate) into a single w_qkv Linear.

    Concatenates output channels with TP interleaving.
    Layout per tp-shard: [Q | K | V] or [Q | K | V | Gate].
    """
    tensors = [t for t in (q, k, v, gate) if t is not None]
    parts = [t.view(t.size(0), tp, -1) for t in tensors]
    merged = torch.cat(parts, dim=-1)
    return merged.view(-1, merged.size(-1) * tp)


# ---------------------------------------------------------------------------
# AttentionBuilder
# ---------------------------------------------------------------------------


class AttentionBuilder(Builder):
    """Attention weight loading builder."""

    _PARAM_TP_RULES: dict[str, SplitSide] = {
        'sinks': SplitSide.OUTPUT,
    }

    def add_qkv_proj(self, q, k, v, *, gate=None):
        """Fuse Q/K/V into a single w_qkv with TP interleave, commit.

        Pipeline: dequant_mixed -> repeat_kv_for_tp -> fuse_qkv -> commit.
        """
        q, k, v, gate = dequant_mixed(q, k, v, gate, data_type=self.config.data_type)
        k, v = repeat_kv_for_tp(k, v, tp=self._tp,
                                head_dim=self.config.head_dim)
        # After KV head repeat, push the padded-global kv_head_num onto
        # config so that C++ module creation sees the correct head count.
        self.config.kv_head_num = _infer_heads(k, self.config.head_dim)
        merged = fuse_qkv(q, k, v, tp=self._tp, gate=gate)
        self._add_linear('w_qkv', merged, SplitSide.OUTPUT)

    def add_o_proj(self, o):
        """Shard along input dim, commit."""
        self._add_linear('wo', o, SplitSide.INPUT)

    def add_param(self, name, tensor):
        """Commit a direct parameter.

        Builder determines split side.
        """
        split_side = self._PARAM_TP_RULES.get(name)
        self._add_tensor(name, tensor, split_side)
