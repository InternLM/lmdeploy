# Copyright (c) OpenMMLab. All rights reserved.
"""DeltaNet weight loading builder and GDN input-projection fusion helpers.

Provides ``DeltaNetBuilder`` for committing DeltaNet weights (GDN input
projections, scalar params, conv1d) and helper functions ``split_qkv`` and
``fuse_gdn`` for merging in_proj_qkv/z/b/a into a single ``in_proj_all``
with TP interleaving.
"""
from __future__ import annotations

import torch

from ..linear import Linear
from ._base import Builder, SplitSide, _ensure_compatible_formats


def tp_interleave_tensor(t: torch.Tensor, tp: int, d: int) -> torch.Tensor:
    """Reshape dim *d* as [tp, per_tp] for TP-rank interleaving."""
    shape = list(t.shape)
    return t.reshape(shape[:d] + [tp, shape[d] // tp] + shape[d + 1:])


def split_qkv(linear: Linear,
              qkv_split: tuple[int, int, int]) -> tuple[Linear, Linear, Linear]:
    """Split combined QKV linear into Q, K, V linears along output dim."""
    wfmt = linear.weight_format
    block_out = (wfmt.block_out or 0) if wfmt is not None else 0
    new_linears = []
    offset = 0
    for dim in qkv_split:
        tensors = {}
        for kind, t in linear.tensors.items():
            out_dim = t.dim() - 1
            if kind in ('scales', 'zeros') and block_out > 0:
                block_offset = offset // block_out
                block_len = dim // block_out
                tensors[kind] = t.narrow(out_dim, block_offset, block_len).contiguous()
            else:
                tensors[kind] = t.narrow(out_dim, offset, dim).contiguous()
        new_linears.append(Linear(tensors=tensors,
                                  weight_format=linear.weight_format))
        offset += dim
    return tuple(new_linears)


def fuse_gdn(q: Linear, k: Linear, v: Linear,
             z: Linear, b: Linear, a: Linear, *,
             tp: int) -> Linear:
    """Fuse GDN input projections with TP interleaving.

    Layout per tp-shard: [Q | K | V | Z | B | A].
    For tp=1 reduces to simple concat along output dim.
    """
    components = [q, k, v, z, b, a]

    if tp <= 1:
        return Linear.concat_out_dim(components)

    first = components[0]
    fused_tensors: dict[str, torch.Tensor] = {}
    for kind in first.tensors:
        parts = []
        all_1d = True
        d = -1
        for lin in components:
            t = lin.tensors.get(kind)
            if t is None:
                continue
            if t.dim() > 1:
                this_d = t.dim() - 1
                if d >= 0 and this_d != d:
                    raise ValueError(
                        f'Inconsistent tensor dims for kind={kind}: '
                        f'{this_d} vs {d}')
                d = this_d
                all_1d = False
                parts.append(tp_interleave_tensor(t, tp, d))
            else:
                # 1-D tensors (bias): simple concat
                parts.append(t)
        if not parts:
            continue
        if all_1d:
            fused_tensors[kind] = torch.cat(parts, dim=0)
        else:
            fused = torch.cat(parts, dim=d + 1)
            shape = list(fused.shape)
            final = shape[:d] + [shape[d] * shape[d + 1]] + shape[d + 2:]
            fused_tensors[kind] = fused.reshape(final)

    return Linear(tensors=fused_tensors, weight_format=first.weight_format)


def fuse_qkv_conv1d(t: torch.Tensor, qkv_split: tuple[int, int, int],
                     tp: int) -> torch.Tensor:
    """Split conv1d into Q/K/V parts, TP-interleave each, concatenate back."""
    q_dim, k_dim, _ = qkv_split
    d_conv = t.shape[0]
    q_part = tp_interleave_tensor(t[:, :q_dim], tp, 1)
    k_part = tp_interleave_tensor(t[:, q_dim:q_dim + k_dim], tp, 1)
    v_part = tp_interleave_tensor(t[:, q_dim + k_dim:], tp, 1)
    return torch.cat([q_part, k_part, v_part], dim=2).reshape(d_conv, -1).contiguous()


# ---------------------------------------------------------------------------
# DeltaNetBuilder -- Gated Delta Net input projections, scalar params, conv1d
# ---------------------------------------------------------------------------


class DeltaNetBuilder(Builder):
    """DeltaNet (Gated Delta Net) weight loading builder."""

    def add_input_projections(self, *, in_proj_qkv, in_proj_z=None,
                              in_proj_b=None, in_proj_a=None, out_proj=None,
                              qkv_split):
        """Fuse GDN input projections via pipeline, commit all linears.

        Pipeline: split_qkv -> ensure_compatible_formats -> fuse_gdn -> commit.
        """
        q, k, v = split_qkv(in_proj_qkv, qkv_split)
        group = _ensure_compatible_formats(
            {'q': q, 'k': k, 'v': v, 'z': in_proj_z, 'b': in_proj_b, 'a': in_proj_a},
            data_type=self.config.data_type)
        fused = fuse_gdn(group['q'], group['k'], group['v'],
                         group['z'], group['b'], group['a'],
                         tp=self._tp)
        self._add_linear('in_proj_all', fused, SplitSide.OUTPUT)
        if out_proj is not None:
            self._add_linear('out_proj', out_proj, SplitSide.INPUT)

    def add_scalar_params(self, a_log=None, dt_bias=None):
        """Commit A_log and dt_bias as OUTPUT-split tensors."""
        if a_log is not None:
            self._add_tensor('A_log', a_log, split_side=SplitSide.OUTPUT)
        if dt_bias is not None:
            self._add_tensor('dt_bias', dt_bias, split_side=SplitSide.OUTPUT)

    def add_conv1d(self, conv1d, qkv_split):
        """Transpose HF layout to TM layout, TP-interleave Q/K/V, commit."""
        if conv1d.ndim == 3 and conv1d.shape[1] == 1:
            conv1d = conv1d.squeeze(1)
        conv1d = conv1d.t().contiguous()
        conv1d = fuse_qkv_conv1d(conv1d, qkv_split, self._tp)
        self._add_tensor('conv1d', conv1d, split_side=SplitSide.OUTPUT)
