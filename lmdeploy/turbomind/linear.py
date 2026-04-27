# Copyright (c) OpenMMLab. All rights reserved.
"""Linear weight bundle and composable dimension operations.

Two weight types flow through the TurboMind weight loading pipeline:

- ``Linear`` -- a bundle of tensors for a single linear layer (weight +
  optional scales, zeros, bias).
- Raw ``torch.Tensor`` -- everything else (norms, embeddings, scalars).

**Tensor functions** ``pad_out_dim`` and ``pad_in_dim`` accept an explicit
``dim`` argument and operate on a single ``torch.Tensor``.

**Linear.concat_out_dim** joins ``Linear`` bundles along the output
dimension, handling all component tensors correctly regardless of
quantization-induced dimension scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .weight_format import WeightFormat


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _norm(dim: int, ndim: int) -> int:
    """Normalise a possibly-negative dimension index."""
    return dim if dim >= 0 else dim + ndim


def _pad_1d(t: Tensor, dim: int, target: int) -> Tensor:
    """Pad one dimension of *t* to *target* size with zeros."""
    deficit = target - t.size(dim)
    if deficit <= 0:
        return t
    pad = [0] * (2 * t.dim())
    # F.pad expects pairs in reverse dim order
    pad[2 * (t.dim() - 1 - dim) + 1] = deficit
    return torch.nn.functional.pad(t, pad, 'constant', 0)


# ---------------------------------------------------------------------------
# Tensor functions
# ---------------------------------------------------------------------------


def pad_out_dim(t: Tensor, target: int, dim: int) -> Tensor:
    """Pad *dim* to *target* size with zeros."""
    return _pad_1d(t, _norm(dim, t.dim()), target)


def pad_in_dim(t: Tensor, target: int, dim: int) -> Tensor:
    """Pad *dim* to *target* size with zeros."""
    return _pad_1d(t, _norm(dim, t.dim()), target)


# ---------------------------------------------------------------------------
# Linear dataclass with methods
# ---------------------------------------------------------------------------


@dataclass
class Linear:
    """Bundle of tensors for a single linear layer.

    ``tensors`` maps a closed-set TM weight kind (e.g. ``"weight"``,
    ``"scales"``, ``"zeros"``, ``"bias"``, ``"qweight"``) to the actual
    tensor.

    **Layout contract**: all ``Linear`` objects are in TM layout with
    axis 0 as the input dimension and axis -1 as the output dimension.
    ``commit_linear`` assumes this layout and does not re-transpose.
    1-D tensors (e.g. bias) only have an output dimension (axis 0).
    """

    tensors: dict[str, Tensor]
    weight_format: WeightFormat = field(compare=False, repr=False)
    @classmethod
    def concat_out_dim(cls, xs: list[Linear]) -> Linear:
        """Concatenate along output dim."""
        first = xs[0]
        result: dict[str, Tensor] = {}
        for kind in first.tensors:
            t = first.tensors[kind]
            result[kind] = torch.cat([x.tensors[kind] for x in xs], dim=t.dim() - 1)
        wfmts = {x.weight_format for x in xs}
        assert len(wfmts) == 1, (
            'concat_out_dim requires uniform weight_format; '
            'call dequant_mixed first if formats differ.')
        return Linear(tensors=result,
                      weight_format=next(iter(wfmts)))
