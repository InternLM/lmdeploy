# Copyright (c) OpenMMLab. All rights reserved.
"""Linear weight bundle and composable dimension operations.

Two weight types flow through the TurboMind weight loading pipeline:

- ``Linear`` -- a bundle of tensors for a single linear layer (weight +
  optional scales, zeros, bias).
- Raw ``torch.Tensor`` -- everything else (norms, embeddings, scalars).

**Tensor functions** ``pad_out_dim`` and ``pad_in_dim`` accept an explicit
``dim`` argument and operate on a single ``torch.Tensor``.

**concat_out_dim** joins ``Linear`` bundles along the output
dimension, handling all component tensors correctly regardless of
quantization-induced dimension scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import functools
import inspect

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


def concat_out_dim(xs: list[Linear]) -> Linear:
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


# ---------------------------------------------------------------------------
# Format / compatibility utilities
# ---------------------------------------------------------------------------


def _dequant_linear(linear: Linear, *, data_type) -> Linear:
    """Dequantize a quantized Linear to trivial.

    ``TrivialFormat.dequant`` is identity, so already-trivial inputs round-trip
    safely.  ``AWQFormat.dequant`` and ``FP8Format.dequant`` do real work.
    GPTQ / CompressedTensor / MXFP4 inherit the base-class
    ``NotImplementedError`` — calling ``_dequant_linear`` on one of those is a
    broken-fusion-group configuration, and the raise names it at the call site.
    """
    from .weight_format import TrivialFormat

    fmt = linear.weight_format
    new_tensors = fmt.dequant(linear.tensors, data_type)
    trivial = TrivialFormat()
    return Linear(tensors=new_tensors, weight_format=trivial)


def dequant_mixed(*linears: Linear | None, data_type) -> tuple[Linear | None, ...]:
    """Dequantize linears to a common trivial format when formats differ.

    Trivial inputs round-trip safely through ``_dequant_linear``.
    None args pass through unchanged.
    """
    formats = {l.weight_format.name for l in linears if l is not None}
    if len(formats) <= 1:
        return linears
    return tuple(
        _dequant_linear(l, data_type=data_type) if l is not None else l
        for l in linears
    )


# ---------------------------------------------------------------------------
# Linear-level transform decorators
# ---------------------------------------------------------------------------


def transform_output_dim(fn):
    """Decorator that lifts a tensor-level transform to Linear-level.

    For output-dim operations: 1-D tensors (bias) are unsqueezed to 2-D
    before calling *fn*, then squeezed back.  Convention: args that are
    ``Linear`` instances are treated as tensor inputs; all other args pass
    through unchanged.  Return type is detected at runtime:
    ``Tensor`` -> single ``Linear``, ``tuple`` -> tuple of ``Linear`` objects.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        first = next(
            v for v in bound.arguments.values() if isinstance(v, Linear)
        )
        out_buckets = None

        for kind in first.tensors:
            was_1d = False
            fn_kwargs = {}

            for name, val in bound.arguments.items():
                if isinstance(val, Linear):
                    t = val.tensors[kind]
                    if t.dim() == 1:
                        was_1d = True
                        t = t.unsqueeze(0)
                    fn_kwargs[name] = t
                else:
                    fn_kwargs[name] = val

            result = fn(**fn_kwargs)
            if not isinstance(result, tuple):
                result = (result,)
            if out_buckets is None:
                out_buckets = [{} for _ in result]
            for i, item in enumerate(result):
                out_buckets[i][kind] = item.squeeze(0) if was_1d else item

        outputs = tuple(
            Linear(ts, weight_format=first.weight_format) for ts in out_buckets
        )
        return outputs if len(outputs) > 1 else outputs[0]

    return wrapper


def transform_input_dim(fn):
    """Decorator that lifts a tensor-level transform to Linear-level.

    For input-dim operations: 1-D tensors (bias) have no input dimension
    and are **passed through unchanged**.  The inner function only ever
    sees 2-D tensors for each kind.  For multi-output functions, 1-D
    tensors are duplicated into every output bucket.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        first = next(
            v for v in bound.arguments.values() if isinstance(v, Linear)
        )
        out_buckets = None
        deferred_1d: list[str] = []

        for kind in first.tensors:
            fn_kwargs = {}
            is_1d = False

            for name, val in bound.arguments.items():
                if isinstance(val, Linear):
                    t = val.tensors[kind]
                    if t.dim() < 2:
                        is_1d = True
                        break
                    fn_kwargs[name] = t
                else:
                    fn_kwargs[name] = val

            if is_1d:
                deferred_1d.append(kind)
                continue

            result = fn(**fn_kwargs)
            if not isinstance(result, tuple):
                result = (result,)
            if out_buckets is None:
                out_buckets = [{} for _ in result]
            for i, item in enumerate(result):
                out_buckets[i][kind] = item

        if out_buckets is None:
            out_buckets = [{}]
        for kind in deferred_1d:
            for bucket in out_buckets:
                bucket[kind] = first.tensors[kind]

        outputs = tuple(
            Linear(ts, weight_format=first.weight_format) for ts in out_buckets
        )
        return outputs if len(outputs) > 1 else outputs[0]

    return wrapper
