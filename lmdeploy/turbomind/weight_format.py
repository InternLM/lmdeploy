# Copyright (c) OpenMMLab. All rights reserved.
"""Weight format resolution for TurboMind checkpoint loading.

Exports:

- ``WeightFormat`` (ABC) and six concrete subclasses: ``TrivialFormat``,
  ``AWQFormat``, ``GPTQFormat``, ``CompressedTensorFormat``, ``FP8Format``,
  ``MXFP4Format``. Each subclass declares its ``name``, ``suffix_map``,
  ``weight_dtype`` (``_tm.DataType`` or ``None``), ``has_zero_point`` flag,
  and overrides ``accepts`` + ``normalize``. Optional overrides: ``pack``
  (identity default), ``synthesize_zeros`` (raises by default), ``dequant``
  (raises by default; ``TrivialFormat.dequant`` is identity).

- ``WeightFormatResolver``: holds the model compute dtype plus an ordered
  list of candidate formats. ``resolve(params, prefix, *, index=None,
  optional=False)`` returns a ``Linear`` bundle in TM layout or raises
  (``KeyError`` on missing tensors without ``optional``, ``ValueError`` when
  tensors exist but no candidate matches).

- ``pack_u4_row``: uint8 → int32 row packer used by quantized ``pack``
  overrides and by downstream callers that pack packed-expert weights
  after slicing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

import _turbomind as _tm
import torch
from torch import Tensor

from .linear import Linear


class PackedTensor(NamedTuple):
    tensor:      torch.Tensor
    alloc_shape: list[int] | None       # None = inherit from packed tensor
    alloc_dtype: _tm.DataType | None  # None = inherit from packed tensor


# ---------------------------------------------------------------------------
# Low-level u4 packing / unpacking helpers (reused across normalize / pack)
# ---------------------------------------------------------------------------


def _get_u4_slices(x: Tensor, dtype: torch.dtype) -> list[Tensor]:
    MAP = {torch.int32: 8, torch.uint8: 2}
    xs = []
    for _ in range(MAP[x.dtype]):
        xs.append((x & 15).to(dtype))
        x = x >> 4
    return xs


def _unpack_awq_gemm(x: Tensor) -> Tensor:
    xs = _get_u4_slices(x, torch.uint8)
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    ys = [xs[i] for i in order]
    return torch.stack(ys, dim=-1).view(*x.shape[:-1], -1)


def pack_u4_row(x: torch.Tensor) -> torch.Tensor:
    """Pack uint8 4-bit values into int32 rows along the last dim.

    Used by every int4 format's ``pack`` override and by callers that
    re-pack tensors after slicing (e.g. packed-MoE expert split).
    """
    assert x.dtype == torch.uint8, f'x.dtype: {x.dtype}'
    xs = x.view(*x.shape[:-1], -1, 8).split(1, dim=-1)
    a = torch.zeros(xs[0].shape, dtype=torch.int32, device=x.device)
    for t in reversed(xs):
        a = (a << 4) | t
    return a.squeeze(dim=-1)


def _zeros_int4_symmetric(scales: Tensor) -> Tensor:
    """Synthesize symmetric int4 zero-points (value = 8) matching *scales*
    shape."""
    return torch.full(scales.shape, 8, dtype=torch.uint8, device=scales.device)


# ---------------------------------------------------------------------------
# WeightFormat ABC
# ---------------------------------------------------------------------------


class WeightFormat(ABC):
    """Abstract per-format policy object.

    Class attributes (override in subclasses):

    - ``name``: canonical format name used for string comparisons.
    - ``suffix_map``: ``{checkpoint_suffix: tm_kind}``. Drives which
      checkpoint tensors each format ingests at a given prefix.
    - ``weight_dtype``: ``_tm.DataType`` for the weight storage dtype;
      ``None`` for trivial (weight dtype equals compute dtype).
    - ``has_zero_point``: ``True`` when the format uses a zero-point
      tensor; gates the resolver's ``synthesize_zeros`` call.

    Instance attributes (set by subclass ``__init__``):

    - ``block_in``, ``block_out``: quantization block sizes. ``None`` for
      dimensions without blocking.

    Methods:

    - ``accepts`` (abstract): classify a checkpoint suffix dict.
    - ``normalize`` (abstract): raw-checkpoint tensor → TM layout.
    - ``pack``: optional commit-time packer. Identity default.
    - ``synthesize_zeros``: fabricate a zeros tensor when the checkpoint
      omits it. Raises ``NotImplementedError`` by default.
    - ``dequant``: produce a trivial ``{weight, bias?}`` dict from TM
      tensors for mixed-format fusion. Raises ``NotImplementedError`` by
      default. ``TrivialFormat.dequant`` is identity.
    - ``make_data_format``: build the ``_tm.DataFormat`` descriptor.

    Equality / hashing: two WeightFormats are equal iff they share class
    and block sizes. This matters for the set-based uniformity checks in
    ``concat_out_dim``.
    """

    name:           ClassVar[str]
    suffix_map:     ClassVar[dict[str, str]]
    weight_dtype:   ClassVar[_tm.DataType | None]
    has_zero_point: ClassVar[bool]

    block_in:  int | None
    block_out: int | None

    def __init__(self, *, block_in: int | None = None,
                 block_out: int | None = None):
        self.block_in  = block_in
        self.block_out = block_out

    @abstractmethod
    def accepts(self, available: dict[str, Tensor]) -> bool: ...

    @abstractmethod
    def normalize(self, tensor: Tensor, kind: str) -> Tensor: ...

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        return PackedTensor(tensor, None, None)

    def synthesize_zeros(self, scales: Tensor) -> Tensor:
        raise NotImplementedError(
            f'{type(self).__name__}.synthesize_zeros not implemented')

    def dequant(self, tensors: dict[str, Tensor],
                data_type) -> dict[str, Tensor]:
        raise NotImplementedError(
            f'{type(self).__name__}.dequant not implemented')

    def make_data_format(self, data_type) -> _tm.DataFormat:
        if self.weight_dtype is None:
            return _tm.ResolveLinearWeightFormat(data_type, data_type, 1, 1)
        return _tm.ResolveLinearWeightFormat(
            data_type, self.weight_dtype,
            self.block_in  or 1, self.block_out or 1)

    def __eq__(self, other) -> bool:
        if not isinstance(other, WeightFormat):
            return NotImplemented
        return (type(self) is type(other)
                and self.block_in  == other.block_in
                and self.block_out == other.block_out)

    def __hash__(self) -> int:
        return hash((type(self), self.block_in, self.block_out))


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------


class TrivialFormat(WeightFormat):
    name           = 'trivial'
    suffix_map     = {'.weight': 'weight', '.bias': 'bias'}
    weight_dtype   = None
    has_zero_point = False

    def accepts(self, available: dict[str, Tensor]) -> bool:
        if not (available.keys() <= {'.weight', '.bias'}):
            return False
        w = available.get('.weight')
        return w is None or w.dtype.is_floating_point

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        x = x.cuda()
        if x.dim() >= 2:
            x = x.t()
        return x

    def dequant(self, tensors, data_type):
        # Already trivial — nothing to undo. Identity override for mixed
        # fusion groups.
        return tensors


class AWQFormat(WeightFormat):
    name           = 'awq'
    suffix_map     = {'.qweight': 'weight', '.scales': 'scales',
                      '.qzeros': 'zeros',   '.bias': 'bias'}
    weight_dtype   = _tm.DataType.TYPE_UINT4
    has_zero_point = True

    def __init__(self, *, block_in: int):
        super().__init__(block_in=block_in, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        qw = available.get('.qweight')
        if qw is None or qw.dtype != torch.int32:
            return False
        scales = available.get('.scales')
        if scales is not None and qw.ndim >= 2 and scales.ndim >= 2:
            return qw.shape[-1] * 8 == scales.shape[-1]
        return True

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        # AWQ checkpoints store weights in TM-native layout:
        #   qweight: [K, N//8] int32 → unpack → [K, N] (TM, no .t())
        #   scales:  [K//g, N] float16 → already TM
        #   zeros:   [K//g, N//8] int32 → unpack → [K//g, N]
        x = x.cuda()
        if x.dtype == torch.int32:
            x = _unpack_awq_gemm(x)
        if kind == 'zeros':
            x = x.to(torch.float16)
        return x

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == 'weight' and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor),
                                list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)

    def dequant(self, tensors, data_type):
        from lmdeploy.pytorch.backends.default.awq_modules import dequantize_gemm

        qweight = tensors['weight']
        scales  = tensors['scales']
        qzeros  = tensors['zeros']
        group_size = qweight.shape[0] // scales.shape[0]
        w = dequantize_gemm(qweight, qzeros, scales, 4, group_size)
        result: dict[str, Tensor] = {'weight': w}
        if 'bias' in tensors:
            result['bias'] = tensors['bias']
        return result


class GPTQFormat(WeightFormat):
    name           = 'gptq'
    suffix_map     = {'.qweight': 'weight', '.scales': 'scales',
                      '.qzeros': 'zeros',   '.bias': 'bias'}
    weight_dtype   = _tm.DataType.TYPE_UINT4
    has_zero_point = True

    def __init__(self, *, block_in: int):
        super().__init__(block_in=block_in, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        qw = available.get('.qweight')
        if qw is None or qw.dtype != torch.int32:
            return False
        scales = available.get('.scales')
        if scales is not None and qw.ndim >= 2 and scales.ndim >= 2:
            return qw.shape[-1] == scales.shape[-1]
        return True

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        # GPTQ checkpoint stores weights in TM-native layout:
        #   qweight: [K//8, N] int32 → unpack → [K, N]
        #   scales:  [K//g, N] float16 → already TM
        #   zeros:   [K//g, N//8] int32 → unpack → [K//g, N] (+1 offset)
        x = x.cuda()
        if x.dtype == torch.int32:
            xs = _get_u4_slices(x, torch.uint8)
            if kind == 'weight':
                x = torch.stack(xs, dim=1).view(-1, x.size(-1))
            else:
                x = torch.stack(xs, dim=-1).view(x.size(0), -1) + 1
        if kind == 'zeros':
            x = x.to(torch.float16)
        return x

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == 'weight' and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor),
                                list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)

    def synthesize_zeros(self, scales: Tensor) -> Tensor:
        return _zeros_int4_symmetric(scales)


class CompressedTensorFormat(WeightFormat):
    name           = 'compressed-tensors'
    suffix_map     = {'.weight_packed':     'weight',
                      '.weight_scale':      'scales',
                      '.weight_zero_point': 'zeros',
                      '.bias':              'bias'}
    weight_dtype   = _tm.DataType.TYPE_UINT4
    has_zero_point = True

    def __init__(self, *, block_in: int):
        super().__init__(block_in=block_in, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        wp = available.get('.weight_packed')
        return wp is not None and wp.dtype == torch.int32

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        x = x.cuda()
        if x.dtype == torch.int32:
            xs = _get_u4_slices(x, torch.uint8)
            if kind == 'weight':
                x = torch.stack(xs, dim=-1).view(*x.shape[:-1], -1)
            elif kind == 'zeros':
                x = torch.stack(xs, dim=1).view(-1, x.size(-1))
        if kind == 'zeros':
            x = x.to(torch.float16)
        if x.dim() >= 2:
            x = x.t()
        return x

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == 'weight' and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor),
                                list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)

    def synthesize_zeros(self, scales: Tensor) -> Tensor:
        return _zeros_int4_symmetric(scales)


class FP8Format(WeightFormat):
    name           = 'fp8'
    suffix_map     = {'.weight':           'weight',
                      '.weight_scale_inv': 'scales',
                      '.bias':             'bias'}
    weight_dtype   = _tm.DataType.TYPE_FP8_E4M3
    has_zero_point = False

    def __init__(self):
        super().__init__(block_in=128, block_out=128)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        if '.weight_scale_inv' not in available:
            return False
        w = available.get('.weight')
        return w is None or w.dtype in (torch.float8_e4m3fn, torch.uint8)

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        x = x.cuda()
        if x.dtype == torch.float8_e4m3fn:
            x = x.view(dtype=torch.uint8)
        if x.dim() >= 2:
            x = x.t()
        return x

    def dequant(self, tensors, data_type):
        from .builders._base import _CPP_TO_TORCH

        weight = tensors['weight']
        scales = tensors['scales']
        block_size = 128
        fp8_weight = weight.view(torch.float8_e4m3fn).float()
        scale = scales.float()
        scale = scale.repeat_interleave(block_size, dim=0)
        scale = scale.repeat_interleave(block_size, dim=1)
        scale = scale[: fp8_weight.shape[0], : fp8_weight.shape[1]]
        target_dtype = _CPP_TO_TORCH[data_type]
        result: dict[str, Tensor] = {'weight': (fp8_weight * scale).to(target_dtype)}
        if 'bias' in tensors:
            result['bias'] = tensors['bias']
        return result

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == 'weight':
            return PackedTensor(tensor, list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)


class MXFP4Format(WeightFormat):
    name           = 'mxfp4'
    suffix_map     = {'.blocks': 'weight', '.scales': 'scales', '.bias': 'bias'}
    weight_dtype   = _tm.DataType.TYPE_FP4_E2M1
    has_zero_point = False

    def __init__(self):
        super().__init__(block_in=32, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        if '.scales' not in available:
            return False
        w = available.get('.blocks')
        return w is None or w.dtype == torch.uint8

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        x = x.cuda()
        if kind == 'weight':
            xs = _get_u4_slices(torch.flatten(x, start_dim=-2), torch.uint8)
            x = torch.flatten(torch.stack(xs, dim=-1), start_dim=-2)
        if x.dim() >= 2:
            x = x.t()
        return x

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == 'weight' and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor),
                                list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class WeightFormatResolver:
    """Resolve a checkpoint prefix to a ``Linear`` bundle in TM layout.

    Holds the model compute dtype and an ordered list of candidate
    formats. ``resolve(params, prefix)`` probes the checkpoint at the
    given prefix, dispatches to the first candidate whose ``accepts``
    returns True, and constructs a ``Linear`` with the format's
    ``make_data_format`` descriptor.

    The suffix probe is scoped to the union of candidate ``suffix_map``
    keys only — not a global "every format ever" list — so adding a new
    format elsewhere does not widen the probe.

    Priority is encoded by list order. The converter puts quantized
    candidates first and ``TrivialFormat()`` last: a prefix that only
    matches trivial (router, norm-like linears in a quantized model)
    deterministically falls through.

    Failure modes are loud and distinct:

    - ``optional=False`` (default) + no tensors at prefix → ``KeyError``
      with candidate suffix list.
    - Tensors present but no candidate accepts → ``ValueError`` with
      available keys and candidate names.
    - Only "no tensors AND optional=True" returns ``None``.
    """

    def __init__(self, *, data_type: _tm.DataType,
                 formats: list[WeightFormat]):
        self._data_type = data_type
        self._formats   = formats
        self._suffixes  = frozenset(
            s for f in formats for s in f.suffix_map)

    @property
    def data_type(self) -> _tm.DataType:
        return self._data_type

    def resolve(self, params: dict[str, Tensor], prefix: str, *,
                index: int | None = None,
                optional: bool = False) -> Linear | None:
        available = {s: params[prefix + s]
                     for s in self._suffixes if (prefix + s) in params}
        if index is not None:
            available = {s: t[index] for s, t in available.items()}

        if not available:
            if optional:
                return None
            raise KeyError(
                f'no checkpoint tensors found at prefix {prefix!r} '
                f'(candidate suffixes: {sorted(self._suffixes)})')

        for fmt in self._formats:
            if fmt.accepts(available):
                return self._build_linear(fmt, available)

        raise ValueError(
            f'no weight format accepts tensors at {prefix!r}: '
            f'got {sorted(available)}, '
            f'tried {[f.name for f in self._formats]}')

    def _build_linear(self, fmt: WeightFormat,
                      available: dict[str, Tensor]) -> Linear:
        tensors = {
            kind: fmt.normalize(available[s], kind)
            for s, kind in fmt.suffix_map.items()
            if s in available
        }
        if fmt.has_zero_point and 'zeros' not in tensors:
            tensors['zeros'] = fmt.synthesize_zeros(tensors['scales'])
        return Linear(tensors=tensors,
                      weight_format=fmt)
