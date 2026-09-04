# Copyright (c) OpenMMLab. All rights reserved.
"""Weight format resolution for TurboMind checkpoint loading.

Exports:

- ``WeightFormat`` (ABC) and six concrete subclasses: ``TrivialFormat``,
  ``AWQFormat``, ``GPTQFormat``, ``CompressedTensorFormat``, ``FP8Format``,
  ``MXFP4Format``. Each subclass declares its ``name``, ``suffix_map``,
  complete data-format dtypes, and overrides ``accepts`` + ``normalize``.
  Optional overrides: ``pack``
  (identity default), ``synthesize_zeros`` (raises by default), ``dequant``
  (raises by default; ``TrivialFormat.dequant`` is identity).

- ``WeightFormatResolver``: holds an ordered list of candidate formats.
  ``resolve(params, prefix, *, index=None,
  optional=False)`` returns the selected format and raw checkpoint tensors or raises
  (``KeyError`` on missing tensors without ``optional``, ``ValueError`` when
  tensors exist but no candidate matches).

- ``pack_u4_row``: uint8 → int32 row packer used by quantized ``pack``
  overrides and by downstream callers that pack packed-expert weights
  after slicing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

import torch
from torch import Tensor

from . import _tm

_GENERIC_FLOAT_DTYPES = frozenset({
    torch.float16,
    torch.bfloat16,
    torch.float32,
})

_TRIVIAL_DTYPES = {
    _tm.DataType.TYPE_FP16: torch.float16,
    _tm.DataType.TYPE_BF16: torch.bfloat16,
    _tm.DataType.TYPE_FP32: torch.float32,
}


class PackedTensor(NamedTuple):
    tensor: torch.Tensor
    alloc_shape: list[int] | None  # None = inherit from packed tensor
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
    assert x.dtype == torch.uint8, f"x.dtype: {x.dtype}"
    xs = x.view(*x.shape[:-1], -1, 8).split(1, dim=-1)
    a = torch.zeros(xs[0].shape, dtype=torch.int32, device=x.device)
    for t in reversed(xs):
        a = (a << 4) | t
    return a.squeeze(dim=-1)


def _zeros_int4_symmetric(scales: Tensor) -> Tensor:
    """Synthesize normalized symmetric int4 zero-points (value = 8) matching
    *scales* shape."""
    return torch.full(scales.shape, 8, dtype=scales.dtype, device=scales.device)


# ---------------------------------------------------------------------------
# WeightFormat ABC
# ---------------------------------------------------------------------------


class WeightFormat(ABC):
    """Abstract per-format policy object.

    Class attributes (override in subclasses):

    - ``name``: canonical format name used for string comparisons.
    - ``suffix_map``: ``{checkpoint_suffix: tm_kind}``. Drives which
      checkpoint tensors each format ingests at a given prefix.
    - ``weight_dtype``: weight storage dtype.
    - ``scales_dtype`` / ``zeros_dtype``: qparameter format dtypes or
      ``TYPE_INVALID`` when absent.

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

    Equality / hashing cover the complete normalized format. This matters
    for the set-based uniformity checks in ``concat_out_dim``.
    """

    name: ClassVar[str]
    suffix_map: ClassVar[dict[str, str]]
    weight_dtype: _tm.DataType
    scales_dtype: ClassVar[_tm.DataType]
    zeros_dtype: ClassVar[_tm.DataType]

    block_in: int | None
    block_out: int | None

    def __init__(self, *, block_in: int | None = None, block_out: int | None = None):
        self.block_in = block_in
        self.block_out = block_out

    @abstractmethod
    def accepts(self, available: dict[str, Tensor]) -> bool: ...

    @abstractmethod
    def normalize(self, tensor: Tensor, kind: str) -> Tensor: ...

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        return PackedTensor(tensor, None, None)

    def synthesize_zeros(self, scales: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.synthesize_zeros not implemented")

    def dequant(self, tensors: dict[str, Tensor], dtype: torch.dtype) -> dict[str, Tensor]:
        raise NotImplementedError(f"{type(self).__name__}.dequant not implemented")

    def make_data_format(self) -> _tm.DataFormat:
        return _tm.DataFormat(
            self.weight_dtype,
            [self.block_in or 1, self.block_out or 1],
            self.scales_dtype,
            self.zeros_dtype,
        )

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self.weight_dtype == other.weight_dtype
            and self.scales_dtype == other.scales_dtype
            and self.zeros_dtype == other.zeros_dtype
            and self.block_in == other.block_in
            and self.block_out == other.block_out
        )

    def __hash__(self) -> int:
        return hash((
            type(self),
            self.weight_dtype,
            self.scales_dtype,
            self.zeros_dtype,
            self.block_in,
            self.block_out,
        ))


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------


class TrivialFormat(WeightFormat):
    name = "trivial"
    suffix_map = {".weight": "weight", ".bias": "bias"}
    scales_dtype = _tm.DataType.TYPE_INVALID
    zeros_dtype = _tm.DataType.TYPE_INVALID

    def __init__(self, *, weight_dtype: _tm.DataType):
        self.weight_dtype = weight_dtype
        super().__init__()

    def accepts(self, available: dict[str, Tensor]) -> bool:
        if not (available.keys() <= {".weight", ".bias"}):
            return False
        w = available.get(".weight")
        return w is None or w.dtype.is_floating_point

    def normalize(self, tensor: Tensor, kind: str) -> Tensor:
        tensor = tensor.to(_TRIVIAL_DTYPES[self.weight_dtype])
        if tensor.dim() >= 2:
            tensor = tensor.t()
        return tensor

    def dequant(self, tensors, dtype):
        # Already trivial — nothing to undo. Identity override for mixed
        # fusion groups.
        return tensors


class AWQFormat(WeightFormat):
    name = "awq"
    suffix_map = {".qweight": "weight", ".scales": "scales", ".qzeros": "zeros", ".bias": "bias"}
    weight_dtype = _tm.DataType.TYPE_UINT4
    scales_dtype = _tm.DataType.TYPE_GENERIC_FLOAT
    zeros_dtype = _tm.DataType.TYPE_GENERIC_FLOAT

    def __init__(self, *, block_in: int):
        super().__init__(block_in=block_in, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        weight = available.get(".qweight")
        scales = available.get(".scales")
        zeros = available.get(".qzeros")
        if weight is None or weight.dtype != torch.int32:
            return False
        if scales is None or scales.dtype not in _GENERIC_FLOAT_DTYPES:
            return False
        if zeros is None or zeros.dtype != torch.int32:
            return False
        if weight.ndim >= 2 and scales.ndim >= 2:
            return weight.shape[-1] * 8 == scales.shape[-1]
        return True

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        # AWQ checkpoints store weights in TM-native layout:
        #   qweight: [K, N//8] int32 → unpack → [K, N] (TM, no .t())
        #   scales:  [K//g, N] float16 → already TM
        #   zeros:   [K//g, N//8] int32 → unpack → [K//g, N]
        if x.dtype == torch.int32:
            x = _unpack_awq_gemm(x)
        return x

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor), list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)

    def dequant(self, tensors, dtype):
        qweight = tensors["weight"]
        scales = tensors["scales"]
        qzeros = tensors["zeros"]
        group_size = qweight.shape[0] // scales.shape[0]
        w = qweight.unflatten(0, (-1, group_size))
        w = (w - qzeros[:, None]) * scales[:, None]
        w = w.flatten(0, 1)
        result: dict[str, Tensor] = {"weight": w}
        if "bias" in tensors:
            result["bias"] = tensors["bias"]
        return result


class GPTQFormat(WeightFormat):
    name = "gptq"
    suffix_map = {".qweight": "weight", ".scales": "scales", ".qzeros": "zeros", ".bias": "bias"}
    weight_dtype = _tm.DataType.TYPE_UINT4
    scales_dtype = _tm.DataType.TYPE_GENERIC_FLOAT
    zeros_dtype = _tm.DataType.TYPE_GENERIC_FLOAT

    def __init__(self, *, block_in: int):
        super().__init__(block_in=block_in, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        qw = available.get(".qweight")
        if qw is None or qw.dtype != torch.int32:
            return False
        scales = available.get(".scales")
        if scales is None or scales.dtype not in _GENERIC_FLOAT_DTYPES:
            return False
        zeros = available.get(".qzeros")
        if zeros is not None and zeros.dtype != torch.int32:
            return False
        if qw.ndim >= 2 and scales.ndim >= 2:
            return qw.shape[-1] == scales.shape[-1]
        return True

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        # GPTQ checkpoint stores weights in TM-native layout:
        #   qweight: [K//8, N] int32 → unpack → [K, N]
        #   scales:  [K//g, N] float16 → already TM
        #   zeros:   [K//g, N//8] int32 → unpack → [K//g, N] (+1 offset)
        if x.dtype == torch.int32:
            xs = _get_u4_slices(x, torch.uint8)
            if kind == "weight":
                x = torch.stack(xs, dim=1).view(-1, x.size(-1))
            else:
                x = torch.stack(xs, dim=-1).view(x.size(0), -1) + 1
        return x

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor), list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)

    def synthesize_zeros(self, scales: Tensor) -> Tensor:
        return _zeros_int4_symmetric(scales)


class CompressedTensorFormat(WeightFormat):
    name = "compressed-tensors"
    suffix_map = {".weight_packed": "weight", ".weight_scale": "scales", ".weight_zero_point": "zeros", ".bias": "bias"}
    weight_dtype = _tm.DataType.TYPE_UINT4
    scales_dtype = _tm.DataType.TYPE_GENERIC_FLOAT
    zeros_dtype = _tm.DataType.TYPE_GENERIC_FLOAT

    def __init__(self, *, block_in: int):
        super().__init__(block_in=block_in, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        weight = available.get(".weight_packed")
        scales = available.get(".weight_scale")
        zeros = available.get(".weight_zero_point")
        if weight is None or weight.dtype != torch.int32:
            return False
        if scales is None or scales.dtype not in _GENERIC_FLOAT_DTYPES:
            return False
        if zeros is not None and zeros.dtype != torch.int32:
            return False
        return True

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        if x.dtype == torch.int32:
            xs = _get_u4_slices(x, torch.uint8)
            if kind == "weight":
                x = torch.stack(xs, dim=-1).view(*x.shape[:-1], -1)
            elif kind == "zeros":
                x = torch.stack(xs, dim=1).view(-1, x.size(-1))
        if x.dim() >= 2:
            x = x.t()
        return x

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor), list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)

    def synthesize_zeros(self, scales: Tensor) -> Tensor:
        return _zeros_int4_symmetric(scales)

    def dequant(self, tensors, dtype):
        weight = tensors["weight"]
        scales = tensors["scales"]
        zeros = tensors["zeros"]

        out_size = weight.shape[-1]
        zeros = zeros[..., :out_size]

        scales = scales.repeat_interleave(self.block_in, dim=0)[: weight.shape[0]]
        zeros = zeros.repeat_interleave(self.block_in, dim=0)[: weight.shape[0]]
        w = (weight.to(scales.dtype) - zeros.to(scales.dtype)) * scales
        result: dict[str, Tensor] = {"weight": w}
        if "bias" in tensors:
            result["bias"] = tensors["bias"]
        return result


class FP8Format(WeightFormat):
    name = "fp8"
    suffix_map = {".weight": "weight", ".weight_scale_inv": "scales", ".bias": "bias"}
    weight_dtype = _tm.DataType.TYPE_FP8_E4M3
    scales_dtype = _tm.DataType.TYPE_GENERIC_FLOAT
    zeros_dtype = _tm.DataType.TYPE_INVALID

    def __init__(self, *, block_out: int):
        if block_out not in (1, 128):
            raise ValueError(f"unsupported_fp8_block_out_{block_out}")
        super().__init__(block_in=128, block_out=block_out)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        scales = available.get(".weight_scale_inv")
        if scales is None or scales.dtype not in _GENERIC_FLOAT_DTYPES:
            return False
        w = available.get(".weight")
        if w is None:
            return False
        if w.dtype not in (torch.float8_e4m3fn, torch.uint8):
            return False
        if w.dim() < 2 or scales.dim() < 2:
            return False
        expected = (
            (w.shape[-2] + self.block_out - 1) // self.block_out,
            (w.shape[-1] + self.block_in - 1) // self.block_in,
        )
        return scales.shape[-2:] == expected

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        if x.dtype == torch.float8_e4m3fn:
            x = x.view(dtype=torch.uint8)
        if x.dim() >= 2:
            x = x.t()
        return x

    def dequant(self, tensors, dtype):
        weight = tensors["weight"]
        scales = tensors["scales"]
        fp8_weight = weight.view(torch.float8_e4m3fn).float()
        scale = scales.float()
        scale = scale.repeat_interleave(self.block_in, dim=0)
        scale = scale.repeat_interleave(self.block_out or 1, dim=1)
        scale = scale[: fp8_weight.shape[0], : fp8_weight.shape[1]]
        result: dict[str, Tensor] = {"weight": (fp8_weight * scale).to(dtype)}
        if "bias" in tensors:
            result["bias"] = tensors["bias"]
        return result

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == "weight":
            return PackedTensor(tensor, list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)


class MXFP4Format(WeightFormat):
    name = "mxfp4"
    suffix_map = {
        ".blocks": "weight",
        ".scales": "scales",
        ".bias": "bias",
    }
    weight_dtype = _tm.DataType.TYPE_FP4_E2M1
    scales_dtype = _tm.DataType.TYPE_UINT8
    zeros_dtype = _tm.DataType.TYPE_INVALID

    def __init__(self):
        super().__init__(block_in=32, block_out=None)

    def accepts(self, available: dict[str, Tensor]) -> bool:
        scales = available.get(".scales")
        if scales is None or scales.dtype != torch.uint8:
            return False
        w = available.get(".blocks")
        if w is None or w.dtype != torch.uint8:
            return False
        return w.numel() == scales.numel() * 16

    def normalize(self, x: Tensor, kind: str) -> Tensor:
        if kind == "weight":
            xs = _get_u4_slices(torch.flatten(x, start_dim=-2), torch.uint8)
            x = torch.flatten(torch.stack(xs, dim=-1), start_dim=-2)
        if x.dim() >= 2:
            x = x.t()
        return x

    def pack(self, tensor: Tensor, kind: str) -> PackedTensor:
        if kind == "weight" and tensor.dtype == torch.uint8:
            return PackedTensor(pack_u4_row(tensor), list(tensor.shape), self.weight_dtype)
        return PackedTensor(tensor, None, None)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class WeightFormatResolver:
    """Resolve a checkpoint prefix to a weight format and raw tensors.

    Holds an ordered list of candidate formats. ``resolve(params, prefix)``
    probes the checkpoint at the
    given prefix, dispatches to the first candidate whose ``accepts``
    returns True, and returns that format with the tensors it accepted.

    The suffix probe is scoped to the union of candidate ``suffix_map``
    keys only — not a global "every format ever" list — so adding a new
    format elsewhere does not widen the probe.

    Priority is encoded by list order. The converter puts quantized
    candidates first and a concrete ``TrivialFormat`` last: a prefix that only
    matches trivial (router, norm-like linears in a quantized model)
    deterministically falls through.

    Failure modes are loud and distinct:

    - ``optional=False`` (default) + no tensors at prefix → ``KeyError``
      with candidate suffix list.
    - Tensors present but no candidate accepts → ``ValueError`` with
      available keys and candidate names.
    - Only "no tensors AND optional=True" returns ``None``.
    """

    def __init__(self, *, formats: list[WeightFormat]):
        self._formats = formats
        self._suffixes = frozenset(s for f in formats for s in f.suffix_map)

    def resolve(self, pfx, *, index: int | None = None, optional: bool = False) -> tuple[WeightFormat, dict[str, Tensor]] | None:
        """Resolve the selected format and its raw checkpoint tensors."""
        read = pfx.get if index is not None else pfx.pop
        available = {s: read(s, sep="", index=index) for s in self._suffixes if pfx.has(s, sep="")}

        if not available:
            if optional:
                return None
            raise KeyError(
                f"no checkpoint tensors found at prefix {pfx.prefix!r} (candidate suffixes: {sorted(self._suffixes)})"
            )

        for fmt in self._formats:
            if fmt.accepts(available):
                return fmt, available

        raise ValueError(
            f"no weight format accepts tensors at {pfx.prefix!r}: "
            f"got {sorted(available)}, "
            f"tried {[f.name for f in self._formats]}"
        )
