"""Torch-facing TurboMind linear API.

Use ``Linear.get_weight_plan()`` followed by ``prepare_weight()`` or ``fuse_weight()`` to create a reusable ``Weight``, then obtain an ``ExecPlan`` with ``get_exec_plan()`` or ``tune()`` and execute with ``Linear`` or ``forward_moe()``. Execution uses the current Torch CUDA stream and allocates outputs from the plan when destinations are omitted.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

import torch

from lmdeploy.turbomind import _tm

if TYPE_CHECKING:
    from lmdeploy.turbomind.weight_format import WeightFormat

__all__ = ['ExecPlan', 'Linear', 'Weight', 'WeightPlan']


_TORCH_TO_TM_NAME = {torch.uint8: 'TYPE_UINT8', torch.int32: 'TYPE_INT32', torch.float16: 'TYPE_FP16', torch.bfloat16: 'TYPE_BF16', torch.float32: 'TYPE_FP32', torch.float8_e4m3fn: 'TYPE_FP8_E4M3'}


def _to_tm_dtype(dtype):
    """Convert a Torch dtype to its TurboMind data type."""
    return getattr(_tm.DataType, _TORCH_TO_TM_NAME[dtype])


def _to_torch_dtype(dtype):
    """Convert a TurboMind data type to its Torch dtype."""
    return next(torch_dtype for torch_dtype, name in _TORCH_TO_TM_NAME.items() if dtype == getattr(_tm.DataType, name))


class WeightPlan:
    """Selected family and preparation contract for a source weight format."""

    __slots__ = ('_impl', '_weight_format', '_dtype', '_grouped', '_shape_constraints')

    @property
    def shape_constraints(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return minimum and alignment constraints for ``(K, N)``."""
        return self._shape_constraints


class ExecPlan:
    """Selected kernel and output allocation contract for one execution problem."""

    __slots__ = ('_impl',)

    @property
    def output(self) -> torch.Tensor:
        """Return a meta tensor describing the required output."""
        spec = self._impl
        return torch.empty_strided(spec.output_shape, spec.output_stride, dtype=_to_torch_dtype(spec.output_dtype), device='meta')

    @property
    def output_scales(self) -> torch.Tensor | None:
        """Return a meta tensor describing output scales, if required."""
        spec = self._impl
        if spec.output_scales_dtype == _tm.DataType.TYPE_INVALID:
            return None
        return torch.empty_strided(spec.output_scales_shape, spec.output_scales_stride, dtype=_to_torch_dtype(spec.output_scales_dtype), device='meta')


class Weight:
    """Prepared native weight handle owned by the caller."""

    __slots__ = ('_impl', '_experts')

    @property
    def is_graph_compatible(self) -> bool:
        """Return whether the prepared family supports CUDA graph capture."""
        return bool(self._impl.is_graph_compatible)

    def close(self) -> None:
        """Release the prepared weight and any retained expert weights."""
        experts = self._experts
        self._impl = None
        self._experts = None
        if experts is not None:
            for expert in experts:
                expert.close()


class Linear:
    """Plan, prepare, tune, and execute TurboMind linear operations from Torch."""

    __slots__ = ('device', '_impl', '_context')

    def __init__(self, device: torch.device | str | int = 'cuda'):
        """Create an executor for a CUDA device."""
        if isinstance(device, int):
            resolved_device = torch.device('cuda', device)
        else:
            resolved_device = torch.device(device)
        if resolved_device.index is None:
            resolved_device = torch.device('cuda', torch.cuda.current_device())
        self.device = resolved_device
        self._impl = None
        self._context = None
        with self._activate():
            try:
                self._impl = _tm.LlamaLinear()
            except Exception:
                self._impl = None
                raise

    @contextmanager
    def _activate(self):
        """Activate the cached native context for the current Torch CUDA stream."""
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            context = self._context
            if context is None or context.stream_ptr != stream.cuda_stream:
                context = _tm.create_device_context(stream.cuda_stream)
                self._context = context
            with context:
                yield stream

    def close(self) -> None:
        """Release the native executor on its device."""
        with self._activate():
            self._impl = None
        self._context = None

    def get_weight_plan(self, *, weight_format: WeightFormat, dtype: torch.dtype, input_dtype: torch.dtype | None = None, output_dtype: torch.dtype | None = None, grouped: bool = False, fusion_type: Literal['silu'] | None = None) -> WeightPlan:
        """Select a weight family and expose its source-shape constraints."""
        weight_format = copy.copy(weight_format)
        data_format = weight_format.make_data_format()
        query = _tm.WeightQuery()
        query.weight_format = data_format
        query.data_type = _to_tm_dtype(dtype)
        query.input_dtype = _to_tm_dtype(dtype if input_dtype is None else input_dtype)
        query.output_dtype = _to_tm_dtype(dtype if output_dtype is None else output_dtype)
        query.grouped = grouped

        impl = self._impl.get_weight_plan(query)
        if impl is None:
            raise NotImplementedError(f'no GEMM family accepts format={data_format}, dtype={dtype}, input_dtype={input_dtype}, output_dtype={output_dtype}, grouped={grouped}')

        minimum, alignment = impl.shape_constraints
        min_k, min_n = minimum
        align_k, align_n = alignment
        alignment = (align_k, math.lcm(align_n, weight_format.block_out or 1))

        if fusion_type == 'silu':
            block_out = weight_format.block_out or 1
            if block_out == 128 and (dtype if input_dtype is None else input_dtype) == torch.bfloat16:
                raise NotImplementedError('BF16-input block-out-128 FP8 SiLU fusion is not supported')
            minimum = (min_k, (min_n + 1) // 2)
            alignment = (align_k, math.lcm(block_out, align_n // math.gcd(align_n, 2)))

        result = WeightPlan()
        result._impl = impl
        result._weight_format = weight_format
        result._dtype = dtype
        result._grouped = grouped
        result._shape_constraints = (minimum, alignment)
        return result

    def _normalize_params(self, weight_format, weight, scales, zeros):
        """Normalize source weight components into TurboMind logical layouts."""
        raw = {'weight': weight, 'scales': scales, 'zeros': zeros}
        raw = {kind: tensor for kind, tensor in raw.items() if tensor is not None}
        normalized = {kind: weight_format.normalize(tensor, kind).contiguous() for kind, tensor in raw.items()}
        if weight_format.zeros_dtype != _tm.DataType.TYPE_INVALID and 'zeros' not in normalized:
            normalized['zeros'] = weight_format.synthesize_zeros(normalized['scales']).contiguous()
        if 'zeros' in normalized:
            normalized['zeros'] = normalized['zeros'].to(normalized['scales'].dtype).contiguous()
        return normalized

    def _copy_param(self, impl, name, src, *, logical_shape, logical_dtype, stream):
        """Allocate a native parameter and copy its source bytes on the active stream."""
        dst = impl.param(name).alloc(logical_shape, logical_dtype)
        _tm.copy_bytes_on_stream(src, dst, stream.cuda_stream)

    def _prepare_one(self, normalized, *, weight_format, plan, dtype, stored_output_dim, stream):
        """Allocate and pack one normalized weight with a selected native plan."""
        source_weight = normalized['weight']
        input_dim = source_weight.shape[0]

        data_format = weight_format.make_data_format()
        config = _tm.LinearConfig()
        config.input_dim = input_dim
        config.output_dim = stored_output_dim
        config.data_type = _to_tm_dtype(dtype)
        config.format = data_format
        config.has_bias = False

        impl = _tm.LinearWeight(config)
        impl.set_plan(plan)
        packed = {kind: weight_format.pack(tensor, kind) for kind, tensor in normalized.items()}
        for kind, item in packed.items():
            tensor = item.tensor
            logical_shape = (list(tensor.shape) if item.alloc_shape is None else list(item.alloc_shape))
            logical_dtype = (_to_tm_dtype(tensor.dtype) if item.alloc_dtype is None else item.alloc_dtype)
            self._copy_param(impl, kind, tensor, logical_shape=logical_shape, logical_dtype=logical_dtype, stream=stream)

        impl.prepare()

        handle = Weight()
        handle._impl = impl
        handle._experts = None
        return handle

    def _prepare_grouped(self, normalized_experts, *, weight_format, plan, dtype, stored_output_dim, stream):
        """Prepare expert weights and link them into one grouped weight handle."""
        experts = []
        try:
            for params in normalized_experts:
                experts.append(self._prepare_one(params, weight_format=weight_format, plan=plan, dtype=dtype, stored_output_dim=stored_output_dim, stream=stream))

            impl = _tm.LinkLinearExperts([expert._impl for expert in experts])
            handle = Weight()
            handle._impl = impl
            handle._experts = experts
            return handle
        except Exception:
            for expert in reversed(experts):
                expert.close()
            raise

    def prepare_weight(self, weight: torch.Tensor | Sequence[torch.Tensor], *, plan: WeightPlan, scales: torch.Tensor | Sequence[torch.Tensor] | None = None, zeros: torch.Tensor | Sequence[torch.Tensor] | None = None) -> Weight:
        """Normalize and pack a dense weight or sequence of expert weights."""
        weight_format = plan._weight_format
        dtype = plan._dtype
        impl_plan = plan._impl

        if not plan._grouped:
            with self._activate() as stream:
                normalized = self._normalize_params(weight_format, weight, scales, zeros)
                return self._prepare_one(normalized, weight_format=weight_format, plan=impl_plan, dtype=dtype, stored_output_dim=normalized['weight'].shape[1], stream=stream)

        weights = list(weight)
        expert_scales = [None] * len(weights) if scales is None else list(scales)
        expert_zeros = [None] * len(weights) if zeros is None else list(zeros)

        with self._activate() as stream:
            normalized_experts = [self._normalize_params(weight_format, expert_weight, expert_scale, expert_zero) for expert_weight, expert_scale, expert_zero in zip(weights, expert_scales, expert_zeros)]
            return self._prepare_grouped(normalized_experts, weight_format=weight_format, plan=impl_plan, dtype=dtype, stored_output_dim=normalized_experts[0]['weight'].shape[1], stream=stream)

    def _interleave_gate_up(self, gate, up, groups):
        """Interleave gate and up components in the selected family block layout."""
        gate_groups = gate.unflatten(-1, (groups, -1))
        up_groups = up.unflatten(-1, (groups, -1))
        return torch.stack((gate_groups, up_groups), dim=-2).flatten(-3, -1).contiguous()

    def fuse_weight(self, weight: tuple[torch.Tensor | Sequence[torch.Tensor], torch.Tensor | Sequence[torch.Tensor]], *, plan: WeightPlan, scales: tuple[torch.Tensor | Sequence[torch.Tensor], torch.Tensor | Sequence[torch.Tensor]] | None = None, zeros: tuple[torch.Tensor | Sequence[torch.Tensor], torch.Tensor | Sequence[torch.Tensor]] | None = None) -> Weight:
        """Normalize, interleave, and pack gate/up weights for fused SiLU execution."""
        weight_format = plan._weight_format
        dtype = plan._dtype
        impl_plan = plan._impl
        gate_weight, up_weight = weight

        if plan._grouped:
            gate_weights = list(gate_weight)
            up_weights = list(up_weight)
        else:
            gate_weights = [gate_weight]
            up_weights = [up_weight]

        count = len(gate_weights)
        if scales is None:
            gate_scales = [None] * count
            up_scales = [None] * count
        else:
            gate_scale_arg, up_scale_arg = scales
            gate_scales = list(gate_scale_arg) if plan._grouped else [gate_scale_arg]
            up_scales = list(up_scale_arg) if plan._grouped else [up_scale_arg]

        if zeros is None:
            gate_zeros = [None] * count
            up_zeros = [None] * count
        else:
            gate_zero_arg, up_zero_arg = zeros
            gate_zeros = list(gate_zero_arg) if plan._grouped else [gate_zero_arg]
            up_zeros = list(up_zero_arg) if plan._grouped else [up_zero_arg]

        with self._activate() as stream:
            normalized_pairs = []
            for gate_weight, up_weight, gate_scale, up_scale, gate_zero, up_zero in zip(gate_weights, up_weights, gate_scales, up_scales, gate_zeros, up_zeros):
                gate = self._normalize_params(weight_format, gate_weight, gate_scale, gate_zero)
                up = self._normalize_params(weight_format, up_weight, up_scale, up_zero)
                normalized_pairs.append((gate, up))

            projection_n = normalized_pairs[0][0]['weight'].shape[1]
            activation = _tm.ActivationType.kSilu
            gate_up_block = impl_plan.gate_up(activation, projection_n)
            if not gate_up_block:
                raise NotImplementedError('selected family cannot represent SiLU fusion')
            groups = projection_n // gate_up_block
            combined_experts = []
            for gate, up in normalized_pairs:
                combined = {kind: self._interleave_gate_up(gate[kind], up[kind], groups) for kind in gate}
                combined_experts.append(combined)

            stored_output_dim = projection_n * 2
            if plan._grouped:
                return self._prepare_grouped(combined_experts, weight_format=weight_format, plan=impl_plan, dtype=dtype, stored_output_dim=stored_output_dim, stream=stream)
            return self._prepare_one(combined_experts[0], weight_format=weight_format, plan=impl_plan, dtype=dtype, stored_output_dim=stored_output_dim, stream=stream)

    def get_exec_plan(self, x: torch.Tensor, weight: Weight, *, offsets: torch.Tensor | None = None, indices: torch.Tensor | None = None) -> ExecPlan:
        """Select an immutable execution plan for an input and prepared weight."""
        tm = _tm
        impl = self._impl.get_exec_plan(weight._impl, tm.from_dlpack_with_strides(x), None if indices is None else tm.from_dlpack_with_strides(indices), None if offsets is None else tm.from_dlpack_with_strides(offsets))
        if impl is None:
            raise NotImplementedError('no GEMM kernel accepts the execution problem')
        exec_plan = ExecPlan()
        exec_plan._impl = impl
        return exec_plan

    def _allocate_output(self, spec, out: torch.Tensor | None, out_scales: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Allocate any omitted output tensors according to an execution plan."""
        if out is None:
            out = torch.empty_strided(spec.output_shape, spec.output_stride, dtype=_to_torch_dtype(spec.output_dtype), device=self.device)
        if out_scales is None and spec.output_scales_dtype != _tm.DataType.TYPE_INVALID:
            out_scales = torch.empty_strided(spec.output_scales_shape, spec.output_scales_stride, dtype=_to_torch_dtype(spec.output_scales_dtype), device=self.device)
        return out, out_scales

    def __call__(self, x: torch.Tensor, weight: Weight, *, exec_plan: ExecPlan, out: torch.Tensor | None = None, input_scales: torch.Tensor | None = None, out_scales: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Execute a dense linear operation with an explicit execution plan."""
        with self._activate():
            out, out_scales = self._allocate_output(exec_plan._impl, out, out_scales)
            self._impl.forward_dense(exec_plan._impl, _tm.from_dlpack_with_strides(x), weight._impl, _tm.from_dlpack_with_strides(out), None if input_scales is None else _tm.from_dlpack_with_strides(input_scales), None if out_scales is None else _tm.from_dlpack_with_strides(out_scales))
        return out, out_scales

    def forward_moe(self, x: torch.Tensor, weight: Weight, *, exec_plan: ExecPlan, offsets: torch.Tensor, out: torch.Tensor | None = None, indices: torch.Tensor | None = None, input_scales: torch.Tensor | None = None, out_scales: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Execute a grouped linear operation with optional indexed input."""
        with self._activate():
            out, out_scales = self._allocate_output(exec_plan._impl, out, out_scales)
            self._impl.forward_moe(exec_plan._impl, _tm.from_dlpack_with_strides(x), weight._impl, None if indices is None else _tm.from_dlpack_with_strides(indices), _tm.from_dlpack_with_strides(offsets), _tm.from_dlpack_with_strides(out), None if input_scales is None else _tm.from_dlpack_with_strides(input_scales), None if out_scales is None else _tm.from_dlpack_with_strides(out_scales))
        return out, out_scales

    def tune(self, x: torch.Tensor, weight: Weight, *, offsets: torch.Tensor | None = None, indices: torch.Tensor | None = None, out: torch.Tensor | None = None, input_scales: torch.Tensor | None = None, out_scales: torch.Tensor | None = None) -> tuple[ExecPlan, torch.Tensor, torch.Tensor | None]:
        """Measure feasible kernels and return the selected execution plan and outputs."""
        tm = _tm
        input_impl = tm.from_dlpack_with_strides(x)
        indices_impl = None if indices is None else tm.from_dlpack_with_strides(indices)
        offsets_impl = None if offsets is None else tm.from_dlpack_with_strides(offsets)
        input_scales_impl = None if input_scales is None else tm.from_dlpack_with_strides(input_scales)
        with self._activate():
            output_spec = self._impl._get_output_spec(weight._impl, input_impl, indices_impl)
            out, out_scales = self._allocate_output(output_spec, out, out_scales)
            impl = self._impl.tune(input_impl, weight._impl, indices_impl, offsets_impl, tm.from_dlpack_with_strides(out), input_scales_impl, None if out_scales is None else tm.from_dlpack_with_strides(out_scales))
        if impl is None:
            raise NotImplementedError('no GEMM kernel accepts the execution problem')
        exec_plan = ExecPlan()
        exec_plan._impl = impl
        return exec_plan, out, out_scales

    def import_records(self, path: str | os.PathLike[str]) -> int:
        """Import cached kernel-selection records from a file."""
        return int(self._impl.import_records(os.fspath(path)))

    def export_records(self, path: str | os.PathLike[str]) -> int:
        """Export cached kernel-selection records to a file."""
        return int(self._impl.export_records(os.fspath(path)))
