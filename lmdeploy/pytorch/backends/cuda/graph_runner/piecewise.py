# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Protocol, TypeVar, overload

import torch
from torch.profiler import record_function

# Eager boundary declaration and output binding.


class UnsupportedBoundaryError(RuntimeError):
    """Raised when an eager boundary cannot feed a later graph safely."""


class PiecewiseGraphGuardError(RuntimeError):
    """Raised before replay when a request does not satisfy plan guards."""


def _make_weak_cuda_view(tensor: torch.Tensor) -> torch.Tensor:
    """Keep tensor metadata without owning graph-pool storage.

    A captured graph keeps its private pool alive. This view lets later eager replay use the captured address without
    preventing that pool from reusing the allocation after the eager call's position in the recorded order.
    """
    if tensor.device.type != 'cuda' or tensor.numel() == 0:
        return tensor

    storage = tensor.untyped_storage()
    weak_storage = torch._C._construct_storage_from_data_pointer(
        storage.data_ptr(),
        tensor.device,
        storage.nbytes(),
    )
    return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
        weak_storage,
        storage_offset=tensor.storage_offset(),
        size=tensor.size(),
        stride=tensor.stride(),
    )


class BoundaryAdapter(Protocol):
    """Stabilize one eager result for a following CUDA graph piece."""

    def allocate(
        self,
        output: Any,
        boundary_input_storages: frozenset[int],
        bridge_pool: ReusableBridgePool | None = None,
    ) -> Any:
        ...

    def copy(self, destination: Any, source: Any) -> None:
        ...


class FixedOutputAdapter:
    """Require an eager result with a fixed tensor-pytree contract."""

    def allocate(
        self,
        output: Any,
        boundary_input_storages: frozenset[int],
        _bridge_pool: ReusableBridgePool | None = None,
    ) -> Any:
        return _allocate_bridge(output, boundary_input_storages)

    def copy(self, destination: Any, source: Any) -> None:
        _copy_tree(destination, source)


@dataclass(frozen=True)
class PiecewiseGraphExecution:
    """Logical token extent for one bucket-shaped construction or replay."""

    raw_tokens: int
    token_bucket: int


_ACTIVE_EXECUTION: ContextVar[PiecewiseGraphExecution | None] = ContextVar(
    'piecewise_cuda_graph_execution', default=None)


@contextmanager
def piecewise_graph_execution(raw_tokens: int, token_bucket: int):
    """Bind logical token metadata for eager boundaries and their adapters."""
    execution = PiecewiseGraphExecution(raw_tokens=raw_tokens, token_bucket=token_bucket)
    token = _ACTIVE_EXECUTION.set(execution)
    try:
        yield execution
    finally:
        _ACTIVE_EXECUTION.reset(token)


def get_piecewise_graph_execution() -> PiecewiseGraphExecution | None:
    """Return the active bucket execution, if piecewise execution owns it."""
    return _ACTIVE_EXECUTION.get()


class PaddedTensorOutputAdapter:
    """Bind a raw-token eager tensor into bucket-shaped stable storage."""

    def __init__(self, token_axis: int = 0) -> None:
        self.token_axis = token_axis

    def allocate(
        self,
        output: Any,
        boundary_input_storages: frozenset[int],
        bridge_pool: ReusableBridgePool | None = None,
    ) -> torch.Tensor:
        execution = _ACTIVE_EXECUTION.get()
        if execution is None:
            raise RuntimeError('padded output allocation requires an active piecewise execution')
        if not isinstance(output, torch.Tensor):
            raise UnsupportedBoundaryError('padded output adapter requires one tensor output')
        if output.layout is not torch.strided:
            raise UnsupportedBoundaryError(f'only strided tensor outputs are supported, got {output.layout}')
        aliases_boundary_input = output.untyped_storage().data_ptr() in boundary_input_storages
        if aliases_boundary_input or output._is_view() or output._base is not None or output.storage_offset() != 0:
            raise UnsupportedBoundaryError('view or aliased tensor outputs are not supported by the padded adapter')
        if output.ndim == 0 or not -output.ndim <= self.token_axis < output.ndim:
            raise UnsupportedBoundaryError('padded output adapter has an invalid token axis')
        token_axis = self.token_axis % output.ndim
        if output.size(token_axis) != execution.raw_tokens:
            raise UnsupportedBoundaryError('eager output does not match the active raw-token extent')
        if not output.is_contiguous():
            raise UnsupportedBoundaryError('padded output adapter requires a contiguous eager output')

        shape = list(output.shape)
        shape[token_axis] = execution.token_bucket
        if bridge_pool is None:
            return output.new_empty(tuple(shape))
        return bridge_pool.allocate_padded_tensor(output, shape, token_axis)

    def copy(self, destination: Any, source: Any) -> None:
        execution = _ACTIVE_EXECUTION.get()
        token_axis = self.token_axis % destination.ndim
        destination.narrow(token_axis, 0, execution.raw_tokens).copy_(source)
        if execution.raw_tokens < execution.token_bucket:
            destination.narrow(token_axis, execution.raw_tokens,
                               execution.token_bucket - execution.raw_tokens).zero_()


@dataclass(frozen=True)
class BoundaryDeclaration:
    """Immutable metadata attached to one eager boundary."""

    func: Callable[..., Any]
    adapter_factory: Callable[[], BoundaryAdapter] | None
    reuse_bridge_after_next_step: bool


class _Builder(Protocol):
    """Minimum interface used by an eager boundary wrapper."""

    inside_eager_boundary: bool

    def call_eager(self, declaration: BoundaryDeclaration, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        ...


_ACTIVE_BUILDER: ContextVar[_Builder | None] = ContextVar('piecewise_cuda_graph_builder', default=None)
_CallableT = TypeVar('_CallableT', bound=Callable[..., Any])


@overload
def eager_boundary(func: _CallableT) -> _CallableT:
    ...


@overload
def eager_boundary(
    func: None = None,
    *,
    adapter_factory: Callable[[], BoundaryAdapter] | None = None,
    reuse_bridge_after_next_step: bool = False,
    eager_only_output: bool = False,
) -> Callable[[_CallableT], _CallableT]:
    ...


def eager_boundary(
    func: _CallableT | None = None,
    *,
    adapter_factory: Callable[[], BoundaryAdapter] | None = None,
    reuse_bridge_after_next_step: bool = False,
    eager_only_output: bool = False,
) -> _CallableT | Callable[[_CallableT], _CallableT]:
    """Declare a function that must execute eagerly between graph pieces.

    The wrapper is transparent unless a piecewise plan builder is active. Nested declarations belong to the outermost
    eager boundary.
    ``eager_only_output`` keeps a dynamic Python result for later eager steps;
    that result must never be consumed by a captured graph piece.
    """

    if func is None:
        return lambda decorated: eager_boundary(
            decorated,
            adapter_factory=adapter_factory,
            reuse_bridge_after_next_step=reuse_bridge_after_next_step,
            eager_only_output=eager_only_output,
        )

    if eager_only_output and (adapter_factory is not None or reuse_bridge_after_next_step):
        raise ValueError('an eager-only output cannot also use a graph bridge')

    declaration = BoundaryDeclaration(
        func=func,
        adapter_factory=None if eager_only_output else adapter_factory or FixedOutputAdapter,
        reuse_bridge_after_next_step=reuse_bridge_after_next_step,
    )

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        builder = _ACTIVE_BUILDER.get()
        if builder is None or builder.inside_eager_boundary:
            return func(*args, **kwargs)
        return builder.call_eager(declaration, args, kwargs)

    return wrapped  # type: ignore[return-value]


@dataclass(frozen=True)
class _FrameValueRef:
    """Reference one live input from the current request frame."""

    name: str


@dataclass
class _EagerValueSlot:
    """Hold a Python value produced and consumed only by eager steps."""

    capture_value: Any
    current: Any

    def bind(self, output: Any) -> None:
        self.current = output


_EAGER_CONSTANT_TYPES = (type(None), bool, int, float, str, bytes, torch.dtype, torch.device)


def _resolve_eager_argument(value: Any, frame_inputs: Mapping[str, Any]) -> Any:
    if isinstance(value, _FrameValueRef):
        return frame_inputs[value.name]
    if isinstance(value, _EagerValueSlot):
        return value.current
    return value


@dataclass(frozen=True)
class _BridgeOutput:
    """Copy an eager tensor result into stable graph-visible storage."""

    bridge: Any
    adapter: BoundaryAdapter

    def bind(self, output: Any) -> None:
        self.adapter.copy(self.bridge, output)


_EagerOutputBinding = _BridgeOutput | _EagerValueSlot


def _tensor_storage_pointers(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return (value.untyped_storage().data_ptr(), )
    if isinstance(value, (tuple, list)):
        pointers: list[int] = []
        for item in value:
            pointers.extend(_tensor_storage_pointers(item))
        return tuple(pointers)
    if isinstance(value, dict):
        pointers = []
        for item in value.values():
            pointers.extend(_tensor_storage_pointers(item))
        return tuple(pointers)
    return ()


@dataclass
class _ReusableTensorSlot:
    tensor: torch.Tensor
    available: bool = True


class ReusableBridgePool:
    """Own stable tensor storage shared by mutually exclusive plans.

    Releasing a bridge only makes its address eligible for a later boundary; the storage remains alive for every CUDA
    graph that captured that address.
    """

    def __init__(self) -> None:
        self._tensor_slots: list[_ReusableTensorSlot] = []

    def begin_plan(self) -> None:
        """Make all slots available while constructing another serial plan."""
        for slot in self._tensor_slots:
            slot.available = True

    def allocate_padded_tensor(self, output: torch.Tensor, shape: Sequence[int], token_axis: int = 0) -> torch.Tensor:
        """Return a contiguous slot, allowing a larger token axis to back
        it."""
        shape = tuple(shape)
        token_axis %= len(shape)
        candidates = [
            slot for slot in self._tensor_slots
            if slot.available and slot.tensor.dtype == output.dtype and slot.tensor.device == output.device
            and slot.tensor.ndim == len(shape)
            and all(slot.tensor.size(axis) == size for axis, size in enumerate(shape) if axis != token_axis)
            and slot.tensor.size(token_axis) >= shape[token_axis]
        ]
        if candidates:
            slot = min(candidates, key=lambda item: item.tensor.size(token_axis))
            slot.available = False
            return slot.tensor.narrow(token_axis, 0, shape[token_axis])

        tensor = output.new_empty(shape)
        self._tensor_slots.append(_ReusableTensorSlot(tensor=tensor, available=False))
        return tensor

    def release(self, bridge: Any) -> None:
        """Logically release pool-owned tensors contained in a bridge tree."""
        pointers = frozenset(_tensor_storage_pointers(bridge))
        for slot in self._tensor_slots:
            if slot.tensor.untyped_storage().data_ptr() in pointers:
                slot.available = True

    def reset(self) -> None:
        self._tensor_slots.clear()

    @property
    def allocated_bytes(self) -> int:
        """Return physical bridge capacity retained by this pool."""
        return sum(slot.tensor.numel() * slot.tensor.element_size() for slot in self._tensor_slots)


def _allocate_bridge(value: Any, boundary_input_storages: frozenset[int]) -> Any:
    if isinstance(value, torch.Tensor):
        if value.layout is not torch.strided:
            raise UnsupportedBoundaryError(f'only strided tensor outputs are supported, got {value.layout}')
        aliases_boundary_input = value.untyped_storage().data_ptr() in boundary_input_storages
        if aliases_boundary_input or value._is_view() or value._base is not None or value.storage_offset() != 0:
            raise UnsupportedBoundaryError('view or aliased tensor outputs require an operator-owned adapter')
        return torch.empty_strided(value.size(), value.stride(), dtype=value.dtype, device=value.device)
    if isinstance(value, tuple) and not hasattr(value, '_fields'):
        return tuple(_allocate_bridge(item, boundary_input_storages) for item in value)
    if isinstance(value, list):
        return [_allocate_bridge(item, boundary_input_storages) for item in value]
    if isinstance(value, dict):
        return {key: _allocate_bridge(item, boundary_input_storages) for key, item in value.items()}
    raise UnsupportedBoundaryError(f'eager output type {type(value).__name__} is not a supported tensor pytree')


def _copy_tree(destination: Any, source: Any) -> None:
    if isinstance(destination, torch.Tensor):
        if not isinstance(source, torch.Tensor):
            raise UnsupportedBoundaryError('eager output structure changed at replay')
        if (destination.shape != source.shape or destination.stride() != source.stride()
                or destination.dtype != source.dtype or destination.device != source.device):
            raise UnsupportedBoundaryError('eager output shape, stride, dtype, or device changed at replay')
        destination.copy_(source)
        return
    if isinstance(destination, tuple):
        if not isinstance(source, tuple) or len(destination) != len(source):
            raise UnsupportedBoundaryError('eager output tuple changed at replay')
        for destination_item, source_item in zip(destination, source):
            _copy_tree(destination_item, source_item)
        return
    if isinstance(destination, list):
        if not isinstance(source, list) or len(destination) != len(source):
            raise UnsupportedBoundaryError('eager output list changed at replay')
        for destination_item, source_item in zip(destination, source):
            _copy_tree(destination_item, source_item)
        return
    if isinstance(destination, dict):
        if not isinstance(source, dict) or destination.keys() != source.keys():
            raise UnsupportedBoundaryError('eager output mapping changed at replay')
        for key in destination:
            _copy_tree(destination[key], source[key])
        return
    raise AssertionError(f'unexpected bridge type: {type(destination).__name__}')


@dataclass(frozen=True)
class GraphStep:
    """One captured graph executable in an ordered piecewise plan."""

    index: int
    graph: torch.cuda.CUDAGraph

    @property
    def label(self) -> str:
        return f'graph:{self.index}'

    def run(self, _frame_inputs: Mapping[str, Any] | None = None) -> None:
        with record_function(f'piecewise::{self.label}'):
            self.graph.replay()


@dataclass(frozen=True)
class EagerStep:
    """One eager boundary and its plan-owned result binding."""

    declaration: BoundaryDeclaration
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output: _EagerOutputBinding

    @property
    def label(self) -> str:
        return f'eager:{self.declaration.func.__qualname__}'

    def run(self, frame_inputs: Mapping[str, Any] | None = None) -> None:
        frame_inputs = {} if frame_inputs is None else frame_inputs
        with record_function(f'piecewise::{self.label}'):
            args = tuple(_resolve_eager_argument(value, frame_inputs) for value in self.args)
            kwargs = {name: _resolve_eager_argument(value, frame_inputs) for name, value in self.kwargs.items()}
            self.output.bind(self.declaration.func(*args, **kwargs))


PlanStep = GraphStep | EagerStep


# Captured plan construction and replay.


class PiecewiseGraphPlan:
    """Own fixed inputs, captured pieces, eager calls, and replay ordering."""

    def __init__(
        self,
        *,
        static_inputs: Sequence[torch.Tensor],
        output: Any,
        steps: Iterable[PlanStep],
        stream: torch.cuda.Stream,
    ) -> None:
        self._static_inputs = tuple(static_inputs)
        self.output = output
        self.steps = tuple(steps)
        self._stream = stream
        self._device = next(value.device for value in self._static_inputs if value.device.type == 'cuda')

    def replay(self, *inputs: torch.Tensor) -> Any:
        """Fill static inputs and replay graph/eager steps in original
        order."""
        if len(inputs) != len(self._static_inputs):
            raise PiecewiseGraphGuardError(f'expected {len(self._static_inputs)} inputs, got {len(inputs)}')
        for static, current in zip(self._static_inputs, inputs):
            if (static.shape != current.shape or static.stride() != current.stride() or static.dtype != current.dtype
                    or static.device != current.device):
                raise PiecewiseGraphGuardError('input shape, stride, dtype, or device differs from the trace')

        def bind(static_inputs: tuple[torch.Tensor, ...]) -> None:
            for static, current in zip(static_inputs, inputs):
                static.copy_(current)

        return self.replay_with_input_binder(bind)

    def replay_with_input_binder(
        self,
        bind: Callable[[tuple[torch.Tensor, ...]], None],
        frame_inputs: Mapping[str, Any] | None = None,
    ) -> Any:
        """Fill plan-owned inputs on the replay stream, then execute the
        plan."""
        frame_inputs = {} if frame_inputs is None else frame_inputs
        caller_stream = torch.cuda.current_stream(self._device)
        self._stream.wait_stream(caller_stream)
        with torch.cuda.stream(self._stream), torch.inference_mode():
            bind(self._static_inputs)
            for step in self.steps:
                step.run(frame_inputs)
        caller_stream.wait_stream(self._stream)
        return self.output


class _CaptureBuilder:
    """Ephemeral direct-execution tracer for one piecewise plan."""

    def __init__(
        self,
        *,
        graph_pool: tuple[int, int],
        stream: torch.cuda.Stream,
        bridge_pool: ReusableBridgePool,
        frame_inputs: Mapping[str, Any],
    ) -> None:
        self._graph_pool = graph_pool
        self._stream = stream
        self._bridge_pool = bridge_pool
        self._capture_frame = tuple(frame_inputs.items())
        self._steps: list[PlanStep] = []
        self._eager_values: dict[int, _EagerValueSlot] = {}
        self.inside_eager_boundary = False
        self._active_graph: torch.cuda.CUDAGraph | None = None
        self._previous_reusable_bridge: Any | None = None

    def build(self, func: Callable[..., Any], static_inputs: Sequence[torch.Tensor]) -> PiecewiseGraphPlan:
        self._bridge_pool.begin_plan()
        token = _ACTIVE_BUILDER.set(self)
        try:
            self._begin_piece()
            output = func(*static_inputs)
            self._finish_piece_and_replay()
            # Published steps keep the slots, but never construction metadata.
            for slot in self._eager_values.values():
                slot.capture_value = None
                slot.current = None
        except BaseException:
            self._discard_active_capture()
            self._steps.clear()
            raise
        finally:
            _ACTIVE_BUILDER.reset(token)

        return PiecewiseGraphPlan(
            static_inputs=static_inputs,
            output=output,
            steps=self._steps,
            stream=self._stream,
        )

    def call_eager(self, declaration: BoundaryDeclaration, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self._finish_piece_and_replay()

        self.inside_eager_boundary = True
        try:
            with record_function(f'piecewise::trace_eager:{declaration.func.__qualname__}'):
                output = declaration.func(*args, **kwargs)
        finally:
            self.inside_eager_boundary = False

        if self._previous_reusable_bridge is not None:
            self._bridge_pool.release(self._previous_reusable_bridge)
            self._previous_reusable_bridge = None

        replay_args = tuple(self._bind_eager_argument(value) for value in args)
        replay_kwargs = {name: self._bind_eager_argument(value) for name, value in kwargs.items()}
        result, output_binding = self._bind_eager_output(declaration, output, args, kwargs)
        self._steps.append(EagerStep(declaration, replay_args, replay_kwargs, output_binding))
        if declaration.reuse_bridge_after_next_step:
            self._previous_reusable_bridge = result
        self._begin_piece()
        return result

    def _bind_eager_output(
        self,
        declaration: BoundaryDeclaration,
        output: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, _EagerOutputBinding]:
        """Bind an eager-only value or a graph-visible tensor bridge."""
        adapter_factory = declaration.adapter_factory
        if adapter_factory is None:
            output_binding = _EagerValueSlot(capture_value=output, current=output)
            self._eager_values[id(output)] = output_binding
            return output, output_binding

        input_storages = frozenset(_tensor_storage_pointers(args) + _tensor_storage_pointers(kwargs))
        adapter = adapter_factory()
        pool = self._bridge_pool if declaration.reuse_bridge_after_next_step else None
        bridge = adapter.allocate(output, input_storages, pool)
        adapter.copy(bridge, output)
        return bridge, _BridgeOutput(bridge, adapter)

    def _bind_eager_argument(self, value: Any) -> Any:
        slot = self._eager_values.get(id(value))
        if slot is not None and slot.capture_value is value:
            return slot
        if not isinstance(value, _EAGER_CONSTANT_TYPES):
            for name, captured in self._capture_frame:
                if captured is value:
                    return _FrameValueRef(name)
        if isinstance(value, torch.Tensor):
            return _make_weak_cuda_view(value)
        return value

    def _begin_piece(self) -> None:
        if self._active_graph is not None:
            raise RuntimeError('a graph piece is already being captured')
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin(pool=self._graph_pool, capture_error_mode='thread_local')
        self._active_graph = graph

    def _finish_piece_and_replay(self) -> None:
        graph = self._active_graph
        if graph is None:
            raise RuntimeError('no graph piece is being captured')
        graph.capture_end()
        self._active_graph = None
        index = sum(isinstance(step, GraphStep) for step in self._steps)
        self._steps.append(GraphStep(index, graph))
        graph.replay()

    def _discard_active_capture(self) -> None:
        """End a failed capture best-effort and preserve its original error."""
        graph = self._active_graph
        if graph is None:
            return
        try:
            graph.capture_end()
        except Exception:
            pass
        self._active_graph = None


def trace_piecewise_cuda_graph(
    func: Callable[..., Any],
    example_inputs: Sequence[torch.Tensor],
    *,
    warmup_iterations: int = 3,
    warmup_func: Callable[..., Any] | None = None,
    bridge_pool: ReusableBridgePool | None = None,
    frame_inputs: Mapping[str, Any] | None = None,
    stream: torch.cuda.Stream | None = None,
) -> PiecewiseGraphPlan:
    """Warm up and directly trace a fixed-shape callable into a plan.

    A boundary adapter may change an eager result from its raw shape to its
    captured shape. Such integrations can supply a shape-compatible warmup
    callable while keeping ``func`` as the exact construction callable.
    """
    if not example_inputs:
        raise ValueError('at least one tensor input is required')
    cuda_inputs = [value for value in example_inputs if value.device.type == 'cuda']
    if not cuda_inputs:
        raise ValueError('piecewise CUDA graph tracing requires a CUDA input')
    device = cuda_inputs[0].device
    if any(value.device != device for value in cuda_inputs):
        raise ValueError('all CUDA inputs must be on the same device')
    if warmup_iterations < 0:
        raise ValueError('warmup_iterations must be non-negative')

    static_inputs = tuple(
        torch.empty_strided(value.size(), value.stride(), dtype=value.dtype, device=value.device)
        for value in example_inputs)
    capture_stream = torch.cuda.Stream(device=device) if stream is None else stream
    if capture_stream.device != device:
        raise ValueError('the piecewise stream must use the same device as the CUDA inputs')
    caller_stream = torch.cuda.current_stream(device)
    capture_stream.wait_stream(caller_stream)

    with torch.cuda.stream(capture_stream), torch.inference_mode():
        for static, example in zip(static_inputs, example_inputs):
            static.copy_(example)
        run_warmup = func if warmup_func is None else warmup_func
        for _ in range(warmup_iterations):
            run_warmup(*static_inputs)
        # Warmup may mutate explicit state inputs. Restore the supplied baseline
        # so construction performs exactly one logical forward from that state.
        for static, example in zip(static_inputs, example_inputs):
            static.copy_(example)
    capture_stream.synchronize()

    graph_pool = torch.cuda.graph_pool_handle()
    bridge_pool = ReusableBridgePool() if bridge_pool is None else bridge_pool
    frame_inputs = {} if frame_inputs is None else frame_inputs
    with torch.cuda.stream(capture_stream), torch.inference_mode():
        plan = _CaptureBuilder(graph_pool=graph_pool, stream=capture_stream,
                               bridge_pool=bridge_pool, frame_inputs=frame_inputs).build(func, static_inputs)
    capture_stream.synchronize()
    caller_stream.wait_stream(capture_stream)
    return plan


# Runner-local plan lifecycle.


@dataclass(frozen=True)
class PiecewiseGraphBuild:
    """A publishable plan and the startup output produced while building."""

    plan: Any
    output: Any


class PiecewiseGraphRuntime(Protocol):
    """Model integration consumed by the runner-local plan manager.

    Warmup and build execute startup dummy requests. A serving request only replays a plan that was published
    successfully during startup.
    """

    def get_capture_token_sizes(self) -> Sequence[int]:
        ...

    def get_piecewise_graph_descriptor(self, context: Any, kwargs: Mapping[str, Any]) -> Hashable | None:
        ...

    def warmup(self, descriptor: Any, kwargs: Mapping[str, Any]) -> None:
        ...

    def build(
        self,
        descriptor: Any,
        kwargs: Mapping[str, Any],
        bridge_pool: ReusableBridgePool,
        stream: torch.cuda.Stream,
    ) -> PiecewiseGraphBuild:
        ...

    def replay(self, plan: Any, kwargs: Mapping[str, Any]) -> Any:
        ...


class PiecewiseGraphManager:
    """Own shared PCG resources and every runner-local prepared plan."""

    def __init__(self, runtime: PiecewiseGraphRuntime, stream: torch.cuda.Stream) -> None:
        self._runtime = runtime
        self._plans: dict[Hashable, Any] = {}
        self._bridge_pool = ReusableBridgePool()
        self._stream = stream

    def get_capture_token_sizes(self) -> list[int]:
        """Return every token size that startup must prepare."""
        return list(self._runtime.get_capture_token_sizes())

    def get_piecewise_graph_descriptor(self, context: Any, kwargs: Mapping[str, Any]) -> Hashable | None:
        """Return a supported descriptor without changing runtime state."""
        return self._runtime.get_piecewise_graph_descriptor(context, kwargs)

    def has_plan(self, descriptor: Hashable) -> bool:
        """Return whether startup published this exact plan."""
        return descriptor in self._plans

    def prepare(self, descriptor: Hashable, kwargs: Mapping[str, Any]) -> Any:
        """Warm up and publish one plan; any failure aborts engine warmup."""
        self._runtime.warmup(descriptor, kwargs)
        build = self._runtime.build(descriptor, kwargs, self._bridge_pool, self._stream)
        self._plans[descriptor] = build.plan
        return build.output

    def replay(self, descriptor: Hashable, kwargs: Mapping[str, Any]) -> Any:
        """Replay a prepared plan.

        Replay failures always propagate.
        """
        with record_function('forward_piecewise_cudagraph'):
            return self._runtime.replay(self._plans[descriptor], kwargs)

    def reset(self) -> None:
        """Drop every plan before releasing their shared bridge storage."""
        self._plans.clear()
        self._bridge_pool.reset()
