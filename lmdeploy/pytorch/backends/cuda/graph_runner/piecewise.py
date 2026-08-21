# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Protocol, TypeVar, overload

import torch
from torch.profiler import record_function


class UnsupportedBoundaryError(RuntimeError):
    """Raised when an eager boundary cannot feed a later graph safely."""


class PiecewiseGraphGuardError(RuntimeError):
    """Raised before replay when a request does not satisfy plan guards."""


class BoundaryAdapter(Protocol):
    """Stabilize one eager result for a following CUDA graph piece."""

    def allocate(self, output: Any, boundary_input_storages: frozenset[int]) -> Any:
        ...

    def copy(self, destination: Any, source: Any) -> None:
        ...


class FixedOutputAdapter:
    """Require an eager result with a fixed tensor-pytree contract."""

    def allocate(self, output: Any, boundary_input_storages: frozenset[int]) -> Any:
        return _allocate_bridge(output, boundary_input_storages)

    def copy(self, destination: Any, source: Any) -> None:
        _copy_tree(destination, source)


@dataclass(frozen=True)
class PiecewiseGraphExecution:
    """Logical token extent for one bucket-shaped construction or replay."""

    raw_tokens: int
    token_bucket: int

    def __post_init__(self) -> None:
        if not 0 < self.raw_tokens <= self.token_bucket:
            raise ValueError('raw_tokens must be within the token bucket')


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

    def allocate(self, output: Any, boundary_input_storages: frozenset[int]) -> torch.Tensor:
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
        if output.ndim == 0 or output.size(0) != execution.raw_tokens:
            raise UnsupportedBoundaryError('eager output does not match the active raw-token extent')
        if not output.is_contiguous():
            raise UnsupportedBoundaryError('padded output adapter requires a contiguous eager output')

        shape = (execution.token_bucket, *output.shape[1:])
        return output.new_empty(shape)

    def copy(self, destination: Any, source: Any) -> None:
        execution = _ACTIVE_EXECUTION.get()
        if execution is None:
            raise RuntimeError('padded output binding requires an active piecewise execution')
        if not isinstance(destination, torch.Tensor) or not isinstance(source, torch.Tensor):
            raise UnsupportedBoundaryError('padded output adapter requires tensor inputs')
        if source.ndim != destination.ndim or source.shape[1:] != destination.shape[1:]:
            raise UnsupportedBoundaryError('eager output rank or non-token dimensions changed at replay')
        if source.size(0) != execution.raw_tokens or destination.size(0) != execution.token_bucket:
            raise UnsupportedBoundaryError('eager output token extent changed at replay')
        if source.dtype != destination.dtype or source.device != destination.device:
            raise UnsupportedBoundaryError('eager output dtype or device changed at replay')
        if not source.is_contiguous():
            raise UnsupportedBoundaryError('eager output layout changed at replay')

        destination[:execution.raw_tokens].copy_(source)
        if execution.raw_tokens < execution.token_bucket:
            destination[execution.raw_tokens:].zero_()


@dataclass(frozen=True)
class BoundaryDeclaration:
    """Immutable metadata attached to one eager boundary."""

    boundary_id: str
    func: Callable[..., Any]
    adapter_factory: Callable[[], BoundaryAdapter]


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
) -> Callable[[_CallableT], _CallableT]:
    ...


def eager_boundary(
    func: _CallableT | None = None,
    *,
    adapter_factory: Callable[[], BoundaryAdapter] | None = None,
) -> _CallableT | Callable[[_CallableT], _CallableT]:
    """Declare a function that must execute eagerly between graph pieces.

    The wrapper is transparent unless a piecewise plan builder is active. Nested declarations belong to the outermost
    eager boundary.
    """

    if func is None:
        return lambda decorated: eager_boundary(decorated, adapter_factory=adapter_factory)

    declaration = BoundaryDeclaration(
        boundary_id=f'{func.__module__}.{func.__qualname__}',
        func=func,
        adapter_factory=adapter_factory or FixedOutputAdapter,
    )

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        builder = _ACTIVE_BUILDER.get()
        if builder is None or builder.inside_eager_boundary:
            return func(*args, **kwargs)
        return builder.call_eager(declaration, args, kwargs)

    wrapped.__piecewise_cuda_graph_boundary__ = declaration  # type: ignore[attr-defined]
    return wrapped  # type: ignore[return-value]


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
        for dst_item, src_item in zip(destination, source):
            _copy_tree(dst_item, src_item)
        return
    if isinstance(destination, list):
        if not isinstance(source, list) or len(destination) != len(source):
            raise UnsupportedBoundaryError('eager output list changed at replay')
        for dst_item, src_item in zip(destination, source):
            _copy_tree(dst_item, src_item)
        return
    if isinstance(destination, dict):
        if not isinstance(source, dict) or destination.keys() != source.keys():
            raise UnsupportedBoundaryError('eager output mapping changed at replay')
        for key in destination:
            _copy_tree(destination[key], source[key])
        return
    raise AssertionError(f'unexpected bridge type: {type(destination).__name__}')


def _tensor_pointers(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return (value.data_ptr(), )
    if isinstance(value, (tuple, list)):
        pointers: list[int] = []
        for item in value:
            pointers.extend(_tensor_pointers(item))
        return tuple(pointers)
    if isinstance(value, dict):
        pointers = []
        for item in value.values():
            pointers.extend(_tensor_pointers(item))
        return tuple(pointers)
    return ()


@dataclass(frozen=True)
class GraphStep:
    """One captured graph executable in an ordered piecewise plan."""

    index: int
    graph: torch.cuda.CUDAGraph

    @property
    def label(self) -> str:
        return f'graph:{self.index}'

    def run(self) -> None:
        with record_function(f'piecewise::{self.label}'):
            self.graph.replay()


@dataclass(frozen=True)
class EagerStep:
    """One eager boundary and its plan-owned stable output bridge."""

    declaration: BoundaryDeclaration
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    bridge: Any
    adapter: BoundaryAdapter

    @property
    def label(self) -> str:
        return f'eager:{self.declaration.func.__qualname__}'

    def run(self) -> None:
        with record_function(f'piecewise::{self.label}'):
            output = self.declaration.func(*self.args, **self.kwargs)
            self.adapter.copy(self.bridge, output)


PlanStep = GraphStep | EagerStep


class PiecewiseGraphPlan:
    """Own fixed inputs, captured pieces, eager calls, and replay ordering."""

    def __init__(
        self,
        *,
        static_inputs: Sequence[torch.Tensor],
        output: Any,
        steps: Iterable[PlanStep],
        stream: torch.cuda.Stream,
        pool: tuple[int, int],
    ) -> None:
        self._static_inputs = tuple(static_inputs)
        self.output = output
        self.steps = tuple(steps)
        self.stream = stream
        self.pool = pool
        self.device = next(value.device for value in self._static_inputs if value.device.type == 'cuda')

    def describe(self) -> tuple[str, ...]:
        """Describe the immutable replay order."""
        return tuple(step.label for step in self.steps)

    def output_pointers(self) -> tuple[int, ...]:
        """Return final output addresses for capture contract tests."""
        return _tensor_pointers(self.output)

    def bridge_pointers(self) -> tuple[int, ...]:
        """Return eager-to-graph bridge addresses for contract tests."""
        pointers: list[int] = []
        for step in self.steps:
            if isinstance(step, EagerStep):
                pointers.extend(_tensor_pointers(step.bridge))
        return tuple(pointers)

    def replay(self, *inputs: torch.Tensor) -> Any:
        """Fill static inputs and replay graph/eager steps in original
        order."""
        if len(inputs) != len(self._static_inputs):
            raise PiecewiseGraphGuardError(f'expected {len(self._static_inputs)} inputs, got {len(inputs)}')
        for static, current in zip(self._static_inputs, inputs):
            if (static.shape != current.shape or static.stride() != current.stride() or static.dtype != current.dtype
                    or static.device != current.device):
                raise PiecewiseGraphGuardError('input shape, stride, dtype, or device differs from the trace')

        for static, current in zip(self._static_inputs, inputs):
            if static.device.type != 'cuda':
                static.copy_(current)

        def bind(static_inputs: tuple[torch.Tensor, ...]) -> None:
            for static, current in zip(static_inputs, inputs):
                if static.device.type == 'cuda':
                    static.copy_(current)

        return self.replay_with_input_binder(bind)

    def replay_with_input_binder(self, bind: Callable[[tuple[torch.Tensor, ...]], None]) -> Any:
        """Fill plan-owned inputs on the replay stream, then execute the
        plan."""
        caller_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.stream), torch.inference_mode():
            bind(self._static_inputs)
            for step in self.steps:
                step.run()
        caller_stream.wait_stream(self.stream)
        return self.output


class _PlanBuilder:
    """Ephemeral direct-execution tracer for one piecewise plan."""

    def __init__(self, *, pool: tuple[int, int], stream: torch.cuda.Stream) -> None:
        self.pool = pool
        self.stream = stream
        self.steps: list[PlanStep] = []
        self.inside_eager_boundary = False
        self._active_graph: torch.cuda.CUDAGraph | None = None

    def build(self, func: Callable[..., Any], static_inputs: Sequence[torch.Tensor]) -> PiecewiseGraphPlan:
        token = _ACTIVE_BUILDER.set(self)
        try:
            self._begin_piece()
            output = func(*static_inputs)
            self._finish_piece_and_replay()
        except BaseException:
            self._discard_active_capture()
            self.steps.clear()
            raise
        finally:
            _ACTIVE_BUILDER.reset(token)

        return PiecewiseGraphPlan(
            static_inputs=static_inputs,
            output=output,
            steps=self.steps,
            stream=self.stream,
            pool=self.pool,
        )

    def call_eager(self, declaration: BoundaryDeclaration, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self._finish_piece_and_replay()
        self.inside_eager_boundary = True
        try:
            with record_function(f'piecewise::trace_eager:{declaration.func.__qualname__}'):
                output = declaration.func(*args, **kwargs)
        finally:
            self.inside_eager_boundary = False

        boundary_input_storages = frozenset(_tensor_storage_pointers(args) + _tensor_storage_pointers(kwargs))
        adapter = declaration.adapter_factory()
        bridge = adapter.allocate(output, boundary_input_storages)
        adapter.copy(bridge, output)
        self.steps.append(EagerStep(declaration, args, dict(kwargs), bridge, adapter))
        self._begin_piece()
        return bridge

    def _begin_piece(self) -> None:
        if self._active_graph is not None:
            raise RuntimeError('a graph piece is already being captured')
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin(pool=self.pool, capture_error_mode='thread_local')
        self._active_graph = graph

    def _finish_piece_and_replay(self) -> None:
        graph = self._active_graph
        if graph is None:
            raise RuntimeError('no graph piece is being captured')
        graph.capture_end()
        self._active_graph = None
        index = sum(isinstance(step, GraphStep) for step in self.steps)
        self.steps.append(GraphStep(index, graph))
        graph.replay()

    def _discard_active_capture(self) -> None:
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
    capture_stream = torch.cuda.Stream(device=device)
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

    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.stream(capture_stream), torch.inference_mode():
        plan = _PlanBuilder(pool=pool, stream=capture_stream).build(func, static_inputs)
    capture_stream.synchronize()
    caller_stream.wait_stream(capture_stream)
    return plan


@dataclass(frozen=True)
class PiecewiseGraphDescriptor:
    """Side-effect-free, plan-invariant identity for one supported call."""

    key: Hashable


@dataclass(frozen=True)
class PiecewiseGraphHooks:
    """Narrow integration hooks supplied by a supported model/backend pair.

    ``build`` may inspect the current call only to bind stable model/cache
    resources. Request-local values must be replaced with synthetic inputs and
    scratch metadata before model execution.
    """

    get_piecewise_graph_descriptor: Callable[[Any, Mapping[str, Any]], PiecewiseGraphDescriptor | None]
    build: Callable[[PiecewiseGraphDescriptor, Mapping[str, Any]], Any]
    replay: Callable[[Any, PiecewiseGraphDescriptor, Mapping[str, Any]], Any]


class PiecewiseFallbackReason(str, Enum):
    """Reasons for safely selecting the existing eager path."""

    UNSUPPORTED = 'unsupported'
    BUILD_FAILED = 'build_failed'
    GUARD_REJECTED = 'guard_rejected'


@dataclass(frozen=True)
class PiecewiseFallback:
    """Structured reason why no piecewise execution occurred."""

    reason: PiecewiseFallbackReason
    detail: str


@dataclass(frozen=True)
class PiecewiseGraphResult:
    """Result of a manager attempt, including safe pre-execution fallback."""

    executed: bool
    output: Any = None
    fallback: PiecewiseFallback | None = None


class PiecewisePlanState(str, Enum):
    """Observable state of one runner-local plan-cache entry."""

    MISSING = 'missing'
    READY = 'ready'
    NEGATIVE = 'negative'


@dataclass(frozen=True)
class _NegativeEntry:
    fallback: PiecewiseFallback


class PiecewiseGraphManager:
    """Own runner-local piecewise plan construction, caching, and replay."""

    def __init__(self, hooks: PiecewiseGraphHooks, *, max_entries: int = 64) -> None:
        if max_entries < 1:
            raise ValueError('max_entries must be positive')
        self._hooks = hooks
        self._max_entries = max_entries
        self._entries: OrderedDict[PiecewiseGraphDescriptor, Any | _NegativeEntry] = OrderedDict()

    def get_piecewise_graph_descriptor(self, context: Any,
                                       kwargs: Mapping[str, Any]) -> PiecewiseGraphDescriptor | None:
        """Return a supported descriptor without changing runtime state."""
        descriptor = self._hooks.get_piecewise_graph_descriptor(context, kwargs)
        if descriptor is not None:
            hash(descriptor)
        return descriptor

    def run(self, descriptor: PiecewiseGraphDescriptor, kwargs: Mapping[str, Any]) -> PiecewiseGraphResult:
        """Build transactionally on a miss, then replay a ready plan."""
        entry = self._entries.get(descriptor)
        if isinstance(entry, _NegativeEntry):
            self._entries.move_to_end(descriptor)
            return PiecewiseGraphResult(executed=False, fallback=entry.fallback)

        if entry is None:
            try:
                plan = self._hooks.build(descriptor, kwargs)
                if plan is None:
                    raise TypeError('piecewise plan builder returned None')
            except Exception as error:
                fallback = PiecewiseFallback(
                    reason=PiecewiseFallbackReason.BUILD_FAILED,
                    detail=f'{type(error).__name__}: {error}',
                )
                self._publish(descriptor, _NegativeEntry(fallback))
                return PiecewiseGraphResult(executed=False, fallback=fallback)
            # Publication happens only after construction returns successfully.
            self._publish(descriptor, plan)
            entry = plan
        else:
            self._entries.move_to_end(descriptor)

        try:
            output = self._hooks.replay(entry, descriptor, kwargs)
        except PiecewiseGraphGuardError as error:
            fallback = PiecewiseFallback(PiecewiseFallbackReason.GUARD_REJECTED, str(error))
            return PiecewiseGraphResult(executed=False, fallback=fallback)
        return PiecewiseGraphResult(executed=True, output=output)

    def get_plan_state(self, descriptor: PiecewiseGraphDescriptor) -> PiecewisePlanState:
        """Inspect one entry without mutating cache order."""
        entry = self._entries.get(descriptor)
        if entry is None:
            return PiecewisePlanState.MISSING
        if isinstance(entry, _NegativeEntry):
            return PiecewisePlanState.NEGATIVE
        return PiecewisePlanState.READY

    def reset(self) -> None:
        """Drop all graph plans and negative entries."""
        self._entries.clear()

    def _publish(self, descriptor: PiecewiseGraphDescriptor, entry: Any | _NegativeEntry) -> None:
        self._entries[descriptor] = entry
        self._entries.move_to_end(descriptor)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
