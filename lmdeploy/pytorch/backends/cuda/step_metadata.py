# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

import torch

from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager

logger = get_logger('lmdeploy')

GraphBufferT = TypeVar('GraphBufferT')
MetaT = TypeVar('MetaT')


@dataclass(frozen=True)
class CudaSequenceMetadata:
    """Sequence layout shared by CUDA step-metadata builders."""

    block_offsets: torch.Tensor
    q_start_loc: torch.Tensor
    q_seqlens: torch.Tensor
    kv_start_loc: torch.Tensor | None
    kv_seqlens: torch.Tensor
    kv_flatten_size: int | None
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_kv_seqlen: int

    @classmethod
    def from_step_context(cls, step_context: 'StepContext') -> 'CudaSequenceMetadata':
        """Build CUDA's canonical sequence layout once for one step."""
        q_seqlens = step_context.q_seqlens
        kv_seqlens = step_context.kv_seqlens

        # Stack the inputs so both cumulative lengths are produced by the
        # same kernels instead of launching one cumsum per tensor.
        seqlens = torch.stack([q_seqlens, kv_seqlens], dim=0)
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=1, dtype=torch.int32), (1, 0))
        cu_seqlens_q = cu_seqlens[0]
        cu_seqlens_k = cu_seqlens[1]

        kv_start_loc = None
        kv_flatten_size = None
        if not step_context.is_decoding:
            kv_start_loc = cu_seqlens_k[:-1].to(kv_seqlens.dtype)
            kv_flatten_size = step_context.sum_kv_seqlen

        block_offsets = step_context.block_offsets
        if block_offsets.dtype != torch.int32:
            block_offsets = block_offsets.to(torch.int32)

        return cls(
            block_offsets=block_offsets,
            q_start_loc=step_context.q_start_loc,
            q_seqlens=q_seqlens,
            kv_start_loc=kv_start_loc,
            kv_seqlens=kv_seqlens,
            kv_flatten_size=kv_flatten_size,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_kv_seqlen=step_context.max_kv_seqlen,
        )


class CudaAttentionMetaBuilder(ABC, Generic[GraphBufferT, MetaT]):
    """Build typed metadata from one operator-owned graph buffer type."""

    @property
    @abstractmethod
    def key(self) -> Hashable:
        """Return the stable key used to deduplicate compatible builders."""
        raise NotImplementedError

    @abstractmethod
    def build(self, step_context: 'StepContext', sequence_metadata: CudaSequenceMetadata) -> MetaT:
        """Build metadata for one eager inference step."""
        raise NotImplementedError

    @abstractmethod
    def apply_legacy_metadata(self, attn_metadata, metadata: MetaT) -> None:
        """Expose a single group's result through legacy attention fields."""
        raise NotImplementedError

    @abstractmethod
    def make_cudagraph_buffer(self, graph_meta, input_buffers, step_context) -> GraphBufferT:
        """Allocate runner-owned graph state of the builder's buffer type."""
        raise NotImplementedError

    @abstractmethod
    def fill_cudagraph_buffer(self, graph_meta, input_buffers, step_context, buffer: GraphBufferT) -> MetaT:
        """Fill graph state and return its operator-facing metadata view."""
        raise NotImplementedError

    def prepare_cudagraph_capture(self, graph_meta, input_buffers, step_context, buffer: GraphBufferT) -> None:
        """Prepare runner-owned graph state in place for graph capture."""


class CudaStepMetaUpdater(ABC):
    """Update non-attention metadata for one inference step."""

    output_key: ClassVar[str]
    priority: ClassVar[int]

    @property
    @abstractmethod
    def key(self) -> Hashable:
        """Return the stable key used to deduplicate compatible updaters."""
        raise NotImplementedError

    @abstractmethod
    def update(self, step_context: 'StepContext', sequence_metadata: CudaSequenceMetadata) -> None:
        """Update metadata for one inference step."""
        raise NotImplementedError


@runtime_checkable
class CudaPiecewiseGraphImpl(Protocol):
    """Operator capability required by the CUDA piecewise runtime."""

    def supports_piecewise_cuda_graph(self) -> bool:
        ...

    def enable_piecewise_cuda_graph(self) -> None:
        ...


@dataclass(frozen=True)
class CudaStepMetaGraphBuffers:
    """Runner-owned graph buffers paired with one resolved metadata plan."""

    # One heterogeneous GraphBufferT value per attention builder. Only its
    # owning builder interprets the value.
    attention_buffers: tuple[object, ...]


@dataclass(frozen=True)
class CudaStepMetaPlan:
    """Model-owned CUDA metadata plan resolved from instantiated operators."""

    attention_builders: tuple[CudaAttentionMetaBuilder[Any, Any], ...]
    step_updaters: tuple[CudaStepMetaUpdater, ...]
    fallback_reason: str | None = None
    implementations: tuple[Any, ...] = ()

    @property
    def is_supported(self) -> bool:
        """Whether every discovered CUDA metadata contract is understood."""
        return self.fallback_reason is None

    def enable_piecewise_cuda_graph(self) -> bool:
        """Install eager boundaries when every collected operator supports
        PCG."""
        if not self.is_supported or not self.implementations:
            return False

        installers = []
        for impl in self.implementations:
            if not isinstance(impl, CudaPiecewiseGraphImpl) or not impl.supports_piecewise_cuda_graph():
                return False
            installers.append(impl.enable_piecewise_cuda_graph)

        for install in installers:
            install()
        return True

    @classmethod
    def from_implementations(cls, implementations: Iterable[Any]) -> 'CudaStepMetaPlan':
        """Resolve builders registered while constructing one CUDA model."""
        attention_owners: list[tuple[Any, CudaAttentionMetaBuilder[Any, Any]]] = []
        step_updaters: list[CudaStepMetaUpdater] = []
        updater_keys: set[Hashable] = set()
        output_keys: dict[str, Hashable] = {}
        visited_impls: set[int] = set()
        unique_implementations: list[Any] = []

        for impl in implementations:
            if id(impl) in visited_impls:
                continue
            visited_impls.add(id(impl))
            unique_implementations.append(impl)

            get_provider = getattr(impl, 'get_step_metadata_provider', None)
            if get_provider is None:
                reason = f'missing metadata contract for {type(impl).__qualname__}'
                return cls(tuple(), tuple(), fallback_reason=reason)

            provider = get_provider()
            if provider is None:
                reason = f'unknown metadata contract for {type(impl).__qualname__}'
                return cls(tuple(), tuple(), fallback_reason=reason)

            if isinstance(provider, CudaAttentionMetaBuilder):
                if not callable(getattr(impl, 'bind_step_meta_group', None)):
                    reason = f'missing attention metadata binding for {type(impl).__qualname__}'
                    return cls(tuple(), tuple(), fallback_reason=reason)
                attention_owners.append((impl, provider))
                continue

            if not isinstance(provider, CudaStepMetaUpdater):
                reason = f'unknown metadata provider for {type(impl).__qualname__}'
                return cls(tuple(), tuple(), fallback_reason=reason)

            if provider.key in updater_keys:
                continue
            previous_key = output_keys.get(provider.output_key)
            if previous_key is not None and previous_key != provider.key:
                reason = (f'multiple updaters target {provider.output_key!r}: '
                          f'{previous_key!r} and {provider.key!r}')
                return cls(tuple(), tuple(), fallback_reason=reason)

            output_keys[provider.output_key] = provider.key
            updater_keys.add(provider.key)
            step_updaters.append(provider)

        if not attention_owners:
            return cls(tuple(), tuple(), fallback_reason='no implementation-selected attention metadata contract')

        attention_builders: list[CudaAttentionMetaBuilder[Any, Any]] = []
        attention_groups: dict[Hashable, int] = {}
        bindings: list[tuple[Any, int]] = []
        for impl, builder in attention_owners:
            group_id = attention_groups.get(builder.key)
            if group_id is None:
                group_id = len(attention_builders)
                attention_groups[builder.key] = group_id
                attention_builders.append(builder)
            bindings.append((impl, group_id))

        # Bind only after the complete model has been validated, so a legacy
        # fallback cannot leave partially configured implementations behind.
        for impl, group_id in bindings:
            impl.bind_step_meta_group(group_id)

        step_updaters.sort(key=lambda builder: builder.priority)
        return cls(
            tuple(attention_builders),
            tuple(step_updaters),
            implementations=tuple(unique_implementations),
        )

    def _attach_attention_metadata(self, attn_metadata, metadata: tuple[Any, ...]) -> None:
        """Attach grouped results and expose the legacy single-group view."""
        attn_metadata.kernel_metadata = metadata
        if len(metadata) != 1:
            return

        self.attention_builders[0].apply_legacy_metadata(attn_metadata, metadata[0])

    def prepare(self, step_context: 'StepContext', sequence_metadata: CudaSequenceMetadata, attn_metadata) -> None:
        """Build grouped attention metadata, then run dependent updaters."""
        assert self.is_supported
        metadata = tuple(builder.build(step_context, sequence_metadata) for builder in self.attention_builders)
        self._attach_attention_metadata(attn_metadata, metadata)
        for updater in self.step_updaters:
            updater.update(step_context, sequence_metadata)

    def make_cudagraph_buffers(self, graph_meta, input_buffers,
                               step_context: 'StepContext') -> CudaStepMetaGraphBuffers:
        """Allocate attention buffers owned by one graph runner."""
        assert self.is_supported
        attention_buffers = tuple(
            builder.make_cudagraph_buffer(graph_meta, input_buffers, step_context)
            for builder in self.attention_builders)
        return CudaStepMetaGraphBuffers(attention_buffers=attention_buffers)

    def fill_cudagraph_buffers(self, graph_meta, input_buffers, step_context: 'StepContext',
                               buffers: CudaStepMetaGraphBuffers, attn_metadata) -> None:
        """Fill grouped attention buffers and redirect the step metadata."""
        assert self.is_supported
        metadata = tuple(
            builder.fill_cudagraph_buffer(graph_meta, input_buffers, step_context, buffer)
            for builder, buffer in zip(self.attention_builders, buffers.attention_buffers, strict=True))
        self._attach_attention_metadata(attn_metadata, metadata)

    def prepare_cudagraph_capture(self, graph_meta, input_buffers, step_context: 'StepContext',
                                  buffers: CudaStepMetaGraphBuffers, attn_metadata) -> None:
        """Transition grouped metadata after warmup and before capture."""
        assert self.is_supported
        for builder, buffer in zip(self.attention_builders, buffers.attention_buffers, strict=True):
            builder.prepare_cudagraph_capture(graph_meta, input_buffers, step_context, buffer)

        # Graph allocations and operator metadata are not necessarily the
        # same type. For example, FA3 allocates a scheduler tensor but exposes
        # an FA3AttentionMetadata view from fill_cudagraph_buffers. Builders
        # transition allocations in place, then legacy fields are refreshed
        # from those already-attached typed views.
        self._attach_attention_metadata(attn_metadata, attn_metadata.kernel_metadata)


_active_implementations: ContextVar[list[Any] | None] = ContextVar(
    'cuda_step_meta_implementations', default=None)


def register_step_metadata_impl(impl: Any) -> None:
    """Register an implementation only inside the active CUDA build scope."""
    implementations = _active_implementations.get()
    if implementations is None:
        logger.debug('Ignore CUDA step-metadata implementation %s constructed outside a model-build scope.',
                     type(impl).__qualname__)
        return
    implementations.append(impl)


@contextmanager
def collect_step_metadata(ctx_mgr: 'StepContextManager'):
    """Collect one model's CUDA implementations and attach its resolved
    plan."""
    ctx_mgr.backend_step_meta_plan = None
    implementations: list[Any] = []
    token = _active_implementations.set(implementations)
    try:
        yield
        ctx_mgr.backend_step_meta_plan = CudaStepMetaPlan.from_implementations(implementations)
    finally:
        _active_implementations.reset(token)
