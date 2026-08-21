# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.distributed import get_world_rank
from lmdeploy.pytorch.nn import Attention

from .piecewise import (
    PiecewiseGraphDescriptor,
    PiecewiseGraphGuardError,
    PiecewiseGraphHooks,
    PiecewiseGraphPlan,
    piecewise_graph_execution,
    trace_piecewise_cuda_graph,
)

if TYPE_CHECKING:
    from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionMetadata
    from lmdeploy.pytorch.model_inputs import StepContext
    from lmdeploy.pytorch.models.utils.cudagraph import PiecewiseCudaGraphMixin


_FORWARD_INPUT_NAMES = frozenset({
    'input_ids',
    'position_ids',
    'past_key_values',
    'attn_metadata',
    'inputs_embeds',
})


@dataclass(frozen=True)
class _StandardPiecewiseGraphKey:
    """Facts that keep one standard decoder plan address- and shape-safe."""

    token_bucket: int
    input_contracts: tuple[tuple[Any, ...], ...]
    cache_contracts: tuple[tuple[Any, ...], ...]


@dataclass
class _StandardPiecewiseGraphPlan:
    """Captured plan plus mutable metadata retained by eager steps."""

    graph: PiecewiseGraphPlan
    key: _StandardPiecewiseGraphKey
    attention_metadata: TritonAttentionMetadata
    output_buffers: dict[str, torch.Tensor]


class StandardDecoderPiecewiseGraphRuntime:
    """Run the shared dense-decoder/default-attention prefill PCG path."""

    def __init__(self, model: PiecewiseCudaGraphMixin) -> None:
        from lmdeploy.pytorch.backends.cuda.attention import TritonAttentionImpl, TritonAttentionMetadata

        self.model = model
        self._attention_metadata_type = TritonAttentionMetadata
        self._attentions = tuple(module for module in model.modules() if isinstance(module, Attention))
        self._backend_supported = bool(self._attentions) and all(
            type(attention.impl) is TritonAttentionImpl for attention in self._attentions)
        if self._backend_supported:
            for attention in self._attentions:
                attention.impl.enable_piecewise_cuda_graph()

        self.hooks = PiecewiseGraphHooks(
            get_piecewise_graph_descriptor=self.get_piecewise_graph_descriptor,
            build=self.build,
            replay=self.replay,
        )

    def get_piecewise_graph_descriptor(
        self,
        context: StepContext,
        kwargs: Mapping[str, Any],
    ) -> PiecewiseGraphDescriptor | None:
        """Select the initial BF16, TP1, one-request prefill envelope."""
        key = self._get_supported_key(context, kwargs)
        if key is None:
            return None
        return PiecewiseGraphDescriptor(key)

    def build(
        self,
        descriptor: PiecewiseGraphDescriptor,
        kwargs: Mapping[str, Any],
    ) -> _StandardPiecewiseGraphPlan:
        """Capture with one synthetic token written to reserved page 0."""
        context = self.model.ctx_mgr.current_context()
        current = self.get_piecewise_graph_descriptor(context, kwargs)
        if current != descriptor or not isinstance(descriptor.key, _StandardPiecewiseGraphKey):
            raise PiecewiseGraphGuardError('piecewise construction inputs changed before build')

        key = descriptor.key
        input_ids = kwargs['input_ids']
        position_ids = kwargs['position_ids']
        past_key_values = kwargs['past_key_values']
        live_metadata = kwargs['attn_metadata']
        examples = self._make_construction_inputs(input_ids, position_ids, live_metadata, key.token_bucket)
        attention_metadata = self._make_construction_metadata(examples)

        def model_forward(
            static_input_ids: torch.Tensor,
            static_position_ids: torch.Tensor,
            block_offsets: torch.Tensor,
            q_start_loc: torch.Tensor,
            q_seqlens: torch.Tensor,
            kv_start_loc: torch.Tensor,
            kv_seqlens: torch.Tensor,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
        ) -> torch.Tensor:
            attention_metadata.block_offsets = block_offsets
            attention_metadata.q_start_loc = q_start_loc
            attention_metadata.q_seqlens = q_seqlens
            attention_metadata.kv_start_loc = kv_start_loc
            attention_metadata.kv_seqlens = kv_seqlens
            attention_metadata.cu_seqlens_q = cu_seqlens_q
            attention_metadata.cu_seqlens_k = cu_seqlens_k
            synthetic_context = replace(
                context,
                input_ids=static_input_ids,
                position_ids=static_position_ids,
                attention_mask=None,
                block_offsets=block_offsets,
                q_start_loc=q_start_loc,
                q_seqlens=q_seqlens,
                kv_seqlens=kv_seqlens,
                kv_caches=past_key_values,
                sum_kv_seqlen=1,
                max_kv_seqlen=1,
                max_q_seqlen=1,
                attn_metadata=attention_metadata,
                _outputs={},
            )
            with self.model.ctx_mgr.context(synthetic_context):
                return self.model(
                    input_ids=static_input_ids,
                    position_ids=static_position_ids,
                    past_key_values=past_key_values,
                    attn_metadata=attention_metadata,
                    inputs_embeds=None,
                )

        def trace_model_forward(*static_inputs: torch.Tensor) -> torch.Tensor:
            with piecewise_graph_execution(raw_tokens=1, token_bucket=key.token_bucket):
                return model_forward(*static_inputs)

        graph = trace_piecewise_cuda_graph(
            trace_model_forward,
            examples,
            warmup_iterations=2,
            warmup_func=model_forward,
        )

        return _StandardPiecewiseGraphPlan(
            graph=graph,
            key=key,
            attention_metadata=attention_metadata,
            output_buffers=self.model.make_output_buffers(graph.output),
        )

    def replay(
        self,
        plan: _StandardPiecewiseGraphPlan,
        descriptor: PiecewiseGraphDescriptor,
        kwargs: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Bind one live request, replay in order, and trim the output."""
        context = self.model.ctx_mgr.current_context()
        current = self.get_piecewise_graph_descriptor(context, kwargs)
        if current != descriptor or descriptor.key != plan.key:
            raise PiecewiseGraphGuardError('piecewise replay inputs no longer match the plan')

        input_ids = kwargs['input_ids']
        position_ids = kwargs['position_ids']
        live_metadata = kwargs['attn_metadata']
        raw_tokens = input_ids.size(1)

        plan.attention_metadata.kv_flatten_size = live_metadata.kv_flatten_size
        plan.attention_metadata.max_kv_seqlen = live_metadata.max_kv_seqlen
        plan.attention_metadata.max_q_seqlen = raw_tokens

        def bind(static_inputs: tuple[torch.Tensor, ...]) -> None:
            (static_input_ids, static_position_ids, block_offsets, q_start_loc, q_seqlens, kv_start_loc,
             kv_seqlens, cu_seqlens_q, cu_seqlens_k) = static_inputs

            static_input_ids.zero_()
            static_input_ids[:, :raw_tokens].copy_(input_ids)
            static_position_ids.zero_()
            static_position_ids[:, :raw_tokens].copy_(position_ids)
            block_offsets.copy_(live_metadata.block_offsets)
            q_start_loc.copy_(live_metadata.q_start_loc)
            q_seqlens.copy_(live_metadata.q_seqlens)
            kv_start_loc.copy_(live_metadata.kv_start_loc)
            kv_seqlens.copy_(live_metadata.kv_seqlens)
            cu_seqlens_q.copy_(live_metadata.cu_seqlens_q)
            cu_seqlens_k.copy_(live_metadata.cu_seqlens_k)

        with piecewise_graph_execution(raw_tokens=raw_tokens, token_bucket=plan.key.token_bucket):
            plan.graph.replay_with_input_binder(bind)

        return self.model.get_outputs_cudagraph(plan.output_buffers, input_ids=input_ids)

    def _get_supported_key(
        self,
        context: StepContext,
        kwargs: Mapping[str, Any],
    ) -> _StandardPiecewiseGraphKey | None:
        if not self._backend_supported or kwargs.keys() != _FORWARD_INPUT_NAMES or context is None:
            return None
        if context.global_is_decoding() or context.is_decoding:
            return None
        if (context.dp_meta is not None or context.enable_microbatch or context.is_dummy or context.is_chunk
                or context.is_chunk_multimodal):
            return None
        if (context.local_adapter_ids is not None or context.input_embeddings is not None
                or context.input_embedding_indexing is not None or context.input_multimodals is not None
                or context.vision_inputs is not None):
            return None
        if context.kv_quant_policy != QuantPolicy.NONE or context.cache_config.quant_policy != QuantPolicy.NONE:
            return None
        if context.cache_config.num_reserved_gpu_blocks < 1 or context.cache_config.num_gpu_blocks < 1:
            return None
        if get_world_rank()[0] != 1:
            return None
        if self.model.get_input_embeddings().weight.dtype != torch.bfloat16:
            return None

        input_ids = kwargs['input_ids']
        position_ids = kwargs['position_ids']
        past_key_values = kwargs['past_key_values']
        attention_metadata = kwargs['attn_metadata']
        if kwargs['inputs_embeds'] is not None or type(attention_metadata) is not self._attention_metadata_type:
            return None
        if attention_metadata is not context.attn_metadata:
            return None
        if attention_metadata.is_decoding or attention_metadata.quant_policy != QuantPolicy.NONE:
            return None
        if attention_metadata.kv_flatten_size is None or attention_metadata.max_kv_seqlen is None:
            return None

        token_bucket = context.cache_config.block_size
        raw_tokens = input_ids.size(1) if input_ids.ndim == 2 else 0
        if (input_ids.device.type != 'cuda' or position_ids.device != input_ids.device
                or input_ids.shape != position_ids.shape or input_ids.size(0) != 1
                or input_ids.dtype != torch.int64 or position_ids.dtype != torch.int64
                or not 0 < raw_tokens <= token_bucket):
            return None
        if context.max_q_seqlen != raw_tokens or context.q_seqlens.numel() != 1:
            return None
        if not self._has_supported_metadata(attention_metadata, input_ids.device):
            return None

        cache_contracts = self._get_cache_contracts(
            past_key_values,
            input_ids.device,
            context.cache_config.num_reserved_gpu_blocks,
        )
        if cache_contracts is None:
            return None
        input_contracts = (
            self._tensor_contract(attention_metadata.block_offsets),
            self._tensor_contract(attention_metadata.q_start_loc),
            self._tensor_contract(attention_metadata.q_seqlens),
            self._tensor_contract(attention_metadata.kv_start_loc),
            self._tensor_contract(attention_metadata.kv_seqlens),
            self._tensor_contract(attention_metadata.cu_seqlens_q),
            self._tensor_contract(attention_metadata.cu_seqlens_k),
        )
        return _StandardPiecewiseGraphKey(token_bucket, input_contracts, cache_contracts)

    def _has_supported_metadata(self, metadata: TritonAttentionMetadata, device: torch.device) -> bool:
        tensors = (
            metadata.block_offsets,
            metadata.q_start_loc,
            metadata.q_seqlens,
            metadata.kv_start_loc,
            metadata.kv_seqlens,
            metadata.cu_seqlens_q,
            metadata.cu_seqlens_k,
        )
        if any(not isinstance(value, torch.Tensor) or value.device != device for value in tensors):
            return False
        return (metadata.block_offsets.ndim == 2 and metadata.block_offsets.size(0) == 1
                and metadata.q_start_loc.shape == (1, ) and metadata.q_seqlens.shape == (1, )
                and metadata.kv_start_loc.shape == (1, ) and metadata.kv_seqlens.shape == (1, )
                and metadata.cu_seqlens_q.shape == (2, ) and metadata.cu_seqlens_k.shape == (2, ))

    def _get_cache_contracts(
        self,
        past_key_values: Any,
        device: torch.device,
        reserved_blocks: int,
    ) -> tuple[tuple[Any, ...], ...] | None:
        if not isinstance(past_key_values, (tuple, list)) or len(past_key_values) != len(self._attentions):
            return None

        contracts: list[tuple[Any, ...]] = []
        for layer_cache in past_key_values:
            if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) != 2:
                return None
            for cache in layer_cache:
                if (not isinstance(cache, torch.Tensor) or cache.device != device or cache.dtype != torch.bfloat16
                        or cache.ndim < 2 or cache.size(0) < reserved_blocks or cache.size(1) < 1):
                    return None
                contracts.append(self._cache_contract(cache))
        return tuple(contracts)

    @staticmethod
    def _make_construction_inputs(
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        metadata: TritonAttentionMetadata,
        token_bucket: int,
    ) -> tuple[torch.Tensor, ...]:
        construction_input_ids = input_ids.new_zeros(1, token_bucket)
        construction_position_ids = position_ids.new_zeros(1, token_bucket)
        block_offsets = metadata.block_offsets.new_zeros(metadata.block_offsets.shape)
        q_start_loc = metadata.q_start_loc.new_zeros(metadata.q_start_loc.shape)
        q_seqlens = metadata.q_seqlens.new_ones(metadata.q_seqlens.shape)
        kv_start_loc = metadata.kv_start_loc.new_zeros(metadata.kv_start_loc.shape)
        kv_seqlens = metadata.kv_seqlens.new_ones(metadata.kv_seqlens.shape)
        cu_seqlens_q = metadata.cu_seqlens_q.new_tensor([0, 1])
        cu_seqlens_k = metadata.cu_seqlens_k.new_tensor([0, 1])
        return (construction_input_ids, construction_position_ids, block_offsets, q_start_loc, q_seqlens,
                kv_start_loc, kv_seqlens, cu_seqlens_q, cu_seqlens_k)

    def _make_construction_metadata(self, inputs: tuple[torch.Tensor, ...]) -> TritonAttentionMetadata:
        (_, _, block_offsets, q_start_loc, q_seqlens, kv_start_loc, kv_seqlens, cu_seqlens_q,
         cu_seqlens_k) = inputs
        return self._attention_metadata_type(
            is_decoding=False,
            block_offsets=block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_start_loc=kv_start_loc,
            kv_seqlens=kv_seqlens,
            quant_policy=QuantPolicy.NONE,
            kv_flatten_size=1,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_kv_seqlen=1,
            max_q_seqlen=1,
        )

    @staticmethod
    def _tensor_contract(tensor: torch.Tensor) -> tuple[Any, ...]:
        return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)

    @classmethod
    def _cache_contract(cls, tensor: torch.Tensor) -> tuple[Any, ...]:
        return (tensor.data_ptr(), *cls._tensor_contract(tensor))
