# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.distributed import get_dp_world_rank, get_ep_world_rank

from .piecewise import (
    PiecewiseGraphBuild,
    PiecewiseGraphPlan,
    ReusableBridgePool,
    piecewise_graph_execution,
    trace_piecewise_cuda_graph,
)

if TYPE_CHECKING:
    from lmdeploy.pytorch.model_inputs import StepContext
    from lmdeploy.pytorch.models.utils.cudagraph import PiecewiseCudaGraphMixin


_STANDARD_FORWARD_INPUT_NAMES = frozenset({
    'input_ids',
    'position_ids',
    'past_key_values',
    'attn_metadata',
})
_FORWARD_CONSTANT_TYPES = (bool, int, float, str, bytes, torch.dtype, torch.device)
_DEFAULT_TOKEN_STRIDE = 512


@dataclass(frozen=True)
class _StandardGraphDescriptor:
    """Immutable inputs that select one standard decoder plan."""

    token_bucket: int
    forward_constants: tuple[tuple[str, Any], ...]


@dataclass
class _StandardGraphPlan:
    """Captured plan and reusable output storage for a standard decoder."""

    graph: PiecewiseGraphPlan
    descriptor: _StandardGraphDescriptor
    output_buffers: dict[str, torch.Tensor]


class StandardDecoderPiecewiseGraphRuntime:
    """Run the shared dense-decoder prefill PCG path."""

    def __init__(
        self,
        model: PiecewiseCudaGraphMixin,
        max_capture_tokens: int,
        cache_quant_policy: QuantPolicy = QuantPolicy.NONE,
        token_stride: int = _DEFAULT_TOKEN_STRIDE,
    ) -> None:
        if token_stride < 1:
            raise ValueError('token_stride must be positive')
        self.model = model
        self.max_capture_tokens = max_capture_tokens
        self.cache_quant_policy = cache_quant_policy
        self.token_stride = token_stride

    def get_capture_token_sizes(self) -> list[int]:
        """Return fixed-stride token buckets, including the configured cap."""
        if (self.max_capture_tokens == 0 or self.cache_quant_policy != QuantPolicy.NONE
                or get_dp_world_rank()[0] != 1 or get_ep_world_rank()[0] != 1):
            return []
        sizes = list(range(self.token_stride, self.max_capture_tokens, self.token_stride))
        sizes.append(self.max_capture_tokens)
        return sizes

    def get_piecewise_graph_descriptor(
        self,
        context: StepContext,
        kwargs: Mapping[str, Any],
    ) -> _StandardGraphDescriptor | None:
        """Select one supported single-rank dense-prefill plan."""
        if context.enable_microbatch or context.is_chunk:
            return None
        if context.local_adapter_ids is not None and not context.is_dummy:
            return None

        forward_constants = self._get_forward_constants(kwargs)
        if forward_constants is None:
            return None

        raw_tokens = kwargs['input_ids'].size(1)
        token_bucket = self._round_up_token_bucket(raw_tokens)
        if token_bucket is None:
            return None
        return _StandardGraphDescriptor(token_bucket, forward_constants)

    def warmup(
        self,
        descriptor: _StandardGraphDescriptor,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Run one startup prefill eagerly with bucket-shaped boundary
        semantics."""
        context = self.model.ctx_mgr.current_context()
        input_ids = kwargs['input_ids']
        position_ids = kwargs['position_ids']
        bucket_inputs = self._make_bucket_inputs(input_ids, position_ids, descriptor.token_bucket)

        with torch.inference_mode(), piecewise_graph_execution(
                raw_tokens=input_ids.size(1), token_bucket=descriptor.token_bucket):
            self._forward_bucket(
                context,
                kwargs['past_key_values'],
                kwargs['attn_metadata'],
                descriptor.forward_constants,
                *bucket_inputs,
            )

    def build(
        self,
        descriptor: _StandardGraphDescriptor,
        kwargs: Mapping[str, Any],
        bridge_pool: ReusableBridgePool,
    ) -> PiecewiseGraphBuild:
        """Capture one startup prefill and return its already-materialized
        output."""
        context = self.model.ctx_mgr.current_context()
        input_ids = kwargs['input_ids']
        position_ids = kwargs['position_ids']
        past_key_values = kwargs['past_key_values']
        live_metadata = kwargs['attn_metadata']
        bucket_inputs = self._make_bucket_inputs(input_ids, position_ids, descriptor.token_bucket)
        raw_tokens = input_ids.size(1)

        def capture_forward(
            static_input_ids: torch.Tensor,
            static_position_ids: torch.Tensor,
        ) -> torch.Tensor:
            return self._forward_bucket(
                context,
                past_key_values,
                live_metadata,
                descriptor.forward_constants,
                static_input_ids,
                static_position_ids,
            )

        with piecewise_graph_execution(raw_tokens=raw_tokens, token_bucket=descriptor.token_bucket):
            graph = trace_piecewise_cuda_graph(
                capture_forward,
                bucket_inputs,
                warmup_iterations=0,
                bridge_pool=bridge_pool,
            )

        plan = _StandardGraphPlan(
            graph=graph,
            descriptor=descriptor,
            output_buffers=self.model.make_output_buffers(graph.output),
        )
        output = self.model.get_outputs_cudagraph(plan.output_buffers, input_ids=input_ids)
        return PiecewiseGraphBuild(plan=plan, output=output)

    def replay(
        self,
        plan: _StandardGraphPlan,
        kwargs: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Bind one live batch, replay in order, and trim the output."""
        input_ids = kwargs['input_ids']
        position_ids = kwargs['position_ids']
        raw_tokens = input_ids.size(1)

        def fill_inputs(static_inputs: tuple[torch.Tensor, ...]) -> None:
            self._fill_bucket_inputs(static_inputs, input_ids, position_ids)

        with piecewise_graph_execution(raw_tokens=raw_tokens, token_bucket=plan.descriptor.token_bucket):
            plan.graph.replay_with_input_binder(fill_inputs)

        return self.model.get_outputs_cudagraph(plan.output_buffers, input_ids=input_ids)

    @staticmethod
    def _get_forward_constants(kwargs: Mapping[str, Any]) -> tuple[tuple[str, Any], ...] | None:
        """Keep immutable extra arguments in the plan key and captured call."""
        constants = []
        for name in sorted(kwargs.keys() - _STANDARD_FORWARD_INPUT_NAMES):
            value = kwargs[name]
            if value is not None and not isinstance(value, _FORWARD_CONSTANT_TYPES):
                return None
            constants.append((name, value))
        return tuple(constants)

    def _round_up_token_bucket(self, raw_tokens: int) -> int | None:
        if not 0 < raw_tokens <= self.max_capture_tokens:
            return None
        rounded = ((raw_tokens + self.token_stride - 1) // self.token_stride) * self.token_stride
        return min(rounded, self.max_capture_tokens)

    @classmethod
    def _make_bucket_inputs(
        cls,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_bucket: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bucket_input_ids = input_ids.new_zeros(1, token_bucket)
        bucket_position_ids = position_ids.new_zeros(1, token_bucket)
        bucket_inputs = (bucket_input_ids, bucket_position_ids)
        cls._fill_bucket_inputs(bucket_inputs, input_ids, position_ids)
        return bucket_inputs

    @staticmethod
    def _fill_bucket_inputs(
        bucket_inputs: tuple[torch.Tensor, torch.Tensor],
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Copy one logical request and deterministically clear its tail."""
        bucket_input_ids, bucket_position_ids = bucket_inputs
        raw_tokens = input_ids.size(1)
        bucket_input_ids.zero_()
        bucket_input_ids[:, :raw_tokens].copy_(input_ids)
        bucket_position_ids.zero_()
        bucket_position_ids[:, :raw_tokens].copy_(position_ids)

    def _forward_bucket(
        self,
        context: StepContext,
        past_key_values: Any,
        attention_metadata: Any,
        forward_constants: tuple[tuple[str, Any], ...],
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bucket_context = replace(
            context,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            kv_caches=past_key_values,
            attn_metadata=attention_metadata,
            _outputs={},
        )
        with self.model.ctx_mgr.context(bucket_context):
            model_inputs = dict(forward_constants)
            model_inputs.update(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                attn_metadata=attention_metadata,
            )
            return self.model(**model_inputs)
