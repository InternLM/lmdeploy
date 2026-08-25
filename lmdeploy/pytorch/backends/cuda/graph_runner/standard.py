# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

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


_GRAPH_TOKEN_INPUT_AXES = {
    'input_ids': 1,
    'position_ids': 1,
    'mrope_position_ids': 1,
}
_FRAME_INPUT_NAMES = (
    'past_key_values',
    'attn_metadata',
    'state_ids',
)
_STANDARD_FORWARD_INPUT_NAMES = frozenset(_GRAPH_TOKEN_INPUT_AXES) | frozenset(_FRAME_INPUT_NAMES)
_FORWARD_CONSTANT_TYPES = (bool, int, float, str, bytes, torch.dtype, torch.device)
_DEFAULT_TOKEN_STRIDE = 512


@dataclass(frozen=True)
class _StandardGraphDescriptor:
    """Immutable inputs that select one standard decoder plan."""

    token_bucket: int
    graph_input_names: tuple[str, ...]
    forward_constants: tuple[tuple[str, Any], ...]


@dataclass
class _StandardGraphPlan:
    """Captured plan and reusable output storage for a standard decoder."""

    graph: PiecewiseGraphPlan
    descriptor: _StandardGraphDescriptor
    output_buffers: dict[str, torch.Tensor]


class StandardDecoderPiecewiseGraphRuntime:
    """Run the shared standard-decoder prefill PCG path."""

    def __init__(
        self,
        model: PiecewiseCudaGraphMixin,
        max_capture_tokens: int,
        token_stride: int = _DEFAULT_TOKEN_STRIDE,
    ) -> None:
        if token_stride < 1:
            raise ValueError('token_stride must be positive')
        self.model = model
        self.max_capture_tokens = max_capture_tokens
        self.token_stride = token_stride

    def get_capture_token_sizes(self) -> list[int]:
        """Return fixed-stride token buckets, including the configured cap."""
        if (self.max_capture_tokens == 0
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
        """Select one supported standard-prefill plan."""
        if context.enable_microbatch:
            return None
        if context.local_adapter_ids is not None and not context.is_dummy:
            return None

        forward_constants = self._get_forward_constants(kwargs)
        if forward_constants is None:
            return None

        raw_tokens = kwargs['input_ids'].size(1)
        graph_input_names = tuple(name for name in _GRAPH_TOKEN_INPUT_AXES if kwargs.get(name) is not None)
        token_bucket = self._round_up_token_bucket(raw_tokens)
        if token_bucket is None:
            return None
        return _StandardGraphDescriptor(token_bucket, graph_input_names, forward_constants)

    def warmup(
        self,
        descriptor: _StandardGraphDescriptor,
        kwargs: Mapping[str, Any],
    ) -> None:
        """Run one startup prefill eagerly with bucket-shaped boundary
        semantics."""
        context = self.model.ctx_mgr.current_context()
        input_ids = kwargs['input_ids']
        bucket_inputs = self._make_bucket_inputs(kwargs, descriptor)

        with torch.inference_mode(), piecewise_graph_execution(
                raw_tokens=input_ids.size(1), token_bucket=descriptor.token_bucket):
            self._forward_bucket(
                context,
                self._get_frame_inputs(kwargs),
                descriptor,
                *bucket_inputs,
            )

    def build(
        self,
        descriptor: _StandardGraphDescriptor,
        kwargs: Mapping[str, Any],
        bridge_pool: ReusableBridgePool,
        stream: torch.cuda.Stream,
    ) -> PiecewiseGraphBuild:
        """Capture one startup prefill and return its already-materialized
        output."""
        context = self.model.ctx_mgr.current_context()
        input_ids = kwargs['input_ids']
        frame_inputs = self._get_frame_inputs(kwargs)
        bucket_inputs = self._make_bucket_inputs(kwargs, descriptor)
        raw_tokens = input_ids.size(1)

        def capture_forward(*static_inputs: torch.Tensor) -> torch.Tensor:
            return self._forward_bucket(
                context,
                frame_inputs,
                descriptor,
                *static_inputs,
            )

        with piecewise_graph_execution(raw_tokens=raw_tokens, token_bucket=descriptor.token_bucket):
            graph = trace_piecewise_cuda_graph(
                capture_forward,
                bucket_inputs,
                warmup_iterations=0,
                bridge_pool=bridge_pool,
                frame_inputs=frame_inputs,
                stream=stream,
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
        raw_tokens = input_ids.size(1)
        frame_inputs = self._get_frame_inputs(kwargs)

        def fill_inputs(static_inputs: tuple[torch.Tensor, ...]) -> None:
            self._fill_bucket_inputs(static_inputs, kwargs, plan.descriptor)

        with piecewise_graph_execution(raw_tokens=raw_tokens, token_bucket=plan.descriptor.token_bucket):
            plan.graph.replay_with_input_binder(fill_inputs, frame_inputs=frame_inputs)

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

    @staticmethod
    def _get_frame_inputs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Build the request frame used to late-bind eager call arguments."""
        return {name: kwargs[name] for name in _FRAME_INPUT_NAMES if name in kwargs}

    def _round_up_token_bucket(self, raw_tokens: int) -> int | None:
        if not 0 < raw_tokens <= self.max_capture_tokens:
            return None
        rounded = ((raw_tokens + self.token_stride - 1) // self.token_stride) * self.token_stride
        return min(rounded, self.max_capture_tokens)

    @classmethod
    def _make_bucket_inputs(
        cls,
        inputs: Mapping[str, Any],
        descriptor: _StandardGraphDescriptor,
    ) -> tuple[torch.Tensor, ...]:
        bucket_inputs = []
        for name in descriptor.graph_input_names:
            value = inputs[name]
            shape = list(value.shape)
            shape[_GRAPH_TOKEN_INPUT_AXES[name]] = descriptor.token_bucket
            bucket_inputs.append(value.new_zeros(shape))
        bucket_inputs = tuple(bucket_inputs)
        cls._fill_bucket_inputs(bucket_inputs, inputs, descriptor)
        return bucket_inputs

    @staticmethod
    def _fill_bucket_inputs(
        bucket_inputs: tuple[torch.Tensor, ...],
        inputs: Mapping[str, Any],
        descriptor: _StandardGraphDescriptor,
    ) -> None:
        """Copy one logical request and deterministically clear its tail."""
        for bucket_input, name in zip(bucket_inputs, descriptor.graph_input_names, strict=True):
            value = inputs[name]
            token_axis = _GRAPH_TOKEN_INPUT_AXES[name]
            bucket_input.zero_()
            bucket_input.narrow(token_axis, 0, value.size(token_axis)).copy_(value)

    def _forward_bucket(
        self,
        context: StepContext,
        frame_inputs: Mapping[str, Any],
        descriptor: _StandardGraphDescriptor,
        *graph_inputs: torch.Tensor,
    ) -> torch.Tensor:
        graph_inputs = dict(zip(descriptor.graph_input_names, graph_inputs, strict=True))
        input_ids = graph_inputs['input_ids']
        position_ids = graph_inputs['position_ids']
        bucket_context = replace(
            context,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            kv_caches=frame_inputs['past_key_values'],
            attn_metadata=frame_inputs['attn_metadata'],
            mrope_position_ids=graph_inputs.get('mrope_position_ids'),
        )
        with self.model.ctx_mgr.context(bucket_context):
            model_inputs = dict(descriptor.forward_constants)
            model_inputs.update(frame_inputs)
            model_inputs.update(graph_inputs)
            return self.model(**model_inputs)
