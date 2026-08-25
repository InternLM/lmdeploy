# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.backends.deepep_state import get_deepep_state
from lmdeploy.pytorch.config import (
    BackendConfig,
    CacheConfig,
    ModelConfig,
    normalize_cudagraph_capture_batch_sizes,
)
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from lmdeploy.pytorch.strategies.base import StrategyFactoryBase

from ...graph_runner import GraphRunner, is_preparing_prefill
from .full_graph import CUDASingleGraphRunner, build_decode_model_forward

if TYPE_CHECKING:
    from ..attention import TritonAttentionMetadata
    from .piecewise import PiecewiseGraphManager


@functools.lru_cache
def _get_capture_batch_size_impl(max_batches: int):
    """Capture batch size."""
    ret = []
    batch_size = 1
    batch_step = 256
    # power of 2
    while batch_size <= min(batch_step, max_batches):
        ret.append(batch_size)
        batch_size *= 2

    # step
    ret += list(range(batch_size, max_batches + 1, batch_step))

    if max_batches != ret[-1]:
        ret.append(max_batches)
    return ret


def _false(*args, **kwargs):
    """Disable CUDA graph execution for an unsupported model."""
    return False


def _validate_speculative_decoding(model_config: ModelConfig) -> None:
    """Validate the CUDA attention backend required by speculative decode."""
    if model_config.model_paradigm != 'ar_spec' or model_config.use_flash_mla:
        return

    from ..attention import require_fa3_for_speculative_decoding
    require_fa3_for_speculative_decoding()


def _make_piecewise_graph_manager(model: torch.nn.Module, cache_config: CacheConfig, backend_config: BackendConfig,
                                  device: torch.device) -> PiecewiseGraphManager | None:
    """Build the optional PCG runtime only for an eligible CUDA model."""
    max_capture_tokens = backend_config.piecewise_cudagraph_max_tokens
    if max_capture_tokens is None or backend_config.eager_mode or backend_config.device_type != 'cuda':
        return None

    from lmdeploy.pytorch.models.utils.cudagraph import PiecewiseCudaGraphMixin

    if not isinstance(model, PiecewiseCudaGraphMixin):
        return None

    from .piecewise import PiecewiseGraphManager
    from .standard import StandardDecoderPiecewiseGraphRuntime

    runtime = StandardDecoderPiecewiseGraphRuntime(
        model,
        min(max_capture_tokens, cache_config.max_prefill_token_num),
    )
    if not runtime.get_capture_token_sizes():
        return None

    step_meta_plan = model.ctx_mgr.backend_step_meta_plan
    if not step_meta_plan.enable_piecewise_cuda_graph():
        return None

    return PiecewiseGraphManager(runtime)


def _update_deepep_mode(context: StepContext) -> None:
    """Select the DeepEP communication mode for this model forward."""
    if not get_deepep_state().enabled():
        return

    from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPBuffer, DeepEPMode
    mode = DeepEPMode.LOW_LATENCY if context.global_is_decoding() else DeepEPMode.NORMAL
    DeepEPBuffer.set_deepep_mode(mode)


def _destroy_deepep_buffer() -> None:
    """Destroy the process-wide DeepEP buffer at the graph reset barrier."""
    if not get_deepep_state().enabled():
        return

    from torch import distributed as dist

    from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPBuffer
    DeepEPBuffer.destroy()
    dist.barrier()


class CUDAGraphRunner(GraphRunner):
    """Dispatch model forwards among full graph, piecewise graph, and eager."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                 backend_config: BackendConfig, device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config, device)
        self.num_blocks = cache_config.num_gpu_blocks
        _validate_speculative_decoding(model_config)

        self.enable_graph = self.check_enable_graph()
        self._decode_model_forward: Callable[..., Any] | None = None

        self._full_graph_pool_handle = torch.cuda.graph_pool_handle()
        self._full_graph_runners: dict[Any, CUDASingleGraphRunner] = {}

        self._piecewise_graph_manager = _make_piecewise_graph_manager(model, cache_config, backend_config, device)

        # strategy factory
        build_ctx = model.ctx_mgr.build_ctx
        strategy_factory: StrategyFactoryBase = build_ctx.strategy_factory
        self.cudagraph_strategy = strategy_factory.build_cudagraph_strategy()

    def check_enable_graph(self):
        """Check enable graph."""
        if self.backend_config.eager_mode:
            return _false

        return getattr(self.model, 'support_cuda_graph', _false)

    def _get_decode_model_forward(self) -> Callable[..., Any]:
        """Lazily build the callable used to capture decode graphs."""
        if self._decode_model_forward is None:
            if self.backend_config.device_type == 'cuda':
                self._decode_model_forward = build_decode_model_forward(self.model)
            else:
                # CAMB and MACA reuse this graph runner, but the compiler policy
                # and Inductor options above are CUDA-specific.
                self._decode_model_forward = self.model
        return self._decode_model_forward

    def _get_capture_tokens(self, batch_size: int):
        """Get capture tokens."""
        cap_sizes = self.get_capture_batch_sizes()
        for size in cap_sizes:
            if size >= batch_size:
                return size
        assert False, f'Unsupported batch_size={batch_size}'

    def get_graph_key(self, input_ids: torch.Tensor, position_ids: torch.Tensor, past_key_values: list,
                      attn_metadata: TritonAttentionMetadata, inputs_embeds: torch.Tensor, **kwargs):
        """Get graph key."""
        context = self.ctx_mgr.current_context()
        is_decoding = context.global_is_decoding()
        batch_size = attn_metadata.q_seqlens.size(0)
        meta = self.get_meta()
        enable_microbatch = get_step_ctx_manager().current_context().enable_microbatch
        query_len = input_ids.size(1) // batch_size
        if meta.padding_batch_size is None:
            batch_size = self._get_capture_tokens(batch_size)
        else:
            batch_size = self._get_capture_tokens(meta.padding_batch_size)
        graph_key = (batch_size, is_decoding, enable_microbatch, query_len)
        graph_key += self.model.get_cudagraph_extra_key(**kwargs)
        return graph_key

    def _prepare_inputs(self, **kwargs):
        """Prepare inputs."""
        assert 'attn_metadata' in kwargs, 'attn_metadata is required for cudagraph.'
        attn_metadata: TritonAttentionMetadata = kwargs['attn_metadata']
        if attn_metadata.block_offsets.dtype != torch.int32:
            attn_metadata.block_offsets = attn_metadata.block_offsets.to(torch.int32)
        return kwargs

    def _should_use_full_graph(self, context: StepContext, **kwargs) -> bool:
        """Return whether the existing full CUDA graph path owns this call."""
        return context.global_is_decoding() and self.enable_graph(**kwargs)

    def _get_piecewise_graph_descriptor(self, context: StepContext, **kwargs):
        """Return a prefill descriptor without changing runtime state."""
        manager = self._piecewise_graph_manager
        if manager is None or context.global_is_decoding():
            return None
        return manager.get_piecewise_graph_descriptor(context, kwargs)

    def get_prefill_warmup_token_sizes(self) -> list[int]:
        """Return extra single-batch token sizes needed by this runner."""
        manager = self._piecewise_graph_manager
        if manager is None:
            return []
        return manager.get_capture_token_sizes()

    def _get_max_tokens(self, graph_key: tuple, input_ids: torch.Tensor, q_seqlens: torch.Tensor):
        max_batches = graph_key[0]
        is_decoding = graph_key[1]
        assert is_decoding
        origin_batch_size = q_seqlens.size(0)
        num_tokens = input_ids.size(1)
        return self.cudagraph_strategy.get_max_tokens(max_batches, origin_batch_size, num_tokens)

    def _forward_eager(self, **kwargs):
        """Run the existing eager path."""
        with record_function('forward_eager'):
            output = self.model(**kwargs)
            return self.model.make_output_buffers(output)

    def _forward_full_graph(self, **kwargs):
        """Capture or replay one existing full decode CUDA graph."""
        graph_key = self.get_graph_key(**kwargs)
        runner = self._full_graph_runners.get(graph_key)
        if runner is None:
            return self._capture_full_graph(graph_key, kwargs)

        return runner.forward(**kwargs)

    def _capture_full_graph(self, graph_key: tuple, kwargs: dict[str, Any]):
        """Capture and publish a full graph for one cache miss."""
        runner = CUDASingleGraphRunner(
            self.model,
            model_forward=self._get_decode_model_forward(),
            max_batches=graph_key[0],
            max_tokens=self._get_max_tokens(graph_key, kwargs['input_ids'], kwargs['attn_metadata'].q_seqlens),
            num_blocks=self.num_blocks,
            is_decoding=graph_key[1],
            decode_query_len=graph_key[3],
            pool=self._full_graph_pool_handle,
            model_config=self.model_config,
            device=self.device,
        )
        output = runner.capture(**kwargs)
        self._full_graph_runners[graph_key] = runner
        # SSM capture warmup updates state, so the first call returns that
        # warmup output instead of replaying and applying the update twice.
        return output

    def __call__(self, **kwargs):
        """Run one model forward through the selected execution path."""
        kwargs = self._prepare_inputs(**kwargs)
        context = self.ctx_mgr.current_context()
        _update_deepep_mode(context)

        if self._should_use_full_graph(context, **kwargs):
            return self._forward_full_graph(**kwargs)

        descriptor = self._get_piecewise_graph_descriptor(context, **kwargs)
        if descriptor is not None:
            manager = self._piecewise_graph_manager
            if manager.has_plan(descriptor):
                return manager.replay(descriptor, kwargs)
            if is_preparing_prefill():
                return manager.prepare(descriptor, kwargs)

        # Serving never captures. If startup warmup was skipped or this call is
        # unsupported, eager execution is selected before PCG can mutate state.
        return self._forward_eager(**kwargs)

    @record_function('prepare_inputs_for_generation')
    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare inputs."""
        return self.model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )

    def reset(self):
        """Remove all graphs to prevent hanging on exit."""
        super().reset()
        self._full_graph_runners.clear()
        if self._piecewise_graph_manager is not None:
            self._piecewise_graph_manager.reset()
        _destroy_deepep_buffer()

    def update_inputs(self, inputs):
        """Update inputs."""
        if self.backend_config.eager_mode:
            return inputs
        is_decoding = inputs.global_is_decoding()
        dp_meta = inputs.dp_meta
        if is_decoding and dp_meta is not None:
            meta = self.get_meta()
            padding_batch_size = meta.padding_batch_size
            batch_size = inputs.seq_length.size(0)
            query_len = inputs.input_ids.numel() // batch_size
            tp_size = self._get_capture_tokens(padding_batch_size) * query_len
            dp_meta.sync_tp_size(tp_size)
        return inputs

    def get_capture_batch_sizes(self) -> list[int]:
        """Capture batch sizes."""
        if self.cache_config.cudagraph_capture_batch_sizes is not None:
            self.cache_config.cudagraph_capture_batch_sizes = normalize_cudagraph_capture_batch_sizes(
                self.cache_config.cudagraph_capture_batch_sizes, self.cache_config.max_batches)
            return self.cache_config.cudagraph_capture_batch_sizes
        return _get_capture_batch_size_impl(self.cache_config.max_batches)
