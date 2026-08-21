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
from lmdeploy.pytorch.envs import enable_piecewise_cuda_graph
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from lmdeploy.pytorch.strategies.base import StrategyFactoryBase

from ...graph_runner import GraphRunner
from .full_graph import CUDASingleGraphRunner, build_decode_model_forward

if TYPE_CHECKING:
    from ..attention import TritonAttentionMetadata


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


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
    """Default value of not support cuda graph."""
    return False


class CUDAGraphRunner(GraphRunner):
    """Cuda graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                 backend_config: BackendConfig, device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config, device)
        self.max_batches = cache_config.max_batches
        self.num_blocks = cache_config.num_gpu_blocks

        # Speculative decoding on CUDA requires FlashAttention-3 (FA3),
        # unless the model uses FlashMLA (e.g., DeepSeek MTP) which handles
        # multi-token decoding queries natively.
        # FA3 is available on SM80+ (Ampere and above) GPUs with CUDA >= 12.3.
        # Without FA3, the Triton paged attention kernel cannot handle
        # multi-token decoding queries (max_q_seqlen > 1) used in spec decoding.
        if model_config.model_paradigm == 'ar_spec' and not getattr(model_config, 'use_flash_mla', False):
            from ..attention import use_fa3
            if not use_fa3:
                sm = torch.cuda.get_device_capability()
                cuda_ver = torch.version.cuda or 'N/A'
                raise RuntimeError(
                    f'Speculative decoding on CUDA requires FlashAttention-3 (FA3), '
                    f'which needs SM80+ (Ampere and above) with CUDA >= 12.3 and '
                    f'flash-attn installed. Detected: SM{sm[0]}.{sm[1]}, CUDA {cuda_ver}. '
                    f'Please ensure your GPU meets SM80+, CUDA >= 12.3, and flash-attn '
                    f'is installed, or disable speculative decoding.')

        self.enable_graph = self.check_enable_graph()
        self._decode_model_forward: Callable[..., Any] | None = None

        self._full_graph_pool_handle = torch.cuda.graph_pool_handle()
        self._full_graph_runners: dict[Any, CUDASingleGraphRunner] = dict()

        self._piecewise_graph_manager = None
        if (enable_piecewise_cuda_graph and not backend_config.eager_mode and backend_config.device_type == 'cuda'):
            hooks = getattr(model, 'piecewise_cuda_graph_hooks', None)
            if hooks is not None:
                from .piecewise import PiecewiseGraphManager

                self._piecewise_graph_manager = PiecewiseGraphManager(hooks)

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
        if not attn_metadata.block_offsets.dtype == torch.int32:
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
            max_batches = graph_key[0]
            is_decoding = graph_key[1]
            decode_query_len = graph_key[3]
            max_tokens = self._get_max_tokens(graph_key, kwargs['input_ids'], kwargs['attn_metadata'].q_seqlens)
            runner = CUDASingleGraphRunner(
                self.model,
                model_forward=self._get_decode_model_forward(),
                max_batches=max_batches,
                max_tokens=max_tokens,
                num_blocks=self.num_blocks,
                is_decoding=is_decoding,
                decode_query_len=decode_query_len,
                pool=self._full_graph_pool_handle,
                model_config=self.model_config,
                device=self.device,
            )
            output = runner.capture(**kwargs)
            self._full_graph_runners[graph_key] = runner
            # SSM would update the state in capture(warmup), replay the graph will leads unexpected state update.
            return output

        return runner.forward(**kwargs)

    def _forward_piecewise_graph(self, descriptor, **kwargs):
        """Build or replay one sibling piecewise CUDA graph plan."""
        return self._piecewise_graph_manager.run(descriptor, kwargs)

    def __call__(self, **kwargs):
        """call."""
        kwargs = self._prepare_inputs(**kwargs)
        context = self.ctx_mgr.current_context()
        if get_deepep_state().enabled():
            from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPBuffer, DeepEPMode
            deepep_mode = DeepEPMode.LOW_LATENCY if context.global_is_decoding() else DeepEPMode.NORMAL
            DeepEPBuffer.set_deepep_mode(deepep_mode)

        if self._should_use_full_graph(context, **kwargs):
            return self._forward_full_graph(**kwargs)

        descriptor = self._get_piecewise_graph_descriptor(context, **kwargs)
        if descriptor is not None:
            result = self._forward_piecewise_graph(descriptor, **kwargs)
            if result.executed:
                return result.output

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
        if get_deepep_state().enabled():
            from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPBuffer

            if hasattr(DeepEPBuffer, 'destroy'):
                from torch import distributed as dist

                DeepEPBuffer.destroy()
                dist.barrier()

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
