# Copyright (c) OpenMMLab. All rights reserved.
import functools
from typing import Any, Callable, Dict, List

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.backends.deepep_moe_checker import get_moe_backend
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.pytorch.strategies.base import StrategyFactoryBase
from lmdeploy.utils import get_logger

from ...graph_runner import GraphRunner
from ..attention import TritonAttentionMetadata
from .piecewise_backend import create_backend

logger = get_logger('lmdeploy')

PREFILL_FULLGRAPH = True
DECODING_FULLGRAPH = True


def get_attn_metadata(kwargs: Dict[str, Any], context: StepContext) -> TritonAttentionMetadata:
    """Get attention metadata from kwargs or context."""
    if 'attn_metadata' in kwargs:
        attn_metadata: TritonAttentionMetadata = kwargs['attn_metadata']
    else:
        attn_metadata: TritonAttentionMetadata = context.attn_metadata
    return attn_metadata


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


def mark_static_kv_cache(kv_caches: List[List[torch.Tensor]]):
    for kv_cache in kv_caches:
        for cache in kv_cache:
            torch._dynamo.mark_static_address(cache)


class TorchCompileSinglePrefillRunner:

    def __init__(
        self,
        model: torch.nn.Module,
        graph: Callable,
        key: Any,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        block_size: int,
        model_config: ModelConfig,
        device: torch.device,
        decode_query_len: int = 1,
    ):
        self.model = model
        self.graph = graph
        self.ctx_mgr = model.ctx_mgr
        self.model_config = model_config
        self.key = key

        self.meta = CudaGraphMeta(
            max_batchs=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            block_size=block_size,
            is_decoding=True,
            device=device,
            input_buffers=dict(),
            output_buffers=dict(),
            vocab_size=self.model_config.vocab_size,
            use_mla_fp8_cache=getattr(self.model_config, 'use_mla_fp8_cache', False),
            use_flash_mla=getattr(self.model_config, 'use_flash_mla', False),
            mla_index_topk=getattr(self.model_config, 'mla_index_topk', None),
            decode_query_len=decode_query_len,
            use_fa3_decoding=model_config.model_paradigm == 'ar_spec',
        )
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = True

    def forward(self, **kwargs):
        """call."""
        # make buffer is not exist
        # same inputs as cudagraph
        if len(self.meta.input_buffers) == 0:
            self.meta.input_buffers = self.model.make_buffers_tokens(self.meta, **kwargs)

        # fill buffers
        padded_kwargs = self.model.fill_buffers_tokens(self.meta, **kwargs)

        context = self.ctx_mgr.current_context()
        self.model.mark_dynamic_inputs(self.meta, **kwargs)
        self.model.mark_dynamic_context(self.meta, context)

        output = self.graph(**padded_kwargs)
        output = self.model.make_output_buffers(output)
        output = self.model.get_outputs_cudagraph(output, **kwargs)

        return output


class TorchCompilePrefillRunner:

    def __init__(self, base_runner: 'TorchCompileRunner', model: torch.nn.Module, model_config: ModelConfig,
                 cache_config: CacheConfig, backend_config: BackendConfig, graph_pool_handle: Any) -> None:
        self.base_runner = base_runner
        self.model = model
        self.model_config = model_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.graph_pool_handle = graph_pool_handle

        self._runner_map: Dict[Any, TorchCompileSinglePrefillRunner] = dict()
        self._compile_backend = create_backend(self.graph_pool_handle, is_decoding=False)

        self.graph = torch.compile(
            self.model.forward,
            fullgraph=PREFILL_FULLGRAPH,
            dynamic=False,
            backend=self._compile_backend,
        )

    def _get_capture_tokens(self, num_tokens: int):
        """Get capture tokens."""
        cap_sizes = self.base_runner.get_capture_prefill_num_tokens()
        for size in cap_sizes:
            if size >= num_tokens:
                return size
        assert False, f'Unsupported batch_size={num_tokens}'

    def get_graph_key(self, input_ids: torch.Tensor, **kwargs):
        """Get graph key."""
        num_tokens = input_ids.size(-1)
        enable_microbatch = get_step_ctx_manager().current_context().enable_microbatch
        # TODO: for draft model to distinguish inputs from target model and itself
        num_tokens = self._get_capture_tokens(num_tokens)
        return (num_tokens, enable_microbatch)

    @record_function('forward_prefill')
    def forward(self, **kwargs):
        """call."""
        if 'past_key_values' in kwargs:
            mark_static_kv_cache(kwargs['past_key_values'])
        graph_key = self.get_graph_key(**kwargs)
        max_tokens = graph_key[0]
        max_batches = self.cache_config.max_batches
        # create runner if not exist
        if graph_key not in self._runner_map:
            runner = TorchCompileSinglePrefillRunner(
                self.model,
                self.graph,
                key=graph_key,
                max_batches=max_batches,
                max_tokens=max_tokens,
                num_blocks=self.cache_config.num_gpu_blocks,
                block_size=self.cache_config.block_size,
                model_config=self.model_config,
                device=self.base_runner.device,
                decode_query_len=1,
            )
            self._runner_map[graph_key] = runner
        runner = self._runner_map[graph_key]
        self._compile_backend.set_key(graph_key)
        return runner.forward(**kwargs)

    def reset(self):
        """Reset."""
        self._runner_map.clear()
        self._compile_backend.reset()


class TorchCompileSingleDecodingRunner:

    def __init__(
        self,
        model: torch.nn.Module,
        graph: Callable,
        key: Any,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        block_size: int,
        model_config: ModelConfig,
        device: torch.device,
        decode_query_len: int = 1,
    ):
        self.model = model
        self.graph = graph
        self.ctx_mgr = model.ctx_mgr
        self.model_config = model_config
        self.key = key

        self.meta = CudaGraphMeta(
            max_batchs=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            block_size=block_size,
            is_decoding=True,
            device=device,
            input_buffers=dict(),
            output_buffers=dict(),
            vocab_size=self.model_config.vocab_size,
            use_mla_fp8_cache=getattr(self.model_config, 'use_mla_fp8_cache', False),
            use_flash_mla=getattr(self.model_config, 'use_flash_mla', False),
            mla_index_topk=getattr(self.model_config, 'mla_index_topk', None),
            decode_query_len=decode_query_len,
            use_fa3_decoding=model_config.model_paradigm == 'ar_spec',
        )
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = True

    def forward(self, **kwargs):
        """call."""
        # make buffer is not exist
        # same inputs as cudagraph
        if len(self.meta.input_buffers) == 0:
            self.meta.input_buffers = self.model.make_buffers_cudagraph(self.meta, **kwargs)

        # fill buffers
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)

        output = self.graph(**padded_kwargs)
        output = self.model.make_output_buffers(output)
        output = self.model.get_outputs_cudagraph(output, **kwargs)

        return output


class TorchCompileDecodingRunner:

    def __init__(self, base_runner: 'TorchCompileRunner', model: torch.nn.Module, model_config: ModelConfig,
                 cache_config: CacheConfig, backend_config: BackendConfig, graph_pool_handle: Any) -> None:
        self.base_runner = base_runner
        self.model = model
        self.model_config = model_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.graph_pool_handle = graph_pool_handle

        self._runner_map: Dict[Any, TorchCompileSingleDecodingRunner] = dict()

        self._compile_backend = create_backend(self.graph_pool_handle, is_decoding=True)
        self.graph = torch.compile(
            self.model.forward,
            fullgraph=DECODING_FULLGRAPH,
            dynamic=False,
            backend=self._compile_backend,
        )

    def _get_capture_tokens(self, batch_size: int):
        """Get capture tokens."""
        cap_sizes = self.base_runner.get_capture_batch_sizes()
        for size in cap_sizes:
            if size >= batch_size:
                return size
        assert False, f'Unsupported batch_size={batch_size}'

    def get_graph_key(self, input_ids: torch.Tensor, **kwargs):
        """Get graph key."""
        context = get_step_ctx_manager().current_context()
        attn_metadata = get_attn_metadata(kwargs, context)
        batch_size = attn_metadata.q_seqlens.size(0)
        meta = self.base_runner.get_meta()
        enable_microbatch = context.enable_microbatch
        # for draft model to distinguish inputs from target model and itself
        query_len = input_ids.size(1) // batch_size
        if meta.padding_batch_size is None:
            batch_size = self._get_capture_tokens(batch_size)
        else:
            batch_size = self._get_capture_tokens(meta.padding_batch_size)
        return (batch_size, enable_microbatch, query_len)

    def _get_max_tokens(self, graph_key: tuple, input_ids: torch.Tensor, q_seqlens: torch.Tensor):
        max_batches = graph_key[0]
        origin_batch_size = q_seqlens.size(0)
        num_tokens = input_ids.size(1)
        return self.base_runner.cudagraph_strategy.get_max_tokens(max_batches, origin_batch_size, num_tokens)

    @record_function('forward_decoding')
    def forward(self, **kwargs):
        """call."""
        if 'past_key_values' in kwargs:
            mark_static_kv_cache(kwargs['past_key_values'])
        graph_key = self.get_graph_key(**kwargs)
        max_batches = graph_key[0]
        decode_query_len = graph_key[2]
        # create runner if not exist
        if graph_key not in self._runner_map:
            attn_metadata = get_attn_metadata(kwargs, get_step_ctx_manager().current_context())
            max_tokens = self._get_max_tokens(graph_key, kwargs['input_ids'], attn_metadata.q_seqlens)
            runner = TorchCompileSingleDecodingRunner(
                self.model,
                self.graph,
                key=graph_key,
                max_batches=max_batches,
                max_tokens=max_tokens,
                num_blocks=self.cache_config.num_gpu_blocks,
                block_size=self.cache_config.block_size,
                model_config=self.model_config,
                device=self.base_runner.device,
                decode_query_len=decode_query_len,
            )
            self._runner_map[graph_key] = runner
        runner = self._runner_map[graph_key]
        self._compile_backend.set_key(graph_key)
        return runner.forward(**kwargs)

    def reset(self):
        """Reset."""
        self._runner_map.clear()
        self._compile_backend.reset()


class TorchCompileRunner(GraphRunner):
    """Cuda graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                 backend_config: BackendConfig, device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config, device)
        self.max_batches = cache_config.max_batches
        self.max_tokens = cache_config.max_prefill_token_num
        self.num_blocks = cache_config.num_gpu_blocks

        self.enable_graph = not self.backend_config.eager_mode

        self.graph_pool_handle = torch.cuda.graph_pool_handle()

        # strategy factory
        build_ctx = model.ctx_mgr.build_ctx
        strategy_factory: StrategyFactoryBase = build_ctx.strategy_factory
        self.cudagraph_strategy = strategy_factory.build_cudagraph_strategy()

        # compile
        torch._inductor.config.triton.cudagraph_support_input_mutation = True
        decoding_size_limit = len(self.get_capture_batch_sizes())
        prefill_size_limit = len(self.get_capture_prefill_num_tokens())
        torch._dynamo.config.cache_size_limit = (decoding_size_limit + prefill_size_limit) * 2
        torch._dynamo.config.accumulated_cache_size_limit = (decoding_size_limit + prefill_size_limit) * 2
        self.prefill_runner = TorchCompilePrefillRunner(
            self,
            self.model,
            model_config=self.model_config,
            cache_config=self.cache_config,
            backend_config=self.backend_config,
            graph_pool_handle=self.graph_pool_handle,
        )
        self.decoding_runner = TorchCompileDecodingRunner(
            self,
            self.model,
            model_config=self.model_config,
            cache_config=self.cache_config,
            backend_config=self.backend_config,
            graph_pool_handle=self.graph_pool_handle,
        )

    def _prepare_inputs(self, **kwargs):
        """Prepare inputs."""
        step_ctx = get_step_ctx_manager().current_context()
        attn_metadata: TritonAttentionMetadata = step_ctx.attn_metadata
        if not attn_metadata.block_offsets.dtype == torch.int32:
            attn_metadata.block_offsets = attn_metadata.block_offsets.to(torch.int32)
        return kwargs

    def __call__(self, **kwargs):
        """call."""

        kwargs = self._prepare_inputs(**kwargs)
        input_ids = kwargs['input_ids']
        long_inputs = input_ids.numel() > self.cache_config.max_prefill_token_num
        enable_graph = self.enable_graph and not long_inputs

        if not enable_graph:
            with record_function('foward'):
                output = self.model(**kwargs)
                return self.model.make_output_buffers(output)

        context = self.ctx_mgr.current_context()
        is_decoding = context.is_decoding

        if is_decoding:
            return self.decoding_runner.forward(**kwargs)
        else:
            return self.prefill_runner.forward(**kwargs)

    @record_function('prepare_inputs_for_generation')
    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare inputs."""

        if get_moe_backend().use_deepep_moe_backend():
            from dlblas.layers.moe.token_dispatcher import DeepEPBuffer, DeepEPMode
            deepep_mode = DeepEPMode.LOW_LATENCY if context.is_decoding else DeepEPMode.NORMAL
            DeepEPBuffer.set_deepep_mode(deepep_mode)

        return self.model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )

    def reset(self):
        """Remove all graphs to prevent hanging on exit."""
        self.prefill_runner.reset()
        self.decoding_runner.reset()

    def update_inputs(self, inputs):
        """Update inputs."""
        if self.backend_config.eager_mode:
            return inputs
        is_decoding = inputs.is_decoding
        dp_meta = inputs.dp_meta
        if dp_meta is not None:
            if is_decoding:
                # pad inputs to same tokens
                meta = self.get_meta()
                padding_batch_size = meta.padding_batch_size
                tp_size = self.decoding_runner._get_capture_tokens(padding_batch_size)
                dp_meta.sync_tp_size(tp_size)
            else:
                # pad inputs to next capture size
                tp_sizes = dp_meta.tp_sizes
                moe_tp_sizes = dp_meta.moe_tp_sizes
                tp_sizes = [self.prefill_runner._get_capture_tokens(size) for size in tp_sizes]
                moe_tp_sizes = [self.prefill_runner._get_capture_tokens(size) for size in moe_tp_sizes]
                dp_meta.set_tp_sizes(tp_sizes, moe_tp_sizes)
        return inputs

    def get_capture_batch_sizes(self) -> List[int]:
        """Capture batch sizes."""
        output = _get_capture_batch_size_impl(self.cache_config.max_batches)

        # torch compile would specialize batchsize 1
        # since there are no big difference between 1 and 2
        # we would just skip it
        if 1 in output:
            output.remove(1)
        return output
