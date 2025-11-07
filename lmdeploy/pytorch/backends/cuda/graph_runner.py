# Copyright (c) OpenMMLab. All rights reserved.
import functools
from typing import Any, Dict, List, Tuple

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.backends.selector import get_backend
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.pytorch.strategies.base import StrategyFactoryBase
from lmdeploy.utils import get_logger

from ..graph_runner import GraphRunner
from .attention import TritonAttentionMetadata

logger = get_logger('lmdeploy')


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


class CUDASingleGraphRunner:
    """Cuda single graph runner."""

    def __init__(
        self,
        model: torch.nn.Module,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        is_decoding: bool,
        pool: Tuple[int, int],
        model_config: ModelConfig,
        device: torch.device,
    ):
        self.model = model
        self.ctx_mgr = model.ctx_mgr
        self.model_config = model_config

        self.meta = CudaGraphMeta(
            max_batchs=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=is_decoding,
            device=device,
            input_buffers=dict(),
            output_buffers=dict(),
            vocab_size=self.model_config.vocab_size,
        )
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = is_decoding
        self.pool = pool
        self._graph: torch.cuda.CUDAGraph = None

    def make_output_buffers(self, output):
        """Make output buffers."""
        output_buffers = dict(logits=output)
        return output_buffers

    def slice_output(self, output_buffers: Dict[str, Any], inputs: Dict[str, Any]):
        """Slice output."""
        num_tokens = inputs['input_ids'].size(-1)
        return output_buffers['logits'][:, :num_tokens]

    @record_function('capture_cudagraph')
    def capture(self, **kwargs):
        """Capture graph."""
        logger.debug(f'Capturing graph with meta: {self.meta}')
        self.meta.input_buffers = self.model.make_buffers_cudagraph(self.meta, **kwargs)
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        current_stream = torch.cuda.current_stream()

        # warmup
        warmup_output = self.model(**padded_kwargs)
        warmup_buffers = self.make_output_buffers(warmup_output)

        self._graph = torch.cuda.CUDAGraph()
        # unsafe kernel call in other thread might invalid the capture
        # so we set thread_safe capture mode here.
        with torch.cuda.graph(self._graph, pool=self.pool, stream=current_stream, capture_error_mode='thread_local'):
            output = self.model(**padded_kwargs)

        output_buffers = self.make_output_buffers(output)
        self.meta.output_buffers = output_buffers
        output = self.slice_output(warmup_buffers, kwargs)
        return output

    @record_function('forward_cudagraph')
    def forward(self, **kwargs):
        """forward."""
        assert self._graph is not None
        self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        self._graph.replay()

        output_buffers = self.meta.output_buffers
        output = self.slice_output(output_buffers, kwargs)
        return output

    def __del__(self):
        """del."""
        del self._graph


class CUDAGraphRunner(GraphRunner):
    """Cuda graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                 backend_config: BackendConfig, device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config, device)
        self.max_batches = cache_config.max_batches
        self.max_tokens = cache_config.max_prefill_token_num
        self.num_blocks = cache_config.num_gpu_blocks

        self.enable_graph = self.check_enable_graph()

        self.graph_pool_handle = torch.cuda.graph_pool_handle()
        self._runner_map: Dict[Any, CUDASingleGraphRunner] = dict()
        self.has_try_compile_model: bool = False

        # strategy factory
        build_ctx = model.ctx_mgr.build_ctx
        strategy_factory: StrategyFactoryBase = build_ctx.strategy_factory
        self.cudagraph_strategy = strategy_factory.build_cudagraph_strategy()

    def check_enable_graph(self):
        """Check enable graph."""
        if self.backend_config.eager_mode:
            return _false

        return getattr(self.model, 'support_cuda_graph', _false)

    def _try_compile_model_once(self):
        if self.has_try_compile_model:
            return

        # TODO: recovery it when torch.compile is stable (should be add a flag to enable it?)
        # if hasattr(self.model, 'compile_model'):
        #     method = getattr(self.model, 'compile_model')
        #     method()

        self.has_try_compile_model = True

    def _get_capture_tokens(self, batch_size: int):
        """Get capture tokens."""
        cap_sizes = self.get_capture_batch_sizes()
        for size in cap_sizes:
            if size >= batch_size:
                return size
        assert False, f'Unsupported batch_size={batch_size}'

    def get_graph_key(self, input_ids: torch.Tensor, position_ids: torch.Tensor, past_key_values: List,
                      attn_metadata: TritonAttentionMetadata, inputs_embeds: torch.Tensor, **kwargs):
        """Get graph key."""
        context = self.ctx_mgr.current_context()
        is_decoding = context.is_decoding
        batch_size = attn_metadata.q_seqlens.size(0)
        meta = self.get_meta()
        enable_microbatch = get_step_ctx_manager().current_context().enable_microbatch
        if meta.padding_batch_size is None:
            batch_size = self._get_capture_tokens(batch_size)
        else:
            batch_size = self._get_capture_tokens(meta.padding_batch_size)
        return (batch_size, is_decoding, enable_microbatch)

    def _get_max_tokens(self, graph_key: tuple):
        max_batches = graph_key[0]
        is_decoding = graph_key[1]
        assert is_decoding
        return self.cudagraph_strategy.get_max_tokens(max_batches)

    def __call__(self, **kwargs):
        """call."""
        if not self.backend_config.eager_mode and get_backend().get_name() == 'cuda':
            self._try_compile_model_once()

        enable_graph = self.enable_graph(**kwargs)

        if not enable_graph:
            with record_function('forward_eager'):
                return self.model(**kwargs)

        graph_key = self.get_graph_key(**kwargs)
        max_batches = graph_key[0]
        is_decoding = graph_key[1]
        if graph_key not in self._runner_map:
            max_tokens = self._get_max_tokens(graph_key)
            runner = CUDASingleGraphRunner(self.model,
                                           max_batches=max_batches,
                                           max_tokens=max_tokens,
                                           num_blocks=self.num_blocks,
                                           is_decoding=is_decoding,
                                           pool=self.graph_pool_handle,
                                           model_config=self.model_config,
                                           device=self.device)
            output = runner.capture(**kwargs)
            self._runner_map[graph_key] = runner
            # SSM would update the state in capture(warmup), replay the graph will leads unexpected state update.
            return output
        else:
            runner = self._runner_map[graph_key]
            output = runner.forward(**kwargs)
            return output

    @record_function('prepare_inputs_for_generation')
    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
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
        self._runner_map.clear()

    def update_inputs(self, inputs):
        """Update inputs."""
        if self.backend_config.eager_mode:
            return inputs
        is_decoding = inputs.is_decoding
        dp_meta = inputs.dp_meta
        if is_decoding and dp_meta is not None:
            meta = self.get_meta()
            padding_batch_size = meta.padding_batch_size
            tp_size = self._get_capture_tokens(padding_batch_size)
            dp_meta.tp_sizes = [tp_size] * len(dp_meta.tp_sizes)
        return inputs

    def get_capture_batch_sizes(self) -> List[int]:
        """Capture batch sizes."""
        return _get_capture_batch_size_impl(self.cache_config.max_batches)
