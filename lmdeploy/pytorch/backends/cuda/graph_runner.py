# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Tuple

import torch
from torch.profiler import record_function

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.backends.selector import get_backend
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.utils import get_logger

from ..graph_runner import GraphRunner

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
        self.model(**padded_kwargs)

        self._graph = torch.cuda.CUDAGraph()
        # unsafe kernel call in other thread might invalid the capture
        # so we set thread_safe capture mode here.
        with torch.cuda.graph(self._graph, pool=self.pool, stream=current_stream, capture_error_mode='thread_local'):
            output = self.model(**padded_kwargs)

        output_buffers = dict(logits=output)
        self.meta.output_buffers = output_buffers
        return output

    @record_function('forward_cudagraph')
    def forward(self, **kwargs):
        """forward."""
        num_tokens = kwargs['input_ids'].size(-1)
        assert self._graph is not None
        self.model.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        self._graph.replay()

        output = self.meta.output_buffers['logits'][:, :num_tokens]
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

    def check_enable_graph(self):
        """Check enable graph."""
        if self.backend_config.eager_mode:
            return _false

        return getattr(self.model, 'support_cuda_graph', _false)

    def _try_compile_model_once(self):
        if self.has_try_compile_model:
            return

        if hasattr(self.model, 'compile_model') and envs.enable_cuda_torch_compile:
            method = getattr(self.model, 'compile_model')
            method()

        self.has_try_compile_model = True

    def get_graph_key(self, input_ids: torch.Tensor, position_ids: torch.Tensor, past_key_values: List,
                      attn_metadata: Any, inputs_embeds: torch.Tensor, **kwargs):
        """Get graph key."""
        context = self.ctx_mgr.current_context()
        is_decoding = context.is_decoding
        num_tokens = input_ids.numel()
        meta = self.get_meta()
        if meta.padding_batch_size is None:
            new_num_tokens = next_power_of_2(num_tokens)
        else:
            new_num_tokens = next_power_of_2(meta.padding_batch_size)
        return (new_num_tokens, is_decoding)

    def __call__(self, **kwargs):
        """call."""
        if not self.backend_config.eager_mode and get_backend().get_name() == 'cuda':
            self._try_compile_model_once()

        enable_graph = self.enable_graph(**kwargs)

        if not enable_graph:
            with record_function('forward_eager'):
                return self.model(**kwargs)

        graph_key = self.get_graph_key(**kwargs)
        max_tokens = graph_key[0]
        is_decoding = graph_key[1]
        if graph_key not in self._runner_map:
            max_batches = max_tokens if is_decoding else self.max_batches
            runner = CUDASingleGraphRunner(self.model,
                                           max_batches=max_batches,
                                           max_tokens=max_tokens,
                                           num_blocks=self.num_blocks,
                                           is_decoding=is_decoding,
                                           pool=self.graph_pool_handle,
                                           model_config=self.model_config,
                                           device=self.device)
            runner.capture(**kwargs)
            self._runner_map[graph_key] = runner
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
            tp_size = next_power_of_2(padding_batch_size)
            dp_meta.tp_sizes = [tp_size] * len(dp_meta.tp_sizes)
        return inputs
