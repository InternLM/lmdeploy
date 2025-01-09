# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext
from lmdeploy.pytorch.models.utils.cudagraph import (CudaGraphMeta,
                                                     next_power_of_2)
from lmdeploy.utils import get_logger

from ...graph_runner import GraphRunner

logger = get_logger('lmdeploy')

BuffType = Dict[str, Tensor]


def _false(*args, **kwargs):
    """default value of not support cuda graph."""
    return False


class CAMBSingleGraphRunner:
    """camb single graph runner."""

    def __init__(
        self,
        model: torch.nn.Module,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        pool: Tuple[int, int],
        device: torch.device,
    ):
        self.model = model
        self.ctx_mgr = model.ctx_mgr
        self.meta = CudaGraphMeta(
            max_batchs=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=True,
            device=device,
            input_buffers=dict(),
            output_buffers=dict(),
        )
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.pool = pool
        self._graph: torch.mlu.CUDAGraph = None

    def capture(self, **kwargs):
        """capture graph."""
        self.meta.input_buffers = self.model.make_buffers_cudagraph(
            self.meta, **kwargs)
        padded_kwargs = self.model.fill_buffers_cudagraph(self.meta, **kwargs)

        context = self.ctx_mgr.current_context()
        self.model.update_context_cudagraph(self.meta, context)
        current_stream = torch.mlu.current_stream()

        # warmup
        self.model(**padded_kwargs)

        self._graph = torch.mlu.CUDAGraph()
        # unsafe kernel call in other thread might invalid the capture
        # so we set thread_safe capture mode here.
        with torch.mlu.graph(self._graph,
                             pool=self.pool,
                             stream=current_stream,
                             capture_error_mode='thread_local'):
            output = self.model(**padded_kwargs)

        output_buffers = dict(logits=output)
        self.meta.output_buffers = output_buffers
        return output

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


class CAMBGraphRunner(GraphRunner):
    """CAMB graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig,
                 cache_config: CacheConfig, backend_config: BackendConfig,
                 device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config,
                         device)
        self.max_batches = cache_config.max_batches
        self.max_tokens = cache_config.max_prefill_token_num
        self.num_blocks = cache_config.num_gpu_blocks

        self.enable_graph = self.check_enable_graph()

        self.graph_pool_handle = torch.mlu.graph_pool_handle()
        self._runner_map: Dict[Any, CAMBSingleGraphRunner] = dict()

    def check_enable_graph(self):
        """check enable graph."""
        if self.backend_config.eager_mode:
            return _false

        return getattr(self.model, 'support_cuda_graph', _false)

    def get_graph_key(self, input_ids: torch.Tensor,
                      position_ids: torch.Tensor, past_key_values: List,
                      attn_metadata: Any, inputs_embeds: torch.Tensor,
                      **kwargs):
        """get graph key."""
        context = self.ctx_mgr.current_context()
        is_decoding = context.is_decoding
        num_tokens = input_ids.numel()
        new_num_tokens = next_power_of_2(num_tokens)
        return (new_num_tokens, is_decoding)

    def __call__(self, **kwargs):
        """call."""
        enable_graph = self.enable_graph(**kwargs)
        graph_key = self.get_graph_key(**kwargs)
        max_tokens = graph_key[0]
        is_decoding = graph_key[1]

        # only enable graph when decoding
        if (not enable_graph) or (not is_decoding):
            return self.model(**kwargs)

        if graph_key not in self._runner_map:
            max_batches = max_tokens
            runner = CAMBSingleGraphRunner(self.model,
                                           max_batches=max_batches,
                                           max_tokens=max_tokens,
                                           num_blocks=self.num_blocks,
                                           pool=self.graph_pool_handle,
                                           device=self.device)
            runner.capture(**kwargs)
            self._runner_map[graph_key] = runner
        else:
            runner = self._runner_map[graph_key]

        output = runner.forward(**kwargs)
        return output

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """prepare inputs."""
        return self.model.prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )
