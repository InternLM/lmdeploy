# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta
from lmdeploy.utils import get_logger

from ...graph_runner import GraphRunner

logger = get_logger('lmdeploy')

BuffType = Dict[str, Tensor]


def round_up_to_multiple_of_8(n: int):
    return (n + 7) // 8 * 8


def _false(*args, **kwargs):
    """default value of not support cuda graph."""
    return False


class MACASingleGraphRunner:
    """MACA single graph runner."""

    def __init__(
        self,
        model: torch.nn.Module,
        max_batches: int,
        max_tokens: int,
        num_blocks: int,
        is_decoding: bool,
        pool: Tuple[int, int],
        device: torch.device,
    ):
        self.model = model
        self.ctx_mgr = model.ctx_mgr
        self.meta = CudaGraphMeta(
            max_batchs=max_batches,
            max_tokens=max_tokens,
            num_blocks=num_blocks,
            is_decoding=is_decoding,
            device=device,
            input_buffers=dict(),
            output_buffers=dict(),
        )
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = is_decoding
        self.pool = pool
        self._graph: torch.cuda.CUDAGraph = None

    def capture(self, **kwargs):
        """capture graph."""
        self.meta.input_buffers = self.make_buffers_cudagraph(
            self.meta, **kwargs)
        padded_kwargs = self.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.update_context_cudagraph(self.meta, context)
        current_stream = torch.cuda.current_stream()

        output = self.model(**padded_kwargs)

        # warmup
        self._graph = torch.cuda.CUDAGraph()
        # unsafe kernel call in other thread might invalid the capture
        # so we set thread_safe capture mode here.
        with torch.cuda.graph(self._graph,
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
        self.fill_buffers_cudagraph(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.update_context_cudagraph(self.meta, context)

        self._graph.replay()
        output = self.meta.output_buffers['logits'][:, :num_tokens]
        return output

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, *args,
                               **kwargs) -> BuffType:
        """make cudagraph buffers from forward inputs."""
        max_batches = graph_meta.max_batchs
        max_tokens = graph_meta.max_tokens
        num_blocks = graph_meta.num_blocks
        device = graph_meta.device

        input_buffers: BuffType = dict()
        input_buffers['input_ids'] = torch.zeros(1,
                                                 max_tokens,
                                                 dtype=torch.int32,
                                                 device=device)
        input_buffers['position_ids'] = torch.ones((1, max_tokens),
                                                   dtype=torch.int32,
                                                   device=device)

        input_buffers['block_offsets'] = torch.zeros((max_batches, num_blocks),
                                                     dtype=torch.int32,
                                                     device=device)

        input_buffers['q_start_loc'] = torch.arange(max_batches + 1,
                                                    dtype=torch.int32,
                                                    device=device)

        input_buffers['q_seqlens'] = torch.ones(max_batches,
                                                dtype=torch.int32,
                                                device=device)

        input_buffers['kv_seqlens'] = torch.ones(max_batches,
                                                 dtype=torch.int32,
                                                 device=device)

        input_buffers['kv_start_indices'] = torch.ones((max_batches, 1),
                                                       dtype=torch.int64,
                                                       device=device)

        input_buffers['local_adapter_ids'] = torch.zeros(max_batches,
                                                         dtype=torch.int32,
                                                         device=device)
        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta,
                               input_ids: Tensor, position_ids: Tensor,
                               past_key_values: List, attn_metadata: Any,
                               inputs_embeds: Tensor,
                               **kwargs) -> Dict[str, Tensor]:
        """fill cudagraph buffers from forward inputs."""
        is_decoding = graph_meta.is_decoding
        block_offsets: Tensor = attn_metadata.block_offsets
        q_start_loc: Tensor = attn_metadata.q_start_loc
        q_seqlens: Tensor = attn_metadata.q_seqlens
        kv_seqlens: Tensor = attn_metadata.kv_seqlens
        kv_start_indices: Tensor = attn_metadata.kv_start_indices

        input_buffers: BuffType = graph_meta.input_buffers

        batch_size, num_blocks = block_offsets.size()
        num_tokens = input_ids.size(-1)
        q_start_loc_size = q_start_loc.size(0)

        # fill buffer
        input_buffers['input_ids'][:, :num_tokens] = input_ids
        input_buffers['position_ids'][:, :num_tokens] = position_ids
        input_buffers[
            'block_offsets'][:batch_size, :num_blocks] = block_offsets
        input_buffers['q_seqlens'][:batch_size] = q_seqlens
        input_buffers['kv_seqlens'][:batch_size] = kv_seqlens

        input_buffers['q_start_loc'][:q_start_loc_size] = q_start_loc
        input_buffers['kv_start_indices'][:batch_size] = kv_start_indices

        if inputs_embeds is not None:
            emb_size = inputs_embeds.size(-1)
            if 'inputs_embeds' not in input_buffers:
                max_num_tokens = input_buffers['input_ids'].size(-1)
                input_buffers['inputs_embeds'] = inputs_embeds.new_zeros(
                    1, max_num_tokens, emb_size)
            input_buffers['inputs_embeds'][:, :num_tokens] = inputs_embeds

        # create inputs
        new_batch_size = round_up_to_multiple_of_8(batch_size)
        q_start_loc_size = round_up_to_multiple_of_8(q_start_loc_size)

        attn_metadata.block_offsets = input_buffers[
            'block_offsets'][:new_batch_size]
        attn_metadata.q_start_loc = input_buffers[
            'q_start_loc'][:q_start_loc_size]
        attn_metadata.q_seqlens = input_buffers['q_seqlens'][:new_batch_size]
        attn_metadata.kv_seqlens = input_buffers['kv_seqlens'][:new_batch_size]
        attn_metadata.kv_start_indices = input_buffers[
            'kv_start_indices'][:new_batch_size]

        new_inputs = dict(
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )

        if is_decoding:
            new_inputs['input_ids'] = input_buffers[
                'input_ids'][:, :new_batch_size]
            new_inputs['position_ids'] = input_buffers[
                'position_ids'][:, :new_batch_size]
        else:
            new_inputs['input_ids'] = input_buffers['input_ids']
            new_inputs['position_ids'] = input_buffers['position_ids']

        if inputs_embeds is not None:
            if is_decoding:
                new_inputs['inputs_embeds'] = input_buffers[
                    'inputs_embeds'][:, :new_batch_size]
            else:
                new_inputs['inputs_embeds'] = input_buffers['inputs_embeds']

        new_inputs.update(kwargs)
        return new_inputs

    def update_context_cudagraph(self, graph_meta, context):
        """update step context with input buffers."""
        input_buffers = graph_meta.input_buffers
        context.q_seqlens = input_buffers['q_seqlens']
        context.kv_seqlens = input_buffers['kv_seqlens']
        context.q_start_loc = input_buffers['q_start_loc']
        context.kv_start_indices = input_buffers['kv_start_indices']

    def __del__(self):
        """del."""
        del self._graph


class MACAGraphRunner(GraphRunner):
    """MACA graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig,
                 cache_config: CacheConfig, backend_config: BackendConfig,
                 device: torch.device):
        super().__init__(model, model_config, cache_config, backend_config,
                         device)
        self.max_batches = cache_config.max_batches
        self.max_tokens = cache_config.max_prefill_token_num
        self.num_blocks = cache_config.num_gpu_blocks

        self.enable_graph = self.check_enable_graph()

        self.graph_pool_handle = torch.cuda.graph_pool_handle()
        self._runner_map: Dict[Any, MACASingleGraphRunner] = dict()

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
        new_num_tokens = round_up_to_multiple_of_8(num_tokens)
        return (new_num_tokens, is_decoding)

    def __call__(self, **kwargs):
        """call."""
        enable_graph = self.enable_graph(**kwargs)
        graph_key = self.get_graph_key(**kwargs)
        max_tokens = graph_key[0]
        is_decoding = graph_key[1]

        if (not enable_graph) or (not is_decoding):
            return self.model(**kwargs)

        if graph_key not in self._runner_map:
            max_batches = max_tokens if is_decoding else self.max_batches
            runner = MACASingleGraphRunner(self.model,
                                           max_batches=max_batches,
                                           max_tokens=max_tokens,
                                           num_blocks=self.num_blocks,
                                           is_decoding=is_decoding,
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
