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
    return (n + 7) // 8 * 8 + 8


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
        self.meta.input_buffers = self.make_camb_buffers(self.meta, **kwargs)
        padded_kwargs = self.update_camb_buffer(self.meta, **kwargs)

        context = self.ctx_mgr.current_context()
        self.update_camb_context(self.meta, context)
        current_stream = torch.mlu.current_stream()

        # warmup
        output = self.model(**padded_kwargs)

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

    def make_camb_buffers(self, graph_meta: CudaGraphMeta, *args,
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

        input_buffers['q_start_loc'] = torch.arange(max_batches,
                                                    dtype=torch.int32,
                                                    device=device)

        input_buffers['q_seqlens'] = torch.ones(max_batches,
                                                dtype=torch.int32,
                                                device=device)

        input_buffers['kv_seqlens'] = torch.ones(max_batches,
                                                 dtype=torch.int32,
                                                 device=device)
        # critical to set negative for kv_start_indices
        # if we don't set it, two batches with same input tokens
        # will result in different answer
        input_buffers['kv_start_indices'] = -torch.ones(
            (max_batches * max_tokens), dtype=torch.int32, device=device)

        input_buffers['local_adapter_ids'] = torch.zeros(max_batches,
                                                         dtype=torch.int32,
                                                         device=device)
        # create buffer for mrope for qwen2VL here
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers['mrope_position_ids'] = torch.ones(3,
                                                             max_tokens,
                                                             dtype=torch.int32,
                                                             device=device)
        return input_buffers

    def update_camb_buffer(self, graph_meta: CudaGraphMeta, input_ids: Tensor,
                           position_ids: Tensor, past_key_values: List,
                           attn_metadata: Any, inputs_embeds: Tensor,
                           **kwargs) -> Dict[str, Tensor]:
        """fill cudagraph buffers from forward inputs."""
        block_offsets: Tensor = attn_metadata.block_offsets
        q_start_loc: Tensor = attn_metadata.q_start_loc
        q_seqlens: Tensor = attn_metadata.q_seqlens
        kv_seqlens: Tensor = attn_metadata.kv_seqlens
        kv_start_indices: Tensor = attn_metadata.kv_start_indices

        input_buffers: BuffType = graph_meta.input_buffers

        batch_size, num_blocks = block_offsets.size()
        num_tokens = input_ids.size(-1)
        # fill buffer
        input_buffers['input_ids'][:, :num_tokens] = input_ids
        input_buffers['position_ids'][:, :num_tokens] = position_ids
        input_buffers[
            'block_offsets'][:batch_size, :num_blocks] = block_offsets
        input_buffers['q_seqlens'][:batch_size] = q_seqlens
        input_buffers['kv_seqlens'][:batch_size] = kv_seqlens
        input_buffers['q_start_loc'][:batch_size + 1] = q_start_loc
        input_buffers[
            'kv_start_indices'][:num_tokens] = kv_start_indices[:num_tokens]

        if inputs_embeds is not None:
            emb_size = inputs_embeds.size(-1)
            if 'inputs_embeds' not in input_buffers:
                max_num_tokens = input_buffers['input_ids'].size(-1)
                input_buffers['inputs_embeds'] = inputs_embeds.new_zeros(
                    1, max_num_tokens, emb_size)
            input_buffers['inputs_embeds'][:, :num_tokens] = inputs_embeds

        # create inputs
        new_num_tokens = round_up_to_multiple_of_8(num_tokens)
        new_batch_size = new_num_tokens

        attn_metadata.block_offsets = input_buffers[
            'block_offsets'][:new_batch_size]
        attn_metadata.q_start_loc = input_buffers[
            'q_start_loc'][:new_batch_size]
        attn_metadata.q_seqlens = input_buffers['q_seqlens'][:new_batch_size]
        attn_metadata.kv_seqlens = input_buffers['kv_seqlens'][:new_batch_size]

        attn_metadata.kv_start_indices = input_buffers[
            'kv_start_indices'][:new_num_tokens]
        new_inputs = dict(
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )

        # is_decoding:
        new_inputs['input_ids'] = input_buffers[
            'input_ids'][:, :new_batch_size]
        new_inputs['position_ids'] = input_buffers[
            'position_ids'][:, :new_batch_size]

        if inputs_embeds is not None:
            new_inputs['inputs_embeds'] = input_buffers[
                'inputs_embeds'][:, :new_batch_size]

        new_inputs.update(kwargs)

        # mrope for qwen2VL
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers[
                'mrope_position_ids'][:, :num_tokens] = mrope_position_ids
            new_inputs['mrope_position_ids'] = input_buffers[
                'mrope_position_ids'][:, :new_batch_size]
        return new_inputs

    def update_camb_context(self, graph_meta, context):
        """update step context with input buffers."""
        input_buffers = graph_meta.input_buffers
        local_adapter_ids = context.local_adapter_ids
        if local_adapter_ids is not None:
            if input_buffers['local_adapter_ids'].data_ptr(
            ) != local_adapter_ids.data_ptr():
                input_buffers['local_adapter_ids'].fill_(0)
            batch_size = local_adapter_ids.size(0)
            input_buffers['local_adapter_ids'][:batch_size] = local_adapter_ids
            context.local_adapter_ids = input_buffers['local_adapter_ids']
        context.q_seqlens = input_buffers['q_seqlens']
        context.kv_seqlens = input_buffers['kv_seqlens']
        context.q_start_loc = input_buffers['q_start_loc']

    def forward(self, **kwargs):
        """forward."""
        num_tokens = kwargs['input_ids'].size(-1)
        assert self._graph is not None
        self.update_camb_buffer(self.meta, **kwargs)
        context = self.ctx_mgr.current_context()
        self.update_camb_context(self.meta, context)

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
        new_num_tokens = round_up_to_multiple_of_8(num_tokens)
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
