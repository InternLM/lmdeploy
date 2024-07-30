# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Tuple

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import StepContext
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


class CUDASingleGraphRunner:
    """cuda single graph runner."""

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
        self.device = device
        self.max_batches = max_batches
        self.max_tokens = max_tokens
        self.num_blocks = num_blocks
        self.is_decoding = is_decoding
        self.pool = pool
        self._graph: torch.cuda.CUDAGraph = None

        self.input_buffers = dict()
        self.output_buffers = dict()
        self.make_buffers()

    def make_buffers(self):
        """make cache step context."""
        max_batches = self.max_batches
        max_tokens = self.max_tokens
        num_blocks = self.num_blocks
        device = self.device
        is_decoding = self.is_decoding

        if is_decoding:
            self.input_buffers['input_ids'] = torch.zeros(1,
                                                          max_batches,
                                                          dtype=torch.int64,
                                                          device=device)
            self.input_buffers['position_ids'] = torch.zeros((1, max_batches),
                                                             dtype=torch.int64,
                                                             device=device)
        else:
            self.input_buffers['input_ids'] = torch.zeros(1,
                                                          max_tokens,
                                                          dtype=torch.int64,
                                                          device=device)
            self.input_buffers['position_ids'] = torch.zeros((1, max_tokens),
                                                             dtype=torch.int64,
                                                             device=device)

        self.input_buffers['block_offsets'] = torch.zeros(
            (max_batches, num_blocks), dtype=torch.int64, device=device)
        self.input_buffers['q_start_loc'] = torch.zeros(max_batches,
                                                        dtype=torch.int64,
                                                        device=device)
        self.input_buffers['q_seqlens'] = torch.zeros(max_batches,
                                                      dtype=torch.int64,
                                                      device=device)
        self.input_buffers['kv_seqlens'] = torch.zeros(max_batches,
                                                       dtype=torch.int64,
                                                       device=device)
        self.input_buffers['local_adapter_ids'] = torch.zeros(
            max_batches, dtype=torch.int64, device=device) - 1

    def _fill_inputs(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                     past_key_values: List, attn_metadata: Any,
                     inputs_embeds: torch.Tensor, **kwargs):
        """fill input."""
        is_decoding = self.is_decoding
        block_offsets = attn_metadata.block_offsets
        q_start_loc = attn_metadata.q_start_loc
        q_seqlens = attn_metadata.q_seqlens
        kv_seqlens = attn_metadata.kv_seqlens

        batch_size, num_blocks = block_offsets.size()
        num_tokens = input_ids.size(-1)

        # fill buffer
        self.input_buffers['input_ids'][:, :num_tokens] = input_ids
        self.input_buffers['position_ids'][:, :num_tokens] = position_ids
        self.input_buffers[
            'block_offsets'][:batch_size, :num_blocks] = block_offsets
        self.input_buffers['q_seqlens'][:batch_size] = q_seqlens
        self.input_buffers['kv_seqlens'][:batch_size] = kv_seqlens
        self.input_buffers['q_start_loc'][:batch_size] = q_start_loc
        if inputs_embeds is not None:
            emb_size = inputs_embeds.size(-1)
            if 'inputs_embeds' not in self.input_buffers:
                max_num_tokens = self.input_buffers['input_ids'].size(-1)
                self.input_buffers['inputs_embeds'] = inputs_embeds.new_zeros(
                    1, max_num_tokens, emb_size)
            self.input_buffers['inputs_embeds'][:, :num_tokens] = inputs_embeds

        # create inputs
        new_batch_size = next_power_of_2(batch_size)
        attn_metadata.block_offsets = self.input_buffers[
            'block_offsets'][:new_batch_size]
        attn_metadata.q_start_loc = self.input_buffers[
            'q_start_loc'][:new_batch_size]
        attn_metadata.q_seqlens = self.input_buffers[
            'q_seqlens'][:new_batch_size]
        attn_metadata.kv_seqlens = self.input_buffers[
            'kv_seqlens'][:new_batch_size]

        new_inputs = dict(
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )

        if is_decoding:
            new_inputs['input_ids'] = self.input_buffers[
                'input_ids'][:, :new_batch_size]
            new_inputs['position_ids'] = self.input_buffers[
                'position_ids'][:, :new_batch_size]
        else:
            new_inputs['input_ids'] = self.input_buffers['input_ids']
            new_inputs['position_ids'] = self.input_buffers['position_ids']

        if inputs_embeds is not None:
            if is_decoding:
                new_inputs['inputs_embeds'] = self.input_buffers[
                    'inputs_embeds'][:, :new_batch_size]
            else:
                new_inputs['inputs_embeds'] = self.input_buffers[
                    'inputs_embeds']

        new_inputs.update(kwargs)
        return new_inputs

    def capture(self, **kwargs):
        """capture graph."""
        padded_kwargs = self._fill_inputs(**kwargs)
        current_stream = torch.cuda.current_stream()

        # warmup
        output = self.model(**padded_kwargs)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph,
                              pool=self.pool,
                              stream=current_stream):
            output = self.model(**padded_kwargs)

        self.output_buffers['logits'] = output

    def forward(self, **kwargs):
        """forward."""
        num_tokens = kwargs['input_ids'].size(-1)
        assert self._graph is not None
        self._fill_inputs(**kwargs)
        self._graph.replay()

        output = self.output_buffers['logits'][:, :num_tokens].clone()
        return output

    def __del__(self):
        """del."""
        del self._graph


class CUDAGraphRunner(GraphRunner):
    """cuda graph runner."""

    def __init__(self, model: torch.nn.Module, model_config: ModelConfig,
                 cache_config: CacheConfig, backend_config: BackendConfig,
                 device: torch.device):
        self.model = model
        self.ctx_mgr = model.ctx_mgr
        self.device = device
        self.model_config = model_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.max_batches = cache_config.max_batches
        self.max_tokens = cache_config.max_prefill_token_num
        self.num_blocks = cache_config.num_gpu_blocks

        self.enable_graph = self.check_enable_graph()

        self.graph_pool_handle = torch.cuda.graph_pool_handle()
        self._runner_map: Dict[Any, CUDASingleGraphRunner] = dict()

    def check_enable_graph(self):
        """check enable graph."""
        if self.backend_config.eager_mode:
            return False

        if getattr(self.model, 'support_cuda_graph', False):
            return False

        # TODO: should we enable cudagraph awq now?
        # hf_config = self.model_config.hf_config
        # quantization_config = getattr(hf_config,
        #    'quantization_config', dict())
        # quant_method = quantization_config.get('quant_method', None)
        # if quant_method == 'awq':
        #     logger.warning('AWQ model does not support cuda graph yet.')
        #     return False
        return True

    def get_graph_key(self, input_ids: torch.Tensor,
                      position_ids: torch.Tensor, past_key_values: List,
                      attn_metadata: Any, inputs_embeds: torch.Tensor,
                      **kwargs):
        """get graph key."""
        context = self.ctx_mgr.current_context()
        is_decoding = context.is_decoding
        block_offsets = attn_metadata.block_offsets
        batch_size, _ = block_offsets.size()
        new_batch_size = next_power_of_2(batch_size)
        return (new_batch_size, is_decoding)

    def __call__(self, **kwargs):
        """call."""
        if not self.enable_graph:
            return self.model(**kwargs)

        graph_key = self.get_graph_key(**kwargs)
        max_batches = graph_key[0]
        is_decoding = graph_key[1]
        if graph_key not in self._runner_map:
            runner = CUDASingleGraphRunner(self.model,
                                           max_batches=max_batches,
                                           max_tokens=self.max_tokens,
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
