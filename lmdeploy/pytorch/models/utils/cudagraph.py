# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch import Tensor

from lmdeploy.pytorch.model_inputs import StepContext

BuffType = Dict[str, Tensor]


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


@dataclass
class CudaGraphMeta:
    """Meta info of cudagraph."""
    max_batchs: int
    max_tokens: int
    num_blocks: int
    is_decoding: int
    device: torch.device
    input_buffers: BuffType = None
    output_buffers: BuffType = None


class CudaGraphMixin:
    """mixin class to support cudagraph."""

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """return True is model support cudagraph."""
        seq_lens = input_ids.size(1)
        return seq_lens <= 256

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
                                                 dtype=torch.int64,
                                                 device=device)
        input_buffers['position_ids'] = torch.zeros((1, max_tokens),
                                                    dtype=torch.int64,
                                                    device=device)

        input_buffers['block_offsets'] = torch.zeros((max_batches, num_blocks),
                                                     dtype=torch.int64,
                                                     device=device)
        input_buffers['q_start_loc'] = torch.zeros(max_batches,
                                                   dtype=torch.int64,
                                                   device=device)
        input_buffers['q_seqlens'] = torch.zeros(max_batches,
                                                 dtype=torch.int64,
                                                 device=device)
        input_buffers['kv_seqlens'] = torch.zeros(max_batches,
                                                  dtype=torch.int64,
                                                  device=device)
        input_buffers['local_adapter_ids'] = torch.zeros(max_batches,
                                                         dtype=torch.int64,
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
        input_buffers: BuffType = graph_meta.input_buffers

        batch_size, num_blocks = block_offsets.size()
        num_tokens = input_ids.size(-1)

        # fill buffer
        input_buffers['input_ids'][:, :num_tokens] = input_ids
        input_buffers['position_ids'][:, :num_tokens] = position_ids
        input_buffers[
            'block_offsets'][:batch_size, :num_blocks] = block_offsets
        if q_seqlens.data_ptr() != input_buffers['q_seqlens'].data_ptr():
            input_buffers['q_seqlens'].zero_()
        input_buffers['q_seqlens'][:batch_size] = q_seqlens
        if kv_seqlens.data_ptr() != input_buffers['kv_seqlens'].data_ptr():
            input_buffers['kv_seqlens'].zero_()
        input_buffers['kv_seqlens'][:batch_size] = kv_seqlens
        input_buffers['q_start_loc'][:batch_size] = q_start_loc
        if inputs_embeds is not None:
            emb_size = inputs_embeds.size(-1)
            if 'inputs_embeds' not in input_buffers:
                max_num_tokens = input_buffers['input_ids'].size(-1)
                input_buffers['inputs_embeds'] = inputs_embeds.new_zeros(
                    1, max_num_tokens, emb_size)
            input_buffers['inputs_embeds'][:, :num_tokens] = inputs_embeds

        # create inputs
        new_batch_size = next_power_of_2(batch_size)
        attn_metadata.block_offsets = input_buffers[
            'block_offsets'][:new_batch_size]
        attn_metadata.q_start_loc = input_buffers[
            'q_start_loc'][:new_batch_size]
        attn_metadata.q_seqlens = input_buffers['q_seqlens'][:new_batch_size]
        attn_metadata.kv_seqlens = input_buffers['kv_seqlens'][:new_batch_size]

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

    def update_context_cudagraph(self, graph_meta: CudaGraphMeta,
                                 context: StepContext):
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
