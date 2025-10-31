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
    vocab_size: int = 1


class CudaGraphMixin:
    """Mixin class to support cudagraph."""

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """Return True is model support cudagraph."""
        return attn_metadata.is_decoding

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, *args, **kwargs) -> BuffType:
        """Make cudagraph buffers from forward inputs."""
        max_batches = graph_meta.max_batchs
        max_tokens = graph_meta.max_tokens
        num_blocks = graph_meta.num_blocks
        device = graph_meta.device

        input_buffers: BuffType = dict()
        input_buffers['input_ids'] = torch.randint(0,
                                                   graph_meta.vocab_size, (1, max_tokens),
                                                   dtype=torch.int64,
                                                   device=device)
        input_buffers['position_ids'] = torch.zeros((1, max_tokens), dtype=torch.int64, device=device)
        if getattr(self.config, 'use_flash_mla', False) is True:
            import flash_mla

            # create buffers for flash mla
            input_buffers['tile_scheduler_metadata'], input_buffers['num_splits'] = flash_mla.get_mla_metadata(
                torch.ones(max_batches, dtype=torch.int32, device=device), self.config.num_attention_heads, 1)

        # flash_mla requires block_offsets and kv_lens int32
        input_buffers['block_offsets'] = torch.zeros((max_batches, num_blocks), dtype=torch.int32, device=device)
        input_buffers['qkv_lens'] = torch.zeros(3, max_batches, dtype=torch.int32, device=device)

        input_buffers['q_start_loc'] = input_buffers['qkv_lens'][0]
        input_buffers['q_seqlens'] = input_buffers['qkv_lens'][1]
        input_buffers['kv_seqlens'] = input_buffers['qkv_lens'][2]
        input_buffers['local_adapter_ids'] = torch.zeros(max_batches, dtype=torch.int64, device=device)
        # create buffer for cross_attn_metadata here
        input_buffers['fill_seqlens'] = torch.zeros(max_batches, dtype=torch.int64, device=device)

        input_buffers['cu_seqlens'] = torch.zeros(2, max_batches + 1, dtype=torch.int32, device=device)
        input_buffers['cu_seqlens_q'] = input_buffers['cu_seqlens'][0]
        input_buffers['cu_seqlens_k'] = input_buffers['cu_seqlens'][1]

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, input_ids: Tensor, position_ids: Tensor,
                               past_key_values: List, attn_metadata: Any, inputs_embeds: Tensor,
                               **kwargs) -> Dict[str, Tensor]:
        """Fill cudagraph buffers from forward inputs."""

        is_decoding = graph_meta.is_decoding
        block_offsets: Tensor = attn_metadata.block_offsets
        q_start_loc: Tensor = attn_metadata.q_start_loc
        q_seqlens: Tensor = attn_metadata.q_seqlens
        kv_seqlens: Tensor = attn_metadata.kv_seqlens
        input_buffers: BuffType = graph_meta.input_buffers

        batch_size, num_blocks = block_offsets.size()
        num_tokens = input_ids.size(-1)

        # fill buffer
        input_buffers['input_ids'].random_(0, graph_meta.vocab_size)
        input_buffers['input_ids'][:, :num_tokens] = input_ids
        input_buffers['position_ids'][:, :num_tokens] = position_ids
        input_buffers['block_offsets'][:batch_size, :num_blocks] = block_offsets

        qkv = torch.stack((q_start_loc, q_seqlens, kv_seqlens))
        input_buffers['qkv_lens'].zero_()
        input_buffers['q_seqlens'].fill_(graph_meta.max_tokens // graph_meta.max_batchs)
        input_buffers['qkv_lens'][:, :batch_size] = qkv
        input_buffers['cu_seqlens_q'][1:batch_size + 1] = input_buffers['q_seqlens'][:batch_size].cumsum(0)
        input_buffers['cu_seqlens_k'][1:batch_size + 1] = input_buffers['kv_seqlens'][:batch_size].cumsum(0)
        if inputs_embeds is not None:
            emb_size = inputs_embeds.size(-1)
            if 'inputs_embeds' not in input_buffers:
                max_num_tokens = input_buffers['input_ids'].size(-1)
                input_buffers['inputs_embeds'] = inputs_embeds.new_zeros(1, max_num_tokens, emb_size)
            input_buffers['inputs_embeds'][:, :num_tokens] = inputs_embeds

        # create inputs
        new_batch_size = input_buffers['block_offsets'].size(0)
        attn_metadata.block_offsets = input_buffers['block_offsets']
        attn_metadata.q_start_loc = input_buffers['q_start_loc']
        attn_metadata.q_seqlens = input_buffers['q_seqlens']
        attn_metadata.kv_seqlens = input_buffers['kv_seqlens']
        attn_metadata.cu_seqlens_q = input_buffers['cu_seqlens_q']
        attn_metadata.cu_seqlens_k = input_buffers['cu_seqlens_k']
        if getattr(self.config, 'use_flash_mla', False) is True:
            import flash_mla
            tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(attn_metadata.kv_seqlens.to(torch.int32),
                                                                             self.config.num_attention_heads, 1)
            # here we use copy_ instead of = to avoid using new allocated mem for cuda graph
            input_buffers['tile_scheduler_metadata'].copy_(tile_scheduler_metadata)
            input_buffers['num_splits'][:new_batch_size + 1].copy_(num_splits[:new_batch_size + 1])
            attn_metadata.tile_scheduler_metadata = input_buffers['tile_scheduler_metadata']
            attn_metadata.num_splits = input_buffers['num_splits']

        new_inputs = dict(
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )

        cross_attn_metadata = kwargs.get('cross_attn_metadata', None)
        if cross_attn_metadata is not None:
            # TODO: update cross_attn_metadata here
            new_inputs['cross_attn_metadata'] = cross_attn_metadata

        if is_decoding:
            new_inputs['input_ids'] = input_buffers['input_ids']
            new_inputs['position_ids'] = input_buffers['position_ids']
        else:
            new_inputs['input_ids'] = input_buffers['input_ids']
            new_inputs['position_ids'] = input_buffers['position_ids']

        if inputs_embeds is not None:
            if is_decoding:
                new_inputs['inputs_embeds'] = input_buffers['inputs_embeds']
            else:
                new_inputs['inputs_embeds'] = input_buffers['inputs_embeds']

        new_inputs.update(kwargs)
        return new_inputs

    def update_context_cudagraph(self, graph_meta: CudaGraphMeta, context: StepContext):
        """Update step context with input buffers."""
        input_buffers = graph_meta.input_buffers
        local_adapter_ids = context.local_adapter_ids
        if local_adapter_ids is not None:
            if input_buffers['local_adapter_ids'].data_ptr() != local_adapter_ids.data_ptr():
                input_buffers['local_adapter_ids'].fill_(0)
            batch_size = local_adapter_ids.size(0)
            input_buffers['local_adapter_ids'][:batch_size] = local_adapter_ids
            context.local_adapter_ids = input_buffers['local_adapter_ids']
        context.q_seqlens = input_buffers['q_seqlens']
        context.kv_seqlens = input_buffers['kv_seqlens']
        context.q_start_loc = input_buffers['q_start_loc']
