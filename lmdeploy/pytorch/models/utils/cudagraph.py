# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.profiler import record_function

from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager

BuffType = Dict[str, Tensor]


def _get_meta_flashattn(
        batch_size: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        num_heads_q: int,
        num_heads_kv: int,
        headdim: int,
        cache_seqlens: torch.Tensor,
        qkv_dtype=torch.bfloat16,
        headdim_v=None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k_new: Optional[torch.Tensor] = None,
        page_size: Optional[int] = None,
        causal=True,
        window_size=(-1, -1),  # -1 means infinite context window
        num_splits=0,
):
    """Get scheduler metadata for flash attn."""
    from flash_attn_interface import get_scheduler_metadata

    metadata = get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens,
        qkv_dtype=qkv_dtype,
        headdim_v=headdim_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new,
        page_size=page_size,
        causal=causal,
        window_size=window_size,
        num_splits=num_splits,
    )
    return metadata


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
    use_mla_fp8_cache: bool = False
    use_flash_mla: bool = False
    mla_index_topk: Optional[int] = None
    decode_query_len: int = 1
    use_fa3_decoding: bool = False


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

    def make_output_buffers(self, output):
        """Make output buffers."""
        if isinstance(output, torch.Tensor):
            output_buffers = dict(hidden_states=output)
        else:
            assert isinstance(output, Dict)
            output_buffers = output
        return output_buffers

    def update_meta_flashattn(self, graph_meta: CudaGraphMeta, block_size: int, max_seqlen_k: int,
                              cache_seqlens: torch.Tensor):
        """Update meta flashattn."""
        ctx_mgr = get_step_ctx_manager()
        step_ctx = ctx_mgr.current_context()
        model_config = step_ctx.model_config
        batch_size = graph_meta.max_batchs
        max_seqlen_q = graph_meta.decode_query_len
        sliding_window = model_config.sliding_window
        num_attention_heads = model_config.num_attention_heads
        num_key_value_heads = model_config.num_key_value_heads
        headdim = model_config.head_dim
        torch_dtype = model_config.dtype
        if sliding_window is None:
            window_size = (-1, -1)
        elif isinstance(sliding_window, int):
            window_size = (sliding_window, sliding_window)
        cache_seqlens = cache_seqlens.to(torch.int32)
        scheduler_metadata = _get_meta_flashattn(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=num_attention_heads,
            num_heads_kv=num_key_value_heads,
            headdim=headdim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=torch_dtype,
            page_size=block_size,
            window_size=window_size,
        )
        return scheduler_metadata

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, *args, past_key_values: List, **kwargs) -> BuffType:
        """Make cudagraph buffers from forward inputs."""
        max_batches = graph_meta.max_batchs
        max_tokens = graph_meta.max_tokens
        num_blocks = graph_meta.num_blocks
        device = graph_meta.device
        decode_query_len = graph_meta.decode_query_len

        input_buffers: BuffType = dict()
        input_buffers['input_ids'] = torch.randint(0,
                                                   graph_meta.vocab_size, (1, max_tokens),
                                                   dtype=torch.int64,
                                                   device=device)
        input_buffers['position_ids'] = torch.zeros((1, max_tokens), dtype=torch.int64, device=device)

        # flash_mla requires block_offsets and kv_lens int32
        input_buffers['block_offsets'] = torch.zeros((max_batches, num_blocks), dtype=torch.int32, device=device)
        input_buffers['qkv_lens'] = torch.zeros(3, max_batches, dtype=torch.int32, device=device)

        input_buffers['q_start_loc'] = input_buffers['qkv_lens'][0]
        input_buffers['q_seqlens'] = input_buffers['qkv_lens'][1]
        input_buffers['kv_seqlens'] = input_buffers['qkv_lens'][2]
        input_buffers['qkv_seqlens'] = input_buffers['qkv_lens'][1:]
        input_buffers['local_adapter_ids'] = torch.zeros(max_batches, dtype=torch.int64, device=device)

        input_buffers['cu_seqlens'] = torch.zeros(2, max_batches + 1, dtype=torch.int32, device=device)
        input_buffers['cu_seqlens_q'] = input_buffers['cu_seqlens'][0]
        input_buffers['cu_seqlens_k'] = input_buffers['cu_seqlens'][1]

        if graph_meta.use_flash_mla is True:
            import flash_mla

            # create buffers for flash mla
            num_attention_heads = self.config.num_attention_heads
            index_topk = graph_meta.mla_index_topk
            num_heads_q = None if index_topk is None else num_attention_heads
            input_buffers['tile_scheduler_metadata'], input_buffers['num_splits'] = flash_mla.get_mla_metadata(
                torch.ones(max_batches, dtype=torch.int32, device=device),
                num_attention_heads * decode_query_len,
                num_heads_k=1,
                num_heads_q=num_heads_q,
                is_fp8_kvcache=graph_meta.use_mla_fp8_cache,
                topk=index_topk)

        # use fa3 decode kernel for spec decode
        elif graph_meta.use_fa3_decoding is True:
            block_size = past_key_values[0][0].size(1)
            input_buffers['scheduler_metadata'] = self.update_meta_flashattn(graph_meta,
                                                                             block_size=block_size,
                                                                             max_seqlen_k=decode_query_len,
                                                                             cache_seqlens=input_buffers['kv_seqlens'])

        return input_buffers

    @record_function('fill_buffers_cudagraph')
    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, input_ids: Tensor, position_ids: Tensor,
                               past_key_values: List, attn_metadata: Any, inputs_embeds: Tensor,
                               **kwargs) -> Dict[str, Tensor]:
        """Fill cudagraph buffers from forward inputs."""

        block_offsets: Tensor = attn_metadata.block_offsets
        q_start_loc: Tensor = attn_metadata.q_start_loc
        q_seqlens: Tensor = attn_metadata.q_seqlens
        kv_seqlens: Tensor = attn_metadata.kv_seqlens
        input_buffers: BuffType = graph_meta.input_buffers

        batch_size, num_blocks = block_offsets.size()
        num_tokens = input_ids.size(-1)
        decode_query_len = graph_meta.decode_query_len
        # fill buffer
        input_buffers['input_ids'].random_(0, graph_meta.vocab_size)
        input_buffers['input_ids'][:, :num_tokens] = input_ids
        input_buffers['position_ids'][:, :num_tokens] = position_ids
        input_buffers['block_offsets'][:batch_size, :num_blocks] = block_offsets

        qkv = torch.stack((q_start_loc, q_seqlens, kv_seqlens))
        input_buffers['qkv_lens'].zero_()
        input_buffers['q_seqlens'].fill_(graph_meta.max_tokens // graph_meta.max_batchs)
        input_buffers['qkv_lens'][:, :batch_size] = qkv
        input_buffers['cu_seqlens'][:, 1:] = input_buffers['qkv_seqlens'].cumsum(1)
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

        if graph_meta.use_flash_mla is True:
            import flash_mla
            num_attention_heads = self.config.num_attention_heads
            index_topk = graph_meta.mla_index_topk
            num_heads_q = None if index_topk is None else num_attention_heads
            tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
                attn_metadata.kv_seqlens.to(torch.int32),
                num_attention_heads * decode_query_len,
                num_heads_k=1,
                num_heads_q=num_heads_q,
                is_fp8_kvcache=graph_meta.use_mla_fp8_cache,
                topk=index_topk)
            # here we use copy_ instead of = to avoid using new allocated mem for cuda graph
            input_buffers['tile_scheduler_metadata'].copy_(tile_scheduler_metadata)
            input_buffers['num_splits'][:new_batch_size + 1].copy_(num_splits[:new_batch_size + 1])
            attn_metadata.tile_scheduler_metadata = input_buffers['tile_scheduler_metadata']
            attn_metadata.num_splits = input_buffers['num_splits']

        # use fa3 decode kernel for spec decode
        elif graph_meta.use_fa3_decoding is True:
            block_size = past_key_values[0][0].size(1)
            scheduler_metadata = self.update_meta_flashattn(
                graph_meta,
                block_size=block_size,
                max_seqlen_k=attn_metadata.max_kv_seqlen,
                cache_seqlens=input_buffers['kv_seqlens'],
            )
            assert scheduler_metadata.shape == input_buffers['scheduler_metadata'].shape
            input_buffers['scheduler_metadata'].copy_(scheduler_metadata)
            attn_metadata.scheduler_metadata = input_buffers['scheduler_metadata']

        new_inputs = dict(
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )

        new_inputs['input_ids'] = input_buffers['input_ids']
        new_inputs['position_ids'] = input_buffers['position_ids']

        if inputs_embeds is not None:
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

    def get_outputs_cudagraph(self, output_buffers: Dict[str, torch.Tensor], input_ids: Tensor, **kwargs):
        """Get outputs from buffers."""
        num_tokens = input_ids.size(-1)
        outputs = dict()
        outputs['hidden_states'] = output_buffers['hidden_states'][:, :num_tokens]
        if output_buffers.get('all_routed_experts', None) is not None:
            outputs['all_routed_experts'] = output_buffers['all_routed_experts'][:num_tokens, ...].clone()
        return outputs
