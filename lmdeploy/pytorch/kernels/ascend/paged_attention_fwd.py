# Copyright (c) OpenMMLab. All rights reserved.
import infer_ext.ops as ext_ops
import torch
from torch import Tensor


def flash_context_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attn_output: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    block_size: int,
    kv_cache_len: int,
    context=None,
):
    num_q_heads, dim = query_states.shape[1:3]
    num_kv_heads = value_states.shape[1]

    if context.is_unpaged_prefill:
        ext_ops.prefill_attention(
            query_states,
            key_states,
            value_states,
            q_start_loc,
            q_seq_len,
            context.max_q_seq_length,
            num_q_heads,
            num_kv_heads,
            attn_mask=context.attention_mask,
            attn_output=attn_output,
        )
    else:
        key_cache = key_cache.reshape(1, kv_cache_len, num_kv_heads * dim)
        value_cache = value_cache.reshape(1, kv_cache_len, num_kv_heads * dim)
        ext_ops.paged_prefill_attention(
            query_states,
            key_cache,
            value_cache,
            block_offsets,
            block_size,
            q_start_loc,
            q_seq_len,
            kv_seq_len,
            num_q_heads,
            num_kv_heads,
            attn_mask=context.attention_mask,
            attn_output=attn_output,
        )


def paged_token_attention(q, k_cache, v_cache, attn_output, kv_seq_len,
                          max_kv_seq_len, block_offsets, block_size):
    num_kv_heads, num_q_heads = k_cache.shape[1], q.shape[1]
    ext_ops.paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_offsets,
        block_size,
        kv_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_output=attn_output.view(q.shape),
    )


def paged_attention_fwd(
    query_states: Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    attn_output: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int,
    window_size: int = 1,
    context=None,
):
    is_decoding = query_states.shape[-3] == q_seqlens.size(0)
    block_num, block_size, head, dim = key_cache.size()
    kv_cache_len = block_num * block_size
    k = key_cache.reshape(block_num * block_size, head, dim)
    v = value_cache.reshape(block_num * block_size, head, dim)
    if not is_decoding:
        flash_context_attention(
            query_states,
            key_states,
            value_states,
            attn_output,
            k,
            v,
            block_offsets,
            q_start_loc,
            q_seqlens,
            kv_seqlens,
            block_size,
            kv_cache_len,
            context=context,
        )
    else:
        paged_token_attention(
            query_states,
            k,
            v,
            attn_output,
            kv_seqlens,
            context.max_kv_seq_length,
            block_offsets,
            block_size,
        )
