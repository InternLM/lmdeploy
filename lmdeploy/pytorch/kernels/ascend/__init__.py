# Copyright (c) OpenMMLab. All rights reserved.

import torch
import infer_ext.ops as ext_ops
from torch import Tensor
from ..default import multinomial_sampling

__all__ = [
    'rms_norm',
    'apply_rotary_pos_emb',
    'fused_rotary_emb',
    'fill_kv_cache',
    'paged_attention_fwd',
    'multinomial_sampling',
]

def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float = 1e-6
):
    return ext_ops.rms_norm(hidden_states, weight, epsilon)

def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor,
    position_ids_1d: Tensor,
    q_embed=None,
    k_embed=None,
    context=None,
):
    bs, head, dim = query_states.shape
    num_kv_heads = key_states.shape[1]
    query_states_reshaped = query_states.reshape(1, bs, head, dim)
    key_states_reshaped = key_states.reshape(1, bs, num_kv_heads, dim)
    if not (hasattr(context, 'cos') or hasattr(context, 'sin')):
        cos = cos[position_ids_1d].view(1, bs, 1, -1)
        sin = sin[position_ids_1d].view(1, bs, 1, -1)
        if context:
            setattr(context, 'cos', cos)
            setattr(context, 'sin', sin)
    cached_cos = context.cos if context else cos
    cached_sin = context.sin if context else sin
    ext_ops.apply_rotary_pos_emb(
        query_states_reshaped, key_states_reshaped, cached_cos, cached_sin,
        None, None, None
    )
    if q_embed is None:
        q_embed = query_states
    else:
        q_embed.copy_(query_states)
    if k_embed is None:
        k_embed = key_states
    else:
        k_embed.copy_(key_states)
    return q_embed, k_embed

def fused_rotary_emb(
    query_states: Tensor,
    key_states: Tensor,
    position_ids: torch.LongTensor,
    inv_freq: Tensor,
    scaling_factor: float,
    out_q: Tensor = None,
    out_k: Tensor = None,
    context=None,
):
    batch, seqlen, head, dim = query_states.shape
    num_kv_heads = key_states.shape[-2]
    query_states_reshaped = query_states.view(batch, seqlen, head, dim)
    key_states_reshaped = key_states.view(batch, seqlen, num_kv_heads, dim)
    position_ids = position_ids.squeeze(0).unsqueeze(-1)
    pos_freq = position_ids / scaling_factor * inv_freq
    if not (hasattr(context, 'cos') or hasattr(context, 'sin')):
        cos = (torch.cos(pos_freq).view(batch, seqlen, 1, -1)
                                  .repeat(1, 1, 1, 2).to(query_states.dtype))
        sin = (torch.sin(pos_freq).view(batch, seqlen, 1, -1)
                                  .repeat(1, 1, 1, 2).to(query_states.dtype))
        if context:
            setattr(context, 'cos', cos)
            setattr(context, 'sin', sin)
    cached_cos = context.cos if context else cos
    cached_sin = context.sin if context else sin
    ext_ops.apply_rotary_pos_emb(query_states_reshaped, key_states_reshaped,
                                 cached_cos, cached_sin, None, None, None)
    if out_q is None:
        out_q = query_states
    else:
        out_q.copy_(query_states)
    if out_k is None:
        out_k = key_states
    else:
        out_k.copy_(key_states)
    return out_q, out_k

def fill_kv_cache(
    key_states: Tensor,
    value_states: Tensor,
    key_caches: Tensor,
    value_caches: Tensor,
    q_start_loc: Tensor,
    q_seq_length: Tensor,
    kv_seq_length: Tensor,
    max_q_seq_length: int,
    block_offsets: Tensor,
    context: None,
):
    """fill key/value state to cache for paged attention."""
    ext_ops.fill_kv_cache(key_states, value_states, key_caches,
                          value_caches, context.kv_start_indices)

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
    batch = q_start_loc.shape[0]    

    for i in range(batch):
        if torch.equal(q_seq_len[i], kv_seq_len[i]):
            ext_ops.context_attention(
                attn_output,
                query_states,
                key_states,
                value_states,
                q_start_loc[i:i+1],
                q_seq_len[i:i+1],
                num_q_heads,
                num_kv_heads,
                context.attention_mask[i:i+1],
            )
        else:
            key_cache = key_cache.reshape(1, kv_cache_len, num_kv_heads * dim)
            value_cache = value_cache.reshape(1, kv_cache_len, num_kv_heads * dim)
            ext_ops.paged_prefill_attention(
                attn_output,
                query_states,
                key_cache,
                value_cache,
                block_offsets,
                block_size,
                q_start_loc[i:i+1],
                q_seq_len[i:i+1],
                kv_seq_len[i:i+1],
                num_q_heads,
                num_kv_heads,
                context.attention_mask[i:i+1],
            )

def paged_token_attention(q, k_cache, v_cache, attn_output, kv_seq_len,
                          block_offsets, block_size):
    num_kv_heads, num_q_heads = k_cache.shape[1], q.shape[1]
    ext_ops.paged_decode_attention(
        attn_output.view(q.shape),
        q,
        k_cache,
        v_cache,
        block_offsets,
        block_size,
        kv_seq_len,
        num_q_heads,
        num_kv_heads,
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
            block_offsets,
            block_size,
        )
