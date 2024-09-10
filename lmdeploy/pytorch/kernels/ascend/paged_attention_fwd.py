# Copyright (c) OpenMMLab. All rights reserved.
import torch
import infer_ext.ops as ext_ops
from infer_ext.utils.type_annotation import Tensor, Optional, Sequence


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
    attn_mask: Sequence[Optional[Tensor]],
    is_unpaged_prefill: Optional[bool],
):
    num_q_heads = query_states.shape[1]
    num_kv_heads = value_states.shape[1]

    if is_unpaged_prefill:
        return ext_ops.context_attention(
            query_states,
            key_states,
            value_states,
            q_start_loc,
            q_seq_len,
            num_q_heads,
            num_kv_heads,
            attn_mask,
            attn_output=attn_output,
        )
    else:
        return ext_ops.paged_prefill_attention(
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
            attn_mask,
            attn_output=attn_output,
        )


def paged_token_attention(q, k_cache, v_cache, attn_output, kv_seq_len,
                          block_offsets, block_size):
    num_q_heads, q_head_dim = q.shape[1:3]
    num_kv_heads = k_cache.shape[-1] // q_head_dim
    return ext_ops.paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_offsets,
        block_size,
        kv_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_output=attn_output,
    )


def paged_attention_fwd(
    query_states: Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attn_output: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    is_decoding: bool,
    block_size: int,
    attn_mask: Sequence[Optional[Tensor]] = (),
    is_unpaged_prefill: Optional[bool] = None,
):
    if not is_decoding:
        return flash_context_attention(
            query_states,
            key_states,
            value_states,
            attn_output,
            key_cache,
            value_cache,
            block_offsets,
            q_start_loc,
            q_seqlens,
            kv_seqlens,
            block_size,
            attn_mask,
            is_unpaged_prefill,
        )
    else:
        return paged_token_attention(
            query_states,
            key_cache,
            value_cache,
            attn_output,
            kv_seqlens,
            block_offsets,
            block_size,
        )
