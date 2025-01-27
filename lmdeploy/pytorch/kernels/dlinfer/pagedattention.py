# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
import torch
from dlinfer.utils.type_annotation import Optional, Sequence, Tensor


def prefill_attention(
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
    cu_seq_lens_kv: Tensor,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    block_size: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    is_unpaged_prefill: Optional[bool],
    kv_scales: Optional[Tensor],
    kv_zeros: Optional[Tensor],
    quant_bits: Optional[int],
) -> Tensor:
    num_q_heads = query_states.shape[1]
    num_kv_heads = value_states.shape[1]

    if is_unpaged_prefill:
        return ext_ops.prefill_attention(
            query_states,
            key_states,
            value_states,
            q_start_loc,
            q_seq_len,
            max_q_seq_len,
            num_q_heads,
            num_kv_heads,
            attn_mask,
            softmax_scale=softmax_scale,
            attn_output=attn_output,
        )
    else:
        return ext_ops.paged_prefill_attention(
            query_states,
            key_states,
            value_states,
            key_cache,
            value_cache,
            block_offsets,
            block_size,
            q_start_loc,
            q_seq_len,
            kv_seq_len,
            cu_seq_lens_kv,
            max_q_seq_len,
            max_kv_seq_len,
            num_q_heads,
            num_kv_heads,
            attn_mask,
            softmax_scale=softmax_scale,
            attn_output=attn_output,
            kv_scales=kv_scales,
            kv_zeros=kv_zeros,
            quant_bits=quant_bits,
        )


def paged_token_attention(
    q,
    k_cache,
    v_cache,
    attn_output,
    kv_seq_len,
    max_kv_seq_len,
    block_offsets,
    block_size,
    softmax_scale: Optional[float],
    kv_scales: Optional[Tensor],
    kv_zeros: Optional[Tensor],
    quant_bits: Optional[int],
):
    num_q_heads, q_head_dim = q.shape[1:3]
    num_kv_heads = k_cache.shape[-1] // q_head_dim
    return ext_ops.paged_decode_attention(
        q,
        k_cache,
        v_cache,
        block_offsets,
        block_size,
        kv_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        softmax_scale=softmax_scale,
        attn_output=attn_output,
        kv_scales=kv_scales,
        kv_zeros=kv_zeros,
        quant_bits=quant_bits,
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
    cu_seq_lens_kv: Tensor,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    is_decoding: bool,
    block_size: int,
    attn_mask: Sequence[Optional[Tensor]] = (),
    softmax_scale: Optional[float] = None,
    is_unpaged_prefill: Optional[bool] = None,
    kv_scales: Optional[Tensor] = None,
    kv_zeros: Optional[Tensor] = None,
    quant_bits: Optional[int] = 0,
):
    if not is_decoding:
        return prefill_attention(
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
            cu_seq_lens_kv,
            max_q_seq_len,
            max_kv_seq_len,
            block_size,
            attn_mask,
            softmax_scale,
            is_unpaged_prefill,
            kv_scales=kv_scales,
            kv_zeros=kv_zeros,
            quant_bits=quant_bits,
        )
    else:
        return paged_token_attention(
            query_states,
            key_cache,
            value_cache,
            attn_output,
            kv_seqlens,
            max_kv_seq_len,
            block_offsets,
            block_size,
            softmax_scale=softmax_scale,
            kv_scales=kv_scales,
            kv_zeros=kv_zeros,
            quant_bits=quant_bits,
        )
