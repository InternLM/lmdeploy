# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
import torch
from torch import Tensor


def sparse_attention_fwd(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    sparse_indices: Tensor,
    block_table: Tensor,
    actual_seq_lengths_q: Tensor,
    kv_seqlens: Tensor,
    value_head_size: int,
    softmax_scale: float,
    attn_output: Tensor = None,
) -> Tensor:
    """Run Ascend sparse MLA attention over split noPE and RoPE caches."""
    if query.size(-1) <= value_head_size:
        raise ValueError(
            'sparse MLA query must contain both latent and RoPE dimensions')
    if sparse_indices.dim() == 2:
        sparse_indices = sparse_indices.unsqueeze(1)
    if sparse_indices.dim() != 3 or sparse_indices.size(1) != 1:
        raise ValueError(
            f'sparse indices must have shape [tokens, 1, topk], got '
            f'{tuple(sparse_indices.shape)}')

    return sparse_attention_fwd_split(
        query[..., :value_head_size], query[..., value_head_size:], key_cache,
        value_cache, sparse_indices, block_table, actual_seq_lengths_q,
        kv_seqlens, softmax_scale, attn_output=attn_output)


def sparse_attention_fwd_split(
    query_nope: Tensor,
    query_rope: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    sparse_indices: Tensor,
    block_table: Tensor,
    actual_seq_lengths_q: Tensor,
    kv_seqlens: Tensor,
    softmax_scale: float,
    attn_output: Tensor = None,
) -> Tensor:
    # Run SFA with separate latent and RoPE query tensors.
    if query_nope.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f'sparse MLA query must be BF16/FP16, got {query_nope.dtype}')
    if query_nope.shape[:-1] != query_rope.shape[:-1]:
        raise ValueError('q_nope and q_rope must have matching leading shapes')
    if sparse_indices.dim() == 2:
        sparse_indices = sparse_indices.unsqueeze(1)
    if sparse_indices.dim() != 3 or sparse_indices.size(1) != 1:
        raise ValueError(
            f'sparse indices must have shape [tokens, 1, topk], got '
            f'{tuple(sparse_indices.shape)}')

    output = ext_ops.sparse_flash_attention(
        query_nope.contiguous(),
        value_cache,
        value_cache,
        sparse_indices.contiguous(),
        softmax_scale,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_q,
        kv_seqlens=kv_seqlens,
        query_rope=query_rope.contiguous(),
        key_rope=key_cache,
    )
    if attn_output is not None:
        attn_output.copy_(output)
        return attn_output
    return output
