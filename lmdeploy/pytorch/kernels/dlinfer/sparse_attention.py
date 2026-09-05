# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
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

    q_nope = query[..., :value_head_size].contiguous()
    q_rope = query[..., value_head_size:].contiguous()

    output = ext_ops.sparse_flash_attention(
        q_nope,
        value_cache,
        value_cache,
        sparse_indices,
        softmax_scale,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_q,
        kv_seqlens=kv_seqlens,
        query_rope=q_rope,
        key_rope=key_cache,
    )
    if attn_output is not None:
        attn_output.copy_(output)
        return attn_output
    return output
