# Copyright (c) OpenMMLab. All rights reserved.
import threading

import dlinfer.ops as ext_ops
from torch import Tensor

# aclnnSparseFlashAttention hardcodes rope_head_dim=64 for sparse MLA
# (attention_mode=2): it rejects query_rope=None with "Shape of tensor
# queryRope is nullptr" and every other width with "rope_head_dim should be
# 64, but got N".  Ropeless (NoPE) models therefore pass zero tensors of
# exactly this width - the zero content adds a constant 0 to the rope dot
# product, which is exact NoPE semantics.  vLLM-Ascend's GLM NoPE path relies
# on the same contract; its _C_ascend extension materialises these zeros
# internally, while torch_npu's direct binding requires the caller to.
SPARSE_MLA_ROPE_HEAD_DIM = 64

_ZERO_KEY_ROPE: dict[tuple, Tensor] = {}
_ZERO_KEY_ROPE_LOCK = threading.Lock()


def _nope_rope_placeholders(query: Tensor, key_cache: Tensor) -> tuple[Tensor, Tensor]:
    """Build the zero rope stand-ins NoPE sparse MLA must supply.

    ``key_rope`` mirrors the key cache's paged layout because the operator
    gathers it through the block table; since it is never written, one shared
    zero buffer per cache shape serves every layer.
    """
    q_rope = query.new_zeros(query.shape[0], query.shape[1],
                             SPARSE_MLA_ROPE_HEAD_DIM)
    with _ZERO_KEY_ROPE_LOCK:
        key_rope = _ZERO_KEY_ROPE.get(key_cache.shape)
        if key_rope is None or key_rope.device != key_cache.device:
            key_rope = key_cache.new_zeros(*key_cache.shape[:-1],
                                           SPARSE_MLA_ROPE_HEAD_DIM)
            _ZERO_KEY_ROPE[key_cache.shape] = key_rope
    return q_rope, key_rope


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
    if query.size(-1) < value_head_size:
        raise ValueError(
            'sparse MLA query must contain at least the latent dimensions')
    if sparse_indices.dim() == 2:
        sparse_indices = sparse_indices.unsqueeze(1)
    if sparse_indices.dim() != 3 or sparse_indices.size(1) != 1:
        raise ValueError(
            f'sparse indices must have shape [tokens, 1, topk], got '
            f'{tuple(sparse_indices.shape)}')

    if query.size(-1) == value_head_size:
        # NoPE attention (GLM-5.3 keeps qk_rope_head_dim == 0): the query is
        # the absorbed latent vector and the latent paged cache is used for
        # both key and value, with no separate RoPE streams.  The operator
        # still demands 64-wide rope tensors, so supply zero stand-ins
        # instead of the None that vLLM-Ascend passes to its own extension.
        q_nope = query.contiguous()
        q_rope, key_rope = _nope_rope_placeholders(query, key_cache)
    else:
        q_nope = query[..., :value_head_size].contiguous()
        q_rope = query[..., value_head_size:].contiguous()
        key_rope = key_cache

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
        key_rope=key_rope,
    )
    if attn_output is not None:
        attn_output.copy_(output)
        return attn_output
    return output
