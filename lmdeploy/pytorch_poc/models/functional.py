# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Any, Callable, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch import distributed as dist

from lmdeploy.pytorch_poc.kernels import (alibi_paged_attention_fwd,
                                          apply_rotary_pos_emb, fill_kv_cache,
                                          paged_attention_fwd)

__all__ = ['apply_rotary_pos_emb']


def generate_batched_mask(q_lens,
                          k_lens,
                          max_q_len: int = None,
                          max_k_len: int = None,
                          device='cuda'):
    """Generate batched mask."""
    if max_q_len is None:
        max_q_len = max(q_lens)

    if max_k_len is None:
        max_k_len = max(k_lens)

    q_range = torch.arange(max_q_len).to(device)
    k_range = torch.arange(max_k_len).to(device)

    cross = k_range.unsqueeze(0) - q_range.unsqueeze(1)
    cross = cross.unsqueeze(0)

    threshold = (k_lens - q_lens).view(-1, 1, 1)
    mask = torch.where(cross <= threshold, 1, 0).to(device)
    for idx, q_len in enumerate(q_lens):
        mask[idx, q_len:, :] = 0
    return mask


def get_slopes(n: int):
    """Get alibi slopes."""

    def _get_interleave_power_of_2(n):
        start = 2**(-(2**-(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n))
        return (
            _get_interleave_power_of_2(closest_power_of_2) +
            get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2])


@torch.no_grad()
def get_alibi_biases(n_heads: int, mask: torch.Tensor):
    """Get alibi bias."""
    m = torch.tensor(get_slopes(n_heads)).to(mask.device)
    distance = mask.cumsum(dim=-1) - 1
    return distance * m[None, :, None, None]


@torch.no_grad()
def attention_forward_with_paged_attention(
    hidden_states: Tensor,
    history_lengths: Sequence,
    block_offsets: Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    position_ids: torch.LongTensor,
    past_key_value: Tuple[Tensor],
    context: Any = None,
    q_proj: Optional[Callable] = None,
    k_proj: Optional[Callable] = None,
    v_proj: Optional[Callable] = None,
    qkv_proj: Optional[Callable] = None,
    o_proj: Optional[Callable] = None,
    rotary_emb_fn: Optional[Callable] = None,
    bias_type: str = 'default',
) -> Tensor:
    """Attention module forward with paced attention.

    Args:
        hidden_states (Tensor): Input of attention layer.
        history_lengths (Sequence): Cache lengths of each data in batch.
        block_offsets (Tensor): Block table of the key/value caches,
            used by paged attention.
        num_heads (int): numbers of query heads.
        num_kv_heads (int): numbers of key/value heads.
        head_dim (int): Feature dimension of heads.
        position_ids (LongTensor): position ids of the input.
        past_key_value (Tuple[Tensor]): key value cache.
        q_proj (Callable): query project module/function.
        k_proj (Callable): key project module/function.
        v_proj (Callable): value project module/function.
        qkv_proj (Callable): query/key/value project module/function.
        o_proj (Callable): output project module/function.
        rotary_emb_fn (Callable): rotary embedding callback.
        bias_type (str): type of attention bias. support ['default', 'alibi'].
    """
    max_seq_len = position_ids.size(-1)

    if qkv_proj is not None:
        assert q_proj is None
        assert k_proj is None
        assert v_proj is None
        query_states, key_states, value_states = qkv_proj(hidden_states)
    else:
        assert qkv_proj is None
        assert q_proj is not None
        assert k_proj is not None
        assert v_proj is not None

        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)

    query_states = query_states.view(-1, num_heads, head_dim)
    key_states = key_states.view(-1, num_kv_heads, head_dim)
    value_states = value_states.view(-1, num_kv_heads, head_dim)

    if rotary_emb_fn is not None:
        query_states, key_states, value_states = rotary_emb_fn(
            query_states, key_states, value_states)

    kv_seq_length = position_ids[..., -1] + 1

    q_seq_length = getattr(context, 'seq_length', None)
    if q_seq_length is None:
        q_seq_length = kv_seq_length - kv_seq_length.new_tensor(
            history_lengths)

    q_start_loc = getattr(context, 'q_start_loc', None)
    if q_start_loc is None:
        q_start_loc = q_seq_length.cumsum(0)
        q_start_loc = torch.cat([q_start_loc.new_zeros(1), q_start_loc[:-1]])

    fill_kv_cache(key_states,
                  value_states,
                  past_key_value[0],
                  past_key_value[1],
                  q_start_loc,
                  q_seq_length,
                  block_offsets=block_offsets,
                  history_lengths=history_lengths,
                  context=context)

    attn_output = torch.empty_like(query_states)

    block_size = past_key_value[0].size(1)

    bias_type = bias_type.lower()
    if bias_type == 'default':
        paged_attention_fwd(
            query_states,
            past_key_value[0],
            past_key_value[1],
            attn_output,
            block_offsets,
            b_start_loc=q_start_loc,
            b_seq_len=q_seq_length,
            b_kv_seq_len=kv_seq_length,
            max_input_len=max_seq_len,
            BLOCK=block_size,
        )
    else:
        if bias_type == 'alibi':
            num_heads_full = num_heads
            head_offset = 0
            if dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                num_heads_full = num_heads * world_size
                head_offset = num_heads * rank
            alibi_paged_attention_fwd(
                query_states,
                past_key_value[0],
                past_key_value[1],
                attn_output,
                block_offsets,
                b_start_loc=q_start_loc,
                b_seq_len=q_seq_length,
                b_kv_seq_len=kv_seq_length,
                max_input_len=max_seq_len,
                head_offset=head_offset,
                num_heads=num_heads_full,
                BLOCK=block_size,
            )
        else:
            raise ValueError(f'Unknown bias type: {bias_type}')
    hidden_size = num_heads * head_dim
    attn_output = attn_output.reshape(*hidden_states.shape[:-1], hidden_size)

    if o_proj is not None:
        attn_output = o_proj(attn_output)
    return attn_output
