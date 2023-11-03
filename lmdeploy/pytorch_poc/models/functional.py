# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import distributed as dist

from lmdeploy.pytorch_poc.kernels import (alibi_paged_attention_fwd,
                                          apply_rotary_pos_emb, fill_kv_cache, paged_attention_fwd,
                                          rerope_attention_fwd)
__all__ = ['apply_rotary_pos_emb']


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (num_key_value_heads, seqlen, head_dim) to
    (num_attention_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:,
                                  None, :, :].expand(num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(num_key_value_heads * n_rep, slen, head_dim)


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


@torch.no_grad()
def attention_forward_with_rerope(
    hidden_states: Tensor,
    history_lengths: Sequence,
    block_offsets: Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    position_ids: torch.LongTensor,
    past_key_value: Tuple[Tensor],
    attention_mask: Tensor,
    context: Any = None,
    q_proj: Optional[Callable] = None,
    k_proj: Optional[Callable] = None,
    v_proj: Optional[Callable] = None,
    qkv_proj: Optional[Callable] = None,
    o_proj: Optional[Callable] = None,
    rotary_emb_context_fn: Optional[Callable] = None,
    rotary_emb_generate_fn: Optional[Callable] = None,
    bias_type: str = 'default',
    training_length=4096,
    window=512,
) -> Tensor:
    """Attention module forward with ReRoPE.

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
        bias_type (str): type of attention bias. support ['default'].
    """
    hidden_size = -1
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

    hidden_size = num_heads * head_dim
    query_states = query_states.view(-1, num_heads, head_dim)
    key_states = key_states.view(-1, num_kv_heads, head_dim)
    value_states = value_states.view(-1, num_kv_heads, head_dim)
    query_states *= ((position_ids.flatten() + 1)[:, None, None].log() /
                     np.log(training_length)).clip(1).to(query_states.dtype)

    kv_seq_length = (position_ids[..., -1] + 1).item()

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

    bsz, q_len, _ = hidden_states.size()
    if bias_type.lower() == 'default':

        if q_len == 1:
            key_states = past_key_value[0][block_offsets].view(
                -1, num_heads, head_dim)[0:history_lengths[-1] + 1]
            value_states = past_key_value[1][block_offsets].view(
                -1, num_heads, head_dim)[0:history_lengths[-1] + 1]

            full_position_ids = torch.arange(
                position_ids.item() + 1,
                device=position_ids.device).unsqueeze(0)

            key_states = rotary_emb_generate_fn(key_states, value_states,
                                                full_position_ids, window)
            attn_weights = torch.matmul(query_states.transpose(
                0, 1), key_states.permute(1, 2, 0)) / math.sqrt(head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = torch.nn.functional.softmax(attn_weights,
                                                       dim=-1,
                                                       dtype=torch.float32).to(
                                                           query_states.dtype)
            attn_output = torch.matmul(attn_weights,
                                       value_states.transpose(0, 1))

        else:

            query_states1, query_states2, key_states1, key_states2, value_states = rotary_emb_context_fn(
                query_states, key_states, value_states, position_ids, window)

            sm_scale = 1.0 / math.sqrt(head_dim)

            PADDING_UNIT = past_key_value[0].shape[1]
            assert PADDING_UNIT in {16, 32, 64, 128, 256}
            padding_len = -query_states1.shape[2] % PADDING_UNIT

            query_states1 = F.pad(query_states1,
                                  (0, 0, 0, padding_len)).contiguous()
            query_states2 = F.pad(query_states2,
                                  (0, 0, 0, padding_len)).contiguous()
            key_states1 = F.pad(key_states1,
                                (0, 0, 0, padding_len)).contiguous()
            key_states2 = F.pad(key_states2,
                                (0, 0, 0, padding_len)).contiguous()
            value_states = F.pad(value_states,
                                 (0, 0, 0, padding_len)).contiguous()

            attn_output = rerope_attention_fwd(query_states1,
                                               query_states2,
                                               key_states1,
                                               key_states2,
                                               value_states,
                                               True,
                                               sm_scale,
                                               window,
                                               BLOCK_M=PADDING_UNIT).squeeze(0)

            attn_output = attn_output[:, 0:q_len]

        if attn_output.size() != (num_heads, q_len, head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is'
                f' {attn_output.size()}')

        attn_output = attn_output.transpose(0, 1).reshape(
            bsz, q_len, hidden_size).contiguous()
    else:
        raise ValueError(f'Unknown bias type: {bias_type}')

    if o_proj is not None:
        attn_output = o_proj(attn_output)
    return attn_output
