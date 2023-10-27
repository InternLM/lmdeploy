# Copyright (c) OpenMMLab. All rights reserved.
import math
import pdb
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import distributed as dist

from lmdeploy.pytorch_poc.kernels import (alibi_paged_attention_fwd,
                                          fill_kv_cache, paged_attention_fwd)


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


def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor,
                         k: Tensor,
                         cos: Tensor,
                         sin: Tensor,
                         position_ids: Tensor,
                         position_ids_1d: Tensor = None):
    """Apply rotary positional embedding on query and key.

    Args:
        q (Tensor): Query state.
        k (Tensor): Key state.
        cos (Tensor): cosine matrix (seq_len, dim).
        sin (Tensor): sine matrix (seq_len, dim).
        position_ids (Tensor): Position ids of q and k.
        position_ids_1d (Tensor): 1d Position ids.

    Returns:
        Tuple[Tensor, Tensor]: Embedded query and key.
    """
    # The first two dimensions of cos and sin are always 1,
    # so we can `squeeze` them.
    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=q.device, dtype=q.dtype)
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    if position_ids_1d is None:
        seq_length = position_ids[..., -1] + 1
        position_ids_1d = [ids[:l] for ids, l in zip(position_ids, seq_length)]
        position_ids_1d = torch.cat(position_ids_1d)
    cos = cos[position_ids_1d].unsqueeze(1)
    sin = sin[position_ids_1d].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


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
        bias_type (str): type of attention bias. support ['default'].
    """
    # max_seq_len = position_ids.size(-1)
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

    # attn_output = torch.empty_like(query_states)

    # block_size = past_key_value[0].size(1)
    bsz, q_len, _ = hidden_states.size()
    bias_type = bias_type.lower()
    if bias_type == 'default':

        if q_len == 1:
            key_states = past_key_value[0][block_offsets].view(
                -1, num_heads, head_dim)[0:history_lengths[-1] + 1]
            value_states = past_key_value[1][block_offsets].view(
                -1, num_heads, head_dim)[0:history_lengths[-1] + 1]
            full_position_ids = torch.arange(
                position_ids.item() + 1,
                device=position_ids.device).unsqueeze(0)

            key_states, value_states = rotary_emb_generate_fn(
                key_states, value_states, full_position_ids, window)
            attn_weights = torch.matmul(query_states.transpose(
                0, 1), key_states.permute(1, 2, 0)) / math.sqrt(head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
        else:
            query_states1, query_states2, key_states1, key_states2, value_states = rotary_emb_context_fn(
                query_states, key_states, value_states, position_ids,
                window + 1)
            attn_weights1 = torch.matmul(query_states1.transpose(
                0, 1), key_states1.permute(1, 2, 0)) / math.sqrt(head_dim)
            attn_weights2 = torch.matmul(query_states2.transpose(
                0, 1), key_states2.permute(1, 2, 0)) / math.sqrt(head_dim)
            rectified_mask = (position_ids[:, -q_len:, None] -
                              position_ids[:, None]).abs() < window
            attn_weights = torch.where(rectified_mask, attn_weights1,
                                       attn_weights2)
            if attn_weights.size() != (num_heads, q_len, kv_seq_length):
                raise ValueError(
                    f'Attention weights should be of size {(num_heads, q_len, kv_seq_length)}, but is'
                    f' {attn_weights.size()}')

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            else:
                tgt_len = attn_weights.shape[-1]
                dtype = attn_weights.dtype
                device = attn_weights.device
                mask = torch.full((tgt_len, tgt_len),
                                  torch.finfo(dtype).min,
                                  device=device)
                mask_cond = torch.arange(mask.size(-1), device=device)
                mask.masked_fill_(
                    mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                mask = mask.to(dtype)
                attn_weights = attn_weights + mask

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states.transpose(0, 1))

        if attn_output.size() != (num_heads, q_len, head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is'
                f' {attn_output.size()}')

        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    else:
        raise ValueError(f'Unknown bias type: {bias_type}')
    # attn_output = attn_output.reshape(*hidden_states.shape[:-1], hidden_size)

    if o_proj is not None:
        attn_output = o_proj(attn_output)
    return attn_output
