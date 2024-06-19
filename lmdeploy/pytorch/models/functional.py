# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Any

import torch

from ..kernels import apply_rotary_pos_emb

__all__ = ['apply_rotary_pos_emb']


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


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


def quant_kv(key: torch.Tensor, value: torch.Tensor, out_type: torch.dtype):
    """Quantize key and value of attention to `out_type`.

    Args:
        key (torch.Tensor): Attention key.
        value (torch.Tensor): Attention value.
        out_type (torch.dtype): Output data type.
    """
    assert out_type is torch.int8
    # quantize key and value
    _min = torch.min(key, axis=-1).values
    _max = torch.max(key, axis=-1).values
    key_zp = (_min + _max) / 2
    key_scale = (_max - key_zp) / 127
    key_int8 = torch.round(
        (key - key_zp[:, :, None]) / key_scale[:, :, None]).to(out_type)

    _min = torch.min(value, axis=-1).values
    _max = torch.max(value, axis=-1).values
    value_zp = (_min + _max) / 2
    value_scale = (_max - value_zp) / 127
    value_int8 = torch.round(
        (value - value_zp[:, :, None]) / value_scale[:, :, None]).to(out_type)

    # wrap zp and scale to qparams
    qparams = {
        'key_zp': key_zp,
        'key_scale': key_scale,
        'value_zp': value_zp,
        'value_scale': value_scale,
    }
    return key_int8, value_int8, qparams


def dequant_kv(context: Any, layer_id: str, key_int8: torch.Tensor,
               value_int8: torch.Tensor, out_type: torch.dtype):
    """Dequantize key and value of attention to `out_type`.

    Args:
        context (Any): StepContext during inference.
        layer_id (str): Layer object id.
        key (torch.Tensor): Quantized attention key.
        value (torch.Tensor): Quantized attention value.
        out_type (torch.dtype): output data type.
    """
    qparams = context.get_output(layer_id)

    key_scale = qparams['key_scale']
    key_zp = qparams['key_zp']
    key_float = (key_int8 * key_scale[:, :, None] +
                 key_zp[:, :, None]).to(out_type)

    value_scale = qparams['value_scale']
    value_zp = qparams['value_zp']
    value_float = (value_int8 * value_scale[:, :, None] +
                   value_zp[:, :, None]).to(out_type)
    return key_float, value_float


def sync_qparam_to_context(context: Any, layer_id: str, qparams: dict):
    """Merge quantization param to context.

    Args:
        context (Any): StepContext during inference.
        layer_id (str): Layer object id.
        qparams (dict): Quantization param of current step.
    """
    if context.inputs.meta is not None:
        last_qparam = context.inputs.meta[layer_id]
        for _k in last_qparam.keys():
            _v = torch.concat([last_qparam[_k], qparams[_k]], axis=0)
            last_qparam[_k] = _v
        context.set_output(layer_id, last_qparam)
    else:
        context.set_output(layer_id, qparams)
