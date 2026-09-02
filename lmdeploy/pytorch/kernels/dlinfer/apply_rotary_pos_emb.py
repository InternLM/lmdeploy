# Copyright (c) OpenMMLab. All rights reserved.

import dlinfer.ops as ext_ops
import torch
from torch import Tensor


def apply_rotary_pos_emb(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    q_embed: Tensor | None,
    k_embed: Tensor | None,
) -> tuple[Tensor, Tensor]:
    query_states_embed, key_states_embed = \
        ext_ops.apply_rotary_pos_emb(query_states,
                                     key_states,
                                     cos, sin)
    if q_embed is None:
        q_embed = query_states_embed.view(query_states.shape)
    elif q_embed is not query_states:
        q_embed.copy_(query_states_embed.view(query_states.shape))

    if k_embed is None:
        k_embed = key_states_embed.view(key_states.shape)
    elif k_embed is not key_states:
        k_embed.copy_(key_states_embed.view(key_states.shape))

    return q_embed, k_embed


def apply_rotary_pos_emb_interleaved(
    query_states: Tensor,
    key_states: Tensor,
    cos: Tensor,
    sin: Tensor,
    q_embed: Tensor | None,
    k_embed: Tensor | None,
    return_native_layout: bool = True,
) -> tuple[Tensor, Tensor]:
    """Apply adjacent-pair RoPE through DLINFER's native vendor operator.

    return_native_layout: When true, keep the vendor's front/back-half output layout;
        when false, restore adjacent pairs for the public helper contract.
    """

    rope_dim = query_states.size(-1)
    num_tokens = query_states.size(1)
    q_heads = query_states.size(-2)
    k_heads = key_states.size(-2)

    query_native = query_states.reshape(-1, q_heads, 1, rope_dim)
    key_native = key_states.reshape(-1, k_heads, 1, rope_dim)
    native_input = torch.cat((query_native, key_native), dim=1)
    assert cos.size(0) == num_tokens
    cos = cos.reshape(num_tokens, 1, 1, rope_dim)
    sin = sin.reshape(num_tokens, 1, 1, rope_dim)

    output = ext_ops.apply_rotary_pos_emb_interleaved(
        native_input,
        cos,
        sin,
        return_native_layout,
    )

    query_states_embed = output[:, :q_heads].reshape(
        query_states.shape)
    key_states_embed = output[:, q_heads:].reshape(
        key_states.shape)

    if q_embed is None:
        q_embed = query_states_embed
    else:
        q_embed.copy_(query_states_embed)
    if k_embed is None:
        k_embed = key_states_embed
    else:
        k_embed.copy_(key_states_embed)
    return q_embed, k_embed
