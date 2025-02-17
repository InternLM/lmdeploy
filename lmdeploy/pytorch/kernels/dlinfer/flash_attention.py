# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from dlinfer.utils.type_annotation import Tensor


def flash_attention_fwd(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attn_output: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_start_loc: Tensor,
    kv_seqlens: Tensor,
    max_q_seqlen: int = None,
    window_size: int = None,
    sm_scale: float = None,
    logit_softcapping: float = None,
    causal: bool = True,
):
    num_q_heads = query_states.shape[1]
    num_kv_heads = value_states.shape[1]
    return ext_ops.prefill_attention(
        query_states,
        key_states,
        value_states,
        q_start_loc,
        q_seqlens,
        max_q_seqlen,
        num_q_heads,
        num_kv_heads,
        attn_mask=[],
        softmax_scale=sm_scale,
        attn_output=attn_output,
    )
