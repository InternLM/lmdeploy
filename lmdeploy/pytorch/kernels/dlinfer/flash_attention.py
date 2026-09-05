# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor


def flash_attention_fwd(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attn_output: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_start_loc: Tensor,
    kv_seqlens: Tensor,
    num_heads: int,
    num_kv_heads: int,
    actual_seq_lengths_cpu: Tensor,
    max_q_seqlen: int = None,
    window_size: int = None,
    sm_scale: float = None,
    logit_softcapping: float = None,
    causal: bool = True,
):
    actual_seq_lengths_cpu = (q_start_loc + q_seqlens).cpu()
    return ext_ops.prefill_attention(
        query_states,
        key_states,
        value_states,
        None,
        None,
        q_seqlens,
        kv_seqlens,
        max_q_seqlen,
        num_heads,
        num_kv_heads,
        attn_mask=[],
        softmax_scale=sm_scale,
        attn_output=attn_output,
        actual_seq_lengths_cpu=actual_seq_lengths_cpu,
    )
