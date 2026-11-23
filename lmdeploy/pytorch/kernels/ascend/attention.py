# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch_npu

from lmdeploy.pytorch.kernels.ascend.types import Tensor


def decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    scale_value: float,
    block_table: Tensor,
    block_size: int,
    actual_q_seqlens_cpu: Tensor,
    kv_seqlens_cpu: Tensor,
    softmax_scale: float,
    attn_output: Tensor,
):
    _, _, dim = query.shape
    block_num = key_cache.size(0)
    query = query.contiguous()
    attn_output = attn_output.contiguous()
    key_cache = key_cache.view(block_num, block_size, -1)
    value_cache = value_cache.view(block_num, block_size, -1)
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(dim)

    attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
        query=query,
        key=key_cache,
        value=value_cache,
        atten_mask=None,
        block_table=block_table,
        input_layout='TND',
        block_size=block_size,
        actual_seq_lengths=actual_q_seqlens_cpu,
        actual_seq_lengths_kv=kv_seqlens_cpu,
        num_key_value_heads=num_kv_heads,
        num_heads=num_q_heads,
        scale=scale_value,
        sparse_mode=0,
    )
    return attn_output


def decode_attention_mla(
    query: Tensor,
    key_cache: Tensor,
    num_kv_heads: int,
    num_q_heads: int,
    scale_value: float,
    block_table: Tensor,
    kv_seqlens_cpu: Tensor,
    mla_vheadsize: int,
    attn_output: Tensor,
):
    num_tokens = query.shape[0]
    _, block_size = key_cache.shape[:2]

    q_nope = query[..., :mla_vheadsize].view(num_tokens, num_q_heads, 1, mla_vheadsize).contiguous()
    q_rope = query[..., mla_vheadsize:].view(num_tokens, num_q_heads, 1, -1)

    # FIA v2 expects paged KV cache in [block, kv_head, block_size, dim].
    key_cache = key_cache.permute(0, 2, 1, 3)
    k_nope = key_cache[..., :mla_vheadsize]
    k_rope = key_cache[..., mla_vheadsize:]

    fai_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        q_nope,
        k_nope,
        k_nope,
        query_rope=q_rope,
        key_rope=k_rope,
        num_query_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        input_layout='BNSD_NBSD',
        atten_mask=None,
        sparse_mode=0,
        softmax_scale=scale_value,
        block_table=block_table,
        block_size=block_size,
        actual_seq_qlen=None,
        actual_seq_kvlen=kv_seqlens_cpu,
    )

    attn_output.copy_(fai_output.squeeze(2).transpose(0, 1))
    return attn_output
