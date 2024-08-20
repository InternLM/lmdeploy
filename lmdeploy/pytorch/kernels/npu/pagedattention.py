# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch_npu
from torch import Tensor


def flash_context_attention(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    attn_output: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: list,
    kv_seqlens: list,
    block_size: int,
    kv_cache_len: int,
    context=None,
):
    batch, head, dim = (
        q_start_loc.shape[0],
        query_states.shape[1],
        query_states.shape[2],
    )
    numKeyValueHeads = value_states.shape[1]
    assert key_states.shape[1] == value_states.shape[1]
    for i in range(batch):
        start = q_start_loc[i]
        end = start + q_seqlens[i]
        single_seqlen = int(end - start)
        single_q = query_states[start:end].view(1, single_seqlen, -1)
        single_k = key_states[start:end].reshape(1, single_seqlen, -1)
        single_v = value_states[start:end].reshape(1, single_seqlen, -1)
        single_out = attn_output[start:end, :].view(1, single_seqlen, -1)
        mask = context.attention_mask[i]
        if q_seqlens[i] == kv_seqlens[i]:
            new_single_out = torch_npu.npu_prompt_flash_attention(
                single_q,
                single_k,
                single_v,
                atten_mask=mask,
                actual_seq_lengths=[q_seqlens[i]],
                actual_seq_lengths_kv=[kv_seqlens[i]],
                num_heads=head,
                num_key_value_heads=numKeyValueHeads,
                scale_value=1 / math.sqrt(float(dim)),
            )
            single_out.copy_(new_single_out)
        else:
            key_cache = key_cache.reshape(1, kv_cache_len,
                                          numKeyValueHeads * dim)
            value_cache = value_cache.reshape(1, kv_cache_len,
                                              numKeyValueHeads * dim)
            for j in range(q_seqlens[i]):
                single_q = query_states[start + j:start + j + 1].view(1, 1, -1)
                single_out = attn_output[start + j:start + j + 1].view(
                    1, 1, -1)
                new_single_out = torch_npu.npu_incre_flash_attention(
                    single_q,
                    key_cache,
                    value_cache,
                    atten_mask=mask[j:j + 1],
                    actual_seq_lengths=[kv_seqlens[i]],
                    num_heads=head,
                    num_key_value_heads=numKeyValueHeads,
                    scale_value=1 / math.sqrt(float(dim)),
                    block_table=block_offsets[i:i + 1],
                    block_size=block_size,
                )
                single_out.copy_(new_single_out)


def paged_token_attention(q, k_cache, v_cache, attn_output, kv_seqlens,
                          block_table, block_size):
    numKeyValueHeads = k_cache.shape[1]
    assert k_cache.shape[1] == v_cache.shape[1]
    bs, head, dim = q.shape
    kv_cache_len = k_cache.shape[0]
    q = q.reshape(bs, 1, head * dim)
    k_cache = k_cache.reshape(1, kv_cache_len, numKeyValueHeads * dim)
    v_cache = v_cache.reshape(1, kv_cache_len, numKeyValueHeads * dim)
    attn_output = attn_output.view(q.shape)
    new_attn_output = torch_npu.npu_incre_flash_attention(
        q,
        k_cache,
        v_cache,
        atten_mask=None,
        actual_seq_lengths=kv_seqlens,
        num_heads=head,
        num_key_value_heads=numKeyValueHeads,
        scale_value=1 / math.sqrt(float(dim)),
        block_table=block_table,
        block_size=block_size,
    )
    attn_output.copy_(new_attn_output)


def paged_attention_fwd(
    query_states: Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    attn_output: Tensor,
    block_offsets: Tensor,
    q_start_loc: Tensor,
    q_seqlens: Tensor,
    kv_seqlens: Tensor,
    max_seqlen: int,
    window_size: int = 1,
    context=None,
):
    is_decoding = query_states.shape[-3] == q_seqlens.size(0)
    block_num, block_size, head, dim = key_cache.size()
    kv_cache_len = block_num * block_size
    k = key_cache.reshape(block_num * block_size, head, dim)
    v = value_cache.reshape(block_num * block_size, head, dim)
    if not is_decoding:
        flash_context_attention(
            query_states,
            key_states,
            value_states,
            attn_output,
            k,
            v,
            block_offsets.to(torch.int32),
            q_start_loc,
            q_seqlens.tolist(),
            kv_seqlens.tolist(),
            block_size,
            kv_cache_len,
            context=context,
        )
    else:
        paged_token_attention(
            query_states,
            k,
            v,
            attn_output,
            kv_seqlens.tolist(),
            block_offsets.to(torch.int32),
            block_size,
        )
