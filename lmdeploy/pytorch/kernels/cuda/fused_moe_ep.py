# Copyright (c) OpenMMLab. All rights reserved.
# modify from dlblas: https://github.com/DeepLink-org/DLBlas
from typing import List, Optional

import torch
import triton
import triton.language as tl

from .activation import silu_and_mul


@triton.jit
def _fwd_kernel_ep_scatter_step1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)
    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)
    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)
    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )


@triton.jit
def _fwd_kernel_ep_scatter_step2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)
    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE
    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)
            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index.to(tl.int64)
                tl.store(output_index + token_id * output_index_stride0 + topk_index, dest_token_index)
                output_tensor_ptr = output_tensor + dest_token_index * output_tensor_stride0
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)


# copy from https://github.com/ModelTC/lightllm/blob/main/lightllm/common/fused_moe/deepep_scatter_gather.py
def ep_scatter(
    recv_x: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
):
    BLOCK_E = 128  # token num of per expert is aligned to 128
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    grid = num_experts
    assert m_indices.shape[0] % BLOCK_E == 0
    _fwd_kernel_ep_scatter_step1[(grid, )](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    grid = min(recv_topk.shape[0], 1024 * 8)
    _fwd_kernel_ep_scatter_step2[(grid, )](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
    )
    return


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)
    # align with xtuner rl
    compute_dtype = output_tensor.dtype.element_ty
    # compute_dtype = tl.float32

    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=compute_dtype)
        for topk_index in range(0, topk_num):
            expert_id = tl.load(recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index)
            if expert_id >= 0:
                source_token_index = tl.load(input_index + cur_token * input_index_stride0 + topk_index)
                acc_weight = tl.load(recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index)
                tmp = tl.load(input_tensor + source_token_index * input_tensor_stride0 + cur_block * BLOCK_D + off_d)
                accumulator += tmp.to(compute_dtype) * acc_weight.to(compute_dtype)
        tl.store(
            output_tensor + cur_token * output_tensor_stride0 + cur_block * BLOCK_D + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
):
    BLOCK_D = 1024  # block size of quantization
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))
    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return


def _deepgemm_grouped_bf16_nt_contiguous(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    m_indices: torch.Tensor,
):
    from lmdeploy.pytorch.third_party import deep_gemm
    return deep_gemm.m_grouped_bf16_gemm_nt_contiguous(x, w, out, m_indices)


def fused_moe_v3(
    hidden_states: torch.Tensor,
    topk_idx,
    topk_weights,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    num_recv_tokens_per_expert: Optional[List[int]],
):
    if num_recv_tokens_per_expert is None:
        return hidden_states
    all_tokens = sum(num_recv_tokens_per_expert)
    if all_tokens <= 0:
        return hidden_states
    M, K = hidden_states.size()
    N = w13_weight.size(1)
    gather_out = torch.empty_like(hidden_states)
    input_tensor = hidden_states.new_empty((all_tokens, K))
    m_indices = hidden_states.new_empty(all_tokens, dtype=torch.int32)
    output_index = torch.empty_like(topk_idx)
    num_recv_tokens_per_expert_gpu = torch.tensor(
        num_recv_tokens_per_expert,
        dtype=torch.int32,
        pin_memory=True,
        device='cpu',
    ).cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)
    ep_scatter(
        hidden_states,
        topk_idx,
        num_recv_tokens_per_expert_gpu,
        expert_start_loc,
        input_tensor,
        m_indices,
        output_index,
    )
    del hidden_states
    gateup_output = gather_out.new_empty((all_tokens, N))
    _deepgemm_grouped_bf16_nt_contiguous(input_tensor, w13_weight, gateup_output, m_indices)
    down_input = gateup_output.new_empty((
        all_tokens,
        N // 2,
    ))
    down_input = silu_and_mul(gateup_output.view(-1, N), down_input)
    down_output = gather_out.new_empty((all_tokens, K))
    _deepgemm_grouped_bf16_nt_contiguous(
        down_input,
        w2_weight,
        down_output,
        m_indices,
    )
    ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)
    return gather_out
