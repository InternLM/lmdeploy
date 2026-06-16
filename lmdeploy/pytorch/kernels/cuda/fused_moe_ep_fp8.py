# Copyright (c) OpenMMLab. All rights reserved.
# modify from dlblas: https://github.com/DeepLink-org/DLBlas
import torch
import triton
import triton.language as tl

from lmdeploy.pytorch.third_party.deep_gemm import get_mn_major_tma_aligned_tensor

from .activation import silu_and_mul
from .blocked_gemm_fp8 import per_token_group_quant_fp8
from .fused_moe_ep import ep_gather


@triton.jit
def _fwd_kernel_ep_scatter_fp8_step1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(num_recv_tokens_per_expert + offset_cumsum,
                                mask=offset_cumsum < num_experts,
                                other=0)
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)
    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)
    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)
    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(m_indices_start_ptr + start_m + off_expert, cur_expert)


@triton.jit
def _fwd_kernel_ep_scatter_fp8_step2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)
    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE
    offset_scale = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_scale = offset_scale < SCALE_HIDDEN_SIZE
    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        to_copy_scale = tl.load(recv_x_scale + token_id * recv_x_scale_stride0 + offset_scale, mask=mask_scale)
        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)
            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index.to(tl.int64)
                tl.store(output_index + token_id * output_index_stride0 + topk_index, dest_token_index)
                output_tensor_ptr = output_tensor + dest_token_index * output_tensor_stride0
                output_tensor_scale_ptr = output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                tl.store(output_tensor_scale_ptr + offset_scale, to_copy_scale, mask=mask_scale)


@torch.no_grad()
def ep_scatter_fp8(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
):
    block_e = 128
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    scale_hidden_size = recv_x_scale.shape[1]
    assert m_indices.shape[0] % block_e == 0
    _fwd_kernel_ep_scatter_fp8_step1[(num_experts, )](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )
    grid = min(recv_topk.shape[0], 1024 * 8)
    _fwd_kernel_ep_scatter_fp8_step2[(grid, )](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=scale_hidden_size,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(scale_hidden_size),
    )


def _deepgemm_grouped_fp8_nt_contiguous(input_tuple, w_tuple, out: torch.Tensor, m_indices: torch.Tensor):
    from lmdeploy.pytorch.third_party import deep_gemm
    return deep_gemm.m_grouped_fp8_gemm_nt_contiguous(input_tuple, w_tuple, out, m_indices)


def fused_moe_v3_fp8(
    hidden_states_fp8: tuple[torch.Tensor, torch.Tensor],
    topk_idx,
    topk_weights,
    w13_weight_fp8: tuple[torch.Tensor, torch.Tensor],
    w2_weight_fp8: tuple[torch.Tensor, torch.Tensor],
    num_recv_tokens_per_expert: list[int] | None,
):
    hidden_states_fp8, hidden_states_scale = hidden_states_fp8
    if num_recv_tokens_per_expert is None:
        return hidden_states_fp8.to(torch.bfloat16)
    all_tokens = sum(num_recv_tokens_per_expert)
    if all_tokens <= 0:
        return hidden_states_fp8.to(torch.bfloat16)
    m, k = hidden_states_fp8.size()
    n = w13_weight_fp8[0].size(1)
    block_size = k // hidden_states_scale.size(1)
    gather_out = torch.empty_like(hidden_states_fp8, device=hidden_states_fp8.device, dtype=torch.bfloat16)
    input_tensor = torch.empty((all_tokens, k), device=hidden_states_fp8.device, dtype=hidden_states_fp8.dtype)
    input_tensor_scale = torch.empty((all_tokens, k // block_size),
                                    device=hidden_states_fp8.device,
                                    dtype=torch.float32)
    m_indices = torch.empty(all_tokens, device=hidden_states_fp8.device, dtype=torch.int32)
    output_index = torch.empty_like(topk_idx)
    num_recv_tokens_per_expert_gpu = torch.tensor(num_recv_tokens_per_expert,
                                                 dtype=torch.int32,
                                                 pin_memory=True,
                                                 device='cpu').cuda(non_blocking=True)
    expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)
    ep_scatter_fp8(hidden_states_fp8, hidden_states_scale, topk_idx, num_recv_tokens_per_expert_gpu, expert_start_loc,
                   input_tensor, input_tensor_scale, m_indices, output_index)
    del hidden_states_fp8

    gateup_output = torch.empty((all_tokens, n), device=gather_out.device, dtype=torch.bfloat16)
    input_tensor_scale = get_mn_major_tma_aligned_tensor(input_tensor_scale)
    _deepgemm_grouped_fp8_nt_contiguous((input_tensor, input_tensor_scale), w13_weight_fp8, gateup_output, m_indices)

    down_input = torch.empty((all_tokens, n // 2), device=gateup_output.device, dtype=torch.bfloat16)
    silu_and_mul(gateup_output.view(-1, n), down_input)
    del gateup_output
    down_input_fp8, down_input_scale = per_token_group_quant_fp8(down_input, block_size)
    down_input_scale = get_mn_major_tma_aligned_tensor(down_input_scale)
    down_output = torch.empty((all_tokens, k), device=gather_out.device, dtype=torch.bfloat16)
    _deepgemm_grouped_fp8_nt_contiguous((down_input_fp8, down_input_scale), w2_weight_fp8, down_output, m_indices)
    ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)
    return gather_out
