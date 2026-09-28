# Copyright (c) OpenMMLab. All rights reserved.
"""Direct compressed-tensors W4A16 MoE kernels."""

from functools import cache

import torch
import triton
import triton.language as tl

from .activation import silu_and_mul
from .moe.fused_moe import _get_sorted_idx, _get_sorted_idx_blocks, _make_intermediate, _renormalize, moe_reduce


@triton.jit
def _fused_moe_w4a16_kernel(
    A,
    B,
    S,
    C,
    SortedIdx,
    ExpStart,
    ExpEnd,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    M_NP2: tl.constexpr,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    top_k: tl.constexpr,
    reindex_a: tl.constexpr,
    reindex_c: tl.constexpr,
):
    """Multiply routed activations by offset-binary INT4 weights."""
    exp_id = tl.program_id(1)
    pid = tl.program_id(0)

    exp_start = tl.load(ExpStart + exp_id)
    exp_end = tl.load(ExpEnd + exp_id)
    routed_m = exp_end - exp_start
    if routed_m <= 0:
        return

    num_pid_m = tl.cdiv(M_NP2, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    if pid_m * BLOCK_SIZE_M >= routed_m or pid_n * BLOCK_SIZE_N >= N:
        return

    offs_sid = exp_start + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_sid < exp_end
    sid = tl.load(SortedIdx + offs_sid, mask=mask_m, other=0)
    offs_am = sid // top_k if reindex_a else offs_sid
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pack_factor: tl.constexpr = 32 // NUM_BITS
    packed_block_k: tl.constexpr = BLOCK_SIZE_K // pack_factor
    code_mask: tl.constexpr = (1 << NUM_BITS) - 1
    signed_offset: tl.constexpr = 1 << (NUM_BITS - 1)
    shifts = tl.arange(0, pack_factor) * NUM_BITS
    packed_lanes = tl.arange(0, packed_block_k)
    exp_id_i64 = exp_id.to(tl.int64)

    for block_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = block_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # A packed word contains eight adjacent INT4 values and one scale is
        # shared by the complete K=32 tile.  Load each unique value once
        # instead of issuing the same packed/scale address 8/32 times.
        offs_packed_k = block_k * packed_block_k + packed_lanes
        packed = tl.load(
            B + exp_id_i64 * stride_be +
            offs_n[:, None] * stride_bn +
            offs_packed_k[None, :] * stride_bk,
            mask=mask_n[:, None],
            other=0,
        )
        codes = (packed[:, :, None] >> shifts[None, None, :]) & code_mask
        codes = tl.reshape(codes, (BLOCK_SIZE_N, BLOCK_SIZE_K))
        signed_codes = codes - signed_offset

        group_k = block_k * BLOCK_SIZE_K // GROUP_SIZE
        scales = tl.load(
            S + exp_id_i64 * stride_se + offs_n * stride_sn +
            group_k * stride_sg,
            mask=mask_n,
            other=0.0,
        )
        b = (
            signed_codes.to(tl.float32) *
            scales[:, None].to(tl.float32)
        ).to(A.dtype.element_ty)
        accumulator = tl.dot(a, tl.trans(b), acc=accumulator)

    output = accumulator.to(C.dtype.element_ty)
    offs_cm = sid if reindex_c else offs_sid
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, output, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _fused_moe_w4a16_compact_kernel(
    A,
    B,
    S,
    C,
    SortedIdx,
    ExpEnd,
    BlockEnd,
    BlockExpertIds,
    BlockOffsets,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,
    reindex_a: tl.constexpr,
    reindex_c: tl.constexpr,
):
    """Multiply one compact routed block by offset-binary INT4 weights."""
    block_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_blocks = tl.load(BlockEnd + num_experts - 1)
    if block_id >= total_blocks:
        return

    exp_id = tl.load(BlockExpertIds + block_id)
    block_sorted_start = tl.load(BlockOffsets + block_id)
    exp_end = tl.load(ExpEnd + exp_id)

    offs_sid = block_sorted_start + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_sid < exp_end
    sid = tl.load(SortedIdx + offs_sid, mask=mask_m, other=0)
    offs_am = sid // top_k if reindex_a else offs_sid
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    pack_factor: tl.constexpr = 32 // NUM_BITS
    packed_block_k: tl.constexpr = BLOCK_SIZE_K // pack_factor
    code_mask: tl.constexpr = (1 << NUM_BITS) - 1
    signed_offset: tl.constexpr = 1 << (NUM_BITS - 1)
    shifts = tl.arange(0, pack_factor) * NUM_BITS
    packed_lanes = tl.arange(0, packed_block_k)
    exp_id_i64 = exp_id.to(tl.int64)

    for block_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = block_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        offs_packed_k = block_k * packed_block_k + packed_lanes
        packed = tl.load(
            B + exp_id_i64 * stride_be +
            offs_n[:, None] * stride_bn +
            offs_packed_k[None, :] * stride_bk,
            mask=mask_n[:, None],
            other=0,
        )
        codes = (packed[:, :, None] >> shifts[None, None, :]) & code_mask
        codes = tl.reshape(codes, (BLOCK_SIZE_N, BLOCK_SIZE_K))
        signed_codes = codes - signed_offset

        group_k = block_k * BLOCK_SIZE_K // GROUP_SIZE
        scales = tl.load(
            S + exp_id_i64 * stride_se + offs_n * stride_sn +
            group_k * stride_sg,
            mask=mask_n,
            other=0.0,
        )
        b = (
            signed_codes.to(tl.float32) *
            scales[:, None].to(tl.float32)
        ).to(A.dtype.element_ty)
        accumulator = tl.dot(a, tl.trans(b), acc=accumulator)

    output = accumulator.to(C.dtype.element_ty)
    offs_cm = sid if reindex_c else offs_sid
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, output, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _fused_moe_w4a16_route_kernel(
    A,
    B,
    S,
    C,
    RouteExpertIds,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_se: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_sg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    TOP_K: tl.constexpr,
    REINDEX_A: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BITS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
):
    """Multiply one route without sorting or launching empty experts."""
    route_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = tl.load(RouteExpertIds + route_id).to(tl.int64)
    valid_route = (expert_id >= 0) & (expert_id < NUM_EXPERTS)
    expert_id = tl.where(valid_route, expert_id, 0)
    input_row = route_id // TOP_K if REINDEX_A else route_id

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m == 0
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), tl.float32)

    pack_factor: tl.constexpr = 32 // NUM_BITS
    packed_block_k: tl.constexpr = BLOCK_SIZE_K // pack_factor
    code_mask: tl.constexpr = (1 << NUM_BITS) - 1
    signed_offset: tl.constexpr = 1 << (NUM_BITS - 1)
    shifts = tl.arange(0, pack_factor) * NUM_BITS
    packed_lanes = tl.arange(0, packed_block_k)

    for block_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = block_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        activation = tl.load(
            A + input_row * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        offs_packed_k = block_k * packed_block_k + packed_lanes
        packed = tl.load(
            B + expert_id * stride_be +
            offs_n[:, None] * stride_bn +
            offs_packed_k[None, :] * stride_bk,
            mask=valid_route & mask_n[:, None],
            other=0,
        )
        codes = (packed[:, :, None] >> shifts[None, None, :]) & code_mask
        codes = tl.reshape(codes, (BLOCK_SIZE_N, BLOCK_SIZE_K))
        signed_codes = codes - signed_offset

        group_k = block_k * BLOCK_SIZE_K // GROUP_SIZE
        scales = tl.load(
            S + expert_id * stride_se + offs_n * stride_sn +
            group_k * stride_sg,
            mask=valid_route & mask_n,
            other=0.0,
        )
        weight = (
            signed_codes.to(tl.float32) *
            scales[:, None].to(tl.float32)
        ).to(A.dtype.element_ty)
        accumulator = tl.dot(
            activation,
            tl.trans(weight),
            acc=accumulator,
        )

    # Only row zero is real.  Keeping a 16-row tile lets the kernel use tensor
    # cores while still assigning exactly one CTA row to each routed expert.
    tl.store(
        C + route_id * stride_cm + offs_m[:, None] * 0 +
        offs_n[None, :] * stride_cn,
        accumulator.to(C.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@cache
def _is_hopper_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index)[0] >= 9


def _use_hopper_w4a16(device: torch.device) -> bool:
    """Return whether H100/H200-specific launch tuning is available."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _is_hopper_device(device_index)


def _w4a16_block_m(num_tokens: int,
                   num_routes: int | None = None,
                   num_experts: int | None = None,
                   prefer_small_tiles: bool = False) -> int:
    """Choose the routed-M tile shared by sorting and GEMM launch."""
    m_np2 = max(16, triton.next_power_of_2(num_tokens))
    if (prefer_small_tiles and num_routes is not None
            and num_experts is not None
            and num_routes < 15 * num_experts):
        return 16
    return 16 if m_np2 <= 32 else 32


def _should_use_compact_w4a16(num_tokens: int, num_routes: int,
                              num_experts: int,
                              prefer_small_tiles: bool = False) -> bool:
    """Use compact scheduling only when it reduces routed-M block capacity."""
    block_m = _w4a16_block_m(num_tokens, num_routes, num_experts,
                             prefer_small_tiles)
    m_np2 = max(16, triton.next_power_of_2(num_tokens))
    origin_blocks = num_experts * triton.cdiv(m_np2, block_m)
    compact_blocks = triton.cdiv(num_routes, block_m) + num_experts
    return compact_blocks < origin_blocks


def _should_use_route_w4a16(num_routes: int,
                            num_experts: int,
                            allow_invalid_routes: bool = False) -> bool:
    """Use route-major scheduling for small, ordinary TP decode batches."""
    return (not allow_invalid_routes and num_routes <= 64
            and num_routes < num_experts)


def fused_moe_w4a16_route_launcher(
    hidden_states: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    route_expert_ids: torch.Tensor,
    top_k: int,
    reindex_a: bool,
    num_bits: int = 4,
    group_size: int = 32,
):
    """Launch a graph-safe route-major W4A16 projection."""
    if num_bits != 4 or group_size != 32:
        raise ValueError(
            f'Only INT4 group-size 32 is supported, got bits={num_bits}, group_size={group_size}'
        )
    if hidden_states.dim() < 2 or output.dim() < 2:
        raise ValueError(
            'Activations and output must have at least two dimensions')
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f'hidden_states must be float16 or bfloat16, got {hidden_states.dtype}'
        )
    if output.dtype != hidden_states.dtype:
        raise ValueError(
            f'Output dtype must match activation dtype, got {output.dtype} and {hidden_states.dtype}'
        )
    if weight_packed.dtype != torch.int32 or weight_scale.dtype != torch.bfloat16:
        raise ValueError(
            'W4A16 kernel requires int32 packed weights and bfloat16 scales')
    if weight_packed.dim() != 3 or weight_scale.dim() != 3:
        raise ValueError(
            'W4A16 MoE weights and scales must be 3D [E, N, packed/grouped K] tensors'
        )
    if weight_packed.shape[:2] != weight_scale.shape[:2]:
        raise ValueError(
            'Packed weight and scale expert/output dimensions must match')
    if route_expert_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f'route_expert_ids must be int32 or int64, got {route_expert_ids.dtype}'
        )
    if any(tensor.device != hidden_states.device
           for tensor in (weight_packed, weight_scale, output,
                          route_expert_ids)):
        raise ValueError(
            'Activations, weights, output, and routes must be on the same device'
        )
    if not output.is_contiguous():
        raise ValueError('Output must be contiguous')
    if top_k < 1:
        raise ValueError(f'top_k must be positive, got {top_k}')

    pack_factor = 32 // num_bits
    num_experts, out_features, packed_k = weight_packed.shape
    in_features = weight_scale.shape[-1] * group_size
    if packed_k * pack_factor != in_features:
        raise ValueError(
            'Packed and scale K dimensions describe different logical weights')
    if hidden_states.shape[-1] != in_features:
        raise ValueError(
            f'Activation K={hidden_states.shape[-1]} does not match weight K={in_features}'
        )
    if output.shape[-1] != out_features:
        raise ValueError(
            f'Output N={output.shape[-1]} does not match weight N={out_features}'
        )

    hidden_states = hidden_states.flatten(0, -2)
    output = output.flatten(0, -2)
    route_expert_ids = route_expert_ids.reshape(-1)
    num_routes = output.size(0)
    if route_expert_ids.numel() != num_routes:
        raise ValueError(
            f'route_expert_ids has {route_expert_ids.numel()} routes but output has {num_routes}'
        )
    if reindex_a and num_routes % top_k != 0:
        raise ValueError(
            f'Route count {num_routes} must be divisible by top_k={top_k}')
    expected_rows = num_routes // top_k if reindex_a else num_routes
    if hidden_states.size(0) != expected_rows:
        raise ValueError(
            f'Activation has {hidden_states.size(0)} rows but route projection expects {expected_rows}'
        )

    is_hopper = _use_hopper_w4a16(hidden_states.device)
    if in_features >= 1024:
        block_n, num_warps = 32, 4
    else:
        block_n = 64
        num_warps = 2 if is_hopper else 4
    grid = (num_routes, triton.cdiv(out_features, block_n))
    _fused_moe_w4a16_route_kernel[grid](
        hidden_states,
        weight_packed,
        weight_scale,
        output,
        route_expert_ids,
        N=out_features,
        K=in_features,
        stride_am=hidden_states.stride(0),
        stride_ak=hidden_states.stride(1),
        stride_be=weight_packed.stride(0),
        stride_bn=weight_packed.stride(1),
        stride_bk=weight_packed.stride(2),
        stride_se=weight_scale.stride(0),
        stride_sn=weight_scale.stride(1),
        stride_sg=weight_scale.stride(2),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        TOP_K=top_k,
        REINDEX_A=reindex_a,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=32,
        NUM_BITS=num_bits,
        GROUP_SIZE=group_size,
        NUM_EXPERTS=num_experts,
        num_warps=num_warps,
        num_stages=3,
    )


def fused_moe_w4a16_kernel_launcher(
    hidden_states: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    sorted_idx: torch.Tensor,
    exp_start: torch.Tensor,
    exp_end: torch.Tensor,
    top_k: int = 1,
    num_tokens: int | None = None,
    reindex_a: bool = True,
    reindex_c: bool = True,
    num_bits: int = 4,
    group_size: int = 32,
    block_end: torch.Tensor | None = None,
    block_expert_ids: torch.Tensor | None = None,
    block_offsets: torch.Tensor | None = None,
    block_m: int | None = None,
):
    """Launch one routed W4A16 GEMM directly from checkpoint layout."""
    if num_tokens is None:
        num_tokens = hidden_states.size(0)
    if num_bits != 4 or group_size != 32:
        raise ValueError(
            f'Only INT4 group-size 32 is supported, got bits={num_bits}, group_size={group_size}'
        )
    if hidden_states.dim() < 2 or output.dim() < 2:
        raise ValueError(
            'Activations and output must have at least two dimensions')
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f'hidden_states must be float16 or bfloat16, got {hidden_states.dtype}'
        )
    if output.dtype != hidden_states.dtype:
        raise ValueError(
            f'Output dtype must match activation dtype, got {output.dtype} and {hidden_states.dtype}'
        )
    if weight_packed.dtype != torch.int32 or weight_scale.dtype != torch.bfloat16:
        raise ValueError(
            'W4A16 kernel requires int32 packed weights and bfloat16 scales')
    if weight_packed.dim() != 3 or weight_scale.dim() != 3:
        raise ValueError(
            'W4A16 MoE weights and scales must be 3D [E, N, packed/grouped K] tensors'
        )
    if weight_packed.shape[:2] != weight_scale.shape[:2]:
        raise ValueError(
            'Packed weight and scale expert/output dimensions must match')
    if any(tensor.device != hidden_states.device
           for tensor in (weight_packed, weight_scale, output, sorted_idx,
                          exp_start, exp_end)):
        raise ValueError(
            'Activations, weights, output, and routing metadata must be on the same device'
        )
    if sorted_idx.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f'sorted_idx must be int32 or int64, got {sorted_idx.dtype}')
    if exp_start.dtype not in (
            torch.int32, torch.int64) or exp_end.dtype != exp_start.dtype:
        raise ValueError(
            'exp_start and exp_end must have the same int32 or int64 dtype')
    if not output.is_contiguous():
        raise ValueError('Output must be contiguous')
    if top_k < 1 or num_tokens < 1:
        raise ValueError(
            f'top_k and num_tokens must be positive, got {top_k} and {num_tokens}'
        )

    pack_factor = 32 // num_bits
    num_experts, out_features, packed_k = weight_packed.shape
    in_features = weight_scale.shape[-1] * group_size
    if packed_k * pack_factor != in_features:
        raise ValueError(
            'Packed and scale K dimensions describe different logical weights')
    if hidden_states.shape[-1] != in_features:
        raise ValueError(
            f'Activation K={hidden_states.shape[-1]} does not match weight K={in_features}'
        )
    if output.shape[-1] != out_features:
        raise ValueError(
            f'Output N={output.shape[-1]} does not match weight N={out_features}'
        )
    num_routes = output.numel() // out_features
    if exp_start.numel() != num_experts or exp_end.numel() != num_experts:
        raise ValueError(
            'Expert routing metadata does not match the number of local experts'
        )
    if sorted_idx.numel() != num_routes:
        raise ValueError(
            f'sorted_idx has {sorted_idx.numel()} routes but output has {num_routes}'
        )
    if num_tokens > hidden_states.numel() // in_features:
        raise ValueError('num_tokens exceeds the number of activation rows')

    compact_meta = (block_end, block_expert_ids, block_offsets)
    use_compact = any(tensor is not None for tensor in compact_meta)
    if use_compact:
        if not all(tensor is not None for tensor in compact_meta):
            raise ValueError(
                'Compact routing requires block_end, block_expert_ids, and block_offsets'
            )
        if block_end.dim() != 1 or block_end.numel() != num_experts:
            raise ValueError(
                'block_end must contain one cumulative block count per expert')
        if (block_expert_ids.dim() != 1 or block_offsets.dim() != 1
                or block_expert_ids.numel() != block_offsets.numel()):
            raise ValueError(
                'Compact block expert ids and offsets must be matching 1D tensors'
            )
        if any(tensor.device != hidden_states.device
               for tensor in compact_meta):
            raise ValueError(
                'Compact routing metadata must be on the activation device')
        if any(tensor.dtype != exp_end.dtype for tensor in compact_meta):
            raise ValueError(
                'Compact routing metadata must have the routing index dtype')
        prefer_small_tiles = _use_hopper_w4a16(hidden_states.device)
        expected_block_m = _w4a16_block_m(num_tokens, num_routes,
                                           num_experts,
                                           prefer_small_tiles)
        if block_m is None:
            block_m = expected_block_m
        if block_m != expected_block_m:
            raise ValueError(
                f'block_m={block_m} does not match the routing tile {expected_block_m}'
            )

    hidden_states = hidden_states.flatten(0, -2)
    output = output.flatten(0, -2)
    m_np2 = max(16, triton.next_power_of_2(num_tokens))
    prefer_small_tiles = _use_hopper_w4a16(hidden_states.device)
    if block_m is None:
        block_m = _w4a16_block_m(num_tokens, num_routes, num_experts,
                                  prefer_small_tiles)
    block_n = 64
    block_k = 32
    if use_compact:
        grid = (block_expert_ids.numel(), triton.cdiv(out_features, block_n))
        _fused_moe_w4a16_compact_kernel[grid](
            hidden_states,
            weight_packed,
            weight_scale,
            output,
            sorted_idx,
            exp_end,
            block_end,
            block_expert_ids,
            block_offsets,
            N=out_features,
            K=in_features,
            stride_am=hidden_states.stride(0),
            stride_ak=hidden_states.stride(1),
            stride_be=weight_packed.stride(0),
            stride_bn=weight_packed.stride(1),
            stride_bk=weight_packed.stride(2),
            stride_se=weight_scale.stride(0),
            stride_sn=weight_scale.stride(1),
            stride_sg=weight_scale.stride(2),
            stride_cm=output.stride(0),
            stride_cn=output.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            NUM_BITS=num_bits,
            GROUP_SIZE=group_size,
            num_experts=num_experts,
            top_k=top_k,
            reindex_a=reindex_a,
            reindex_c=reindex_c,
            num_warps=2 if prefer_small_tiles else 4,
            num_stages=3,
        )
        return

    grid = (triton.cdiv(m_np2, block_m) * triton.cdiv(out_features, block_n),
            num_experts)
    _fused_moe_w4a16_kernel[grid](
        hidden_states,
        weight_packed,
        weight_scale,
        output,
        sorted_idx,
        exp_start,
        exp_end,
        N=out_features,
        K=in_features,
        stride_am=hidden_states.stride(0),
        stride_ak=hidden_states.stride(1),
        stride_be=weight_packed.stride(0),
        stride_bn=weight_packed.stride(1),
        stride_bk=weight_packed.stride(2),
        stride_se=weight_scale.stride(0),
        stride_sn=weight_scale.stride(1),
        stride_sg=weight_scale.stride(2),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        M_NP2=m_np2,
        NUM_BITS=num_bits,
        GROUP_SIZE=group_size,
        top_k=top_k,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        num_warps=2 if prefer_small_tiles else 4,
        num_stages=3,
    )


def fused_moe_w4a16_masked(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    masked_m: torch.Tensor,
    num_bits: int = 4,
    group_size: int = 32,
) -> torch.Tensor:
    """Run fixed-capacity, unweighted local experts for DeepEP decode.

    ``hidden_states`` uses DeepEP's low-latency layout
    ``[local_experts, capacity, hidden]``. Only the first ``masked_m[e]``
    rows of expert ``e`` are valid. The returned BF16 tensor keeps the same
    fixed layout; DeepEP applies global router weights while combining it.
    """
    if hidden_states.dim() != 3:
        raise ValueError(
            'masked W4A16 hidden_states must be [experts, capacity, hidden]')
    if not hidden_states.is_contiguous():
        raise ValueError('masked W4A16 hidden_states must be contiguous')
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(
            f'DeepEP masked W4A16 requires bfloat16 activations, got {hidden_states.dtype}'
        )
    num_experts, capacity, hidden_dim = hidden_states.shape
    if num_experts < 1 or capacity < 1:
        raise ValueError(
            f'masked W4A16 requires positive experts and capacity, got {hidden_states.shape}'
        )
    if masked_m.dim() != 1 or masked_m.numel() != num_experts:
        raise ValueError(
            f'masked_m must have one count per local expert ({num_experts})')
    if masked_m.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f'masked_m must be int32 or int64, got {masked_m.dtype}')
    if masked_m.device != hidden_states.device:
        raise ValueError(
            'masked_m and hidden_states must be on the same device')
    local_weight_tensors = (
        gate_up_packed,
        gate_up_scale,
        down_packed,
        down_scale,
    )
    if any(tensor.shape[0] != num_experts
           for tensor in local_weight_tensors):
        raise ValueError(
            'masked W4A16 weights must match the local expert count')
    if gate_up_packed.shape[1] % 2 != 0:
        raise ValueError('Gate-up output dimension must be even')
    ffn_dim = gate_up_packed.shape[1] // 2
    if down_packed.shape[1] != hidden_dim:
        raise ValueError('Down output dimension must match hidden size')
    if down_scale.shape[-1] * group_size != ffn_dim:
        raise ValueError(
            'Down input dimension must match half of gate-up output')

    route_count = num_experts * capacity
    sorted_idx = torch.arange(
        route_count,
        dtype=masked_m.dtype,
        device=hidden_states.device,
    )
    exp_start = torch.arange(
        num_experts,
        dtype=masked_m.dtype,
        device=hidden_states.device,
    ) * capacity
    exp_end = exp_start + masked_m.clamp(min=0, max=capacity)

    gate_up = hidden_states.new_zeros(
        (num_experts, capacity, 2 * ffn_dim))
    fused_moe_w4a16_kernel_launcher(
        hidden_states,
        gate_up_packed,
        gate_up_scale,
        gate_up,
        sorted_idx,
        exp_start,
        exp_end,
        top_k=1,
        # This controls the routed-M launch capacity per expert. The routing
        # indices still address the full flattened [E, capacity] tensor.
        num_tokens=capacity,
        reindex_a=True,
        reindex_c=True,
        num_bits=num_bits,
        group_size=group_size,
    )

    activated = silu_and_mul(
        gate_up.flatten(0, 1)).unflatten(0, (num_experts, capacity))
    expert_output = hidden_states.new_zeros(
        (num_experts, capacity, hidden_dim))
    fused_moe_w4a16_kernel_launcher(
        activated,
        down_packed,
        down_scale,
        expert_output,
        sorted_idx,
        exp_start,
        exp_end,
        top_k=1,
        num_tokens=capacity,
        reindex_a=True,
        reindex_c=True,
        num_bits=num_bits,
        group_size=group_size,
    )
    return expert_output


def fused_moe_w4a16(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    renormalize: bool = False,
    num_bits: int = 4,
    group_size: int = 32,
    allow_invalid_routes: bool = False,
) -> torch.Tensor:
    """Run an eager routed MoE without materializing BF16 expert weights."""
    if hidden_states.dim() != 2:
        raise ValueError(
            f'hidden_states must be 2D [tokens, hidden], got {hidden_states.shape}'
        )
    if topk_ids.dim() != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            'topk_weights and topk_ids must have the same 2D shape')
    if topk_ids.size(0) != hidden_states.size(0) or topk_ids.size(1) != topk:
        raise ValueError(
            'Routing dimensions must match the token count and top_k')
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f'topk_ids must be int32 or int64, got {topk_ids.dtype}')
    if topk_weights.device != hidden_states.device or topk_ids.device != hidden_states.device:
        raise ValueError(
            'Activations and routing tensors must be on the same device')
    if gate_up_packed.shape[0] != down_packed.shape[0]:
        raise ValueError(
            'Gate-up and down weights must contain the same number of experts')
    if gate_up_packed.shape[1] % 2 != 0:
        raise ValueError('Gate-up output dimension must be even')
    ffn_dim = gate_up_packed.shape[1] // 2
    hidden_dim = hidden_states.shape[-1]
    if down_packed.shape[1] != hidden_dim:
        raise ValueError('Down output dimension must match hidden size')
    if down_scale.shape[-1] * group_size != ffn_dim:
        raise ValueError(
            'Down input dimension must match half of gate-up output')

    num_tokens = hidden_states.size(0)
    num_experts = gate_up_packed.size(0)
    if topk > num_experts and not allow_invalid_routes:
        raise ValueError(
            f'top_k={topk} cannot exceed num_experts={num_experts}')
    if num_tokens == 0:
        return hidden_states.new_empty((0, hidden_dim))
    if allow_invalid_routes and renormalize:
        raise ValueError(
            'Invalid-route masking cannot renormalize an EP-local route subset')
    valid_routes = None
    routing_topk_ids = topk_ids
    routing_num_experts = num_experts
    if allow_invalid_routes:
        valid_routes = (topk_ids >= 0) & (topk_ids < num_experts)
        topk_weights = torch.where(
            valid_routes,
            topk_weights,
            torch.zeros((), dtype=topk_weights.dtype, device=topk_weights.device),
        )
        # Keep invalid DeepEP-local routes away from the shared sorter.  A
        # private sentinel expert preserves the flattened route positions and
        # is excluded from the local W4 launch metadata.
        routing_topk_ids = torch.where(
            valid_routes,
            topk_ids,
            torch.full_like(topk_ids, num_experts),
        ).reshape(-1, 1)
        routing_num_experts += 1
    topk_weights = _renormalize(topk_weights, renormalize)
    num_routes = topk_ids.numel()
    if _should_use_route_w4a16(num_routes, num_experts,
                               allow_invalid_routes):
        # Decode has very few active routes relative to Kimi's 384 experts.
        # Read dynamic expert ids in-device so CUDA Graph replay can change
        # routing without rebuilding metadata or launching empty expert CTAs.
        route_expert_ids = topk_ids.reshape(-1)
        gate_up = _make_intermediate(
            (num_tokens, topk, ffn_dim * 2),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            zeros=False,
        )
        fused_moe_w4a16_route_launcher(
            hidden_states,
            gate_up_packed,
            gate_up_scale,
            gate_up,
            route_expert_ids,
            top_k=topk,
            reindex_a=True,
            num_bits=num_bits,
            group_size=group_size,
        )
        activated = silu_and_mul(gate_up.flatten(0, -2))
        expert_output = _make_intermediate(
            (num_tokens, topk, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            zeros=False,
        )
        fused_moe_w4a16_route_launcher(
            activated,
            down_packed,
            down_scale,
            expert_output,
            route_expert_ids,
            top_k=1,
            reindex_a=False,
            num_bits=num_bits,
            group_size=group_size,
        )
        return moe_reduce(expert_output, topk_weights, fp32_acc=True)

    # PyTorch promotes int32 cumsums to int64.  Normalize only paths that use
    # the shared sorter so all routing metadata keeps a homogeneous dtype.
    if (routing_topk_ids.dtype != torch.int64
            and not (num_experts == 1 and not allow_invalid_routes)):
        routing_topk_ids = routing_topk_ids.to(torch.int64)

    prefer_small_tiles = _use_hopper_w4a16(hidden_states.device)
    if num_experts == 1 and not allow_invalid_routes:
        # The shared Triton sorter cannot compile a width-one tl.sort. With one
        # expert, top_k is necessarily one and the route order is already sorted.
        sorted_idx = torch.arange(num_tokens,
                                  dtype=topk_ids.dtype,
                                  device=topk_ids.device)
        exp_start = torch.zeros(1,
                                dtype=topk_ids.dtype,
                                device=topk_ids.device)
        exp_end = torch.full((1, ),
                             num_tokens,
                             dtype=topk_ids.dtype,
                             device=topk_ids.device)
        compact_meta = {}
    elif _should_use_compact_w4a16(num_tokens, num_routes, num_experts,
                                   prefer_small_tiles):
        block_m = _w4a16_block_m(num_tokens, num_routes, num_experts,
                                  prefer_small_tiles)
        (sorted_idx, exp_start, exp_end, block_end, block_expert_ids,
         block_offsets) = _get_sorted_idx_blocks(
             routing_topk_ids,
             routing_num_experts,
             num_experts,
             0,
             block_m,
         )
        exp_start = exp_start[:num_experts]
        exp_end = exp_end[:num_experts]
        compact_meta = dict(block_end=block_end,
                            block_expert_ids=block_expert_ids,
                            block_offsets=block_offsets,
                            block_m=block_m)
    else:
        sorted_idx, exp_start, exp_end = _get_sorted_idx(
            routing_topk_ids, routing_num_experts)
        exp_start = exp_start[:num_experts]
        exp_end = exp_end[:num_experts]
        compact_meta = {}

    gate_up = _make_intermediate(
        (num_tokens, topk, ffn_dim * 2),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
        zeros=allow_invalid_routes,
    )
    fused_moe_w4a16_kernel_launcher(
        hidden_states,
        gate_up_packed,
        gate_up_scale,
        gate_up,
        sorted_idx,
        exp_start,
        exp_end,
        top_k=topk,
        num_tokens=num_tokens,
        reindex_a=True,
        reindex_c=False,
        num_bits=num_bits,
        group_size=group_size,
        **compact_meta,
    )

    routed_shape = gate_up.shape[:-1]
    activated = silu_and_mul(gate_up.flatten(0, -2)).unflatten(0, routed_shape)
    expert_output = _make_intermediate(
        (num_tokens, topk, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
        zeros=allow_invalid_routes,
    )
    fused_moe_w4a16_kernel_launcher(
        activated,
        down_packed,
        down_scale,
        expert_output,
        sorted_idx,
        exp_start,
        exp_end,
        top_k=1,
        num_tokens=num_tokens,
        reindex_a=False,
        reindex_c=True,
        num_bits=num_bits,
        group_size=group_size,
        **compact_meta,
    )
    if valid_routes is not None:
        expert_output.masked_fill_(~valid_routes[..., None], 0)
    return moe_reduce(expert_output, topk_weights, fp32_acc=True)
