# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache

import torch
import triton
import triton.language as tl

from ..gated_delta_rule import GatedDeltaRuleBuilder, GatedDeltaRuleImpl
from .utils import has_tilelang


@lru_cache
def has_fla():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        return True
    except Exception:
        return False


@triton.jit
def _state_select_kernel(
    state_ptr,
    out_ptr,
    state_indices_ptr,
    spec_offsets_ptr,
    stride_s0,
    stride_s1,
    stride_o0,
    INNER_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused state select: out[b] = state[state_indices[b],
    spec_offsets[b]]."""
    batch_id = tl.program_id(0).to(tl.int64)
    block_id = tl.program_id(1)

    state_idx = tl.load(state_indices_ptr + batch_id)
    spec_off = tl.load(spec_offsets_ptr + batch_id)

    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < INNER_SIZE

    src_ptr = state_ptr + state_idx * stride_s0 + spec_off * stride_s1 + offs
    dst_ptr = out_ptr + batch_id * stride_o0 + offs

    data = tl.load(src_ptr, mask=mask)
    tl.store(dst_ptr, data, mask=mask)


@triton.jit
def _state_scatter_kernel(
    state_ptr,
    src_ptr,
    state_indices_ptr,
    spec_offsets_ptr,
    stride_s0,
    stride_s1,
    stride_i0,
    INNER_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused state scatter: state[si[b], so[b]] = src[b]."""
    batch_id = tl.program_id(0).to(tl.int64)
    block_id = tl.program_id(1)

    state_idx = tl.load(state_indices_ptr + batch_id)
    spec_off = tl.load(spec_offsets_ptr + batch_id)

    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < INNER_SIZE

    in_ptr = src_ptr + batch_id * stride_i0 + offs
    dst_ptr = state_ptr + state_idx * stride_s0 + spec_off * stride_s1 + offs

    data = tl.load(in_ptr, mask=mask)
    tl.store(dst_ptr, data, mask=mask)


def _state_select(state, state_indices, spec_offsets):
    """Fused state select: out = state[state_indices, spec_offsets].

    Requires inner dims [2:] to be contiguous.
    """
    B = state_indices.shape[0]
    inner_shape = state.shape[2:]
    inner_size = 1
    for s in inner_shape:
        inner_size *= s

    out = state.new_empty((B, *inner_shape))

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(inner_size, BLOCK_SIZE)
    grid = (B, num_blocks)

    _state_select_kernel[grid](
        state,
        out,
        state_indices,
        spec_offsets,
        state.stride(0),
        state.stride(1),
        out.stride(0),
        INNER_SIZE=inner_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def _state_scatter(state, state_indices, spec_offsets, src):
    """Fused state scatter: state[state_indices, spec_offsets] =
    src.to(state.dtype).

    Requires inner dims [2:] to be contiguous.
    """
    if src.dtype != state.dtype:
        src = src.to(state.dtype)

    inner_size = 1
    for s in state.shape[2:]:
        inner_size *= s

    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(inner_size, BLOCK_SIZE)
    B = state_indices.shape[0]
    grid = (B, num_blocks)

    _state_scatter_kernel[grid](
        state,
        src,
        state_indices,
        spec_offsets,
        state.stride(0),
        state.stride(1),
        src.stride(0),
        INNER_SIZE=inner_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


class CudaGatedDeltaRuleImpl(GatedDeltaRuleImpl):

    def __init__(self):
        if not has_fla() or not has_tilelang():
            raise ImportError('fla and tilelang is required for CudaGatedDeltaRuleImpl')
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule
        self.chunk_func = chunk_gated_delta_rule
        self.recurrent_func = fused_recurrent_gated_delta_rule

    def chunk_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        scale: float | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        output_final_state: bool = False,
        spec_state_offsets: torch.Tensor | None = None,
    ):

        assert initial_state is not None
        recurrent_state = initial_state

        if spec_state_offsets is not None:
            spec_read_offsets = spec_state_offsets[0]
            init_state = _state_select(recurrent_state, state_indices, spec_read_offsets)
        else:
            batch_state = recurrent_state.index_select(0, state_indices)
            init_state = batch_state

        if use_qk_l2norm_in_kernel:
            # l2norm in fla would recompile when seqlen changed.
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)
        core_attn_out, last_state = self.chunk_func(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=init_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens,
        )
        if spec_state_offsets is not None:
            # write to next slots
            spec_write_offsets = spec_state_offsets[1]
            _state_scatter(recurrent_state, state_indices, spec_write_offsets, last_state)
        else:
            last_state = recurrent_state.index_copy_(0, state_indices, last_state.to(recurrent_state.dtype))
        if not output_final_state:
            last_state = None
        return core_attn_out, last_state

    def fused_recurrent_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        scale: float | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        output_final_state: bool = False,
        cache_seqlens: torch.Tensor | None = None,
    ):
        return self.recurrent_func(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            state_indices=state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output_final_state=output_final_state,
            cache_seqlens=cache_seqlens,
        )


class CudaGatedDeltaRuleBuilder(GatedDeltaRuleBuilder):

    @staticmethod
    def build() -> GatedDeltaRuleImpl:
        return CudaGatedDeltaRuleImpl()
