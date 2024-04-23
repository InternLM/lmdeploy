# Copyright (c) OpenMMLab. All rights reserved.
import triton
import triton.language as tl
from torch import Tensor

from .utils import get_kernel_meta


@triton.jit
def _div_up(val, other):
    return (val + other - 1) // other


@triton.jit
def _fill_kv_cache_kernel(
    KStates,
    VStates,
    KCaches,
    VCaches,
    QStartLoc,
    QSeqLens,
    KVSeqLens,
    BlockOffsets,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_kss,
    stride_ksh,
    stride_ksd,
    stride_vss,
    stride_vsh,
    stride_vsd,
    stride_kcn: tl.constexpr,
    stride_kcb: tl.constexpr,
    stride_kch: tl.constexpr,
    stride_kcd: tl.constexpr,
    stride_vcn: tl.constexpr,
    stride_vcb: tl.constexpr,
    stride_vch: tl.constexpr,
    stride_vcd: tl.constexpr,
    stride_boff,
    BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """fill kv cache kernel."""
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)

    # initialize
    h_off = tl.arange(0, BLOCK_H)
    d_off = tl.arange(0, BLOCK_D)

    q_startloc = tl.load(QStartLoc + batch_id)
    q_seqlen = tl.load(QSeqLens + batch_id)
    kv_seqlen = tl.load(KVSeqLens + batch_id)
    history_seqlen = kv_seqlen - q_seqlen

    block0_first_tokenloc = history_seqlen % BLOCK

    state_token_offset = tl.maximum(block_id * BLOCK - block0_first_tokenloc,
                                    0)
    kv_block_id = _div_up(history_seqlen + 1, BLOCK) - 1 + block_id
    kv_block_id = min(kv_block_id, stride_boff - 1)
    block_off = tl.load(BlockOffsets + batch_id * stride_boff + kv_block_id)

    cur_startloc = q_startloc + state_token_offset
    ks_ptr = KStates + cur_startloc * stride_kss
    vs_ptr = VStates + cur_startloc * stride_vss

    kc_ptr = KCaches + block_off * stride_kcn
    vc_ptr = VCaches + block_off * stride_vcn

    c_first_tokenloc = block0_first_tokenloc
    if block_id != 0:
        c_first_tokenloc *= 0
    c_last_tokenloc = tl.minimum(
        BLOCK, q_seqlen + block0_first_tokenloc - block_id * BLOCK)

    for bidx in range(c_first_tokenloc, c_last_tokenloc):
        sidx = bidx - c_first_tokenloc
        mask = (h_off[:, None] < num_heads) & (d_off[None, :] < head_dim)
        k = tl.load(ks_ptr + sidx * stride_kss + h_off[:, None] * stride_ksh +
                    d_off[None, :] * stride_ksd,
                    mask=mask)
        tl.store(kc_ptr + bidx * stride_kcb + h_off[:, None] * stride_kch +
                 d_off[None, :] * stride_kcd,
                 k,
                 mask=mask)

        v = tl.load(vs_ptr + sidx * stride_vss + h_off[:, None] * stride_vsh +
                    d_off[None, :] * stride_vsd,
                    mask=mask)
        tl.store(vc_ptr + bidx * stride_vcb + h_off[:, None] * stride_vch +
                 d_off[None, :] * stride_vcd,
                 v,
                 mask=mask)


def fill_kv_cache(k_states: Tensor, v_states: Tensor, k_caches: Tensor,
                  v_caches: Tensor, q_start_loc: Tensor, q_seq_length: Tensor,
                  kv_seq_length: Tensor, max_q_seq_length: int,
                  block_offsets: Tensor):
    """fill key/value state to cache for paged attention."""

    block_offsets = block_offsets.contiguous()
    batch_size = block_offsets.size(0)
    block_size, num_heads, head_dim = k_caches.size()[1:]
    max_num_blocks = triton.cdiv(max_q_seq_length, block_size) + 1

    BLOCK = block_size
    BLOCK_H = triton.next_power_of_2(num_heads)
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = [batch_size, max_num_blocks]
    kernel_meta = get_kernel_meta(k_states)
    _fill_kv_cache_kernel[grid](
        k_states,
        v_states,
        k_caches,
        v_caches,
        q_start_loc,
        q_seq_length,
        kv_seq_length,
        block_offsets,
        num_heads=num_heads,
        head_dim=head_dim,
        stride_kss=k_states.stride(-3),
        stride_ksh=k_states.stride(-2),
        stride_ksd=k_states.stride(-1),
        stride_vss=v_states.stride(-3),
        stride_vsh=v_states.stride(-2),
        stride_vsd=v_states.stride(-1),
        stride_kcn=k_caches.stride(0),
        stride_kcb=k_caches.stride(1),
        stride_kch=k_caches.stride(2),
        stride_kcd=k_caches.stride(3),
        stride_vcn=v_caches.stride(0),
        stride_vcb=v_caches.stride(1),
        stride_vch=v_caches.stride(2),
        stride_vcd=v_caches.stride(3),
        stride_boff=block_offsets.stride(0),
        BLOCK=BLOCK,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        **kernel_meta,
    )
