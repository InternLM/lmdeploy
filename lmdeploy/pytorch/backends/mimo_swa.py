# Copyright (c) OpenMMLab. All rights reserved.
"""MiMo-V2-Flash SWA backed by a sequence-scoped BF16 state ring."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MiMoSWAAttentionMetadata:
    """Layer-invariant sequence layout for MiMo's SWA state ring."""

    state_slots: torch.Tensor
    start_positions: torch.Tensor
    q_seqlens: torch.Tensor
    cu_q_seqlens: torch.Tensor
    history_lens: torch.Tensor
    kv_seqlens: torch.Tensor
    cu_kv_seqlens: torch.Tensor
    max_q_seqlen: int
    max_kv_seqlen: int
    window_size: int

    @classmethod
    def from_step_context(
        cls,
        attn_metadata: Any,
        step_context: Any,
        state_slots: torch.Tensor,
        num_state_slots: int,
        window_size: int,
    ) -> 'MiMoSWAAttentionMetadata':
        """Build once per model step, then reuse for all SWA layers."""
        if attn_metadata is None:
            raise RuntimeError('MiMo SWA state attention requires attention metadata.')
        if not isinstance(window_size, int) or window_size <= 1:
            raise ValueError(f'MiMo SWA window_size must be greater than one, got {window_size!r}.')

        q_seqlens = getattr(attn_metadata, 'q_seqlens', None)
        kv_seqlens = getattr(attn_metadata, 'kv_seqlens', None)
        if q_seqlens is None or kv_seqlens is None:
            raise RuntimeError('MiMo SWA state attention requires q_seqlens and kv_seqlens.')
        q_seqlens = q_seqlens.to(dtype=torch.int32)
        kv_seqlens = kv_seqlens.to(device=q_seqlens.device, dtype=torch.int64)
        state_slots = state_slots.to(device=q_seqlens.device, dtype=torch.int64)
        if state_slots.dim() != 1 or state_slots.numel() != q_seqlens.numel():
            raise ValueError(
                f'MiMo state_slots must have shape [{q_seqlens.numel()}], got {tuple(state_slots.shape)}.'
            )

        start_positions = kv_seqlens - q_seqlens.to(torch.int64)
        history_limit = window_size - 1
        valid_slots = (state_slots >= 0) & (state_slots < num_state_slots) & (start_positions >= 0)
        history_lens = start_positions.clamp(min=0, max=history_limit).to(torch.int32)
        history_lens = torch.where(valid_slots, history_lens, 0)

        cu_q_seqlens = torch.zeros(q_seqlens.numel() + 1, dtype=torch.int32, device=q_seqlens.device)
        torch.cumsum(q_seqlens, dim=0, out=cu_q_seqlens[1:])
        ring_kv_seqlens = history_lens + q_seqlens
        cu_kv_seqlens = torch.zeros_like(cu_q_seqlens)
        torch.cumsum(ring_kv_seqlens, dim=0, out=cu_kv_seqlens[1:])

        max_q_seqlen = getattr(step_context, 'max_q_seqlen', None)
        if max_q_seqlen is None:
            max_q_seqlen = getattr(attn_metadata, 'max_q_seqlen', None)
        if max_q_seqlen is None:
            # A synchronization-free upper bound.  It is exact for batch=1
            # and only affects launch metadata for ragged batches.
            max_q_seqlen = int(step_context.input_ids.numel())
        max_q_seqlen = int(max_q_seqlen)

        return cls(
            state_slots=state_slots,
            start_positions=start_positions,
            q_seqlens=q_seqlens,
            cu_q_seqlens=cu_q_seqlens,
            history_lens=history_lens,
            kv_seqlens=ring_kv_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=max_q_seqlen + history_limit,
            window_size=window_size,
        )


def mimo_swa_state_attention(
    attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_ring: torch.Tensor,
    v_ring: torch.Tensor,
    metadata: MiMoSWAAttentionMetadata,
    sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attend to chronological ring history plus current K/V, then update it."""
    if k_ring.size(0) != v_ring.size(0) or k_ring.size(1) != v_ring.size(1):
        raise ValueError('MiMo SWA K/V rings must have identical state-slot and window dimensions.')
    if k_ring.size(1) != metadata.window_size:
        raise ValueError(
            f'MiMo SWA metadata window {metadata.window_size} does not match ring window {k_ring.size(1)}.'
        )

    impl = attention.impl
    if not hasattr(impl, 'flash_attention_fwd'):
        raise RuntimeError(f'MiMo SWA state ring requires Triton varlen attention, got {type(impl).__name__}.')
    if getattr(impl, 'alibi', False):
        raise RuntimeError('MiMo SWA state ring does not support ALiBi.')

    # Delay CUDA/Triton kernel imports until a real SWA forward.  Model/config
    # inspection on a CPU-only host must not initialize the Triton driver.
    from lmdeploy.pytorch.kernels.cuda.mimo_swa_ring import (
        flatten_mimo_swa_ring,
        scatter_mimo_swa_ring,
    )

    flat_key = flatten_mimo_swa_ring(
        k_ring,
        key,
        metadata.state_slots,
        metadata.start_positions,
        metadata.q_seqlens,
        metadata.cu_q_seqlens,
        metadata.history_lens,
        metadata.cu_kv_seqlens,
        metadata.max_q_seqlen,
    )
    flat_value = flatten_mimo_swa_ring(
        v_ring,
        value,
        metadata.state_slots,
        metadata.start_positions,
        metadata.q_seqlens,
        metadata.cu_q_seqlens,
        metadata.history_lens,
        metadata.cu_kv_seqlens,
        metadata.max_q_seqlen,
    )

    attention._lazy_init(query.device)
    output = impl.flash_attention_fwd(
        query,
        flat_key,
        flat_value,
        cu_seqlens_q=metadata.cu_q_seqlens,
        cu_seqlens_k=metadata.cu_kv_seqlens,
        max_seqlen_q=metadata.max_q_seqlen,
        max_seqlen_k=metadata.max_kv_seqlen,
        window_size=metadata.window_size - 1,
        softmax_scale=impl.scale,
        softcap=impl.logit_softcapping,
        causal=True,
        sinks=sink,
        alibi_slopes=None,
        block_sparse_size=impl.block_sparse_size,
        kv_layout='shd',
    )

    # Preserve the old ring until the attention kernel has consumed it.  CUDA
    # stream ordering then makes these writes visible to the next model step.
    scatter_mimo_swa_ring(
        key,
        k_ring,
        metadata.state_slots,
        metadata.start_positions,
        metadata.q_seqlens,
        metadata.cu_q_seqlens,
    )
    scatter_mimo_swa_ring(
        value,
        v_ring,
        metadata.state_slots,
        metadata.start_positions,
        metadata.q_seqlens,
        metadata.cu_q_seqlens,
    )
    return output
