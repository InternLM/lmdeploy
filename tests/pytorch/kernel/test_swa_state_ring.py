# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import torch

from lmdeploy.pytorch.backends.cuda.attention.swa_state_ring import (
    SWAStateRingMetadata,
    SWAStateRingAttentionImpl,
)
from lmdeploy.pytorch.backends.attention import SWAStateRingAttentionBuildSpec
from lmdeploy.pytorch.kernels.cuda.swa_state_ring import flatten_swa_state_ring, scatter_swa_state_ring


def test_swa_state_ring_flattens_history_then_updates_current_tokens():
    ring = torch.zeros((1, 4, 1, 8), dtype=torch.bfloat16, device='cuda')
    ring[0, 0].fill_(10)
    ring[0, 1].fill_(20)
    current = torch.stack((torch.full((1, 8), 30), torch.full((1, 8), 40))).to(
        device='cuda', dtype=torch.bfloat16)
    state_slots = torch.tensor([0], dtype=torch.int64, device='cuda')
    start_positions = torch.tensor([2], dtype=torch.int64, device='cuda')
    q_seqlens = torch.tensor([2], dtype=torch.int32, device='cuda')
    cu_q_seqlens = torch.tensor([0, 2], dtype=torch.int32, device='cuda')
    history_lens = torch.tensor([2], dtype=torch.int32, device='cuda')
    cu_kv_seqlens = torch.tensor([0, 4], dtype=torch.int32, device='cuda')

    flattened = flatten_swa_state_ring(
        ring,
        current,
        state_slots,
        start_positions,
        q_seqlens,
        cu_q_seqlens,
        history_lens,
        cu_kv_seqlens,
        max_q_seqlen=2,
    )
    expected = torch.tensor([10, 20, 30, 40], dtype=torch.bfloat16, device='cuda')
    torch.testing.assert_close(flattened[:4, 0, 0], expected)

    scatter_swa_state_ring(
        current,
        ring,
        state_slots,
        start_positions,
        q_seqlens,
        cu_q_seqlens,
        max_q_seqlen=2,
    )
    torch.testing.assert_close(ring[0, :, 0, 0], expected)


def test_swa_state_ring_q1_paged_decode_matches_flatten():
    """The q=1 ring-page fast path preserves pre-wrap and wrapped results."""

    from lmdeploy.pytorch.kernels.cuda import flash_attn_varlen_func, flash_attn_with_kvcache

    torch.manual_seed(7)
    batch_size = 2
    window_size = 128
    num_q_heads = 8
    num_kv_heads = 1
    head_dim = 192
    value_dim = 128
    dtype = torch.bfloat16
    device = 'cuda'

    query = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype, device=device)
    current_k = torch.randn(batch_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    current_v = torch.randn(batch_size, num_kv_heads, value_dim, dtype=dtype, device=device)
    k_ring = torch.randn(batch_size, window_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_ring = torch.randn(batch_size, window_size, num_kv_heads, value_dim, dtype=dtype, device=device)
    attn_metadata = SimpleNamespace(
        q_seqlens=torch.ones(batch_size, dtype=torch.int64, device=device),
        # The first sequence has not wrapped; the second has wrapped.
        kv_seqlens=torch.tensor([6, 194], dtype=torch.int64, device=device),
    )
    metadata = SWAStateRingMetadata.from_step_context(
        attn_metadata,
        SimpleNamespace(max_q_seqlen=1),
        state_slots=torch.arange(batch_size, dtype=torch.int64, device=device),
        num_state_slots=batch_size,
        window_size=window_size,
    )
    flat_k = flatten_swa_state_ring(
        k_ring,
        current_k,
        metadata.state_slots,
        metadata.start_positions,
        metadata.q_seqlens,
        metadata.cu_q_seqlens,
        metadata.history_lens,
        metadata.cu_kv_seqlens,
        metadata.max_q_seqlen,
    )
    flat_v = flatten_swa_state_ring(
        v_ring,
        current_v,
        metadata.state_slots,
        metadata.start_positions,
        metadata.q_seqlens,
        metadata.cu_q_seqlens,
        metadata.history_lens,
        metadata.cu_kv_seqlens,
        metadata.max_q_seqlen,
    )
    scale = head_dim**-0.5
    sinks = torch.randn(num_q_heads, dtype=dtype, device=device)
    expected = flash_attn_varlen_func(
        query,
        flat_k,
        flat_v,
        metadata.cu_q_seqlens,
        metadata.cu_kv_seqlens,
        max_seqlen_q=1,
        max_seqlen_k=window_size,
        window_size=window_size - 1,
        softmax_scale=scale,
        causal=True,
        sinks=sinks,
        kv_layout='shd',
    )

    impl = SWAStateRingAttentionImpl(
        SWAStateRingAttentionBuildSpec(
            num_heads=num_q_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            v_head_dim=value_dim,
            scale=scale,
            sliding_window=(window_size - 1, 0),
            learnable_sink=True,
        ))
    actual = impl.forward(
        query,
        current_k,
        current_v,
        k_ring,
        v_ring,
        metadata,
        learnable_sink=sinks,
    )

    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(k_ring[0, 5], current_k[0])
    torch.testing.assert_close(k_ring[1, 65], current_k[1])
