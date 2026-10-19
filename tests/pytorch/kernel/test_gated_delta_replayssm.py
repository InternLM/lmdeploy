import pytest
import torch


def _make_step(batch, h, hv, kdim, vdim, dtype, seed):
    gen = torch.Generator(device='cuda').manual_seed(seed)
    q = torch.randn(batch, 1, h, kdim, device='cuda', dtype=dtype, generator=gen)
    k = torch.randn_like(q)
    v = torch.randn(batch, 1, hv, vdim, device='cuda', dtype=dtype, generator=gen)
    g = -torch.rand(batch, 1, hv, device='cuda', dtype=torch.float32, generator=gen) * 0.2
    beta = torch.sigmoid(torch.randn(batch, 1, hv, device='cuda', dtype=dtype, generator=gen))
    return q, k, v, g.contiguous(), beta.contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('buffer_len', [1, 8, 16, 32])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('channelwise', [False, True])
def test_standard_replayssm_matches_recurrent_across_flushes(buffer_len, dtype, channelwise):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm import (
        advance_gdn_replayssm_standard_cursor,
        fused_recurrent_gdn_replayssm_decode,
    )
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

    batch, h, hv, kdim, vdim = 3, 2, 4, 64, 96
    if channelwise:
        h, hv, kdim, vdim = 2, 2, 128, 128
    state_dtype = torch.float32 if channelwise else dtype
    slots = 5
    state_ids = torch.tensor([1, 3, 4], device='cuda', dtype=torch.int64)
    gen = torch.Generator(device='cuda').manual_seed(123)
    baseline = torch.randn(slots, hv, vdim, kdim, device='cuda', dtype=state_dtype, generator=gen)
    checkpoint = baseline.clone()
    d_ring = torch.zeros(slots, hv, buffer_len, vdim, device='cuda', dtype=state_dtype)
    k_ring = torch.zeros(slots, h, buffer_len, kdim, device='cuda', dtype=state_dtype)
    gate_shape = (slots, hv, buffer_len, kdim) if channelwise else (slots, hv, buffer_len)
    g_ring = torch.zeros(gate_shape, device='cuda', dtype=torch.float32)
    cursor = torch.zeros(slots, 3, device='cuda', dtype=torch.int32)

    for step in range(2 * buffer_len + 3):
        q, k, v, g, beta = _make_step(batch, h, hv, kdim, vdim, dtype, step)
        if channelwise:
            g = -torch.rand(batch, 1, hv, kdim, device='cuda') * 0.2
        q_ref = q.repeat_interleave(hv // h, dim=2).contiguous()
        k_ref = k.repeat_interleave(hv // h, dim=2).contiguous()
        expected, _ = fused_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v,
            g=g,
            beta=beta,
            initial_state=baseline,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            state_indices=state_ids,
            transpose_state_layout=True,
        )
        actual, _ = fused_recurrent_gdn_replayssm_decode(
            q,
            k,
            v,
            g,
            beta,
            checkpoint,
            d_ring,
            k_ring,
            g_ring,
            cursor,
            state_ids,
        )
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        advance_gdn_replayssm_standard_cursor(cursor, state_ids, buffer_len)
        if (step + 1) % buffer_len == 0:
            torch.testing.assert_close(checkpoint[state_ids], baseline[state_ids], atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('channelwise', [False, True])
def test_standard_replayssm_memcheck_matrix_invalid_rows_are_byte_identical(channelwise):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm import (
        advance_gdn_replayssm_standard_cursor,
        fused_recurrent_gdn_replayssm_decode,
    )

    batch, h, hv, kdim, vdim, buffer_len = 2, 2, 4, 64, 80, 8
    state_ids = torch.tensor([1, -1], device='cuda', dtype=torch.int64)
    checkpoint = torch.randn(3, hv, vdim, kdim, device='cuda', dtype=torch.float32)
    d_ring = torch.randn(3, hv, buffer_len, vdim, device='cuda', dtype=torch.float32)
    k_ring = torch.randn(3, h, buffer_len, kdim, device='cuda', dtype=torch.float32)
    g_ring = torch.randn(3, hv, buffer_len, device='cuda', dtype=torch.float32)
    if channelwise:
        g_ring = torch.randn(3, hv, buffer_len, kdim, device='cuda')
    cursor = torch.zeros(3, 3, device='cuda', dtype=torch.int32)
    cursor[0, 0] = buffer_len - 1
    before = tuple(t.clone() for t in (checkpoint, d_ring, k_ring, g_ring, cursor))
    q, k, v, g, beta = _make_step(batch, h, hv, kdim, vdim, torch.float32, 99)

    if channelwise:
        g = -torch.rand(*g.shape, kdim, device='cuda') * 0.2
    out, _ = fused_recurrent_gdn_replayssm_decode(
        q, k, v, g, beta, checkpoint, d_ring, k_ring, g_ring, cursor, state_ids)
    advance_gdn_replayssm_standard_cursor(cursor, state_ids, buffer_len)

    assert torch.count_nonzero(out[1]) == 0
    for tensor, original in zip((checkpoint, d_ring, k_ring, g_ring, cursor), before):
        assert torch.equal(tensor[0], original[0])
        assert torch.equal(tensor[2], original[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
def test_standard_replayssm_force_flush_publishes_checkpoint_boundary():
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm import (
        advance_gdn_replayssm_standard_cursor,
        fused_recurrent_gdn_replayssm_decode,
    )
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

    batch, h, hv, kdim, vdim, buffer_len = 2, 1, 2, 32, 48, 8
    state_ids = torch.arange(batch, device='cuda', dtype=torch.int64)
    baseline = torch.zeros(batch, hv, vdim, kdim, device='cuda', dtype=torch.bfloat16)
    checkpoint = baseline.clone()
    d_ring = torch.zeros(batch, hv, buffer_len, vdim, device='cuda', dtype=torch.bfloat16)
    k_ring = torch.zeros(batch, h, buffer_len, kdim, device='cuda', dtype=torch.bfloat16)
    g_ring = torch.zeros(batch, hv, buffer_len, device='cuda', dtype=torch.float32)
    cursor = torch.zeros(batch, 3, device='cuda', dtype=torch.int32)

    for step in range(4):
        q, k, v, g, beta = _make_step(batch, h, hv, kdim, vdim, torch.bfloat16, 900 + step)
        q_ref = q.repeat_interleave(hv // h, dim=2)
        k_ref = k.repeat_interleave(hv // h, dim=2)
        fused_recurrent_gated_delta_rule(
            q_ref,
            k_ref,
            v,
            g=g,
            beta=beta,
            initial_state=baseline,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            state_indices=state_ids,
            transpose_state_layout=True,
        )
        force_flush = torch.ones(batch, device='cuda', dtype=torch.bool) if step == 3 else None
        fused_recurrent_gdn_replayssm_decode(
            q, k, v, g, beta, checkpoint, d_ring, k_ring, g_ring, cursor,
            state_ids, force_flush=force_flush)
        advance_gdn_replayssm_standard_cursor(
            cursor, state_ids, buffer_len, force_flush=force_flush)

    assert torch.count_nonzero(cursor) == 0
    torch.testing.assert_close(checkpoint, baseline, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
def test_standard_replayssm_graph_save_restore_continues_before_natural_flush():
    """A graph-captured false flag must remain a mutable flush input."""
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm import (
        advance_gdn_replayssm_standard_cursor,
        fused_recurrent_gdn_replayssm_decode,
    )
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule
    from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMeta, CudaGraphMixin

    batch, slots, h, hv, kdim, vdim, buffer_len = 1, 3, 1, 2, 32, 48, 8
    state_ids = torch.zeros(batch, device='cuda', dtype=torch.int64)
    checkpoint = torch.zeros(slots, hv, vdim, kdim, device='cuda', dtype=torch.bfloat16)
    d_ring = torch.zeros(slots, hv, buffer_len, vdim, device='cuda', dtype=torch.bfloat16)
    k_ring = torch.zeros(slots, h, buffer_len, kdim, device='cuda', dtype=torch.bfloat16)
    g_ring = torch.zeros(slots, hv, buffer_len, device='cuda', dtype=torch.float32)
    cursor = torch.zeros(slots, 3, device='cuda', dtype=torch.int32)
    q, k, v, g, beta = _make_step(batch, h, hv, kdim, vdim, torch.bfloat16, 1700)
    q_buf, k_buf, v_buf = q.clone(), k.clone(), v.clone()
    g_buf, beta_buf = g.clone(), beta.clone()

    graph_meta = CudaGraphMeta(max_batchs=batch,
                               max_tokens=batch,
                               num_blocks=1,
                               is_decoding=True,
                               device=torch.device('cuda'),
                               is_ssm=True,
                               is_replayssm_standard=True)
    mixin = CudaGraphMixin()
    graph_meta.input_buffers = mixin.make_buffers_cudagraph(
        graph_meta,
        input_ids=state_ids[None],
        position_ids=state_ids[None],
        past_key_values=[],
        attn_metadata=None)
    force_flush = graph_meta.input_buffers['replayssm_force_flush']

    # Compile both launches before graph capture. Capture itself sees an
    # all-false, but non-None, fixed-address flag exactly like dummy warmup.
    fused_recurrent_gdn_replayssm_decode(
        q_buf, k_buf, v_buf, g_buf, beta_buf, checkpoint, d_ring, k_ring,
        g_ring, cursor, state_ids, force_flush=force_flush)
    advance_gdn_replayssm_standard_cursor(
        cursor, state_ids, buffer_len, force_flush=force_flush)
    torch.cuda.synchronize()
    checkpoint.zero_()
    d_ring.zero_()
    k_ring.zero_()
    g_ring.zero_()
    cursor.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, _ = fused_recurrent_gdn_replayssm_decode(
            q_buf, k_buf, v_buf, g_buf, beta_buf, checkpoint, d_ring,
            k_ring, g_ring, cursor, state_ids, force_flush=force_flush)
        advance_gdn_replayssm_standard_cursor(
            cursor, state_ids, buffer_len, force_flush=force_flush)

    # Graph capture executes once; reset persistent state before the real
    # sequence while retaining graph-owned buffers and addresses.
    checkpoint.zero_()
    d_ring.zero_()
    k_ring.zero_()
    g_ring.zero_()
    cursor.zero_()
    reference = torch.zeros(1, hv, vdim, kdim, device='cuda', dtype=torch.bfloat16)

    def replay_step(seed: int, flush: bool):
        q, k, v, g, beta = _make_step(batch, h, hv, kdim, vdim,
                                      torch.bfloat16, seed)
        q_buf.copy_(q)
        k_buf.copy_(k)
        v_buf.copy_(v)
        g_buf.copy_(g)
        beta_buf.copy_(beta)
        force_flush.fill_(flush)
        expected, _ = fused_recurrent_gated_delta_rule(
            q.repeat_interleave(hv // h, dim=2),
            k.repeat_interleave(hv // h, dim=2),
            v,
            g=g,
            beta=beta,
            initial_state=reference,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            state_indices=torch.zeros(1, device='cuda', dtype=torch.int64),
            transpose_state_layout=True)
        graph.replay()
        torch.testing.assert_close(graph_out, expected, atol=3e-2, rtol=3e-2)

    for step in range(4):
        replay_step(1800 + step, flush=step == 3)

    assert cursor[0].tolist() == [0, 0, 0]
    torch.testing.assert_close(checkpoint[0], reference[0], atol=5e-2, rtol=5e-2)

    # Publish the complete tuple, restore it into another runtime slot, and
    # continue there. This models StateCacheEngine.copy_slots on save/restore.
    for cache in (checkpoint, d_ring, k_ring, g_ring, cursor):
        cache[1].copy_(cache[0])
        cache[2].copy_(cache[1])
    state_ids.fill_(2)
    for step in range(4, 7):
        replay_step(1800 + step, flush=step == 6)

    assert cursor[2].tolist() == [0, 0, 0]
    torch.testing.assert_close(checkpoint[2], reference[0], atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
def test_standard_replayssm_partial_history_materializes_for_continuation():
    from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm import (
        advance_gdn_replayssm_standard_cursor,
        fold_gdn_replayssm_standard_history,
        fused_recurrent_gdn_replayssm_decode,
        reset_gdn_replayssm_cursor,
    )
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

    batch, h, hv, kdim, vdim, buffer_len = 2, 1, 2, 32, 48, 8
    state_ids = torch.arange(batch, device='cuda', dtype=torch.int64)
    baseline = torch.zeros(batch, hv, vdim, kdim, device='cuda', dtype=torch.bfloat16)
    checkpoint = baseline.clone()
    d_ring = torch.zeros(batch, hv, buffer_len, vdim, device='cuda', dtype=torch.bfloat16)
    k_ring = torch.zeros(batch, h, buffer_len, kdim, device='cuda', dtype=torch.bfloat16)
    g_ring = torch.zeros(batch, hv, buffer_len, device='cuda', dtype=torch.float32)
    cursor = torch.zeros(batch, 3, device='cuda', dtype=torch.int32)
    for step in range(3):
        q, k, v, g, beta = _make_step(batch, h, hv, kdim, vdim, torch.bfloat16, 1200 + step)
        fused_recurrent_gated_delta_rule(
            q.repeat_interleave(hv // h, dim=2),
            k.repeat_interleave(hv // h, dim=2),
            v,
            g=g,
            beta=beta,
            initial_state=baseline,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            state_indices=state_ids,
            transpose_state_layout=True,
        )
        fused_recurrent_gdn_replayssm_decode(
            q, k, v, g, beta, checkpoint, d_ring, k_ring, g_ring, cursor, state_ids)
        advance_gdn_replayssm_standard_cursor(cursor, state_ids, buffer_len)

    fold_gdn_replayssm_standard_history(
        checkpoint, d_ring, k_ring, g_ring, cursor, state_ids)
    reset_gdn_replayssm_cursor(cursor, state_ids)
    torch.testing.assert_close(checkpoint, baseline, atol=5e-2, rtol=5e-2)
    assert torch.count_nonzero(cursor) == 0
