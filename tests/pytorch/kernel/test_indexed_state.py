# Copyright (c) OpenMMLab. All rights reserved.
"""Direct recurrent unit addressing, isolated from scheduler integration."""
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule


def _inputs(w, channelwise=False):
    torch.manual_seed(42)
    b, h, d = 2, 2, 128
    q, k, v = [torch.randn(b, w, h, d, device='cuda', dtype=torch.bfloat16) / 8 for _ in range(3)]
    g = -torch.rand((b, w, h, d) if channelwise else (b, w, h), device='cuda')
    beta = torch.rand(b, w, h, device='cuda')
    return q, k, v, g, beta


@pytest.mark.parametrize('w', [1, 2, 4, 6])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('layout', ['default', 'transposed', 'kda'])
def test_indexed_shared_input_matches_ring_and_preserves_source(w, dtype, layout):
    q, k, v, g, beta = _inputs(w, channelwise=layout == 'kda')
    transposed = layout != 'default'
    # Noncontiguous layer view: ID stride must not be inferred from payload size.
    storage = torch.randn(2 * w + 4, 3, 2, 128, 128, device='cuda', dtype=dtype) / 16
    pool = storage[:, 1]
    before = storage.clone()
    src = torch.tensor([2, 2], device='cuda', dtype=torch.int64)
    dst = torch.arange(3, 3 + 2 * w, device='cuda').flip(0).reshape(2, w)
    # Independent legacy representation, with logical-time positions rather
    # than the new physical output lookup. Every intermediate boundary compared.
    ring = torch.zeros(2, w + 1, 2, 128, 128, device='cuda', dtype=dtype)
    ring[:, 0] = pool[2]
    expected, _ = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=ring, output_final_state=True,
        cache_seqlens=torch.zeros(2, device='cuda', dtype=torch.int32),
        transpose_state_layout=transposed)
    actual, returned = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=pool, output_final_state=True,
        state_indices=src, output_state_indices=dst, transpose_state_layout=transposed)
    assert returned.data_ptr() == pool.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for row in range(2):
        for step in range(w):
            torch.testing.assert_close(pool[dst[row, step]], ring[row, step + 1], rtol=0, atol=0)
    untouched = torch.ones(len(pool), device='cuda', dtype=torch.bool)
    untouched[dst.flatten()] = False
    torch.testing.assert_close(pool[untouched], before[:, 1][untouched], rtol=0, atol=0)
    torch.testing.assert_close(storage[:, 0], before[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(storage[:, 2], before[:, 2], rtol=0, atol=0)


@pytest.mark.parametrize('transposed', [False, True])
def test_indexed_dummy_output_ids_cannot_write_reserved_or_input_units(transposed):
    q, k, v, g, beta = _inputs(4)
    pool = torch.randn(8, 2, 128, 128, device='cuda', dtype=torch.bfloat16)
    before = pool.clone()
    src = torch.tensor([2, -1], device='cuda')
    # Device masks are not a substitute for host owner validation. They protect
    # dummy rows and reserved/input storage when padding reaches the primitive.
    dst = torch.tensor([[0, -1, 8, 2], [3, 4, 5, 6]], device='cuda')
    actual, _ = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=pool, output_final_state=True,
        state_indices=src, output_state_indices=dst, transpose_state_layout=transposed)
    torch.testing.assert_close(pool, before, rtol=0, atol=0)
    assert torch.count_nonzero(actual[1]) == 0


def test_indexed_output_metadata_rejected_before_launch():
    q, k, v, g, beta = _inputs(1)
    pool = torch.zeros(4, 2, 128, 128, device='cuda')
    src = torch.tensor([1, 1], device='cuda')
    for dst in (torch.ones(2, device='cuda', dtype=torch.int64),
                torch.ones(2, 1, device='cuda', dtype=torch.int32)):
        with pytest.raises(ValueError, match='Output state IDs'):
            fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=pool, output_final_state=True,
                                             state_indices=src, output_state_indices=dst)
    with pytest.raises(ValueError, match='no ring offsets'):
        fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=pool, output_final_state=True,
                                         state_indices=src, output_state_indices=src[:, None],
                                         cache_seqlens=torch.zeros(2, device='cuda', dtype=torch.int32))


@pytest.mark.parametrize('w', [1, 2, 4, 6])
@pytest.mark.parametrize('width', [2, 4])
@pytest.mark.parametrize('activation', [None, 'silu'])
def test_indexed_conv_shared_input_and_every_boundary(w, width, activation):
    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_update

    torch.manual_seed(49)
    channels = 192
    x = torch.randn(2, w, channels, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
    backing = torch.randn(2 * w + 4, 3, channels, width - 1, device='cuda', dtype=x.dtype)
    pool = backing[:, 1]
    before = backing.clone()
    weight = torch.randn(channels, width, device='cuda', dtype=x.dtype) / 8
    bias = torch.randn(channels, device='cuda', dtype=x.dtype) / 8
    src = torch.tensor([2, 2], device='cuda')
    dst = torch.arange(3, 3 + 2 * w, device='cuda').flip(0).reshape(2, w)
    history = pool[2].expand(2, -1, -1).clone()
    # Scalar-time legacy AR calls provide output parity, while concatenation
    # independently specifies exact history bytes at every logical boundary.
    expected = []
    for step in range(w):
        expected.append(causal_conv1d_update(x[:, :, step:step + 1], history, weight, bias, activation))
    actual = causal_conv1d_update(x, pool, weight, bias, activation,
                                 conv_state_indices=src, output_state_indices=dst)
    torch.testing.assert_close(actual, torch.cat(expected, dim=-1), rtol=0, atol=0)
    for row in range(2):
        tokens = torch.cat((before[2, 1], x[row]), dim=-1)
        for step in range(w):
            torch.testing.assert_close(pool[dst[row, step]], tokens[:, step + 1:step + width], rtol=0, atol=0)
    untouched = torch.ones(len(pool), device='cuda', dtype=torch.bool)
    untouched[dst.flatten()] = False
    torch.testing.assert_close(pool[untouched], before[:, 1][untouched], rtol=0, atol=0)
    torch.testing.assert_close(backing[:, 0], before[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(backing[:, 2], before[:, 2], rtol=0, atol=0)


def test_indexed_conv_dummy_outputs_and_padding():
    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_update

    x = torch.randn(2, 192, 4, device='cuda', dtype=torch.bfloat16)
    pool = torch.randn(8, 192, 3, device='cuda', dtype=x.dtype)
    before = pool.clone()
    weight = torch.randn(192, 4, device='cuda', dtype=x.dtype)
    src = torch.tensor([2, -1], device='cuda')
    dst = torch.tensor([[0, -1, 8, 2], [3, 4, 5, 6]], device='cuda')
    actual = causal_conv1d_update(x, pool, weight, conv_state_indices=src, output_state_indices=dst)
    torch.testing.assert_close(pool, before, rtol=0, atol=0)
    assert torch.count_nonzero(actual[1]) == 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_indexed_default_direct_store_roundtrip(dtype):
    # K=256 forces multiple waves and the direct (not shared-memory-coalesced)
    # default-layout store path. Preserve the dtype roundtrip between tokens.
    torch.manual_seed(83)
    q = torch.randn(2, 4, 2, 256, device='cuda', dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    v = torch.randn(2, 4, 2, 128, device='cuda', dtype=q.dtype) / 8
    g = -torch.rand(2, 4, 2, device='cuda')
    beta = torch.rand_like(g)
    pool = torch.randn(12, 2, 256, 128, device='cuda', dtype=dtype) / 16
    original = pool[2].clone()
    src = torch.tensor([2, 2], device='cuda')
    dst = torch.arange(3, 11, device='cuda').reshape(2, 4)
    ring = torch.zeros(2, 5, 2, 256, 128, device='cuda', dtype=dtype)
    ring[:, 0] = original
    expected, _ = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=ring, output_final_state=True,
        cache_seqlens=torch.zeros(2, device='cuda', dtype=torch.int32))
    actual, _ = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=pool, output_final_state=True,
        state_indices=src, output_state_indices=dst)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool[dst], ring[:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(pool[2], original, rtol=0, atol=0)


def test_indexed_fused_kda_gate_matches_legacy():
    q, k, v, g, beta = _inputs(4, channelwise=True)
    pool = torch.randn(12, 2, 128, 128, device='cuda') / 16
    original = pool[2].clone()
    src = torch.tensor([2, 2], device='cuda')
    dst = torch.arange(3, 11, device='cuda').reshape(2, 4)
    ring = torch.zeros(2, 5, 2, 128, 128, device='cuda')
    ring[:, 0] = original
    kwargs = dict(output_final_state=True, transpose_state_layout=True,
                  a_log=torch.zeros(2, device='cuda'), dt_bias=torch.randn(2, 128, device='cuda'), lower_bound=-5.)
    expected, _ = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=ring,
        cache_seqlens=torch.zeros(2, device='cuda', dtype=torch.int32), **kwargs)
    actual, _ = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=pool, state_indices=src, output_state_indices=dst, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool[dst], ring[:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(pool[2], original, rtol=0, atol=0)


@pytest.mark.parametrize('transposed', [False, True])
def test_indexed_128_rounds_acceptance_and_checkpoint_branches(transposed):
    q, k, v, g, beta = _inputs(4)
    torch.manual_seed(95)
    pool = torch.randn(24, 2, 128, 128, device='cuda', dtype=q.dtype) / 16
    ring = torch.zeros(2, 5, 2, 128, 128, device='cuda', dtype=q.dtype)
    source = [1, 1]
    reference = pool[source].clone()
    bank = [list(range(2, 7)), list(range(7, 12))]
    frozen = {1: pool[1].clone()}
    for iteration in range(128):
        output = [[unit for unit in ids if unit != src][:4] for ids, src in zip(bank, source)]
        src = torch.tensor(source, device='cuda')
        dst = torch.tensor(output, device='cuda')
        ring[:, 0] = reference
        expected, _ = fused_recurrent_gated_delta_rule(
            q, k, v, g, beta, initial_state=ring, output_final_state=True,
            cache_seqlens=torch.zeros(2, device='cuda', dtype=torch.int32),
            transpose_state_layout=transposed)
        actual, _ = fused_recurrent_gated_delta_rule(
            q, k, v, g, beta, initial_state=pool, output_final_state=True,
            state_indices=src, output_state_indices=dst, transpose_state_layout=transposed)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(pool[dst], ring[:, 1:], rtol=0, atol=0)
        for row in range(2):
            accepted = (iteration + row) % 5  # includes no progress, partial and full
            reference[row] = ring[row, accepted]
            if accepted:
                source[row] = output[row][accepted - 1]
        if iteration % 16 == 15:
            # Freeze a materialized unit and replace it before constructing the
            # next output table. Both consumers then branch from that source.
            unit = source[0]
            frozen[unit] = pool[unit].clone()
            bank[0][bank[0].index(unit)] = 12 + iteration // 16
            source = [unit, unit]
            reference[:] = pool[unit]
        for unit, snapshot in frozen.items():
            torch.testing.assert_close(pool[unit], snapshot, rtol=0, atol=0)
