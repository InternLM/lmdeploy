# Copyright (c) OpenMMLab. All rights reserved.
"""Exclusive decode input reuse: exact math, complete backing and graph lifetime."""
from functools import partial

import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_update as baseline_conv
from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule as baseline_gdn

# Explicit authorization is required; default calls remain out-of-place.
candidate_gdn = partial(baseline_gdn, allow_input_alias=True)
candidate_conv = partial(baseline_conv, allow_input_alias=True)


def exact(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('w', [1, 4, 6])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('layout', ['default', 'transposed', 'kda'])
def test_exclusive_recurrent_last_output_alias(w, dtype, layout):
    torch.manual_seed(24017)
    b, h, d = 2, 2, 128
    q, k, v = [
        torch.randn(b, w, h, d, device='cuda', dtype=torch.bfloat16) / 8
        for _ in range(3)
    ]
    g = -torch.rand(
        (b, w, h, d) if layout == 'kda' else (b, w, h), device='cuda')
    beta = torch.rand(b, w, h, device='cuda')
    backing = torch.randn(2 * w + 5, 3, h, d, d, device='cuda',
                          dtype=dtype) / 16
    original = backing.clone()
    src = torch.tensor([1, 2], device='cuda')
    out_ids = torch.arange(3, 3 + 2 * w, device='cuda').reshape(b, w)
    # Golden math is current unaliased indexed kernel; additionally compare
    # with the separate legacy logical-time ring addressing implementation.
    expected_backing = backing.clone()
    kwargs = dict(output_final_state=True,
                  transpose_state_layout=layout != 'default')
    expected, _ = baseline_gdn(q,
                               k,
                               v,
                               g,
                               beta,
                               initial_state=expected_backing[:, 1],
                               state_indices=src,
                               output_state_indices=out_ids,
                               **kwargs)
    ring = torch.zeros(b, w + 1, h, d, d, device='cuda', dtype=dtype)
    ring[:, 0] = original[src, 1]
    legacy, _ = baseline_gdn(q,
                             k,
                             v,
                             g,
                             beta,
                             initial_state=ring,
                             cache_seqlens=torch.zeros(b,
                                                       device='cuda',
                                                       dtype=torch.int32),
                             **kwargs)
    exact(expected, legacy)
    exact(expected_backing[out_ids, 1], ring[:, 1:])
    # Exactly W units per request: W-1 distinct destinations and its input.
    alias_ids = out_ids.clone()
    alias_ids[:, -1] = src
    actual, _ = candidate_gdn(q,
                              k,
                              v,
                              g,
                              beta,
                              initial_state=backing[:, 1],
                              state_indices=src,
                              output_state_indices=alias_ids,
                              **kwargs)
    exact(actual, expected)
    exact(backing[alias_ids, 1], ring[:, 1:])
    expected_entire = original.clone()
    expected_entire[alias_ids, 1] = ring[:, 1:]
    exact(backing, expected_entire)  # zero, unused units and sibling layers
    # All possible positive acceptance boundaries remain selectable.
    for consumed in range(1, w + 1):
        exact(backing[alias_ids[:, consumed - 1], 1], ring[:, consumed])
    # Three-replay graph check, reset outside graph (not a timing result).
    if w == 4 and dtype == torch.bfloat16 and layout == 'default':
        graph = torch.cuda.CUDAGraph()
        backing.copy_(original)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            captured, _ = candidate_gdn(q,
                                        k,
                                        v,
                                        g,
                                        beta,
                                        initial_state=backing[:, 1],
                                        state_indices=src,
                                        output_state_indices=alias_ids,
                                        **kwargs)
        for _ in range(3):
            backing.copy_(original)
            graph.replay()
            exact(captured, expected)
            exact(backing, expected_entire)


@pytest.mark.parametrize('w', [1, 4, 6])
@pytest.mark.parametrize('shared', [False, True])
def test_conv_exact_full_backing(w, shared):
    torch.manual_seed(24018)
    b, c, width = 2, 8192, 4
    x = torch.randn(b, c, w, device='cuda', dtype=torch.bfloat16) / 8
    weights = torch.randn(c, width, device='cuda', dtype=x.dtype) / 8
    bias = torch.randn(c, device='cuda', dtype=x.dtype) / 8
    backing = torch.randn(
        2 * w + 5, 3, c, width - 1, device='cuda', dtype=x.dtype) / 8
    original = backing.clone()
    src = torch.tensor([1, 1] if shared else [1, 2], device='cuda')
    out_ids = torch.arange(3, 3 + 2 * w, device='cuda').reshape(b, w)
    expected_backing = backing.clone()
    expected = baseline_conv(x,
                             expected_backing[:, 1],
                             weights,
                             bias=bias,
                             activation='silu',
                             conv_state_indices=src,
                             output_state_indices=out_ids)
    alias_ids = out_ids.clone()
    if not shared:
        alias_ids[:, -1] = src
    actual = candidate_conv(x,
                            backing[:, 1],
                            weights,
                            bias=bias,
                            activation='silu',
                            conv_state_indices=src,
                            output_state_indices=alias_ids)
    exact(actual, expected)
    # Ordered raw history independently computed for every boundary.
    history = torch.cat([original[src, 1], x], dim=-1)
    expected_entire = original.clone()
    for row in range(b):
        for step in range(w):
            expected_entire[alias_ids[row, step],
                            1] = history[row, :, step + 1:step + width]
    exact(backing, expected_entire)
    exact(backing[alias_ids, 1], expected_backing[out_ids, 1])
    if shared:
        exact(backing[src, 1], original[src, 1])


@pytest.mark.parametrize('w', [1, 4, 6])
def test_shared_recurrent_input_remains_readonly(w):
    torch.manual_seed(24019)
    q, k, v = [
        torch.randn(2, w, 2, 128, device='cuda', dtype=torch.bfloat16) / 8
        for _ in range(3)
    ]
    g = -torch.rand(2, w, 2, device='cuda')
    beta = torch.rand_like(g)
    pool = torch.randn(2 * w + 4, 2, 128, 128, device='cuda') / 16
    before = pool.clone()
    expected_pool = pool.clone()
    src = torch.tensor([1, 1], device='cuda')
    dst = torch.arange(2, 2 + 2 * w, device='cuda').reshape(2, w)
    kw = dict(output_final_state=True,
              state_indices=src,
              output_state_indices=dst)
    expected, _ = baseline_gdn(q,
                               k,
                               v,
                               g,
                               beta,
                               initial_state=expected_pool,
                               **kw)
    actual, _ = candidate_gdn(q, k, v, g, beta, initial_state=pool, **kw)
    exact(actual, expected)
    exact(pool, expected_pool)
    exact(pool[1], before[1])


@pytest.mark.parametrize('w', [1, 4])
@pytest.mark.parametrize('layout', ['default', 'kda'])
def test_minimal_pool_graph_rollover(w, layout):
    from lmdeploy.pytorch.kernels.cuda.compact_state import select_compact_state_ids
    torch.manual_seed(913)
    b, h, d = 2, 2, 128
    dtype = torch.bfloat16 if layout == 'default' else torch.float32
    q, k, v = [
        torch.randn(b, w, h, d, device='cuda', dtype=torch.bfloat16) / 8
        for _ in range(3)
    ]
    g = -torch.rand(
        (b, w, h, d) if layout == 'kda' else (b, w, h), device='cuda')
    beta = torch.rand(b, w, h, device='cuda')
    # Exactly global zero + W private units per request, no hidden candidate unit.
    actual = torch.randn(1 + b * w, h, d, d, device='cuda', dtype=dtype) / 16
    initial = actual.clone()
    expected_pool = torch.zeros(3 + b * w, h, d, d, device='cuda', dtype=dtype)
    expected_src = torch.tensor([1 + b * w, 2 + b * w], device='cuda')
    expected_dst = torch.arange(1, 1 + b * w, device='cuda').reshape(b, w)
    banks = [list(range(1, 1 + w)), list(range(1 + w, 1 + 2 * w))]
    current = [row[0] for row in banks]

    def destinations():
        return [[bank[(bank.index(cur) + 1 + j) % w] for j in range(w)]
                for bank, cur in zip(banks, current)]

    source = torch.tensor(current, device='cuda')
    bank_ids = torch.tensor(banks, device='cuda')
    _, dest = select_compact_state_ids(bank_ids,
                                       source,
                                       w,
                                       allow_input_alias=True)
    kw = dict(output_final_state=True, transpose_state_layout=layout == 'kda')
    expected_pool[expected_src] = initial[source]
    # Warm first; reset capture effects outside the graph, never timed.
    candidate_gdn(q,
                  k,
                  v,
                  g,
                  beta,
                  initial_state=actual,
                  state_indices=source,
                  output_state_indices=dest,
                  **kw)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, dest = select_compact_state_ids(bank_ids,
                                           source,
                                           w,
                                           allow_input_alias=True)
        captured, _ = candidate_gdn(q,
                                    k,
                                    v,
                                    g,
                                    beta,
                                    initial_state=actual,
                                    state_indices=source,
                                    output_state_indices=dest,
                                    **kw)
    actual.copy_(initial)
    for step in range(8):
        rows = destinations()
        source.copy_(torch.tensor(current, device='cuda'))
        expected, _ = baseline_gdn(q,
                                   k,
                                   v,
                                   g,
                                   beta,
                                   initial_state=expected_pool,
                                   state_indices=expected_src,
                                   output_state_indices=expected_dst,
                                   **kw)
        graph.replay()
        assert dest.tolist() == rows
        exact(captured, expected)
        exact(actual[dest], expected_pool[expected_dst])
        exact(actual[0], initial[0])
        take = [step % w, (step + 1) % w]
        # Oracle retains its own candidate values, not readback from actual pool.
        expected_pool[expected_src] = torch.stack([
            expected_pool[expected_dst[r, take[r]]].clone() for r in range(b)
        ])
        current = [rows[r][take[r]] for r in range(b)]


@pytest.mark.parametrize('allow_alias', [False, True])
@pytest.mark.parametrize('layout', ['default', 'transposed', 'kda'])
def test_recurrent_alias_permission_masks_early_reuse_and_reserved_unit(
        allow_alias, layout):
    torch.manual_seed(44)
    q, k, v = [
        torch.randn(1, 4, 2, 128, device='cuda', dtype=torch.bfloat16) / 8
        for _ in range(3)
    ]
    g = -torch.rand(
        (1, 4, 2, 128) if layout == 'kda' else (1, 4, 2), device='cuda')
    beta = torch.rand(1, 4, 2, device='cuda')
    pool = torch.randn(7, 2, 128, 128, device='cuda') / 16
    original, reference = pool.clone(), pool.clone()
    source = torch.tensor([1], device='cuda')
    destinations = torch.tensor([[1, 0, 3, 1]], device='cuda')
    separate = torch.tensor([[2, 3, 4, 5]], device='cuda')
    kwargs = dict(output_final_state=True,
                  transpose_state_layout=layout != 'default',
                  state_indices=source)
    expected, _ = baseline_gdn(q,
                               k,
                               v,
                               g,
                               beta,
                               initial_state=reference,
                               output_state_indices=separate,
                               **kwargs)
    actual, _ = baseline_gdn(q,
                             k,
                             v,
                             g,
                             beta,
                             initial_state=pool,
                             output_state_indices=destinations,
                             allow_input_alias=allow_alias,
                             **kwargs)
    exact(actual, expected)
    original[3] = reference[4]
    if allow_alias:
        original[1] = reference[5]
    exact(pool,
          original)  # Early alias, zero and unused units cannot be written.


@pytest.mark.parametrize('allow_alias', [False, True])
def test_conv_alias_permission_masks_early_reuse_and_reserved_unit(
        allow_alias):
    torch.manual_seed(45)
    x = torch.randn(1, 256, 4, device='cuda', dtype=torch.bfloat16) / 8
    weight = torch.randn(256, 4, device='cuda', dtype=x.dtype) / 8
    pool = torch.randn(7, 256, 3, device='cuda', dtype=x.dtype)
    original, reference = pool.clone(), pool.clone()
    source = torch.tensor([1], device='cuda')
    separate = torch.tensor([[2, 3, 4, 5]], device='cuda')
    destinations = torch.tensor([[1, 0, 3, 1]], device='cuda')
    expected = baseline_conv(x,
                             reference,
                             weight,
                             conv_state_indices=source,
                             output_state_indices=separate)
    actual = baseline_conv(x,
                           pool,
                           weight,
                           conv_state_indices=source,
                           output_state_indices=destinations,
                           allow_input_alias=allow_alias)
    exact(actual, expected)
    original[3] = reference[4]
    if allow_alias:
        original[1] = reference[5]
    exact(pool, original)


def test_exclusive_permission_rejects_legacy_state_addressing():
    q = torch.ones(1, 1, 1, 128, device='cuda', dtype=torch.bfloat16)
    pool = torch.zeros(1, 1, 128, 128, device='cuda', dtype=q.dtype)
    with pytest.raises(ValueError, match='requires indexed'):
        baseline_gdn(q,
                     q,
                     q,
                     initial_state=pool,
                     output_final_state=True,
                     allow_input_alias=True)
    x = torch.ones(1, 256, 1, device='cuda', dtype=torch.bfloat16)
    conv = torch.zeros(1, 256, 3, device='cuda', dtype=x.dtype)
    weight = torch.ones(256, 4, device='cuda', dtype=x.dtype)
    with pytest.raises(ValueError, match='requires indexed'):
        baseline_conv(x, conv, weight, allow_input_alias=True)


@pytest.mark.parametrize('width', [1, 4, 6])
@pytest.mark.parametrize('shared', [False, True])
def test_minimal_id_bank_rotates_and_preserves_every_positive_prefix(
        width, shared):
    from lmdeploy.pytorch.kernels.cuda.compact_state import select_compact_state_ids
    banks = torch.arange(1, 1 + 2 * width, device='cuda').reshape(2, width)
    source = torch.zeros(2, dtype=torch.long,
                         device='cuda') if shared else banks[:,
                                                             0].contiguous()
    current, outputs = select_compact_state_ids(banks,
                                                source,
                                                width,
                                                allow_input_alias=True)
    expected = banks if shared else banks.roll(-1, dims=1)
    exact(current, source)
    exact(outputs, expected)
    for count in range(1, width + 1):
        selected, following = select_compact_state_ids(
            banks,
            current,
            width,
            output_ids=outputs,
            consumed=torch.full_like(source, count),
            allow_input_alias=True)
        exact(selected, outputs[:, count - 1])
        exact(following[:, -1], selected)
        for row in range(2):
            assert set(following[row].tolist()) == set(banks[row].tolist())


@pytest.mark.parametrize('shared', [False, True])
def test_zero_advance_cannot_masquerade_as_retained_overwritten_input(shared):
    from lmdeploy.pytorch.kernels.cuda.compact_state import select_compact_state_ids
    banks = torch.tensor([[1, 2, 3, 4]], device='cuda')
    source = torch.tensor([5 if shared else 1], device='cuda')
    current, outputs = select_compact_state_ids(banks,
                                                source,
                                                4,
                                                allow_input_alias=True)
    selected, following = select_compact_state_ids(banks,
                                                   current,
                                                   4,
                                                   output_ids=outputs,
                                                   consumed=torch.tensor(
                                                       [0], device='cuda'),
                                                   allow_input_alias=True)
    assert selected.tolist() == ([5] if shared else [-1])
    assert following.tolist() == ([[1, 2, 3, 4]] if shared else [[-1] * 4])


def test_full_bank_alias_geometry_requires_explicit_authorization():
    from lmdeploy.pytorch.kernels.cuda.compact_state import select_compact_state_ids
    for width in (1, 4):
        banks = torch.arange(1, width + 1, device='cuda')[None]
        source = banks[:, 0].contiguous()
        with pytest.raises(ValueError, match='bank table|Output width'):
            select_compact_state_ids(banks, source, width)


@pytest.mark.parametrize('width', [1, 4])
def test_minimal_conv_pool_graph_rollover_with_independent_history(width):
    from lmdeploy.pytorch.kernels.cuda.compact_state import select_compact_state_ids
    torch.manual_seed(46)
    channels = 256
    x = torch.randn(2, channels, width, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(channels, 4, device='cuda', dtype=x.dtype) / 8
    backing = torch.randn(1 + 2 * width,
                          3,
                          channels,
                          3,
                          device='cuda',
                          dtype=x.dtype)
    initial = backing.clone()
    banks = torch.arange(1, 1 + 2 * width, device='cuda').reshape(2, width)
    source = banks[:, 0].contiguous()
    histories = backing[source, 1].clone()
    _, dest = select_compact_state_ids(banks,
                                       source,
                                       width,
                                       allow_input_alias=True)
    candidate_conv(x,
                   backing[:, 1],
                   weight,
                   conv_state_indices=source,
                   output_state_indices=dest)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, dest = select_compact_state_ids(banks,
                                           source,
                                           width,
                                           allow_input_alias=True)
        captured = candidate_conv(x,
                                  backing[:, 1],
                                  weight,
                                  conv_state_indices=source,
                                  output_state_indices=dest)
    backing.copy_(initial)
    for step in range(8):
        x.copy_(torch.randn_like(x))
        # Independent chronological histories and legacy conv output math.
        chronological = torch.cat((histories, x), dim=-1)
        reference_state = histories.clone()
        expected = baseline_conv(x, reference_state, weight)
        candidates = torch.stack(
            [chronological[:, :, j + 1:j + 4] for j in range(width)], dim=1)
        graph.replay()
        exact(captured, expected)
        exact(backing[dest, 1], candidates)
        exact(backing[0], initial[0])
        exact(backing[:, 0], initial[:, 0])
        exact(backing[:, 2], initial[:, 2])
        take = [step % width, (step + 1) % width]
        histories = torch.stack(
            [candidates[row, take[row]].clone() for row in range(2)])
        source.copy_(torch.stack([dest[row, take[row]] for row in range(2)]))
