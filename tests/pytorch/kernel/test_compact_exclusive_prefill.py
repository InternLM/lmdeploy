# Copyright (c) OpenMMLab. All rights reserved.
"""Cold and graph-safe prefill reuse with no full-state staging."""
import itertools

import pytest
import torch
import triton


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def packed(lengths):
    counts = [(length + 63) // 64 for length in lengths]
    return dict(cu_seqlens=torch.tensor([0, *itertools.accumulate(lengths)],
                                        device='cuda'),
                chunk_indices=torch.tensor([(row, chunk)
                                            for row, count in enumerate(counts)
                                            for chunk in range(count)],
                                           device='cuda',
                                           dtype=torch.int32),
                chunk_offsets=torch.tensor([0, *itertools.accumulate(counts)],
                                           device='cuda',
                                           dtype=torch.int32))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('width', [2, 4])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_conv_deferred_alias_export_preserves_short_rows_shared_inputs_and_full_backing(
        dtype, width, checkpoint):
    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import indexed_causal_conv1d_fwd
    torch.manual_seed(202618)
    lengths = [0, 1, 2, 3, 15, 16, 17, 64, 72, 129]
    channels = 192  # Includes a partial channel tile in the deferred exporter.
    x = torch.randn(sum(lengths), channels, device='cuda', dtype=dtype).T[None]
    weight = torch.randn(channels, width, device='cuda', dtype=dtype) / 8
    backing = torch.randn(48,
                          3,
                          channels,
                          width - 1,
                          device='cuda',
                          dtype=dtype)
    backing[0].zero_()
    initial, reference = backing.clone(), backing.clone()
    src = torch.tensor([1, 2, 3, 4, 5, 6, 7, 0, 20, 20], device='cuda')
    dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 22, 23], device='cuda')
    ref_dst = torch.arange(24, 34, device='cuda')
    cp = torch.arange(34, 44, device='cuda')
    offsets = torch.tensor([min(64, length) for length in lengths],
                           device='cuda')
    meta = packed(lengths)
    seq = torch.repeat_interleave(
        torch.arange(len(lengths), device='cuda', dtype=torch.int32),
        torch.tensor(lengths, device='cuda'))
    common = dict(seq_idx=seq,
                  cu_seqlens=meta['cu_seqlens'],
                  activation='silu')
    if checkpoint:
        common.update(checkpoint_ids=cp, checkpoint_offsets=offsets)
    expected = indexed_causal_conv1d_fwd(x, weight, reference[:, 1], src,
                                         ref_dst, **common)
    actual = indexed_causal_conv1d_fwd(x,
                                       weight,
                                       backing[:, 1],
                                       src,
                                       dst,

                                       **common)
    exact(actual, expected)
    expected_backing = initial.clone()
    for row, length in enumerate(lengths):
        if not length:
            continue
        expected_backing[dst[row], 1] = reference[ref_dst[row], 1]
        if checkpoint:
            expected_backing[cp[row], 1] = reference[cp[row], 1]
        # Raw history oracle independent of either GPU export implementation.
        start = sum(lengths[:row])
        history = torch.cat(
            (initial[src[row], 1], x[0, :, start:start + length]), dim=-1)
        exact(backing[dst[row], 1], history[:, -(width - 1):])
    exact(backing, expected_backing)
    # Capture includes both the convolution and the ordered final-only export.
    graph = torch.cuda.CUDAGraph()
    backing.copy_(initial)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        captured = indexed_causal_conv1d_fwd(x,
                                             weight,
                                             backing[:, 1],
                                             src,
                                             dst,

                                             **common)
    for _ in range(3):
        backing.copy_(initial)
        graph.replay()
        exact(captured, expected)
        exact(backing, expected_backing)


@pytest.mark.parametrize('kind', ['gdn', 'kda'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('state_v_first', [False, True])
def test_recurrent_cold_alias_bypasses_autotuner_and_matches_full_pipeline(
        monkeypatch, kind, dtype, checkpoint, state_v_first):
    from lmdeploy.pytorch.kernels.cuda import indexed_chunk_state as state_module
    from lmdeploy.pytorch.kernels.cuda.indexed_chunk_gated_delta import (
        indexed_chunk_gated_delta_rule,
        indexed_chunk_kda,
    )
    torch.manual_seed(202619)
    lengths = [0, 1, 72, 137, 72]
    heads, dim = 2, 128
    q, k, v = [
        torch.randn(
            1, sum(lengths), heads, dim, device='cuda', dtype=torch.bfloat16) /
        8 for _ in range(3)
    ]
    beta = torch.randn(1,
                       sum(lengths),
                       heads,
                       device='cuda',
                       dtype=torch.bfloat16)
    options = dict(state_v_first=state_v_first, **packed(lengths))
    if kind == 'gdn':
        q, k = [
            torch.nn.functional.normalize(t.float(), dim=-1).to(t.dtype)
            for t in (q, k)
        ]
        beta = beta.sigmoid()
        g = -torch.rand(1, sum(lengths), heads, device='cuda') / 10
        forward = indexed_chunk_gated_delta_rule
    else:
        g = torch.randn_like(q)
        options.update(use_qk_l2norm_in_kernel=True,
                       use_gate_in_kernel=True,
                       use_beta_sigmoid_in_kernel=True,
                       safe_gate=True,
                       lower_bound=-5.0,
                       A_log=torch.zeros(heads, device='cuda'),
                       dt_bias=torch.randn(heads * dim, device='cuda'))
        forward = indexed_chunk_kda
    backing = torch.randn(24, 3, heads, dim, dim, device='cuda',
                          dtype=dtype) / 16
    initial, reference = backing.clone(), backing.clone()
    src = torch.tensor([1, 2, 3, 6, 6], device='cuda')
    dst = torch.tensor([1, 2, 3, 4, 5], device='cuda')
    ref_dst = torch.tensor([7, 8, 9, 10, 11], device='cuda')
    cp = torch.arange(12, 17, device='cuda')
    if checkpoint:
        options.update(checkpoint_ids=cp,
                       checkpoint_offsets=torch.tensor(
                           [min(64, length) for length in lengths],
                           device='cuda'))
    expected = forward(q, k, v, g, beta, reference[:, 1], src, ref_dst,
                       **options)
    # The first aliased invocation must not enter the trial-launch wrapper.
    assert isinstance(state_module._indexed_chunk_state_kernel.fn, triton.runtime.JITFunction)
    actual = forward(q,
                     k,
                     v,
                     g,
                     beta,
                     backing[:, 1],
                     src,
                     dst,

                     **options)
    exact(actual, expected)
    expected_backing = initial.clone()
    for row, length in enumerate(lengths):
        if length:
            expected_backing[dst[row], 1] = reference[ref_dst[row], 1]
            if checkpoint:
                expected_backing[cp[row], 1] = reference[cp[row], 1]
    exact(backing, expected_backing)
    graph = torch.cuda.CUDAGraph()
    backing.copy_(initial)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        captured = forward(q,
                           k,
                           v,
                           g,
                           beta,
                           backing[:, 1],
                           src,
                           dst,

                           **options)
    for _ in range(3):
        backing.copy_(initial)
        graph.replay()
        exact(captured, expected)
        exact(backing, expected_backing)


@pytest.mark.parametrize('width', [1, 4])
def test_prefill_then_decode_uses_exact_minimal_pool_without_state_conversion(
        width):
    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_update, indexed_causal_conv1d_fwd
    from lmdeploy.pytorch.kernels.cuda.compact_state import select_compact_state_ids
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule
    from lmdeploy.pytorch.kernels.cuda.indexed_chunk_gated_delta import indexed_chunk_gated_delta_rule
    torch.manual_seed(202620)
    batch, heads, dim, channels = 2, 2, 128, 128
    count = 1 + batch * width  # Global zero + W private units per request, nothing else.
    recurrent = torch.randn(
        count, heads, dim, dim, device='cuda', dtype=torch.bfloat16) / 16
    conv = torch.randn(count, channels, 3, device='cuda', dtype=torch.bfloat16)
    recurrent[0].zero_()
    conv[0].zero_()
    ref_recurrent = torch.zeros(count + batch,
                                heads,
                                dim,
                                dim,
                                device='cuda',
                                dtype=recurrent.dtype)
    ref_conv = torch.zeros(count + batch,
                           channels,
                           3,
                           device='cuda',
                           dtype=conv.dtype)
    ref_source = torch.arange(count, count + batch, device='cuda')
    bank = torch.arange(1, count, device='cuda').reshape(batch, width)
    current = torch.zeros(batch, device='cuda', dtype=torch.long)
    final = bank[:, 0].contiguous()
    weight = torch.randn(channels, 4, device='cuda', dtype=conv.dtype) / 8
    for length in (64, 8, 1):
        q, k, v = [
            torch.randn(1,
                        batch * length,
                        heads,
                        dim,
                        device='cuda',
                        dtype=torch.bfloat16) / 8 for _ in range(3)
        ]
        q, k = [
            torch.nn.functional.normalize(t.float(), dim=-1).to(t.dtype)
            for t in (q, k)
        ]
        g = -torch.rand(1, batch * length, heads, device='cuda') / 10
        beta = torch.rand_like(g).to(q.dtype)
        meta = packed([length] * batch)
        x = torch.randn(batch * length,
                        channels,
                        device='cuda',
                        dtype=conv.dtype).T[None]
        seq = torch.arange(batch, device='cuda',
                           dtype=torch.int32).repeat_interleave(length)
        expected = indexed_chunk_gated_delta_rule(q,
                                                  k,
                                                  v,
                                                  g,
                                                  beta,
                                                  ref_recurrent,
                                                  ref_source,
                                                  final,
                                                  state_v_first=True,
                                                  **meta)
        actual = indexed_chunk_gated_delta_rule(q,
                                                k,
                                                v,
                                                g,
                                                beta,
                                                recurrent,
                                                current,
                                                final,

                                                state_v_first=True,
                                                **meta)
        expected_conv = indexed_causal_conv1d_fwd(
            x,
            weight,
            ref_conv,
            ref_source,
            final,
            seq_idx=seq,
            cu_seqlens=meta['cu_seqlens'])
        actual_conv = indexed_causal_conv1d_fwd(x,
                                                weight,
                                                conv,
                                                current,
                                                final,
                                                seq_idx=seq,
                                                cu_seqlens=meta['cu_seqlens'])
        exact(actual, expected)
        exact(actual_conv, expected_conv)
        exact(recurrent[final], ref_recurrent[final])
        exact(conv[final], ref_conv[final])
        # Independent reference retains its own values, not candidate readback.
        ref_recurrent[ref_source] = ref_recurrent[final]
        ref_conv[ref_source] = ref_conv[final]
        current = final
    for iteration in range(width + 1):
        _, outputs = select_compact_state_ids(bank,
                                              current,
                                              width)
        q, k, v = [
            torch.randn(
                batch, width, heads, dim, device='cuda', dtype=torch.bfloat16)
            / 8 for _ in range(3)
        ]
        g = -torch.rand(batch, width, heads, device='cuda') / 10
        beta = torch.rand_like(g)
        x = torch.randn(batch,
                        channels,
                        width,
                        device='cuda',
                        dtype=conv.dtype)
        expected, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=ref_recurrent,
            state_indices=ref_source,
            output_state_indices=outputs,
            output_final_state=True,
            transpose_state_layout=True)
        actual, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=recurrent,
            state_indices=current,
            output_state_indices=outputs,
            output_final_state=True,
            transpose_state_layout=True)
        expected_conv = causal_conv1d_update(x,
                                             ref_conv,
                                             weight,
                                             conv_state_indices=ref_source,
                                             output_state_indices=outputs)
        actual_conv = causal_conv1d_update(x,
                                           conv,
                                           weight,
                                           conv_state_indices=current,
                                           output_state_indices=outputs)
        exact(actual, expected)
        exact(actual_conv, expected_conv)
        exact(recurrent[outputs], ref_recurrent[outputs])
        exact(conv[outputs], ref_conv[outputs])
        take = iteration % width
        current = outputs[:, take].contiguous()
        ref_recurrent[ref_source] = ref_recurrent[current]
        ref_conv[ref_source] = ref_conv[current]
    assert torch.count_nonzero(recurrent[0]).item() == torch.count_nonzero(
        conv[0]).item() == 0


@pytest.mark.parametrize('width', [1, 4])
def test_empty_prefill_retains_private_input_without_claiming_zero_advance_decode(width):
    from lmdeploy.pytorch.kernels.cuda.compact_state import select_compact_state_ids
    bank = torch.tensor([[1] + [-1] * (width - 1)], device='cuda')
    current = torch.tensor([1], device='cuda')
    # The zero-length prefill kernels never wrote this nominal destination.
    previous_output = torch.tensor([[1]], device='cuda')
    retained, following = select_compact_state_ids(
        bank, current, width, output_ids=previous_output, consumed=torch.tensor([0], device='cuda'),
        allow_partial_bank=True)
    assert retained.tolist() == [1]
    assert following.tolist() == ([[1]] if width == 1 else [[-1] * width])
    # A full-bank decode producer did materialize its candidate and cannot
    # claim the overwritten input still represents the previous boundary.
    full_bank = torch.arange(1, width + 1, device='cuda')[None]
    invalid, outputs = select_compact_state_ids(
        full_bank, current, width, output_ids=previous_output, consumed=torch.tensor([0], device='cuda'))
    assert invalid.tolist() == [-1] and outputs.tolist() == [[-1] * width]


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('width', [2, 4])
@pytest.mark.parametrize('alias', [False, True])
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('lengths', [[1], [8], [0, 1, 2, 5], [9]])
def test_short_conv_tile_matches_full_tile_and_complete_storage(
        monkeypatch, dtype, width, alias, checkpoint, lengths):
    from lmdeploy.pytorch.kernels.cuda import causal_conv1d as module
    torch.manual_seed(202620)
    total, count, channels = sum(lengths), len(lengths), 192
    # Non-contiguous packed projection and layer-strided physical state pool.
    x = torch.randn(total, channels + 64, device='cuda', dtype=dtype)[:, :channels].T[None]
    weight = torch.randn(channels, width, device='cuda', dtype=dtype) / 8
    bias = torch.randn(channels, device='cuda', dtype=dtype)
    initial = torch.randn(3 * count + 1, 3, channels, width - 1, device='cuda', dtype=dtype)
    initial[0].zero_()
    source = torch.arange(1, count + 1, device='cuda')
    final = source if alias else source + count
    opts = dict(seq_idx=torch.repeat_interleave(
        torch.arange(count, device='cuda', dtype=torch.int32), torch.tensor(lengths, device='cuda')),
        cu_seqlens=packed(lengths)['cu_seqlens'], bias=bias, activation='silu')
    if checkpoint:
        opts.update(checkpoint_ids=source + 2 * count,
                    checkpoint_offsets=torch.tensor([max(0, n - 1) for n in lengths], device='cuda'))
    candidate, reference = initial.clone(), initial.clone()
    factory = module.causal_conv1d_fwd
    tiles = []

    def observe(*args, **kwargs):
        tiles.append(kwargs['ChunkSizeL'])
        return factory(*args, **kwargs)

    monkeypatch.setattr(module, 'causal_conv1d_fwd', observe)
    actual = module.indexed_causal_conv1d_fwd(x, weight, candidate[:, 1], source, final, **opts)
    assert tiles == [8 if total <= 8 else 16]

    def full_tile(*args, **kwargs):
        kwargs['ChunkSizeL'] = 16
        return factory(*args, **kwargs)

    monkeypatch.setattr(module, 'causal_conv1d_fwd', full_tile)
    expected = module.indexed_causal_conv1d_fwd(x, weight, reference[:, 1], source, final, **opts)
    exact(actual, expected)
    exact(candidate, reference)
    # Independent raw-history oracle; includes chunks shorter than K-1.
    for row, length in enumerate(lengths):
        if length:
            start = sum(lengths[:row])
            raw = torch.cat((initial[row + 1, 1], x[0, :, start:start + length]), dim=-1)
            target = row + 1 if alias else row + 1 + count
            exact(candidate[target, 1], raw[:, -(width - 1):])
    exact(candidate[:, 0], initial[:, 0])
    exact(candidate[:, 2], initial[:, 2])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('id_dtype', [torch.int32, torch.int64])
def test_short_conv_graph_replay_updates_ragged_ids_and_shared_inputs(monkeypatch, dtype, id_dtype):
    from lmdeploy.pytorch.kernels.cuda import causal_conv1d as module
    torch.manual_seed(202621)
    channels = 192
    x = torch.randn(8, channels, device='cuda', dtype=dtype).T[None]
    weight = torch.randn(channels, 4, device='cuda', dtype=dtype) / 8
    initial = torch.randn(12, 3, channels, 3, device='cuda', dtype=dtype)
    initial[0].zero_()
    candidate, reference = initial.clone(), initial.clone()
    source = torch.tensor([1, 2], device='cuda', dtype=id_dtype)
    final = source.clone()
    checkpoint = torch.tensor([9, 10], device='cuda', dtype=id_dtype)
    offsets = torch.tensor([3, 3], device='cuda', dtype=id_dtype)
    cu = torch.tensor([0, 4, 8], device='cuda', dtype=id_dtype)
    seq = torch.tensor([0] * 4 + [1] * 4, device='cuda', dtype=torch.int32)
    options = dict(seq_idx=seq, cu_seqlens=cu, checkpoint_ids=checkpoint,
                   checkpoint_offsets=offsets, activation='silu')
    module.indexed_causal_conv1d_fwd(x, weight, candidate[:, 1], source, final, **options)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        actual = module.indexed_causal_conv1d_fwd(x, weight, candidate[:, 1], source, final, **options)
    factory = module.causal_conv1d_fwd

    def full_tile(*args, **kwargs):
        kwargs['ChunkSizeL'] = 16
        return factory(*args, **kwargs)

    monkeypatch.setattr(module, 'causal_conv1d_fwd', full_tile)
    for lengths, sources, finals in [([0, 8], [1, 2], [1, 2]), ([1, 7], [3, 4], [3, 4]),
                                    ([7, 1], [0, 5], [7, 5]), ([4, 4], [6, 6], [7, 8])]:
        x.normal_()
        source.copy_(torch.tensor(sources, device='cuda', dtype=id_dtype))
        final.copy_(torch.tensor(finals, device='cuda', dtype=id_dtype))
        offsets.copy_(torch.tensor([min(3, n) for n in lengths], device='cuda', dtype=id_dtype))
        cu.copy_(torch.tensor([0, lengths[0], 8], device='cuda', dtype=id_dtype))
        seq.copy_(torch.tensor([0] * lengths[0] + [1] * lengths[1], device='cuda', dtype=torch.int32))
        candidate.copy_(initial)
        reference.copy_(initial)
        expected = module.indexed_causal_conv1d_fwd(x, weight, reference[:, 1], source, final, **options)
        graph.replay()
        exact(actual, expected)
        exact(candidate, reference)
