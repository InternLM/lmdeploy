# Copyright (c) OpenMMLab. All rights reserved.
import itertools

import pytest
import torch
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

from lmdeploy.pytorch.kernels.cuda.indexed_chunk_state import indexed_chunk_state_fwd


@pytest.mark.parametrize('state_v_first', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('gate_kind', ['g', 'gk'])
def test_indexed_chunk_72_checkpoint_is_observational(state_v_first, dtype, gate_kind):
    torch.manual_seed(127)
    b, t, h, hv, kdim, vdim = 2, 72, 2, 2, 128, 128
    k, w, u = [torch.randn(b, t, h, kdim, device='cuda', dtype=torch.bfloat16) / 32 for _ in range(3)]
    shape = (hv, vdim, kdim) if state_v_first else (hv, kdim, vdim)
    backing = torch.randn(12, 3, *shape, device='cuda', dtype=dtype) / 16
    pool = backing[:, 1]  # layer view with nontrivial entry stride
    before = backing.clone()
    src = torch.tensor([2, 2], device='cuda')
    dst = torch.tensor([9, 5], device='cuda')
    cp = torch.tensor([7, 3], device='cuda')
    offsets = torch.tensor([64, 64], device='cuda', dtype=torch.int32)
    g = -((torch.arange(t, device='cuda') % 64 + 1).float() / 100)[None, :, None].expand(b, t, hv).contiguous()
    gates = {'g': g} if gate_kind == 'g' else {'gk': g[..., None].expand(b, t, hv, kdim).contiguous()}
    common = dict(state_v_first=state_v_first, **gates)
    expected_h, expected_v, expected_final = chunk_gated_delta_rule_fwd_h(
        k, w, u, initial_state=pool[src].contiguous().float(), output_final_state=True, **common)
    actual_h, actual_v = indexed_chunk_state_fwd(k, w, u, pool, src, dst, **common)
    final_no_checkpoint = pool[dst].clone()
    h_cp, v_cp = indexed_chunk_state_fwd(k, w, u, pool, src, dst, checkpoint_ids=cp,
                                        checkpoint_offsets=offsets, **common)
    # Same arithmetic/shape: exporting a checkpoint must not change computation.
    for a, e in ((h_cp, actual_h), (v_cp, actual_v), (pool[dst], final_no_checkpoint)):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    for a, e in ((actual_h, expected_h), (actual_v, expected_v), (pool[dst], expected_final.to(dtype))):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    # Independent legacy propagation stopped exactly after the first64 tokens.
    short_gates = {name: gate[:, :64].contiguous() for name, gate in gates.items()}
    _, _, expected_checkpoint = chunk_gated_delta_rule_fwd_h(
        k[:, :64].contiguous(), w[:, :64].contiguous(), u[:, :64].contiguous(),
        initial_state=before[:, 1][src].contiguous(), output_final_state=True,
        state_v_first=state_v_first, **short_gates)
    torch.testing.assert_close(pool[cp], expected_checkpoint.to(dtype), rtol=0, atol=0)
    untouched = torch.ones(len(pool), dtype=torch.bool, device='cuda')
    untouched[torch.cat((dst, cp))] = False
    torch.testing.assert_close(pool[untouched], before[:, 1][untouched], rtol=0, atol=0)
    for layer in (0, 2):
        torch.testing.assert_close(backing[:, layer], before[:, layer], rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('width', [2, 4])
@pytest.mark.parametrize('with_checkpoint', [False, True])
@pytest.mark.parametrize('lengths', [[1, 2, 72, 129, 3], [15, 16, 17, 65, 3]])
def test_indexed_conv_packed_short_and_internal(dtype, width, with_checkpoint, lengths):
    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_fn, indexed_causal_conv1d_fwd

    torch.manual_seed(2112)
    channels = 128
    x = torch.randn(sum(lengths), channels, device='cuda', dtype=dtype).T.unsqueeze(0)
    weight = torch.randn(channels, width, device='cuda', dtype=dtype) / 4
    bias = torch.randn(channels, device='cuda', dtype=dtype)
    backing = torch.randn(20, 3, channels, width - 1, device='cuda', dtype=dtype)
    pool = backing[:, 1]
    before = backing.clone()
    src = torch.tensor([2, 2, 1, 2, 1], device='cuda')
    dst = torch.tensor([8, 10, 12, 14, 16], device='cuda')
    cp = torch.tensor([9, 11, 13, 15, 17], device='cuda')
    cp_offsets = torch.tensor([1, 1, min(64, lengths[2]), min(128, lengths[3] - 1), 2], device='cuda')
    cu = torch.tensor([0] + list(itertools.accumulate(lengths)), device='cuda')
    seq_idx = torch.repeat_interleave(torch.arange(len(lengths), device='cuda', dtype=torch.int32),
                                    torch.tensor(lengths, device='cuda'))
    expected = causal_conv1d_fn(x, weight, bias=bias, seq_idx=seq_idx, initial_states=pool[src].contiguous(),
                               activation='silu')
    common = dict(seq_idx=seq_idx, cu_seqlens=cu, bias=bias, activation='silu')
    actual = indexed_causal_conv1d_fwd(x, weight, pool, src, dst, **common)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if with_checkpoint:
        again = indexed_causal_conv1d_fwd(x, weight, pool, src, dst, checkpoint_ids=cp,
                                          checkpoint_offsets=cp_offsets, **common)
        torch.testing.assert_close(again, actual, rtol=0, atol=0)
    for row, length in enumerate(lengths):
        start = sum(lengths[:row])
        history = torch.cat((before[src[row], 1], x[0, :, start:start + length]), dim=-1)
        torch.testing.assert_close(pool[dst[row]], history[:, -(width - 1):], rtol=0, atol=0)
        if with_checkpoint:
            offset = int(cp_offsets[row])
            history = torch.cat((before[src[row], 1], x[0, :, start:start + offset]), dim=-1)
            torch.testing.assert_close(pool[cp[row]], history[:, -(width - 1):], rtol=0, atol=0)
    untouched = torch.ones(len(pool), dtype=torch.bool, device='cuda')
    untouched[dst] = False
    if with_checkpoint:
        untouched[cp] = False
    torch.testing.assert_close(pool[untouched], before[:, 1][untouched], rtol=0, atol=0)
    for layer in (0, 2):
        torch.testing.assert_close(backing[:, layer], before[:, layer], rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['gdn', 'kda'])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_indexed_full_chunk_pipeline(kind, packed, dtype):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.kda import chunk_kda

    from lmdeploy.pytorch.kernels.cuda.indexed_chunk_gated_delta import (
        indexed_chunk_gated_delta_rule,
        indexed_chunk_kda,
    )

    torch.manual_seed(912)
    lengths = [1, 72, 137] if packed else [72, 72]
    batch, length = (1, sum(lengths)) if packed else (2, 72)
    heads, dim = 2, 128
    q, k, v = [torch.randn(batch, length, heads, dim, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    beta = torch.randn(batch, length, heads, device='cuda', dtype=torch.bfloat16)
    options = dict(state_v_first=True)
    if kind == 'gdn':
        q = torch.nn.functional.normalize(q.float(), dim=-1).to(q.dtype)
        k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
        beta = beta.sigmoid()
        g = -torch.rand(batch, length, heads, device='cuda') / 10
        legacy, indexed = chunk_gated_delta_rule, indexed_chunk_gated_delta_rule
    else:
        g = torch.randn_like(q)
        options.update(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                       use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-5.0,
                       A_log=torch.zeros(heads, device='cuda'), dt_bias=torch.randn(heads * dim, device='cuda'))
        legacy, indexed = chunk_kda, indexed_chunk_kda
    src = torch.tensor([2] * len(lengths), device='cuda', dtype=torch.int32)
    dst = torch.arange(4, 4 + len(lengths), device='cuda', dtype=torch.int32)
    cp = torch.arange(8, 8 + len(lengths), device='cuda', dtype=torch.int32)
    cp_offsets = torch.tensor([min(64, x) for x in lengths], device='cuda', dtype=torch.int32)
    backing = torch.randn(12, 3, heads, dim, dim, device='cuda', dtype=dtype) / 16
    pool = backing[:, 1]
    before = backing.clone()
    extra = {}
    if packed:
        cu_cpu = torch.tensor([0, *itertools.accumulate(lengths)])
        options['cu_seqlens'] = cu_cpu.cuda()
        counts = [(x + 63) // 64 for x in lengths]
        extra['chunk_indices'] = torch.tensor([(r, c) for r, n in enumerate(counts) for c in range(n)],
                                              device='cuda', dtype=torch.int32)
        extra['chunk_offsets'] = torch.tensor([0, *itertools.accumulate(counts)], device='cuda', dtype=torch.int32)
    expected, final = legacy(q, k, v, g, beta, initial_state=pool[src].contiguous().float(),
                             output_final_state=True, **options)
    actual = indexed(q, k, v, g, beta, pool, src, dst, **options, **extra)
    saved_final = pool[dst].clone()
    observed = indexed(q, k, v, g, beta, pool, src, dst, checkpoint_ids=cp, checkpoint_offsets=cp_offsets,
                       **options, **extra)
    for a, e in ((actual, expected), (pool[dst], final.to(dtype)), (actual, observed), (pool[dst], saved_final)):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    # Compare each checkpoint with the complete legacy pipeline stopped there.
    for row, length in enumerate(lengths):
        offset = min(64, length)
        start = sum(lengths[:row]) if packed else 0
        tensors = [t[0:1, start:start + offset].contiguous() if packed else t[row:row + 1, :offset].contiguous()
                   for t in (q, k, v, g, beta)]
        single = {key: value for key, value in options.items() if key != 'cu_seqlens'}
        _, ref = legacy(*tensors, initial_state=before[2:3, 1].contiguous().float(), output_final_state=True, **single)
        torch.testing.assert_close(pool[cp[row]], ref[0].to(dtype), rtol=0, atol=0)
    untouched = torch.ones(len(pool), dtype=torch.bool, device='cuda')
    untouched[torch.cat((dst, cp))] = False
    torch.testing.assert_close(pool[untouched], before[:, 1][untouched], rtol=0, atol=0)
    for layer in (0, 2):
        torch.testing.assert_close(backing[:, layer], before[:, layer], rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('kind', ['gdn', 'kda'])
def test_prefill_split_and_checkpoint_continuation(dtype, kind):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.kda import chunk_kda

    from lmdeploy.pytorch.kernels.cuda.indexed_chunk_gated_delta import (
        indexed_chunk_gated_delta_rule,
        indexed_chunk_kda,
    )

    torch.manual_seed(72)
    dim, heads = 128, 2
    q, k, v = [torch.randn(1, 72, heads, dim, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    q, k = [torch.nn.functional.normalize(t.float(), dim=-1).to(t.dtype) for t in (q, k)]
    beta = torch.rand(1, 72, heads, device='cuda', dtype=torch.bfloat16)
    g = -torch.rand((1, 72, heads) if kind == 'gdn' else (1, 72, heads, dim), device='cuda') / 10
    options = dict(state_v_first=True)
    if kind == 'kda':
        options['safe_gate'] = True
    legacy, indexed = ((chunk_gated_delta_rule, indexed_chunk_gated_delta_rule) if kind == 'gdn'
                       else (chunk_kda, indexed_chunk_kda))
    pool = torch.randn(8, heads, dim, dim, device='cuda', dtype=dtype) / 16
    src, dst, cp = [torch.tensor([i], device='cuda') for i in (1, 2, 3)]
    tensors = (q, k, v, g, beta)
    whole = indexed(*tensors, pool, src, dst, checkpoint_ids=cp,
                    checkpoint_offsets=torch.tensor([64], device='cuda'), **options)
    checkpoint = pool[3].clone()
    first = [t[:, :64].contiguous() for t in tensors]
    last = [t[:, 64:].contiguous() for t in tensors]
    o64, state64 = legacy(*first, initial_state=pool[1:2].float(), output_final_state=True, **options)
    o8, final = legacy(*last, initial_state=state64.to(dtype).float(), output_final_state=True, **options)
    torch.testing.assert_close(pool[3:4], state64.to(dtype), rtol=0, atol=0)
    # Limits are frozen in split-tolerance.md before test execution, not fitted.
    torch.testing.assert_close(whole, torch.cat((o64, o8), dim=1), rtol=0.02, atol=0.02)
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.0001
    torch.testing.assert_close(pool[2:3], final.to(dtype), rtol=tolerance, atol=tolerance)
    # The last8 tokens cannot mutate checkpoint64. Reuse it with a new suffix.
    last[2] = -last[2]
    expected, expected_state = legacy(*last, initial_state=checkpoint[None].float(), output_final_state=True, **options)
    continued = indexed(*last, pool, cp, torch.tensor([4], device='cuda'), **options)
    torch.testing.assert_close(continued, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool[4:5], expected_state.to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(pool[3], checkpoint, rtol=0, atol=0)


@pytest.mark.parametrize('id_dtype', [torch.int32, torch.int64])
def test_indexed_prefill_masks_empty_rows_and_final_dedup(id_dtype):
    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import indexed_causal_conv1d_fwd

    torch.manual_seed(56)
    lengths = [0, 72, 1, 72, 0]
    h, d = 2, 128
    k, w, u = [torch.randn(1, sum(lengths), h, d, device='cuda', dtype=torch.bfloat16) / 32 for _ in range(3)]
    pool = torch.randn(14, h, d, d, device='cuda')
    original = pool.clone()
    src = torch.tensor([1, 0, 2, 2, 3], device='cuda', dtype=id_dtype)
    dst = torch.tensor([4, 5, 6, 0, 7], device='cuda', dtype=id_dtype)
    # row1 c=e dedup; row2 negative mask; row3 unsupported internal56 must not export.
    cp = torch.tensor([8, 5, -1, 10, 11], device='cuda', dtype=id_dtype)
    offset = torch.tensor([0, 72, 1, 56, 0], device='cuda', dtype=id_dtype)
    cu = torch.tensor([0, *itertools.accumulate(lengths)], device='cuda', dtype=id_dtype)
    counts = [(x + 63) // 64 for x in lengths]
    chunks = torch.tensor([(r, c) for r, n in enumerate(counts) for c in range(n)],
                          device='cuda', dtype=id_dtype)
    chunk_offsets = torch.tensor([0, *itertools.accumulate(counts)], device='cuda', dtype=id_dtype)
    args = dict(cu_seqlens=cu, chunk_indices=chunks, chunk_offsets=chunk_offsets)
    indexed_chunk_state_fwd(k, w, u, pool, src, dst, **args)
    expected = pool.clone()
    indexed_chunk_state_fwd(k, w, u, pool, src, dst, checkpoint_ids=cp, checkpoint_offsets=offset, **args)
    torch.testing.assert_close(pool, expected, rtol=0, atol=0)
    untouched = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]
    torch.testing.assert_close(pool[untouched], original[untouched], rtol=0, atol=0)
    # Conv independently supports arbitrary offsets; the planner must intersect
    # capabilities with recurrent and must never publish the unsupported56 above.
    c = 128
    x = torch.randn(sum(lengths), c, device='cuda', dtype=torch.bfloat16).T.unsqueeze(0)
    weight = torch.randn(c, 4, device='cuda', dtype=x.dtype)
    conv = torch.randn(14, c, 3, device='cuda', dtype=x.dtype)
    original_conv = conv.clone()
    seq = torch.repeat_interleave(torch.arange(len(lengths), device='cuda', dtype=torch.int32),
                                 torch.tensor(lengths, device='cuda'))
    no_cp = indexed_causal_conv1d_fwd(x, weight, conv, src, dst, seq_idx=seq, cu_seqlens=cu)
    expected_conv = conv.clone()
    # Disable unsupported common boundary, keep the shared final dedup.
    cp[3] = -1
    with_cp = indexed_causal_conv1d_fwd(x, weight, conv, src, dst, seq_idx=seq, cu_seqlens=cu,
                                      checkpoint_ids=cp, checkpoint_offsets=offset)
    torch.testing.assert_close(no_cp, with_cp, rtol=0, atol=0)
    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
    torch.testing.assert_close(conv[untouched], original_conv[untouched], rtol=0, atol=0)


@pytest.mark.parametrize('activation', [None, 'silu'])
def test_conv_prefill_zero_history_and_channel_tail(activation):
    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_fn, indexed_causal_conv1d_fwd

    torch.manual_seed(32)
    channels, length, width = 67, 19, 4
    x = torch.randn(length, channels, device='cuda', dtype=torch.bfloat16).T.unsqueeze(0)
    weight = torch.randn(channels, width, device='cuda', dtype=x.dtype) / 4
    seq = torch.zeros(length, device='cuda', dtype=torch.int32)
    cu = torch.tensor([0, length], device='cuda', dtype=torch.int32)
    pool = torch.zeros(3, channels, width - 1, device='cuda', dtype=x.dtype)
    src, dst = [torch.tensor([i], device='cuda', dtype=torch.int32) for i in (0, 1)]
    legacy = causal_conv1d_fn(x, weight, seq_idx=seq, activation=activation)
    actual = indexed_causal_conv1d_fwd(x, weight, pool, src, dst, seq_idx=seq, cu_seqlens=cu,
                                     activation=activation)
    expected = torch.nn.functional.conv1d(torch.nn.functional.pad(x.float(), (width - 1, 0)),
                                          weight.float()[:, None], groups=channels)
    if activation:
        expected = torch.nn.functional.silu(expected)
    torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
    torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.01)
    torch.testing.assert_close(pool[1], x[0, :, -(width - 1):], rtol=0, atol=0)
    assert torch.count_nonzero(pool[0]) == 0
    assert torch.count_nonzero(pool[2]) == 0


@pytest.mark.parametrize('steps', [1, 4])
def test_prefill_units_feed_indexed_decode_without_payload_conversion(steps):
    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_update, indexed_causal_conv1d_fwd
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule
    from lmdeploy.pytorch.kernels.cuda.indexed_chunk_gated_delta import indexed_chunk_gated_delta_rule

    torch.manual_seed(2120)
    h, hv, dim, channels = 1, 2, 128, 512
    conv = torch.randn(12, channels, 3, device='cuda', dtype=torch.bfloat16) / 16
    state = torch.randn(12, hv, dim, dim, device='cuda', dtype=torch.bfloat16) / 16
    weight = torch.randn(channels, 4, device='cuda', dtype=conv.dtype) / 4
    x = torch.randn(72, channels, device='cuda', dtype=conv.dtype).T.unsqueeze(0)
    src, dst, cp = [torch.tensor([i], device='cuda', dtype=torch.int64) for i in (1, 2, 3)]
    common = dict(checkpoint_ids=cp, checkpoint_offsets=torch.tensor([64], device='cuda', dtype=torch.int64))

    def qkv(mixed):
        q, k, v = mixed.transpose(1, 2).split([h * dim, h * dim, hv * dim], dim=-1)
        q = torch.nn.functional.normalize(q.reshape(1, -1, h, dim).float(), dim=-1).to(mixed.dtype)
        k = torch.nn.functional.normalize(k.reshape(1, -1, h, dim).float(), dim=-1).to(mixed.dtype)
        return q, k, v.reshape(1, -1, hv, dim).contiguous()

    mixed = indexed_causal_conv1d_fwd(x, weight, conv, src, dst,
                                     seq_idx=torch.zeros(72, device='cuda', dtype=torch.int32),
                                     cu_seqlens=torch.tensor([0, 72], device='cuda', dtype=torch.int64), **common)
    q, k, v = qkv(mixed)
    g = -torch.rand(1, 72, hv, device='cuda') / 10
    beta = torch.rand_like(g)
    indexed_chunk_gated_delta_rule(q, k, v, g, beta, state, src, dst, state_v_first=True, **common)
    checkpoint_conv, checkpoint_state = conv[3].clone(), state[3].clone()
    final_conv, final_state = conv[2].clone(), state[2].clone()
    decode_x = torch.randn(1, steps, channels, device='cuda', dtype=conv.dtype).transpose(1, 2)
    outputs = torch.arange(4, 4 + steps, device='cuda', dtype=torch.int64)[None]
    # The candidate consumes pool IDs directly; only the reference gathers.
    mixed = causal_conv1d_update(decode_x, conv, weight, conv_state_indices=dst, output_state_indices=outputs)
    legacy_conv = final_conv[None].clone()
    ref_mixed = torch.cat([causal_conv1d_update(decode_x[:, :, i:i + 1], legacy_conv, weight)
                          for i in range(steps)], dim=-1)
    torch.testing.assert_close(mixed, ref_mixed, rtol=0, atol=0)
    q, k, v = qkv(mixed)
    g = -torch.rand(1, steps, hv, device='cuda') / 10
    beta = torch.rand_like(g)
    actual, _ = fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=state,
                                                state_indices=dst, output_state_indices=outputs,
                                                output_final_state=True, transpose_state_layout=True)
    ring = torch.zeros(1, steps + 1, hv, dim, dim, device='cuda', dtype=state.dtype)
    ring[:, 0] = final_state
    expected, _ = fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=ring,
                                                  cache_seqlens=torch.zeros(1, device='cuda', dtype=torch.int32),
                                                  output_final_state=True, transpose_state_layout=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state[outputs], ring[:, 1:], rtol=0, atol=0)
    torch.testing.assert_close(conv[outputs[0, -1]], legacy_conv[0], rtol=0, atol=0)
    for tensor, idx, ref in ((conv, 2, final_conv), (conv, 3, checkpoint_conv),
                             (state, 2, final_state), (state, 3, checkpoint_state)):
        torch.testing.assert_close(tensor[idx], ref, rtol=0, atol=0)
