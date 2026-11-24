# Copyright (c) OpenMMLab. All rights reserved.
"""Independent logical-time recurrence and exact indexed-decode
counterexamples.

The reference uses CPU PyTorch, not LMDeploy/FLA/TileLang/Triton or physical IDs. State is [B, HV, K, V]; gates and beta
are already transformed. Output uses the updated FP32 state before each token boundary roundtrips through cache dtype.
Using FP32 cache_dtype also gives an unrounded logical recurrence for prefill.

Controlled GPU cases use dyadic values and zero/underflow decays for exact comparisons. Random-input cases additionally
check fused gate transforms, normalization and chunk-prefill against mathematical tolerances; these do not replace bit-
exact ring parity, checkpoint observability or source immutability. Only GPU cases skip without CUDA/TileLang; analytic
CPU reference checks still run.
"""

import math
from importlib.util import find_spec

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or find_spec('tilelang') is None,
    reason='Indexed recurrent kernel requires CUDA and TileLang',
)


def _recurrent_decode_reference(q, k, v, log_decay, beta, initial, *, cache_dtype, scale=1.0):
    h = initial.float().clone()
    group = v.shape[2] // q.shape[2]
    outputs, boundaries = [], []
    for step in range(q.shape[1]):
        query = q[:, step].float().repeat_interleave(group, dim=1) * scale
        key = k[:, step].float().repeat_interleave(group, dim=1)
        decay = log_decay[:, step].float().exp()
        if decay.ndim == 2:
            decay = decay[..., None]
        decayed = h * decay[..., None]
        prediction = torch.einsum('bhk,bhkv->bhv', key, decayed)
        residual = (v[:, step].float() - prediction) * beta[:, step].float()[..., None]
        updated = decayed + key[..., None] * residual[..., None, :]
        outputs.append(torch.einsum('bhk,bhkv->bhv', query, updated).to(q.dtype))
        boundary = updated.to(cache_dtype)
        boundaries.append(boundary)
        h = boundary.float()
    return torch.stack(outputs, dim=1), torch.stack(boundaries, dim=1)


@pytest.mark.parametrize('dtype,epsilon', [(torch.bfloat16, 2**-8), (torch.float16, 2**-11)])
def test_output_precedes_cache_rounding_and_every_token_roundtrips(dtype, epsilon):
    # At the first state ULP midpoint, round-to-even yields1.0. Output q=.75
    # must use the pre-cast value; retaining that value for token2 is also wrong.
    q = torch.full((1, 2, 1, 1), 0.75, dtype=dtype)
    k = torch.ones_like(q)
    v = torch.full_like(q, 2)
    initial = torch.ones(1, 1, 1, 1, dtype=dtype)
    output, states = _recurrent_decode_reference(
        q, k, v, torch.zeros(1, 2, 1), torch.full((1, 2, 1), epsilon), initial, cache_dtype=dtype
    )
    expected = torch.tensor(0.75 * (1 + epsilon), dtype=dtype)
    assert expected != torch.tensor(0.75, dtype=dtype)
    assert torch.equal(output, expected.expand_as(output))
    assert torch.equal(states, torch.ones_like(states))
    unrounded_second = (1 + epsilon) + epsilon * (2 - (1 + epsilon))
    assert torch.tensor(unrounded_second, dtype=dtype) != 1
    assert torch.equal(initial, torch.ones_like(initial))


@pytest.mark.parametrize('channelwise', [False, True])
def test_analytic_grouped_heads_and_decay_before_prediction(channelwise):
    # One q/k head drives two different value/state heads. Decay must happen
    # before the residual prediction, not merely before adding the outer product.
    q = torch.tensor([[[[1.0, 0.0]]]])
    k = q.clone()
    v = torch.tensor([[[[3.0], [7.0]]]])
    initial = torch.tensor([[[[2.0], [4.0]], [[6.0], [8.0]]]])
    g = torch.full((1, 1, 2, 2) if channelwise else (1, 1, 2), -math.log(2))
    if channelwise:
        g[..., 1] = 0
    output, states = _recurrent_decode_reference(
        q, k, v, g, torch.full((1, 1, 2), 0.5), initial, cache_dtype=torch.float32
    )
    # Decayed first rows1/3; interpolate toward3/7 =>2/5.
    torch.testing.assert_close(output.flatten(), torch.tensor([2.0, 5.0]), rtol=0, atol=1e-6)
    expected = torch.tensor([[[[[2.0], [4.0 if channelwise else 2.0]], [[5.0], [8.0 if channelwise else 4.0]]]]])
    torch.testing.assert_close(states, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize('channelwise', [False, True])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_128_steps_resumable_at_eight_independent_boundaries(channelwise, dtype):
    generator = torch.Generator().manual_seed(2112)
    q = torch.randn(2, 128, 1, 4, generator=generator) / 4
    k = torch.randn(2, 128, 1, 4, generator=generator) / 4
    v = torch.randn(2, 128, 2, 3, generator=generator) / 4
    g = -torch.rand((2, 128, 2, 4) if channelwise else (2, 128, 2), generator=generator)
    beta = torch.rand(2, 128, 2, generator=generator)
    initial = torch.randn(2, 2, 4, 3, generator=generator).to(dtype)
    original = initial.clone()
    output, states = _recurrent_decode_reference(q, k, v, g, beta, initial, cache_dtype=dtype)
    for boundary in range(16, 129, 16):
        short_output, short_states = _recurrent_decode_reference(
            q[:, :boundary],
            k[:, :boundary],
            v[:, :boundary],
            g[:, :boundary],
            beta[:, :boundary],
            initial,
            cache_dtype=dtype,
        )
        assert torch.equal(short_output, output[:, :boundary])
        assert torch.equal(short_states[:, -1], states[:, boundary - 1])
        if boundary < 128:
            continued, tail = _recurrent_decode_reference(
                q[:, boundary:],
                k[:, boundary:],
                v[:, boundary:],
                g[:, boundary:],
                beta[:, boundary:],
                states[:, boundary - 1],
                cache_dtype=dtype,
            )
            assert torch.equal(continued, output[:, boundary:])
            assert torch.equal(tail, states[:, boundary:])
    assert torch.equal(initial, original)


def _logical(tensor, transposed):
    return tensor.transpose(-1, -2) if transposed else tensor


@requires_cuda
@pytest.mark.parametrize('layout', ['default', 'transposed', 'kda'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_128_candidate_boundaries_and_eight_checkpoint_branches(layout, dtype):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

    transposed = layout != 'default'
    generator = torch.Generator().manual_seed(20261017)
    # DistinctK/V positions, groupedq/k/valueheads, stridedlayer andpermutedIDs.
    initial = (torch.randint(-8, 9, (24, 3, 4, 128, 128), generator=generator) / 16).to(dtype)
    backing = initial.cuda()
    pool = backing[:, 1]
    source = [1, 1]
    expected = _logical(initial[source, 1], transposed).clone()
    banks = [list(range(2, 7)), list(range(7, 12))]
    frozen = {1: expected[0].clone()}
    for iteration in range(32):
        q = torch.zeros(2, 4, 2, 128, dtype=torch.bfloat16)
        k = torch.zeros_like(q)
        for row in range(2):
            for step in range(4):
                for head in range(2):
                    coordinate = (iteration * 4 + step + head * 11 + row * 7) % 128
                    k[row, step, head, coordinate] = 1
                    q[row, step, head, (coordinate + (step % 2) * 5) % 128] = 0.5 if step % 2 else 1
        v = (torch.randint(-8, 9, (2, 4, 4, 128), generator=generator) / 16).to(q.dtype)
        gate = torch.zeros((2, 4, 4, 128) if layout == 'kda' else (2, 4, 4))
        if layout == 'kda':
            gate[..., (iteration % 4) :: 4] = -1000
        else:
            gate[:, :, iteration % 4] = -1000
        beta = torch.ones(2, 4, 4)
        beta[:, 0] = 0
        wanted, states = _recurrent_decode_reference(q, k, v, gate, beta, expected, cache_dtype=dtype)
        destinations = [list(reversed([u for u in bank if u != src]))[:4] for bank, src in zip(banks, source)]
        before = backing.clone()
        output, _ = fused_recurrent_gated_delta_rule(
            q.cuda(),
            k.cuda(),
            v.cuda(),
            gate.cuda(),
            beta.cuda(),
            scale=1.0,
            initial_state=pool,
            output_final_state=True,
            state_indices=torch.tensor(source, device='cuda'),
            output_state_indices=torch.tensor(destinations, device='cuda'),
            transpose_state_layout=transposed,
        )
        torch.testing.assert_close(output.cpu(), wanted, rtol=0, atol=0)
        actual_states = _logical(pool[torch.tensor(destinations, device='cuda')], transposed).cpu()
        torch.testing.assert_close(actual_states, states, rtol=0, atol=0)
        untouched = torch.ones(24, device='cuda', dtype=torch.bool)
        untouched[torch.tensor(destinations, device='cuda').flatten()] = False
        assert torch.equal(pool[untouched], before[:, 1][untouched])
        assert torch.equal(backing[:, 0], before[:, 0]) and torch.equal(backing[:, 2], before[:, 2])
        for row in range(2):
            accepted = (iteration + row) % 5
            if row == 0 and iteration % 4 == 3:
                accepted = 1 + (iteration // 4) % 4
            if accepted:
                expected[row] = states[row, accepted - 1]
                source[row] = destinations[row][accepted - 1]
        if iteration % 4 == 3:
            unit = source[0]
            assert unit not in frozen
            frozen[unit] = expected[0].clone()
            banks[0][banks[0].index(unit)] = 12 + iteration // 4
            source = [unit, unit]
            expected = torch.stack((expected[0], expected[0]))
        for unit, state in frozen.items():
            torch.testing.assert_close(_logical(pool[unit], transposed).cpu(), state, rtol=0, atol=0)
    assert len(frozen) == 9  # Initialsource plus8new independentcheckpointbranches.


@requires_cuda
@pytest.mark.parametrize('layout', ['default', 'transposed', 'kda'])
@pytest.mark.parametrize('dtype,epsilon', [(torch.bfloat16, 2**-8), (torch.float16, 2**-11)])
def test_output_before_rounding_and_dtype_roundtrip_mutation_counterexample(layout, dtype, epsilon):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

    transposed = layout != 'default'
    q = torch.zeros(2, 2, 1, 128, dtype=dtype)
    k = torch.zeros_like(q)
    q[..., 0] = 0.75
    k[..., 0] = 1
    v = torch.full_like(q, 2)
    gate = torch.zeros((2, 2, 1, 128) if layout == 'kda' else (2, 2, 1))
    beta = torch.full((2, 2, 1), epsilon)
    initial = torch.ones(2, 1, 128, 128, dtype=dtype)
    wanted, states = _recurrent_decode_reference(q, k, v, gate, beta, initial, cache_dtype=dtype)
    assert torch.all(wanted != torch.tensor(0.75, dtype=dtype))
    pool = torch.ones(8, 1, 128, 128, device='cuda', dtype=dtype)
    output, _ = fused_recurrent_gated_delta_rule(
        q.cuda(),
        k.cuda(),
        v.cuda(),
        gate.cuda(),
        beta.cuda(),
        scale=1.0,
        initial_state=pool,
        output_final_state=True,
        state_indices=torch.tensor([2, 2], device='cuda'),
        output_state_indices=torch.tensor([[3, 4], [5, 6]], device='cuda'),
        transpose_state_layout=transposed,
    )
    torch.testing.assert_close(output.cpu(), wanted, rtol=0, atol=0)
    torch.testing.assert_close(
        _logical(pool[torch.tensor([[3, 4], [5, 6]], device='cuda')], transposed).cpu(), states, rtol=0, atol=0
    )
    assert torch.equal(pool[2].cpu(), initial[0])


def _normalize_reference(x):
    x = x.float()
    return x / (x.square().sum(dim=-1, keepdim=True) + 1e-6).sqrt()


def _assert_math_close(actual, expected, *, chunk=False):
    # Frozen before execution. These bound arithmetic-order/exp/reduction error,
    # not physical-index correctness: same-path ring comparisons stay bit-exact.
    # Chunk math additionally rounds intermediate WY/matmul activations to BF16.
    if chunk or actual.dtype == torch.bfloat16:
        rtol, atol = 0.02, 0.002
    elif actual.dtype == torch.float16:
        rtol, atol = 0.003, 0.0003
    else:
        rtol, atol = 0.0002, 0.000002
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual.cpu(), expected.to(actual.dtype), rtol=rtol, atol=atol)


@requires_cuda
@pytest.mark.parametrize('layout', ['default', 'transposed', 'kda', 'kda_fused'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('normalize', [False, True])
def test_random_decode_against_independent_math_and_exact_ring(layout, dtype, normalize):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

    generator = torch.Generator().manual_seed(20261018)
    q, k = [torch.randn(2, 4, 1, 128, generator=generator).to(torch.bfloat16) / 8 for _ in range(2)]
    v = torch.randn(2, 4, 2, 128, generator=generator).to(torch.bfloat16) / 4
    channelwise, transposed = layout.startswith('kda'), layout != 'default'
    gate = -torch.rand((2, 4, 2, 128) if channelwise else (2, 4, 2), generator=generator)
    beta = torch.rand(2, 4, 2, generator=generator)
    options = dict(transpose_state_layout=transposed, use_qk_l2norm_in_kernel=normalize)
    log_decay, logical_beta = gate, beta
    if layout == 'kda_fused':
        gate = torch.randn(2, 4, 2, 128, generator=generator).to(torch.bfloat16)
        gate[..., 0] = -30  # Both saturation ends coexist with ordinary logits.
        gate[..., 1] = 30
        beta = torch.randn(2, 4, 2, generator=generator).to(torch.bfloat16)
        a_log = torch.tensor([-0.7, 0.4])
        bias = torch.randn(2, 128, generator=generator) / 3
        log_decay = -5 * torch.sigmoid(a_log.exp()[:, None] * (gate.float() + bias))
        logical_beta = beta.float().sigmoid()
        options.update(a_log=a_log.cuda(), dt_bias=bias.cuda(), lower_bound=-5.0)
    initial = torch.randn(12, 3, 2, 128, 128, generator=generator).to(dtype) / 16
    backing = initial.cuda()
    pool = backing[:, 1]
    source = torch.tensor([2, 2], device='cuda')
    destination = torch.tensor([[10, 5, 8, 3], [4, 9, 6, 7]], device='cuda')
    logical = _logical(initial[[2, 2], 1], transposed)
    query, key = (_normalize_reference(x) for x in (q, k)) if normalize else (q, k)
    expected, states = _recurrent_decode_reference(
        query, key, v, log_decay, logical_beta, logical, cache_dtype=dtype, scale=128**-0.5
    )
    tensors = [x.cuda() for x in (q, k, v, gate, beta)]
    actual, _ = fused_recurrent_gated_delta_rule(
        *tensors, initial_state=pool, state_indices=source, output_state_indices=destination,
        output_final_state=True, **options
    )
    _assert_math_close(actual, expected)
    _assert_math_close(_logical(pool[destination], transposed), states)
    # Independent addressing with identical execution math must remain exact.
    ring = torch.zeros(2, 5, 2, 128, 128, device='cuda', dtype=dtype)
    ring[:, 0] = initial[2, 1].cuda()
    baseline, _ = fused_recurrent_gated_delta_rule(
        *tensors, initial_state=ring, cache_seqlens=torch.zeros(2, device='cuda', dtype=torch.int32),
        output_final_state=True, **options
    )
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    torch.testing.assert_close(pool[destination], ring[:, 1:], rtol=0, atol=0)
    untouched = torch.ones(12, dtype=torch.bool)
    untouched[destination.cpu().flatten()] = False
    assert torch.equal(pool.cpu()[untouched], initial[:, 1][untouched])
    assert torch.equal(backing[:, 0].cpu(), initial[:, 0])
    assert torch.equal(backing[:, 2].cpu(), initial[:, 2])


@requires_cuda
@pytest.mark.parametrize('kind', ['gdn', 'kda_fused'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_chunk_72_checkpoint_against_unrounded_independent_recurrence(kind, dtype):
    from lmdeploy.pytorch.kernels.cuda.indexed_chunk_gated_delta import (
        indexed_chunk_gated_delta_rule,
        indexed_chunk_kda,
    )

    generator = torch.Generator().manual_seed(211272)
    q, k, v = [torch.randn(2, 72, 2, 128, generator=generator).to(torch.bfloat16) for _ in range(3)]
    v = v / 4
    query, key = [_normalize_reference(x).to(x.dtype) for x in (q, k)]
    beta = torch.rand(2, 72, 2, generator=generator)
    gate = -torch.rand(2, 72, 2, generator=generator) / 10
    options = dict(state_v_first=True)
    log_decay, logical_beta = gate, beta
    if kind == 'gdn':
        q, k = query, key
        kernel = indexed_chunk_gated_delta_rule
    else:
        kernel = indexed_chunk_kda
        # Use the local generator for reproducibility, not ambient torch RNG.
        gate = torch.randn(q.shape, generator=generator).to(q.dtype)
        beta = torch.randn(beta.shape, generator=generator).to(q.dtype)
        a_log = torch.tensor([-0.7, 0.4])
        bias = torch.randn(2, 128, generator=generator) / 3
        log_decay = -5 * torch.sigmoid(a_log.exp()[:, None] * (gate.float() + bias))
        logical_beta = beta.float().sigmoid()
        options.update(A_log=a_log.cuda(), dt_bias=bias.cuda(), lower_bound=-5.0, safe_gate=True,
                       use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, use_qk_l2norm_in_kernel=True)
    initial = torch.randn(9, 3, 2, 128, 128, generator=generator).to(dtype) / 16
    backing = initial.cuda()
    pool = backing[:, 1]
    source = torch.tensor([2, 2], device='cuda')
    destination = torch.tensor([7, 4], device='cuda')
    checkpoint = torch.tensor([3, 6], device='cuda')
    # FP32 logical recurrence has no cache-dtype rounding inside the72 tokens.
    # Checkpoint export64 is observational, not an intermediate cast in this path.
    expected, states = _recurrent_decode_reference(
        query, key, v, log_decay, logical_beta, initial[[2, 2], 1].transpose(-1, -2),
        cache_dtype=torch.float32, scale=128**-0.5
    )
    tensors = [x.cuda() for x in (q, k, v, gate, beta)]
    actual = kernel(*tensors, pool, source, destination, checkpoint_ids=checkpoint,
                    checkpoint_offsets=torch.tensor([64, 64], device='cuda'), **options)
    _assert_math_close(actual, expected, chunk=True)
    _assert_math_close(pool[destination].transpose(-1, -2), states[:, -1], chunk=True)
    _assert_math_close(pool[checkpoint].transpose(-1, -2), states[:, 63], chunk=True)
    saved_final, saved_checkpoint = pool[destination].clone(), pool[checkpoint].clone()
    without = kernel(*tensors, pool, source, destination, **options)
    torch.testing.assert_close(without, actual, rtol=0, atol=0)
    torch.testing.assert_close(pool[destination], saved_final, rtol=0, atol=0)
    torch.testing.assert_close(pool[checkpoint], saved_checkpoint, rtol=0, atol=0)
    untouched = torch.ones(9, dtype=torch.bool)
    untouched[[7, 4, 3, 6]] = False
    assert torch.equal(pool.cpu()[untouched], initial[:, 1][untouched])
    assert torch.equal(backing[:, 0].cpu(), initial[:, 0])
    assert torch.equal(backing[:, 2].cpu(), initial[:, 2])
