# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch


@pytest.mark.parametrize('steps', [1, 3, 6])
@pytest.mark.parametrize('state_dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('with_bias', [False, True])
def test_fused_kda_gate_preserves_recurrence_and_dummy_state(steps, state_dtype, with_bias):
    from fla.ops.kda.gate import kda_gate_fwd

    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule as run

    torch.manual_seed(7)
    batch, heads, dim, ring = 3, 2, 128, 6
    mixed = torch.randn(batch, steps, 3 * heads * dim, device='cuda', dtype=torch.bfloat16)
    q, k, v = [x.unflatten(-1, (heads, dim)) for x in mixed.chunk(3, dim=-1)]
    raw_gate = torch.randn_like(q).contiguous()
    raw_beta = torch.randn(batch, steps, heads, device='cuda', dtype=torch.bfloat16)
    a_log = torch.randn(heads, device='cuda')
    dt_bias = torch.randn(heads * dim, device='cuda') if with_bias else None
    initial = torch.randn(batch, ring, heads, dim, dim, device='cuda', dtype=state_dtype) * 0.1
    state = initial.clone()
    kwargs = dict(state_indices=torch.tensor([2, -1, 0], device='cuda'),
                  cache_seqlens=torch.tensor([5, 0, 3], device='cuda', dtype=torch.int32),
                  output_final_state=True, transpose_state_layout=True, use_qk_l2norm_in_kernel=True)
    gate = kda_gate_fwd(raw_gate, a_log, dt_bias, lower_bound=-5.0)
    expected, expected_state = run(q, k, v, g=gate, beta=raw_beta.float().sigmoid(),
                                   initial_state=initial.clone(), **kwargs)
    actual, _ = run(q, k, v, g=raw_gate, beta=raw_beta, a_log=a_log, dt_bias=dt_bias,
                    lower_bound=-5.0, initial_state=state, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(state[1], initial[1], rtol=0, atol=0)
    assert torch.count_nonzero(actual[1]) == 0


def test_fused_kda_gate_cuda_graph_replay():
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule as run

    torch.manual_seed(29)
    q, k, v, raw_gate = [torch.randn(1, 6, 2, 128, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
    raw_beta = torch.randn(1, 6, 2, device='cuda', dtype=torch.bfloat16)
    initial = torch.randn(1, 6, 2, 128, 128, device='cuda')
    state = initial.clone()
    history = torch.tensor([5], device='cuda', dtype=torch.int32)
    kwargs = dict(g=raw_gate, beta=raw_beta, a_log=torch.randn(2, device='cuda'),
                  dt_bias=torch.randn(256, device='cuda'), lower_bound=-5.0, cache_seqlens=history,
                  output_final_state=True, transpose_state_layout=True, use_qk_l2norm_in_kernel=True)
    run(q, k, v, initial_state=state, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, _ = run(q, k, v, initial_state=state, **kwargs)
    raw_gate.normal_()
    raw_beta.normal_()
    history.fill_(2)
    expected, expected_state = run(q, k, v, initial_state=initial.clone(), **kwargs)
    state.copy_(initial)
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def test_kda_unbounded_gate_keeps_existing_arithmetic():
    from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl

    torch.manual_seed(43)
    impl = CudaKdaImpl()
    q, k, v, raw_gate = [torch.randn(1, 1, 2, 128, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
    raw_beta = torch.randn(1, 1, 2, device='cuda', dtype=torch.bfloat16)
    a_log = torch.randn(2, device='cuda')
    dt_bias = torch.randn(256, device='cuda')
    initial = torch.randn(1, 2, 128, 128, device='cuda')
    gate = impl.kda_gate(raw_gate, a_log, dt_bias)
    expected, expected_state = impl.recurrent_func(
        q, k, v, g=gate, beta=raw_beta.float().sigmoid(), initial_state=initial.clone(),
        output_final_state=True, transpose_state_layout=True, use_qk_l2norm_in_kernel=True)
    actual, state = impl._decode_recurrent(
        q, k, v, raw_gate, raw_beta, a_log, dt_bias, initial.clone(), lower_bound=None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize('beta_dtype', [torch.bfloat16, torch.float32])
def test_fused_kda_gate_extreme_beta(beta_dtype):
    from fla.ops.kda.gate import kda_gate_fwd

    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule as run

    torch.manual_seed(53)
    q, k, v, raw_gate = [torch.randn(9, 1, 2, 128, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
    raw_beta = torch.tensor([-100, -89, -88, -87.5, -86, 0, 86, 88, 100], device='cuda', dtype=beta_dtype)
    raw_beta = raw_beta[:, None, None].expand(9, 1, 2).contiguous()
    a_log = torch.randn(2, device='cuda')
    initial = torch.randn(9, 2, 128, 128, device='cuda')
    kwargs = dict(output_final_state=True, transpose_state_layout=True, use_qk_l2norm_in_kernel=True)
    expected, expected_state = run(
        q, k, v, g=kda_gate_fwd(raw_gate, a_log, lower_bound=-5.0), beta=raw_beta.float().sigmoid(),
        initial_state=initial.clone(), **kwargs)
    actual, state = run(q, k, v, g=raw_gate, beta=raw_beta, a_log=a_log, lower_bound=-5.0,
                        initial_state=initial.clone(), **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize('invalid', [
    'missing_bound', 'positive_bound', 'missing_beta', 'untransposed', 'missing_a_log'
])
def test_fused_kda_gate_rejects_unsupported_contract(invalid):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule as run

    q = torch.empty(1, 1, 2, 128)
    kwargs = dict(g=q, beta=torch.empty(1, 1, 2), a_log=torch.zeros(2),
                  dt_bias=torch.zeros(256), lower_bound=-5.0, transpose_state_layout=True)
    overrides = dict(missing_bound=dict(lower_bound=None), positive_bound=dict(lower_bound=1.0),
                     missing_beta=dict(beta=None), untransposed=dict(transpose_state_layout=False),
                     missing_a_log=dict(a_log=None))
    kwargs.update(overrides[invalid])
    with pytest.raises(ValueError, match='KDA gating'):
        run(q, q, q, initial_state=torch.empty(1, 2, 128, 128), **kwargs)
