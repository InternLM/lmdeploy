"""Strided speculative decode IO must preserve output and every ring slot."""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('implementation,strided', [('CausalConv1dTilelangImpl', True),
                                                   ('CausalConv1dDaoImpl', False)])
def test_decode_layout_contract(implementation, strided):
    from lmdeploy.pytorch.backends.cuda import causal_conv1d as backend

    impl = getattr(backend, implementation).__new__(getattr(backend, implementation))
    batch, tokens, hidden = 3, 8, 128
    x = torch.randn(1, batch * tokens, hidden)
    captured = {}

    def update(inp, *args, **kwargs):
        captured['input'] = inp
        captured['output'] = torch.empty_like(inp)
        return captured['output']

    impl.update_fn = update
    meta = SimpleNamespace(conv_state_indices=torch.arange(batch, dtype=torch.int32),
                           cache_seqlens=torch.zeros(batch, dtype=torch.int32))
    out = impl.decode(x, torch.empty(hidden, 4), None, torch.empty(batch, hidden, 11), meta, 'silu')
    assert out.shape == x.shape
    assert captured['input'].is_contiguous() is not strided
    assert (captured['input'].data_ptr() == x.data_ptr()) is strided
    assert out.is_contiguous()
    assert (captured['output'].data_ptr() == out.data_ptr()) is strided


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('seqlen', [1, 4, 8])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('activation', [None, 'silu'])
@pytest.mark.parametrize('has_bias', [False, True])
def test_strided_ring_matches_contiguous_and_graph(seqlen, dtype, activation, has_bias):
    pytest.importorskip('tilelang')
    from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_update

    torch.manual_seed(41)
    batch, hidden, width, states = 5, 128, 4, 7
    slots = width + seqlen - 1
    physical = torch.randn(batch, seqlen, hidden, device='cuda', dtype=dtype)
    x = physical.transpose(1, 2)
    weight = torch.randn(hidden, width, device='cuda', dtype=dtype)
    bias = torch.randn(hidden, device='cuda', dtype=dtype) if has_bias else None
    initial = torch.randn(states, hidden, slots, device='cuda', dtype=dtype)
    backing = torch.empty(states, 3, hidden, slots, device='cuda', dtype=dtype)
    state = backing[:, 1]
    state.copy_(initial)
    reference_state = initial.clone()
    ids = torch.tensor([6, 0, -1, 4, 2], device='cuda', dtype=torch.int32)
    lengths = torch.tensor([0, slots - 1, 2, slots + 3, 2 * slots + 1], device='cuda', dtype=torch.int32)

    def run(inp, cache):
        return causal_conv1d_update(inp, cache, weight, bias=bias, activation=activation,
                                   cache_seqlens=lengths, conv_state_indices=ids)

    reference = run(x.contiguous(), reference_state)
    actual = run(x, state)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
    assert actual.transpose(1, 2).is_contiguous()
    assert torch.count_nonzero(actual[2]) == 0

    # Independent grouped-convolution oracle; no optional Dao package needed.
    oracle = torch.zeros_like(x)
    oracle_state = initial.clone()
    for row, (state_id, length) in enumerate(zip([6, 0, -1, 4, 2],
                                               [0, slots - 1, 2, slots + 3, 2 * slots + 1])):
        if state_id < 0:
            continue
        history_ids = [(length - width + 1 + j) % slots for j in range(width - 1)]
        history = initial[state_id, :, history_ids]
        inputs = torch.cat((history, x[row]), dim=-1).float()[None]
        result = torch.nn.functional.conv1d(inputs, weight.float()[:, None],
                                           bias=None if bias is None else bias.float(), groups=hidden)
        if activation == 'silu':
            result = torch.nn.functional.silu(result)
        oracle[row] = result[0].to(dtype)
        for token in range(seqlen):
            oracle_state[state_id, :, (length + token) % slots] = x[row, :, token]
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, oracle, atol=tolerance / 10, rtol=tolerance)
    torch.testing.assert_close(state, oracle_state, atol=0, rtol=0)

    # Capture without counting the capture execution as a replay. Reset both
    # caches, then check repeated in-place ring updates, including wraparound.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(x, state)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run(x, state)
    state.copy_(initial)
    reference_state.copy_(initial)
    for _ in range(3):
        graph.replay()
        reference = run(x.contiguous(), reference_state)
        torch.testing.assert_close(graph_output, reference, atol=0, rtol=0)
        torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
