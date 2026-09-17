"""Split-V decode preserves all FP32 circular snapshots, including rollback."""
import importlib

import pytest
import torch


@pytest.mark.parametrize('change', ['none', 'seqlen', 'heads', 'key_dim', 'input_dtype', 'state_dtype',
                                  'num_states', 'layout', 'circular', 'architecture'])
def test_split_v_policy(monkeypatch, change):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import _use_split_v_spec_decode

    shape = [3, 8, 32, 128]
    if change == 'seqlen':
        shape[1] = 4
    elif change == 'heads':
        shape[2] = 16
    elif change == 'key_dim':
        shape[3] = 64
    q = torch.empty(shape, device='meta', dtype=torch.float16 if change == 'input_dtype' else torch.bfloat16)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a: (8, 0) if change == 'architecture' else (9, 0))
    enabled = _use_split_v_spec_decode(q, torch.bfloat16 if change == 'state_dtype' else torch.float32,
                                       9 if change == 'num_states' else 8, change != 'layout',
                                       None if change == 'circular' else object())
    assert enabled is (change == 'none')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('batch', [1, 3, 8])
@pytest.mark.parametrize('padded', [False, True])
def test_split_v_exact_snapshots_and_graph_rollback(monkeypatch, batch, padded):
    pytest.importorskip('tilelang')
    mod = importlib.import_module('lmdeploy.pytorch.kernels.cuda.gated_delta_rule')
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip('SM90 specialization')
    torch.manual_seed(43)
    q = torch.randn(batch, 8, 32, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn_like(q)
    packed = torch.randn(batch, 8, 8192, device='cuda', dtype=torch.bfloat16)
    v = packed[:, :, 4096:].view(batch, 8, 32, 128)
    g = -torch.rand(batch, 8, 32, device='cuda')
    beta = torch.rand_like(g).bfloat16()
    states = batch + 2
    initial = torch.randn(states, 8, 32, 128, 128, device='cuda') * .01
    backing = torch.empty(states, 3, 8, 32, 128, 128, device='cuda')
    state = backing[:, 1]
    state.copy_(initial)
    ref_state = initial.clone()
    ids = torch.arange(batch, device='cuda', dtype=torch.int64)
    if padded:
        ids[0] = -1
        if batch > 1:
            ids[-1] = states
    lengths = torch.arange(batch, device='cuda', dtype=torch.int32) * 7

    def run(cache):
        return mod.fused_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=cache,
                                                   output_final_state=True, use_qk_l2norm_in_kernel=True,
                                                   state_indices=ids, cache_seqlens=lengths,
                                                   transpose_state_layout=True)[0]

    with monkeypatch.context() as ctx:
        ctx.setattr(mod, '_use_split_v_spec_decode', lambda *a: False)
        reference = run(ref_state)
    actual = run(state)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(state, ref_state, atol=0, rtol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(state)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(state)
    state.copy_(initial)
    ref_state.copy_(initial)
    for accepted in [0, 3, 7, 1]:
        graph.replay()
        with monkeypatch.context() as ctx:
            ctx.setattr(mod, '_use_split_v_spec_decode', lambda *a: False)
            reference = run(ref_state)
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
        torch.testing.assert_close(state, ref_state, atol=0, rtol=0)
        lengths.add_(accepted + 1)
