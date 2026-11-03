# Copyright (c) OpenMMLab. All rights reserved.
"""Offline-selected Hopper launch must preserve indexed KDA state and graph
math."""
import pytest
import torch
import triton


@pytest.mark.parametrize('heads', [8, 64])
@pytest.mark.parametrize('length', [511, 512, 513, 1023, 1024, 8191, 8192, 8193, 32768])
@pytest.mark.parametrize('checkpoint', ['interior', 'final'])
def test_hopper_single_request_kda_tiling_is_bitexact(monkeypatch, heads, length, checkpoint):
    from lmdeploy.pytorch.kernels.cuda import indexed_chunk_state as state_module
    from lmdeploy.pytorch.kernels.cuda.indexed_chunk_gated_delta import indexed_chunk_kda
    if not state_module.IS_NVIDIA_HOPPER or not state_module._is_h200(torch.device('cuda', 0)):
        pytest.skip('Offline launch selection is H200-only.')
    torch.manual_seed(202620)
    q, k, v, gate = [torch.randn(1, length, heads, 128, device='cuda', dtype=torch.bfloat16) / 8
                     for _ in range(4)]
    beta = torch.randn(1, length, heads, device='cuda', dtype=torch.bfloat16)
    alog = torch.zeros(heads, device='cuda')
    bias = torch.randn(heads * 128, device='cuda')
    initial = torch.randn(5, 3, heads, 128, 128, device='cuda') / 16
    initial[0].zero_()
    actual, reference = initial.clone(), initial.clone()
    src = torch.tensor([1], device='cuda')
    dst = torch.tensor([1 if checkpoint == 'interior' else 2], device='cuda')
    cp = torch.tensor([2], device='cuda')
    offset = min(512, ((length - 1) // 64) * 64) if checkpoint == 'interior' else length
    chunks = (length + 63) // 64
    metadata = dict(
        state_v_first=True,
        cu_seqlens=torch.tensor([0, length], device='cuda'),
        chunk_indices=torch.tensor([(0, i) for i in range(chunks)], device='cuda', dtype=torch.int32),
        chunk_offsets=torch.tensor([0, chunks], device='cuda', dtype=torch.int32),
        checkpoint_ids=cp,
        checkpoint_offsets=torch.tensor([offset], device='cuda'),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        A_log=alog,
        dt_bias=bias,

    )
    # Alias mode may never invoke online trial launches on persistent input.
    assert isinstance(state_module._indexed_chunk_state_kernel.fn, triton.runtime.JITFunction)
    original = state_module._indexed_chunk_state_kernel
    launches = []

    class ObservedKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(kwargs['num_warps'])
                return original[grid](*args, **kwargs)
            return launch

    monkeypatch.setattr(state_module, '_indexed_chunk_state_kernel', ObservedKernel())
    monkeypatch.setattr(state_module, 'IS_NVIDIA_HOPPER', False)
    expected = indexed_chunk_kda(q, k, v, gate, beta, reference[:, 1], src, dst, **metadata)
    assert launches[-1] == 2
    monkeypatch.setattr(state_module, 'IS_NVIDIA_HOPPER', True)
    output = indexed_chunk_kda(q, k, v, gate, beta, actual[:, 1], src, dst, **metadata)
    assert launches[-1] == (4 if length >= 512 else 2)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    actual.copy_(initial)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        captured = indexed_chunk_kda(q, k, v, gate, beta, actual[:, 1], src, dst, **metadata)
    for _ in range(2):
        q.mul_(.75)
        alog.add_(.03125)
        reference.copy_(initial)
        monkeypatch.setattr(state_module, 'IS_NVIDIA_HOPPER', False)
        expected = indexed_chunk_kda(q, k, v, gate, beta, reference[:, 1], src, dst, **metadata)
        actual.copy_(initial)
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize('name,expected', [('NVIDIA H200', True), ('NVIDIA H200 NVL', True),
                                          ('NVIDIA H100 80GB HBM3', False), ('NVIDIA B200', False)])
def test_offline_tiling_device_scope(monkeypatch, name, expected):
    from lmdeploy.pytorch.kernels.cuda import indexed_chunk_state as state_module
    state_module._is_h200.cache_clear()
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: name)
    try:
        assert state_module._is_h200(torch.device('cuda', 0)) is expected
    finally:
        state_module._is_h200.cache_clear()
