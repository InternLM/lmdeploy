# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope='module')
def build_prefill_token_meta():
    # This helper only needs Torch. Avoid importing unrelated backend kernels
    # (and initializing CUDA) when running its CPU tests.
    source = Path(__file__).parents[3] / 'lmdeploy/pytorch/backends/cuda/attention/v4_utils.py'
    spec = importlib.util.spec_from_file_location('_test_v4_utils', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module.build_prefill_token_meta
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize('lengths', [[2, 0, 3], [0, 0], [4, 1, 7]])
@pytest.mark.parametrize('supply_cumulative', [False, True])
@pytest.mark.parametrize('supply_total', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_build_prefill_token_meta(build_prefill_token_meta, lengths, supply_cumulative, supply_total, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    q_seqlens = torch.tensor(lengths, dtype=torch.int32, device=device)
    cu_q_seqlens = torch.nn.functional.pad(q_seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    result = build_prefill_token_meta(
        q_seqlens,
        cu_q_seqlens if supply_cumulative else None,
        total_tokens=sum(lengths) if supply_total else None)
    expected_seq = [i for i, length in enumerate(lengths) for _ in range(length)]
    expected_pos = [pos for length in lengths for pos in range(length)]
    assert result.seq_id.cpu().tolist() == expected_seq
    assert result.token_pos.cpu().tolist() == expected_pos
    assert result.seq_id.dtype == torch.int64
    # The existing fallback cumsum promotes integer input to int64.
    expected_dtype = cu_q_seqlens.dtype if supply_cumulative else torch.int64
    assert result.token_pos.dtype == expected_dtype
    assert result.seq_id.device == q_seqlens.device
    assert result.token_pos.device == q_seqlens.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('supply_cumulative', [False, True])
def test_build_prefill_token_meta_host_count_avoids_cuda_sync(build_prefill_token_meta, supply_cumulative):
    q_seqlens = torch.tensor([2, 0, 3], dtype=torch.int32, device='cuda')
    cu_q_seqlens = torch.nn.functional.pad(q_seqlens.cumsum(0, dtype=torch.int32), (1, 0))
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode('error')
        result = build_prefill_token_meta(
            q_seqlens, cu_q_seqlens if supply_cumulative else None, total_tokens=5)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert result.seq_id.cpu().tolist() == [0, 0, 2, 2, 2]
    assert result.token_pos.cpu().tolist() == [0, 1, 0, 1, 2]
