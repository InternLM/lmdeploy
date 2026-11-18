# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

FP8_DTYPE = torch.float8_e4m3fn


def _has_gluon():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return False
    from lmdeploy.pytorch.backends.cuda.blockedf8_modules import _has_gluon as has_gluon
    return has_gluon()


pytestmark = pytest.mark.skipif(not _has_gluon(), reason='Gluon WGMMA requires supported Triton on Hopper')


def _make_inputs(groups, capacity, n, k, column_major):
    torch.manual_seed(5027)
    aq = torch.randn(groups, capacity, k, device='cuda').to(FP8_DTYPE)
    bq = torch.randn(groups, n, k, device='cuda').to(FP8_DTYPE)
    asc = (torch.empty(groups, k // 128, capacity, device='cuda').transpose(1, 2)
           if column_major else torch.empty(groups, capacity, k // 128, device='cuda'))
    bsc = torch.empty(groups, n // 128, k // 128, device='cuda')
    # Vary every expert/row/K-block scale to expose incorrect stride arithmetic.
    asc.uniform_(0.001, 0.02)
    bsc.uniform_(0.001, 0.02)
    masks = torch.zeros(groups, device='cuda', dtype=torch.int32)
    out = torch.full((groups, capacity, n), torch.nan, device='cuda', dtype=torch.bfloat16)
    return (aq, asc), (bq, bsc), out, masks


def _check_valid_rows(a, b, out, masks, expected_m):
    from lmdeploy.pytorch.third_party.deep_gemm import m_grouped_fp8_gemm_nt_masked

    reference = torch.empty_like(out)
    # DeepGEMM requires a positive scheduling hint even when the tested
    # dispatcher accepts zero. Neither hint may limit the actual expert rows.
    m_grouped_fp8_gemm_nt_masked(a, b, reference, masks, max(1, expected_m))
    valid = torch.arange(out.size(1), device='cuda')[None, :] < masks[:, None]
    torch.testing.assert_close(out[valid], reference[valid], atol=0, rtol=0)
    # The last active tile may write padding, but an empty expert has no tiles.
    assert torch.isnan(out[masks == 0]).all()


@pytest.mark.parametrize('k', [2048, 4096])
@pytest.mark.parametrize('capacity', [128, 384])
@pytest.mark.parametrize('column_major', [False, True], ids=['row_major', 'column_major'])
@torch.inference_mode()
def test_grouped_masked_gemm_boundaries_and_graph_updates(k, capacity, column_major):
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_gemm_gluon import grouped_fp8_gemm_nt_masked

    a, b, out, masks = _make_inputs(8, capacity, 256, k, column_major)
    counts = [0, 1, 7, 8, 9, 15, 16, capacity]
    # Warm with active experts, then capture an empty batch. Graph replay must
    # consume new GPU counts, including full experts, without recapture.
    masks.copy_(torch.tensor(counts, device='cuda', dtype=torch.int32))
    grouped_fp8_gemm_nt_masked(a, b, out, masks)
    _check_valid_rows(a, b, out, masks, expected_m=3)
    masks.zero_()
    out.fill_(torch.nan)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped_fp8_gemm_nt_masked(a, b, out, masks)

    for next_counts in ([0] * 8, counts, [17, 31, 32, 33, capacity - 1, capacity, 0, 3], counts[::-1]):
        masks.copy_(torch.tensor(next_counts, device='cuda', dtype=torch.int32))
        a[0].copy_(torch.randn_like(a[0], dtype=torch.bfloat16).to(FP8_DTYPE))
        a[1].uniform_(0.001, 0.02)
        b[1].uniform_(0.001, 0.02)
        out.fill_(torch.nan)
        graph.replay()
        _check_valid_rows(a, b, out, masks, expected_m=3)


@pytest.mark.parametrize('column_major', [False, True], ids=['row_major', 'column_major'])
@pytest.mark.parametrize('decoding,enabled,expected_m,use_gluon', [
    (True, True, 0, True),
    (True, True, 4, True),
    (True, True, 5, False),
    (False, True, 3, False),
    (True, False, 3, False),
])
@torch.inference_mode()
def test_grouped_backend_expected_m_does_not_truncate(
        column_major, decoding, enabled, expected_m, use_gluon, monkeypatch):
    from lmdeploy.pytorch.backends.cuda.moe import blocked_fp8 as backend
    from lmdeploy.pytorch.kernels.cuda import blocked_fp8_gemm_gluon as kernels

    # The backend dispatch targets the actual GLM EP8 shape: 36 local experts.
    a, b, out, masks = _make_inputs(36, 128, 4096, 2048, column_major)
    counts = torch.tensor([0, 1, 7, 8, 9, 31, 65, 127, 128] * 4, device='cuda', dtype=torch.int32)
    masks.copy_(counts)
    impl = object.__new__(backend.FusedMoELowLatency)
    impl._small_m_gluon = enabled
    context = SimpleNamespace(global_is_decoding=lambda: decoding)
    monkeypatch.setattr(backend, 'get_step_ctx_manager', lambda: SimpleNamespace(current_context=lambda: context))
    grouped = kernels.grouped_fp8_gemm_nt_masked
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return grouped(*args, **kwargs)

    monkeypatch.setattr(kernels, 'grouped_fp8_gemm_nt_masked', counted)
    impl._deepgemm_grouped_fp8_nt_masked(a, b, out, masks, expected_m)
    assert bool(calls) == use_gluon
    _check_valid_rows(a, b, out, masks, expected_m)
