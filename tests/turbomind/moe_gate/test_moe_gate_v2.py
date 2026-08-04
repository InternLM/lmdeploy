import pytest
import torch

from tests.turbomind.moe_gate import turbomind_moe_gate
from tests.turbomind.moe_gate.cases import SMOKE_CASES
from tests.turbomind.moe_gate.reference import moe_gate_v2_reference

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@cuda_required
@pytest.mark.parametrize('case', SMOKE_CASES, ids=lambda c: c.name)
def test_moe_gate_v2_matches_reference(case):
    if not turbomind_moe_gate.is_available():
        pytest.skip('TurboMind moe_gate_v2 bridge is unavailable')

    torch.manual_seed(0)
    logits = torch.randn(case.tokens, case.experts, device='cuda', dtype=torch.float32)

    expected = moe_gate_v2_reference(logits, case.top_k)
    actual = turbomind_moe_gate.moe_gate_v2(logits, case.top_k)

    f2n, f2E, en2f, offsets, scales = actual
    e_f2n, e_f2E, e_en2f, e_offsets, e_scales = expected

    assert torch.equal(offsets.cpu(), e_offsets.cpu())
    assert torch.equal(f2n.cpu(), e_f2n.cpu())
    assert torch.equal(f2E.cpu(), e_f2E.cpu())
    assert torch.equal(en2f.cpu(), e_en2f.cpu())
    torch.testing.assert_close(scales.cpu(), e_scales.cpu(), rtol=1e-4, atol=1e-5)


@cuda_required
def test_moe_gate_v2_preallocated_matches_allocate():
    if not turbomind_moe_gate.is_available():
        pytest.skip('TurboMind moe_gate_v2 bridge is unavailable')

    case = SMOKE_CASES[0]
    torch.manual_seed(1)
    logits = torch.randn(case.tokens, case.experts, device='cuda', dtype=torch.float32)

    allocated = turbomind_moe_gate.moe_gate_v2(logits, case.top_k)
    buffers = turbomind_moe_gate.allocate_moe_gate_v2_buffers(case.tokens, case.experts, case.top_k)
    preallocated = turbomind_moe_gate.moe_gate_v2(logits, case.top_k, buffers=buffers)

    for a, b in zip(allocated, preallocated):
        if a.dtype.is_floating_point:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        else:
            assert torch.equal(a, b)


MODE_CASES = (
    # (tokens, experts, top_k, softmax, norm_topk, routed_scale)
    (16, 8, 2, True, False, 1.0),
    (16, 8, 2, True, True, 1.0),
    (16, 8, 2, False, False, 1.0),
    (16, 8, 2, True, False, 1.5),
    (8, 256, 8, True, False, 1.0),
    (8, 256, 8, True, True, 1.0),
    (8, 256, 8, False, False, 1.0),
    (8, 256, 8, True, False, 0.5),
    (8, 2560, 8, True, False, 1.0),
    (8, 2560, 8, True, True, 1.0),
    (8, 2560, 8, False, False, 1.0),
    (8, 2560, 8, True, False, 0.5),
)


def _mode_case_id(case):
    tokens, experts, top_k, softmax, norm_topk, routed_scale = case
    return f't{tokens}_e{experts}_k{top_k}_sm{int(softmax)}_nt{int(norm_topk)}_rs{routed_scale}'


@cuda_required
@pytest.mark.parametrize(
    'tokens,experts,top_k,softmax,norm_topk,routed_scale',
    MODE_CASES,
    ids=[_mode_case_id(c) for c in MODE_CASES],
)
def test_moe_gate_v2_modes(tokens, experts, top_k, softmax, norm_topk, routed_scale):
    if not turbomind_moe_gate.is_available():
        pytest.skip('TurboMind moe_gate_v2 bridge is unavailable')

    torch.manual_seed(2)
    logits = torch.randn(tokens, experts, device='cuda', dtype=torch.float32)

    expected = moe_gate_v2_reference(
        logits, top_k, softmax=softmax, norm_topk=norm_topk, routed_scale=routed_scale)
    actual = turbomind_moe_gate.moe_gate_v2(
        logits, top_k, softmax=softmax, norm_topk=norm_topk, routed_scale=routed_scale)

    f2n, f2E, en2f, offsets, scales = actual
    e_f2n, e_f2E, e_en2f, e_offsets, e_scales = expected

    assert torch.equal(offsets.cpu(), e_offsets.cpu())
    assert torch.equal(f2n.cpu(), e_f2n.cpu())
    assert torch.equal(f2E.cpu(), e_f2E.cpu())
    assert torch.equal(en2f.cpu(), e_en2f.cpu())
    torch.testing.assert_close(scales.cpu(), e_scales.cpu(), rtol=1e-4, atol=1e-5)
