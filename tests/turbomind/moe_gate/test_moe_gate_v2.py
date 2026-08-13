import pytest
import torch

from . import turbomind_moe_gate
from .cases import SMOKE_CASES
from .reference import moe_gate_v2_reference

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


NAN_TOKENS, NAN_EXPERTS, NAN_TOP_K = 16, 64, 4
NAN_INVALID_ROWS = (3, 7, 11)


def _nan_logits() -> torch.Tensor:
    torch.manual_seed(3)
    logits = torch.randn(NAN_TOKENS, NAN_EXPERTS, device='cuda', dtype=torch.float32)
    logits[list(NAN_INVALID_ROWS)] = float('nan')
    return logits


@cuda_required
def test_moe_gate_v2_masked_nan_tokens_route_nowhere():
    """Masked-out tokens produce no routing entries, even with NaN logits.

    Before the token_mask gate, NaN logits made the kernel read an uninitialized expert id and index masks/accum out of
    bounds (illegal memory access).
    """
    if not turbomind_moe_gate.is_available():
        pytest.skip('TurboMind moe_gate_v2 bridge is unavailable')

    logits = _nan_logits()
    token_mask = torch.ones(NAN_TOKENS, device='cuda', dtype=torch.bool)
    token_mask[list(NAN_INVALID_ROWS)] = False

    f2n, f2E, en2f, offsets, scales = turbomind_moe_gate.moe_gate_v2(
        logits, NAN_TOP_K, token_mask=token_mask)
    torch.cuda.synchronize()  # surfaces any async CUDA error here, not in a later test

    valid = [t for t in range(NAN_TOKENS) if t not in NAN_INVALID_ROWS]
    total = int(offsets[-1])
    assert total == len(valid) * NAN_TOP_K
    assert not (set(f2n[:total].cpu().tolist()) & set(NAN_INVALID_ROWS))

    # Routing of the valid tokens matches the reference on the valid subset.
    e_f2n, e_f2E, e_en2f, e_offsets, e_scales = moe_gate_v2_reference(logits[valid], NAN_TOP_K)
    valid_lut = torch.tensor(valid, dtype=torch.int32)
    assert torch.equal(f2n[:total].cpu(), valid_lut[e_f2n.cpu()])
    assert torch.equal(f2E[:total].cpu(), e_f2E.cpu())
    assert torch.equal(offsets.cpu(), e_offsets.cpu())
    assert torch.equal(en2f[:, valid].cpu(), e_en2f.cpu())
    torch.testing.assert_close(scales[:, valid].cpu(), e_scales.cpu(), rtol=1e-4, atol=1e-5)

    # Masked tokens keep the -1 skip sentinel so combine never touches them.
    neg1 = torch.full((NAN_TOP_K, len(NAN_INVALID_ROWS)), -1, dtype=torch.int32)
    assert torch.equal(en2f[:, list(NAN_INVALID_ROWS)].cpu(), neg1)
