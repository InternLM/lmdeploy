# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""S1 comparison and routing tests for GPTQ models on the W4A16 GEMM route.

The pytorch engine has no native GPTQ linear modules; GPTQ linears load
through the AWQ family after the load-time conversion (``gptq_repack``).
The S1 reference is therefore dual:

- elementwise dequant matmul of the raw GPTQ tensors (float32), the
  checkpoint-semantics ground truth;
- the AWQ Triton kernel on the converted tensors, the pytorch-side quant
  path the engine actually exposes.

Thresholds match the AWQ S1 gate in ``test_tm_gemm_linear.py``.
"""

import pytest
import torch

from lmdeploy import turbomind
from lmdeploy.turbomind import gptq_repack

if not turbomind.is_available():
    pytest.skip('TurboMind is not built', allow_module_level=True)

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def _softmax_kl(a, b):
    """KL(P||Q) between the softmax distributions of two logit tensors."""
    p = torch.log_softmax(a.float(), dim=-1)
    q = torch.log_softmax(b.float(), dim=-1)
    return (p.exp() * (p - q)).sum(-1).mean().item()


def _make_gptq(k, n, group_size=128, seed=0):
    """Synthesize GPTQ checkpoint tensors for a [N, K] linear."""
    generator = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=generator)
    qweight, scales, qzeros = gptq_repack.quantize_gptq(w, group=group_size, sym=True)
    return qweight.cuda(), scales.cuda(), qzeros.cuda()


def _supported_on_device(k, n, group_size=128):
    from lmdeploy.pytorch.backends.cuda.tm_probe import turbomind_support_reason

    return turbomind_support_reason(k, n, 4, group_size, torch.float16) is None


# ---------------------------------------------------------------------------
# S1: GPTQ-TM vs dequant reference and AWQ Triton.
# ---------------------------------------------------------------------------


@cuda_required
@pytest.mark.parametrize('k,n,m', [(512, 256, 1), (512, 256, 4), (1024, 512, 4)])
def test_s1_gptq_tm_matches_reference(k, n, m):
    """TurboMind GEMM on converted GPTQ tensors agrees with both references."""
    if not _supported_on_device(k, n):
        pytest.skip('registry rejects the shape on this device')

    group = 128
    qw, sc, qz = _make_gptq(k, n, group, seed=k + n)
    awq_qw, awq_sc, awq_qz = [
        t.cuda() for t in gptq_repack.gptq_to_awq(qw.cpu(), sc.cpu(), qz.cpu())
    ]
    x = torch.randn(m, k, dtype=torch.float16, device='cuda')

    # Reference 1: elementwise dequant matmul of the raw GPTQ tensors.
    ref = gptq_repack.gptq_dequant(qw.cpu(), sc.cpu(), qz.cpu()).cuda()
    out_ref = x.float() @ ref.t()

    # Reference 2: the AWQ Triton kernel on the converted tensors.
    from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Impl
    impl_triton = AwqLinearW4A16Impl(k, n, 4, group)
    out_triton = impl_triton.forward(x, awq_qw, awq_sc, awq_qz).squeeze(0)

    # Route under test: TurboMind GEMM on the same converted tensors.
    from lmdeploy.pytorch.backends.cuda.tm_gemm_modules import TurbomindGemmW4A16Impl
    impl_tm = TurbomindGemmW4A16Impl(k, n, group)
    impl_tm.update_weights(awq_qw, awq_sc, awq_qz)
    with torch.no_grad():
        out_tm = impl_tm.forward(x, awq_qw, awq_sc, awq_qz).squeeze(0)

    for name, out in [('triton', out_triton), ('dequant', out_ref)]:
        kl = _softmax_kl(out_ref if name == 'dequant' else out_triton, out_tm)
        agreement = (out.argmax(-1) == out_tm.argmax(-1)).float().mean().item()
        max_abs = (out.float() - out_tm.float()).abs().max().item()
        assert agreement >= 0.99, f'{name}: argmax agreement {agreement:.4f}, KL {kl:.3e}'
        assert kl <= 1e-2, f'{name}: KL {kl:.3e}'
        assert max_abs <= 1.0, f'{name}: max abs diff {max_abs:.3e}'


# ---------------------------------------------------------------------------
# Probe and routing semantics.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kwargs', [
    dict(in_features=512, out_features=256, w_bit=4, group_size=128),
    dict(in_features=512, out_features=256, w_bit=4, group_size=64),
    dict(in_features=510, out_features=256, w_bit=4, group_size=128),
    dict(in_features=512, out_features=254, w_bit=4, group_size=128),
])
def test_probe_reasons_readable(kwargs):
    """Every rejection names the violated geometry rule.

    g64 is asserted conditionally: the registry decides per family, so the
    test fixes only that the verdict is either acceptance or a readable
    reason, never a silent pass-through.
    """
    from lmdeploy.pytorch.backends.cuda.tm_probe import turbomind_support_reason

    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    reason = turbomind_support_reason(kwargs['in_features'], kwargs['out_features'], kwargs['w_bit'],
                                      kwargs['group_size'], torch.float16)
    if kwargs['group_size'] == 64:
        assert reason is None or 'no GEMM family' in reason, reason
    elif kwargs['in_features'] == 512 and kwargs['out_features'] == 256:
        assert reason is None, reason
    else:
        assert reason is not None and len(reason) > 8, reason
