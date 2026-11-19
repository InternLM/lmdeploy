"""Production-width head rounding, batching, chunking and independent VJPs."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('rows', [1, 17, 33, 65])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('strided', [False, True])
def test_head_batch_chunking_and_fp64_reference(rows, dtype, strided):
    from lmdeploy.pytorch.nn.glm53_alignment import aligned_linear

    torch.manual_seed(354)
    if strided:
        x = torch.randn(rows, 8192, device='cuda', dtype=dtype)[:, ::2]
        w = (torch.randn(4096, 4097, device='cuda') * .01).t()
    else:
        x = torch.randn(rows, 4096, device='cuda', dtype=dtype)
        w = torch.randn(4097, 4096, device='cuda') * .01
    bias = torch.randn(4097, device='cuda') * .01
    output = aligned_linear(x, w, bias)
    reference = F.linear(x.double(), w.double(), bias.double())
    assert (output.double()-reference).norm()/reference.norm() < 5e-7
    torch.testing.assert_close(output.double(), reference, atol=2e-6, rtol=2e-5)
    for chunk in (1, 3, 17):
        actual = torch.cat([aligned_linear(x[i:i+chunk], w, bias) for i in range(0, rows, chunk)])
        torch.testing.assert_close(actual, output, atol=0, rtol=0)


def test_head_independent_vjp_and_compiled_path():
    from lmdeploy.pytorch.nn.glm53_alignment import aligned_linear

    torch.manual_seed(355)
    x = torch.randn(1, 17, 4096, device='cuda', requires_grad=True)
    w = (torch.randn(4097, 4096, device='cuda') * .01).requires_grad_()
    bias = torch.randn(4097, device='cuda', requires_grad=True)
    compiled = torch.compile(aligned_linear, fullgraph=True)
    output = compiled(x, w, bias)
    torch.testing.assert_close(output, aligned_linear(x, w, bias), atol=0, rtol=0)
    xd, wd, bd = [t.detach().double().requires_grad_() for t in (x, w, bias)]
    oracle = F.linear(xd, wd, bd)
    seed = torch.randn_like(output)
    grads = torch.autograd.grad(output, (x, w, bias), seed)
    expected = torch.autograd.grad(oracle, (xd, wd, bd), seed.double())
    for actual, target in zip(grads, expected):
        torch.testing.assert_close(actual.double(), target, atol=3e-5, rtol=3e-5)


def test_head_nonfinite_classification_and_empty_rows():
    from lmdeploy.pytorch.nn.glm53_alignment import aligned_linear

    w = torch.ones(4096, 4096, device='cuda') * .001
    for value in (float('inf'), float('-inf'), float('nan')):
        x = torch.ones(1, 4096, device='cuda')
        x[0, 0] = value
        actual, expected = aligned_linear(x, w), F.linear(x, w)
        assert torch.equal(actual.isnan(), expected.isnan())
        assert torch.equal(actual.isinf(), expected.isinf())
        mask = expected.isinf()
        assert torch.equal(actual[mask].signbit(), expected[mask].signbit())
    assert aligned_linear(x[:0], w).shape == (0, 4096)
