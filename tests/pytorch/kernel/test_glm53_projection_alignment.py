"""Production-geometry mHC projection against independent FP64 references."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('rows', [1, 3, 16, 33, 65, 128])
@pytest.mark.parametrize('strided', [False, True])
def test_mhc_projection_rounding_batch_and_vjp(rows, strided):
    from lmdeploy.pytorch.nn.glm53_alignment import aligned_linear

    torch.manual_seed(346)
    if strided:
        x = torch.randn(rows, 32768, device='cuda')[:, ::2].detach().requires_grad_()
        weight = (torch.randn(16384, 24, device='cuda') * .01).t().detach().requires_grad_()
    else:
        x = torch.randn(rows, 16384, device='cuda', requires_grad=True)
        weight = (torch.randn(24, 16384, device='cuda') * .01).requires_grad_()
    bias = torch.randn(24, device='cuda', requires_grad=True)
    output = aligned_linear(x, weight, bias)
    xd, wd, bd = [v.detach().double().requires_grad_() for v in (x, weight, bias)]
    reference = torch.nn.functional.linear(xd, wd, bd)
    torch.testing.assert_close(output, reference.float(), atol=0, rtol=0)
    torch.testing.assert_close(output[:1], aligned_linear(x[:1], weight, bias), atol=0, rtol=0)
    seed = torch.randn_like(output)
    actual_gradients = torch.autograd.grad(output, (x, weight, bias), seed)
    oracle_gradients = torch.autograd.grad(reference, (xd, wd, bd), seed.double())
    for actual, expected in zip(actual_gradients, oracle_gradients):
        torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=3e-5)


def test_mhc_projection_compiled_backward_and_empty_rows():
    from lmdeploy.pytorch.nn.glm53_alignment import aligned_linear

    x = torch.randn(1, 33, 16384, device='cuda', requires_grad=True)
    w = torch.randn(24, 16384, device='cuda', requires_grad=True)
    compiled = torch.compile(aligned_linear, fullgraph=True)
    eager, actual = aligned_linear(x, w), compiled(x, w)
    torch.testing.assert_close(actual, eager, atol=0, rtol=0)
    seed = torch.randn_like(eager)
    gradients = torch.autograd.grad(actual, (x, w), seed)
    expected = torch.autograd.grad(eager, (x, w), seed)
    for a, b in zip(gradients, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert aligned_linear(x[:, :0], w).shape == (1, 0, 24)
