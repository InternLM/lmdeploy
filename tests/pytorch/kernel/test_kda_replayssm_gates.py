import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')


@pytest.mark.parametrize('shape', [(1, 1, 1, 128), (3, 2, 3, 96), (2, 8, 5, 128),
                                  (64, 6, 16, 128), (224, 6, 16, 128), (512, 6, 16, 128)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize('lower_bound', [-5.0, -0.1])
def test_kda_replay_gates_oracle_and_graph(shape, dtype, lower_bound):
    from lmdeploy.pytorch.kernels.cuda.kda_replayssm import prepare_kda_replay_gates

    torch.manual_seed(20261017)
    b, t, h, k = shape
    raw = torch.randn(shape, device='cuda', dtype=dtype)
    beta = torch.randn((b, t, h), device='cuda', dtype=dtype)
    raw.flatten()[::7] = 80
    raw.flatten()[::11] = -80
    beta.flatten()[::3] = 100
    beta.flatten()[::5] = -100
    a_log = torch.linspace(-5, 5, h, device='cuda')
    bias = torch.randn(h * k, device='cuda')

    def check(outputs):
        expected_g = lower_bound * torch.sigmoid(a_log.exp()[None, None, :, None]
                                                * (raw.float() + bias.reshape(h, k)))
        expected_beta = torch.sigmoid(beta.float())
        assert outputs[0].shape == raw.shape and outputs[1].shape == beta.shape
        assert all(x.dtype == torch.float32 and x.is_contiguous() for x in outputs)
        torch.testing.assert_close(outputs[0], expected_g, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(outputs[1], expected_beta, atol=0, rtol=0)

    check(prepare_kda_replay_gates(raw, beta, a_log, bias, lower_bound))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = prepare_kda_replay_gates(raw, beta, a_log, bias, lower_bound)
    for _ in range(3):
        raw.add_(0.25)
        beta.sub_(0.5)
        graph.replay()
        check(outputs)


@pytest.mark.parametrize('lower_bound', [None, 0.0, 1.0])
def test_kda_replay_gates_reject_unbounded(lower_bound):
    from lmdeploy.pytorch.kernels.cuda.kda_replayssm import prepare_kda_replay_gates

    with pytest.raises(ValueError, match='negative bounded gate'):
        prepare_kda_replay_gates(None, None, None, None, lower_bound)


def test_kda_replay_gates_tail_and_storage_offset():
    from lmdeploy.pytorch.kernels.cuda.kda_replayssm import _prepare_kda_replay_gates

    n, h, k = 9, 3, 96
    raw = torch.randn(n * k + 2, device='cuda', dtype=torch.bfloat16)[1:-1]
    beta = torch.randn(n + 2, device='cuda', dtype=torch.bfloat16)[1:-1]
    a_log = torch.zeros(h, device='cuda')
    bias = torch.zeros(h * k, device='cuda')
    owner_g = torch.full((n * k + 2,), 123.0, device='cuda')
    owner_beta = torch.full((n + 2,), 123.0, device='cuda')
    _prepare_kda_replay_gates[(3,)](
        raw, beta, a_log, bias, owner_g[1:-1], owner_beta[1:-1],
        N=n, H=h, K=k, BH=4, BK=128, LOWER_BOUND=-5.0, num_warps=4)
    assert owner_g[0] == owner_g[-1] == owner_beta[0] == owner_beta[-1] == 123
    torch.testing.assert_close(owner_g[1:-1], -5 * torch.sigmoid(raw.float()), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(owner_beta[1:-1], torch.sigmoid(beta.float()), atol=0, rtol=0)
