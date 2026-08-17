import pytest
import torch


def reference_noaux_tc_routing(
    logits: torch.Tensor,
    bias: torch.Tensor,
    num_experts: int = 256,
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
    renormalize: bool = True,
    routed_scaling_factor: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = logits.shape[0]
    scores = torch.sigmoid(logits.float())
    scores_for_choice = scores + bias[None, :]

    group_size = num_experts // n_group
    grouped_scores = scores_for_choice.view(batch_size, n_group, group_size)
    group_scores = grouped_scores.topk(2, dim=-1)[0].sum(dim=-1)

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)

    score_mask = group_mask.unsqueeze(-1).expand(batch_size, n_group, group_size).reshape(batch_size, -1)
    # Note: Using 0.0 matches the actual inference code in deepseek_v2.py
    # Works correctly because sigmoid scores are always in (0, 1)
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

    _, topk_idx = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)

    if renormalize:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

    return topk_weight * routed_scaling_factor, topk_idx


def assert_routes_close(
    output: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
):
    output_weights, output_ids = output
    expected_weights, expected_ids = expected
    output_order = output_ids.argsort(dim=-1)
    expected_order = expected_ids.argsort(dim=-1)
    output_ids = output_ids.gather(1, output_order)
    expected_ids = expected_ids.gather(1, expected_order)
    output_weights = output_weights.gather(1, output_order)
    expected_weights = expected_weights.gather(1, expected_order)

    torch.testing.assert_close(output_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(output_weights, expected_weights, rtol=1e-4, atol=1e-5)


class TestNoauxTC:

    @pytest.fixture(autouse=True)
    def auto_context(self):
        origin_dtype = torch.get_default_dtype()
        origin_device = torch.get_default_device()
        with torch.inference_mode():
            torch.set_default_dtype(torch.float32)
            torch.set_default_device('cuda')
            try:
                yield
            finally:
                torch.set_default_dtype(origin_dtype)
                torch.set_default_device(origin_device)

    @pytest.fixture(params=[1, 256])
    def batch_size(self, request):
        yield request.param

    @pytest.fixture
    def num_experts(self):
        yield 256

    @pytest.fixture
    def logits(self, batch_size, num_experts):
        yield torch.randn(batch_size, num_experts)

    @pytest.fixture
    def bias(self, num_experts):
        yield torch.empty(num_experts).uniform_(-0.05, 0.05)

    @pytest.fixture(params=[
        {
            'num_experts': 256,
            'n_group': 1,
            'topk_group': 1,
            'top_k': 8,
            'renormalize': True,
            'routed_scaling_factor': 2.5,
        },
        {
            'num_experts': 256,
            'n_group': 8,
            'topk_group': 4,
            'top_k': 8,
            'renormalize': True,
            'routed_scaling_factor': 2.5,
        },
    ])
    def kwargs(self, request):
        yield request.param

    @pytest.fixture
    def gt(self, logits, bias, kwargs):
        yield reference_noaux_tc_routing(logits, bias, **kwargs)

    def test_noaux_tc_router(self, logits, bias, kwargs, gt):
        from lmdeploy.pytorch.kernels.cuda.fused_noaux_tc import fused_noaux_tc_routing

        output = fused_noaux_tc_routing(logits, bias, **kwargs)
        assert_routes_close(output, gt)

    @pytest.mark.parametrize('batch_size', [1, 32])
    def test_kimi_router(self, batch_size):
        from lmdeploy.pytorch.kernels.cuda.fused_noaux_tc import fused_noaux_tc_routing

        torch.manual_seed(batch_size)
        logits = torch.randn(batch_size, 384)
        bias = torch.randn(384)
        kwargs = {
            'num_experts': 384,
            'n_group': 1,
            'topk_group': 1,
            'top_k': 8,
            'renormalize': True,
            'routed_scaling_factor': 2.827,
        }

        output = fused_noaux_tc_routing(logits, bias, **kwargs)
        expected = reference_noaux_tc_routing(logits, bias, **kwargs)
        assert_routes_close(output, expected)

    def test_kimi_router_uses_custom_kernel(self):
        from lmdeploy.pytorch.backends.cuda.moe_router import TritonRouterNoauxTCImpl

        kwargs = {
            'scoring_func': 'sigmoid',
            'top_k': 8,
            'n_group': 1,
            'topk_group': 1,
            'n_routed_experts': 384,
            'routed_scaling_factor': 2.827,
            'renormalize': True,
        }
        router = TritonRouterNoauxTCImpl(**kwargs)
        padded_group_router = TritonRouterNoauxTCImpl(**(kwargs | {
            'n_group': 8,
            'topk_group': 4,
        }))

        assert router.enable_custom_kernel
        assert not padded_group_router.enable_custom_kernel

    @pytest.mark.parametrize('batch_size', [512, 513])
    def test_kimi_router_large_batches(self, batch_size):
        from lmdeploy.pytorch.kernels.cuda.fused_noaux_tc import fused_noaux_tc_routing

        torch.manual_seed(batch_size)
        logits = torch.randn(batch_size, 384)
        bias = torch.randn(384)
        kwargs = {
            'num_experts': 384,
            'n_group': 1,
            'topk_group': 1,
            'top_k': 8,
            'renormalize': True,
            'routed_scaling_factor': 2.827,
        }

        output = fused_noaux_tc_routing(logits, bias, **kwargs)
        expected = reference_noaux_tc_routing(logits, bias, **kwargs)

        assert output[0].dtype == torch.float32
        assert output[1].dtype == torch.int64
        assert_routes_close(output, expected)

    @pytest.mark.parametrize('input_kind', ['saturation', 'near-tie'])
    def test_kimi_router_numerical_edges(self, input_kind):
        from lmdeploy.pytorch.kernels.cuda.fused_noaux_tc import fused_noaux_tc_routing

        if input_kind == 'saturation':
            logits = torch.empty(3, 384)
            logits[:, 0::2] = -80
            logits[:, 1::2] = 80
            bias = torch.linspace(-0.25, 0.25, 384)
        else:
            logits = torch.zeros(3, 384)
            # These differences are small but still distinct in FP32.
            bias = torch.arange(384) * 2**-20
        kwargs = {
            'num_experts': 384,
            'n_group': 1,
            'topk_group': 1,
            'top_k': 8,
            'renormalize': True,
            'routed_scaling_factor': 2.827,
        }

        output = fused_noaux_tc_routing(logits, bias, **kwargs)
        expected = reference_noaux_tc_routing(logits, bias, **kwargs)

        assert_routes_close(output, expected)

    def test_kimi_router_is_repeatable(self):
        from lmdeploy.pytorch.kernels.cuda.fused_noaux_tc import fused_noaux_tc_routing

        torch.manual_seed(11)
        logits = torch.randn(32, 384)
        bias = torch.randn(384)
        kwargs = {
            'num_experts': 384,
            'n_group': 1,
            'topk_group': 1,
            'top_k': 8,
            'renormalize': True,
            'routed_scaling_factor': 2.827,
        }

        outputs = [fused_noaux_tc_routing(logits, bias, **kwargs) for _ in range(3)]
        for output in outputs[1:]:
            torch.testing.assert_close(output[0], outputs[0][0], rtol=0, atol=0)
            torch.testing.assert_close(output[1], outputs[0][1], rtol=0, atol=0)

    def test_kimi_router_fuses_postprocessing(self, monkeypatch):
        from lmdeploy.pytorch.kernels.cuda.fused_noaux_tc import fused_noaux_tc_routing

        logits = torch.randn(1, 384)
        bias = torch.randn(384)
        monkeypatch.setattr(torch, 'topk', lambda *args, **kwargs: pytest.fail('unexpected torch.topk'))

        weights, ids = fused_noaux_tc_routing(
            logits,
            bias,
            num_experts=384,
            n_group=1,
            topk_group=1,
            top_k=8,
            renormalize=True,
            routed_scaling_factor=2.827,
        )

        assert weights.shape == (1, 8)
        assert ids.shape == (1, 8)

    def test_kimi_router_cudagraph_dynamic_logits(self):
        from lmdeploy.pytorch.kernels.cuda.fused_noaux_tc import fused_noaux_tc_routing

        batch_size = 4
        num_experts = 384
        base_logits = torch.linspace(-4, 4, num_experts)
        static_logits = torch.stack([base_logits.roll(i * 17) for i in range(batch_size)])
        bias = torch.linspace(-0.01, 0.01, num_experts)
        kwargs = {
            'num_experts': num_experts,
            'n_group': 1,
            'topk_group': 1,
            'top_k': 8,
            'renormalize': True,
            'routed_scaling_factor': 2.827,
        }

        warm_output = fused_noaux_tc_routing(static_logits, bias, **kwargs)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = fused_noaux_tc_routing(static_logits, bias, **kwargs)

        next_logits = static_logits.flip(-1)
        static_logits.copy_(next_logits)
        graph.replay()
        torch.cuda.synchronize()

        expected = reference_noaux_tc_routing(next_logits, bias, **kwargs)
        assert_routes_close(graph_output, expected)
        assert not torch.equal(
            graph_output[1].sort(dim=-1).values,
            warm_output[1].sort(dim=-1).values,
        )
