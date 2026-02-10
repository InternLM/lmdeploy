import torch
import pytest

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

    @pytest.fixture
    def batch_size(self):
        yield 32
    
    @pytest.fixture
    def num_experts(self):
        yield 256
    
    @pytest.fixture
    def logits(self, batch_size, num_experts):
        yield torch.randn(batch_size, num_experts)
    
    @pytest.fixture
    def bias(self, num_experts):
        yield torch.randn(num_experts)
    
    @pytest.fixture
    def kwargs(self):
        yield {
            'num_experts': 256,
            'n_group': 8,
            'topk_group': 4,
            'top_k': 8,
            'renormalize': True,
            'routed_scaling_factor': 2.5,
        }
    
    @pytest.fixture
    def gt(self, logits, bias, kwargs):
        yield reference_noaux_tc_routing(logits, bias, **kwargs)

    def test_noaux_tc_router(self, logits, bias, kwargs, gt):
        from lmdeploy.pytorch.kernels.cuda.fused_noaux_tc import fused_noaux_tc_routing
        
        out_weights, out_ids = fused_noaux_tc_routing(logits, bias, **kwargs)
        gt_weights, gt_ids = gt

        torch.testing.assert_close(out_weights, gt_weights, rtol=1e-4, atol=1e-5)
        # topk in torch is not stable, so we won't assert ids