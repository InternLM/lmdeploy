# Copyright (c) OpenMMLab. All rights reserved.
"""Reference tests for noaux_tc MoE routing algorithm.

The CUDA kernel MoeGateNoAuxTCKernel in moe_utils_v2.cu implements:
  1. scores = scoring_func(logits)  (sigmoid or softmax)
  2. scores_for_choice = scores + correction_bias
  3. top-k selection on scores_for_choice
  4. weights taken from raw scores (not biased)
  5. optional renormalization of weights
  6. multiply by routed_scale

These tests validate the reference Python implementation matches expected
behavior for deterministic inputs. They do NOT call the CUDA kernel directly
(no Python binding exists); they ensure the algorithm specification is correct.
"""

import pytest
import torch


def noaux_tc_route(logits, correction_bias, top_k, routed_scale, use_sigmoid, norm_topk):
    """Reference implementation of noaux_tc routing.

    Args:
        logits: [tokens, experts] float32
        correction_bias: [experts] float32 or None
        top_k: int
        routed_scale: float
        use_sigmoid: bool (True=sigmoid, False=softmax)
        norm_topk: bool

    Returns:
        topk_idx: [tokens, top_k] int64 - selected expert indices
        topk_weights: [tokens, top_k] float32 - routing weights
    """
    if use_sigmoid:
        scores = torch.sigmoid(logits)
    else:
        # softmax
        scores = torch.softmax(logits, dim=-1)

    if correction_bias is not None:
        scores_for_choice = scores + correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores.clone()

    # Replace non-finite values with -inf for selection
    scores_for_choice = torch.where(
        torch.isfinite(scores_for_choice),
        scores_for_choice,
        torch.tensor(-float('inf'), device=scores_for_choice.device),
    )

    # Top-k on biased scores
    _, topk_idx = scores_for_choice.topk(top_k, dim=-1)

    # Weights from raw (unbiased) scores
    topk_weights = scores.gather(1, topk_idx)

    # Optional renormalization
    if norm_topk:
        wsum = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / wsum.clamp(min=1e-20)

    topk_weights = topk_weights * routed_scale
    return topk_idx, topk_weights


class TestNoAuxTCRouting:
    """Test noaux_tc MoE routing reference implementation."""

    @pytest.fixture(params=[8, 32, 128])
    def num_tokens(self, request):
        yield request.param

    @pytest.fixture(params=[8, 64])
    def num_experts(self, request):
        yield request.param

    @pytest.fixture(params=[2, 4, 6])
    def top_k(self, request):
        yield request.param

    @pytest.fixture(params=[True, False])
    def use_sigmoid(self, request):
        yield request.param

    @pytest.fixture(params=[True, False])
    def norm_topk(self, request):
        yield request.param

    def test_output_shapes(self, num_tokens, num_experts, top_k, use_sigmoid, norm_topk):
        """Verify output tensor shapes."""
        if top_k > num_experts:
            pytest.skip('top_k > num_experts')
        logits = torch.randn(num_tokens, num_experts)
        bias = torch.randn(num_experts) * 0.1
        idx, weights = noaux_tc_route(logits, bias, top_k, 1.0, use_sigmoid, norm_topk)
        assert idx.shape == (num_tokens, top_k)
        assert weights.shape == (num_tokens, top_k)

    def test_weights_nonnegative_sigmoid(self):
        """Sigmoid scores are always in (0, 1), so weights must be >= 0."""
        logits = torch.randn(16, 32)
        bias = torch.randn(32) * 0.5
        _, weights = noaux_tc_route(logits, bias, top_k=4, routed_scale=1.0, use_sigmoid=True, norm_topk=False)
        assert (weights >= 0).all()

    def test_weights_nonnegative_softmax(self):
        """Softmax scores are always >= 0, so weights must be >= 0."""
        logits = torch.randn(16, 32)
        bias = torch.randn(32) * 0.5
        _, weights = noaux_tc_route(logits, bias, top_k=4, routed_scale=1.0, use_sigmoid=False, norm_topk=False)
        assert (weights >= 0).all()

    def test_renormalization(self):
        """With norm_topk=True and routed_scale=1.0, weights sum to 1."""
        logits = torch.randn(16, 64)
        bias = torch.randn(64) * 0.1
        _, weights = noaux_tc_route(logits, bias, top_k=6, routed_scale=1.0, use_sigmoid=True, norm_topk=True)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(
            sums,
            torch.ones_like(sums),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_routed_scale(self):
        """Weights should scale linearly with routed_scale."""
        logits = torch.randn(16, 64)
        bias = torch.randn(64) * 0.1
        _, w1 = noaux_tc_route(logits, bias, top_k=4, routed_scale=1.0, use_sigmoid=True, norm_topk=True)
        _, w2 = noaux_tc_route(logits, bias, top_k=4, routed_scale=2.5, use_sigmoid=True, norm_topk=True)
        torch.testing.assert_close(w2, w1 * 2.5, atol=1e-5, rtol=1e-5)

    def test_bias_affects_selection_not_weights(self):
        """Correction bias changes which experts are selected but weights come
        from raw scores."""
        torch.manual_seed(42)
        logits = torch.randn(1, 8)
        # Large bias on expert 0 should force its selection
        bias = torch.zeros(8)
        bias[0] = 100.0
        idx, weights = noaux_tc_route(logits, bias, top_k=1, routed_scale=1.0, use_sigmoid=True, norm_topk=False)
        assert idx[0, 0] == 0  # Expert 0 must be selected
        # Weight should be sigmoid(logits[0, 0]), not affected by bias
        expected_w = torch.sigmoid(logits[0, 0:1])
        torch.testing.assert_close(weights[0], expected_w, atol=1e-5, rtol=1e-5)

    def test_no_bias(self):
        """When correction_bias is None, selection is purely by scores."""
        logits = torch.randn(4, 16)
        idx_biased, _ = noaux_tc_route(logits, None, top_k=3, routed_scale=1.0, use_sigmoid=True, norm_topk=True)
        # Should match top-k of sigmoid(logits) directly
        scores = torch.sigmoid(logits)
        _, expected_idx = scores.topk(3, dim=-1)
        assert (idx_biased == expected_idx).all()

    def test_unique_experts(self):
        """Each token should select top_k distinct experts."""
        logits = torch.randn(32, 64)
        bias = torch.randn(64) * 0.1
        idx, _ = noaux_tc_route(logits, bias, top_k=6, routed_scale=1.0, use_sigmoid=True, norm_topk=True)
        for row in idx:
            assert len(set(row.tolist())) == 6

    def test_nan_inf_handling(self):
        """Non-finite scores_for_choice should not be selected."""
        logits = torch.randn(4, 8)
        bias = torch.zeros(8)
        bias[0] = float('nan')
        bias[1] = float('inf')
        idx, weights = noaux_tc_route(logits, bias, top_k=2, routed_scale=1.0, use_sigmoid=True, norm_topk=False)
        # Expert 0 (NaN bias) should never be selected
        assert (idx != 0).all()
        # All weights should be finite
        assert torch.isfinite(weights).all()

    def test_glm4_flash_config(self):
        """Test with GLM-4.7-Flash's actual config: 64 experts, top_k=6,
        sigmoid scoring, norm_topk=True, routed_scale=1.0."""
        torch.manual_seed(0)
        logits = torch.randn(32, 64)
        bias = torch.randn(64) * 0.05  # small correction bias
        idx, weights = noaux_tc_route(
            logits,
            bias,
            top_k=6,
            routed_scale=1.0,
            use_sigmoid=True,
            norm_topk=True,
        )
        assert idx.shape == (32, 6)
        assert weights.shape == (32, 6)
        assert (weights >= 0).all()
        # After renorm with scale=1.0, each row sums to 1
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(32),
            atol=1e-5,
            rtol=1e-5,
        )
