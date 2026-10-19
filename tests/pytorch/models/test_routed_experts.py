# Copyright (c) OpenMMLab. All rights reserved.
"""Model forward contracts for target-only routed-expert capture."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lmdeploy.pytorch.models.glm5_next import (
    Glm5NextDecoderLayer,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
    Glm5NextMoE,
    Glm5NextMTPDecoderLayer,
)

SENTINEL = torch.iinfo(torch.uint16).max


def bare(cls):
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    return obj


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.seen = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids, inputs_embeds, all_routed_experts, **kwargs):
        self.seen = all_routed_experts
        if all_routed_experts is not None:
            all_routed_experts[:, 1].fill_(7)
        return self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds


class CaptureLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.is_linear_attention = index != 1

    def forward(self, hidden_states, all_routed_experts, **kwargs):
        all_routed_experts[:, self.index].fill_(self.index)
        return hidden_states, None


class Gate(nn.Module):
    def forward(self, hidden_states, routed_experts=None):
        ids = torch.tensor([[2, 5]], dtype=torch.int64).expand(hidden_states.shape[0], -1)
        if routed_experts is not None:
            routed_experts.copy_(ids)
        return torch.full_like(ids, .5, dtype=torch.float32), ids


class Experts(nn.Module):
    def forward(self, hidden_states, weights, ids):
        return hidden_states + weights.sum(-1, keepdim=True)


class CaptureMoE(Glm5NextMoE):
    def __init__(self):
        nn.Module.__init__(self)
        self.seen = 'not_called'

    def forward(self, x, all_routed_experts=None):
        self.seen = all_routed_experts
        return x


class Attention(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class Norm(nn.Module):
    def forward(self, x, residual=None):
        return x if residual is None else (x, residual)


class TestGlm5NextRoutedExperts:
    @pytest.mark.parametrize('capture', [False, True])
    @pytest.mark.parametrize('mtp_prefill', [False, True])
    @pytest.mark.parametrize('use_embeds', [False, True])
    def test_target_capture_preserves_hidden_states_and_mtp_embeddings(self, capture, mtp_prefill, use_embeds):
        model = bare(Glm5NextForConditionalGeneration)
        model.enable_return_routed_experts = capture
        model.config = SimpleNamespace(num_hidden_layers=3, num_experts_per_tok=2)
        model.model = Backbone()
        ids = torch.tensor([[1, 2, 3]])
        embeds = model.get_input_embeddings()(ids) if use_embeds else None
        out = model(ids, ids, [], inputs_embeds=embeds, return_input_embeds=mtp_prefill)
        expected = model.get_input_embeddings()(ids)
        if not capture and not mtp_prefill:
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            assert model.model.seen is None
            return
        torch.testing.assert_close(out['hidden_states'], expected, rtol=0, atol=0)
        assert ('target_inputs_embeds' in out) == mtp_prefill
        if mtp_prefill:
            torch.testing.assert_close(out['target_inputs_embeds'], expected, rtol=0, atol=0)
        assert ('all_routed_experts' in out) == capture
        if capture:
            routes = out['all_routed_experts']
            assert routes.shape == (3, 3, 2)
            assert routes.dtype == torch.uint16
            assert torch.all(routes[:, 1].int() == 7)
            assert torch.all(routes[:, (0, 2)].int() == SENTINEL)

    def test_cudagraph_routes_are_trimmed_cloned_and_preserve_mtp_embeddings(self):
        model = bare(Glm5NextForConditionalGeneration)
        routes = torch.arange(8 * 3 * 2).reshape(8, 3, 2).to(torch.uint16)
        buffers = dict(hidden_states=torch.randn(1, 8, 4), all_routed_experts=routes,
                       target_inputs_embeds=torch.randn(1, 8, 4))
        out = model.get_outputs_cudagraph(buffers, torch.zeros(1, 3, dtype=torch.long))
        assert out['all_routed_experts'].shape == (3, 3, 2)
        torch.testing.assert_close(out['all_routed_experts'].int(), routes[:3].int())
        routes.zero_()
        assert out['all_routed_experts'].int().sum() > 0
        assert out['target_inputs_embeds'].shape == (1, 3, 4)

    def test_backbone_forwards_routes_across_hybrid_layers(self):
        model = bare(Glm5NextModel)
        model.config = SimpleNamespace(hc_mult=2, linear_conv_kernel_dim=4)
        model.embed_tokens = nn.Embedding(16, 4)
        model.layers = nn.ModuleList([CaptureLayer(i) for i in range(3)])
        model.norm = nn.Identity()
        model.gated_delta_meta_builder = lambda *a: None
        ids = torch.tensor([[1, 2, 3]])
        routes = torch.full((3, 3, 2), SENTINEL, dtype=torch.uint16)
        out = model(ids, None, [(), (None, None), ()], SimpleNamespace(is_decoding=False),
                    state_ids=torch.zeros(1, dtype=torch.long), all_routed_experts=routes)
        torch.testing.assert_close(out, model.embed_tokens(ids), rtol=0, atol=0)
        for layer in range(3):
            assert torch.all(routes[:, layer].int() == layer)

    def test_moe_captures_logical_ids_without_changing_outputs(self):
        moe = bare(Glm5NextMoE)
        moe.layer_idx = 1
        moe.gate = Gate()
        moe.experts = Experts()
        moe.shared_experts = None
        moe._all_reduce = False
        moe._fp32_tp_reduce = False
        x = torch.randn(1, 5, 4)
        routes = torch.full((5, 3, 2), SENTINEL, dtype=torch.uint16)
        expected = moe(x)
        actual = moe(x, routes)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.all(routes[:, 1].int() == torch.tensor([2, 5]))
        assert torch.all(routes[:, (0, 2)].int() == SENTINEL)

    @pytest.mark.parametrize('dense', [False, True])
    def test_decoder_routes_only_sparse_ffn(self, dense):
        layer = bare(Glm5NextDecoderLayer)
        layer.is_linear_attention = True
        layer.self_attn = Attention()
        layer.mlp = nn.Identity() if dense else CaptureMoE()
        for name in ['hc_attn_fn', 'hc_attn_scale', 'hc_attn_base', 'hc_ffn_fn', 'hc_ffn_scale',
                     'hc_ffn_base', 'input_layernorm', 'post_attention_layernorm']:
            setattr(layer, name, None)
        layer._hc_pre = lambda x, *a, **kw: (x, None, None)
        layer._hc_post = lambda x, *a: (x, None)
        routes = torch.zeros(3, 2, 2, dtype=torch.uint16)
        x = torch.randn(1, 3, 4)
        out, _ = layer(x, (), None, None, all_routed_experts=routes)
        torch.testing.assert_close(out, x)
        if not dense:
            assert layer.mlp.seen is routes

    def test_mtp_decoder_does_not_capture_draft_routes(self):
        layer = bare(Glm5NextMTPDecoderLayer)
        layer.self_attn = Attention()
        layer.mlp = CaptureMoE()
        layer.input_layernorm = Norm()
        layer.post_attention_layernorm = Norm()
        layer(torch.randn(1, 3, 4), None, ())
        assert layer.mlp.seen is None

    def test_mtp_wrapper_ignores_target_capture_build_context(self, monkeypatch):
        from lmdeploy.pytorch.model_inputs import BuildModelContext
        from lmdeploy.pytorch.models import glm5_next as mod
        from lmdeploy.pytorch.models.patch import build_model_context

        class Predictor(nn.Module):
            def __init__(self, *a, **kw):
                super().__init__()

            def forward(self, input_ids, position_ids, target_hidden_states, **kwargs):
                assert 'all_routed_experts' not in kwargs
                return target_hidden_states

        monkeypatch.setattr(mod, 'GlmMoeDsaMultiTokenPredictor', Predictor)
        config = SimpleNamespace(text_config=SimpleNamespace(index_share_for_mtp_iteration=False))
        with build_model_context(BuildModelContext(enable_return_routed_experts=True)):
            draft = mod.Glm5NextMTPModel(config, ctx_mgr=None)
        ids = torch.ones(1, 3, dtype=torch.long)
        hidden = torch.randn(1, 3, 4)
        out = draft(ids, ids, hidden, [])
        assert out is hidden
        assert not hasattr(draft, 'enable_return_routed_experts')

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA router kernel required')
    def test_cuda_glm_router_capture_does_not_change_routing(self):
        from lmdeploy.pytorch.models.deepseek_v2 import MoEGate
        from lmdeploy.pytorch.models.glm5_next import Glm5NextNoauxTCRouter

        config = SimpleNamespace(num_experts_per_tok=8, n_routed_experts=288, n_group=1, topk_group=1,
                                 scoring_func='sigmoid', norm_topk_prob=True, model_type='glm5_next_text',
                                 topk_method='noaux_tc', routed_scaling_factor=2.5, router_n_groups=-1,
                                 moe_router_dtype='float32', n_shared_experts=1)
        gate = bare(MoEGate)
        gate.weight = nn.Parameter(torch.randn(288, 128, device='cuda'), requires_grad=False)
        gate.e_score_correction_bias = nn.Parameter(torch.randn(288, device='cuda'), requires_grad=False)
        gate.router_gemm = lambda x, weight: x @ weight.T
        gate.fake_eplb = False
        gate.eplb_dispatch_info = None
        gate.topk_method = 'noaux_tc'
        gate.noaux_tc_router = Glm5NextNoauxTCRouter(config)
        x = torch.randn(17, 128, device='cuda')
        weights, ids = gate(x)
        capture = torch.empty(17, 8, device='cuda', dtype=torch.uint16)
        actual_weights, actual_ids = gate(x, capture)
        torch.testing.assert_close(weights, actual_weights, rtol=0, atol=0)
        torch.testing.assert_close(ids, actual_ids, rtol=0, atol=0)
        torch.testing.assert_close(ids.int(), capture.int(), rtol=0, atol=0)
