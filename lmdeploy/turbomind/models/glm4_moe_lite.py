# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-4 MoE Lite (GLM-4.7-Flash) TextModel for the new pipeline."""
from __future__ import annotations

from typing import TYPE_CHECKING

import _turbomind as _tm

if TYPE_CHECKING:
    from transformers import Glm4MoeLiteConfig

from ..builders import (
    DecoderLayerBuilder,
    DecoderLayerConfig,
    FfnBuilder,
    MLABuilder,
    ModuleListBuilder,
    ModuleListConfig,
    MoeBuilder,
    TextModelBuilder,
    _act_type_id,
)
from ..text_model import TextModel
from .base import INPUT_MODELS
from .utils import (
    make_mla_config,
    make_model_weight_config,
    make_moe_config,
)


@INPUT_MODELS.register_module(name='glm4-moe-lite')
class Glm4MoeLiteModel(TextModel):
    """Weight model for GLM-4 MoE Lite (e.g. GLM-4.7-Flash)."""

    cfg: Glm4MoeLiteConfig

    def __init__(self, cfg: Glm4MoeLiteConfig, *, resolver):
        super().__init__(cfg, resolver=resolver)

        self._attn_cfg = make_mla_config(cfg)

        # ---- FFN template ----
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self.cfg.hidden_size
        self._ffn_cfg.act_type   = _act_type_id('silu')

        # ---- MoE template (GLM-specific: noaux_tc + sigmoid) ----
        if cfg.n_routed_experts > 0:
            self._moe_cfg = make_moe_config(
                cfg,
                experts_per_token=cfg.num_experts_per_tok,
                topk_method='noaux_tc',
                scoring_func='sigmoid',
                routed_scale=cfg.routed_scaling_factor,
                topk_group=cfg.topk_group,
                n_group=cfg.n_group)
            self._moe_cfg.expert_num = cfg.n_routed_experts

        self._tune_layer_num = 2  # GLM-MoE recommends tuning 2 layers

    # ------------------------------------------------------------------
    # model()
    # ------------------------------------------------------------------

    def model(self, pfx):
        root_cfg = make_model_weight_config(self.cfg)
        builder = TextModelBuilder(
            root_cfg, self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size)
        builder.add_token_embeds(pfx.get('model.embed_tokens.weight'))
        builder.norm = self.norm(pfx + 'model.norm')
        builder.add_lm_head(self._linear(pfx + 'lm_head'))  # GLM: never tied
        builder.layers = self.layers(pfx + 'model.layers')
        builder.build()

    # ------------------------------------------------------------------
    # MLA attention (uses MLABuilder + self._attn_cfg clone)
    # ------------------------------------------------------------------

    def attn(self, pfx):
        cfg = self._attn_cfg.clone()
        m = MLABuilder(cfg, self._ctx, tp=self._attn_tp)

        q_b = (self._linear(pfx + 'q_b_proj', optional=True) or
               self._linear(pfx + 'q_proj'))
        m.add_projections(
            q_a_proj=self._linear(pfx + 'q_a_proj'),
            q_b_proj=q_b,
            kv_a_proj=self._linear(pfx + 'kv_a_proj_with_mqa'),
            kv_b_proj=self._linear(pfx + 'kv_b_proj'),
            wo=self._linear(pfx + 'o_proj'),
        )
        m.q_a_layernorm  = self.norm(pfx + 'q_a_layernorm')
        m.kv_a_layernorm = self.norm(pfx + 'kv_a_layernorm')
        return m.build()

    # ------------------------------------------------------------------
    # FFN / MoE factories
    # ------------------------------------------------------------------

    def ffn(self, pfx, inter_size, is_expert=False):
        w1, w3, w2 = [self._linear(pfx + f'{x}_proj') for x in ('gate', 'up', 'down')]

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = inter_size
        cfg.is_expert  = is_expert

        m = FfnBuilder(cfg, self._ctx, tp=self._mlp_tp)
        m.add_ffn(w1, w2, w3)
        return m.build()

    def moe(self, pfx):
        cfg = self._moe_cfg.clone()

        m = MoeBuilder(cfg, self._ctx)

        m.add_gate('gate', self._linear(pfx + 'gate'))

        correction = pfx.pop('gate.e_score_correction_bias')
        m.add_param('score_correction_bias', correction)

        experts = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for e in range(cfg.expert_num):
            experts[e] = self.ffn(pfx + 'experts' + e,
                                  self.cfg.moe_intermediate_size, is_expert=True)
        m.experts = experts.build()

        shared = self.ffn(pfx + 'shared_experts',
                          self.cfg.intermediate_size * self.cfg.n_shared_experts)

        return m.build(), shared

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            d.attention_norm = self.norm(p + 'input_layernorm')
            d.attention = self.attn(p + 'self_attn')
            d.ffn_norm = self.norm(p + 'post_attention_layernorm')
            if self.cfg.mlp_layer_types[i] == 'sparse':
                d.moe_ffn, d.feed_forward = self.moe(p + 'mlp')
            else:
                d.feed_forward = self.ffn(p + 'mlp', self.cfg.intermediate_size)
            layers[i] = d.build()
        return layers.build()
