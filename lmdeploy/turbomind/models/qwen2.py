# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen2 TextModel for the new pipeline.

Handles both dense Qwen2 and Qwen2-MoE variants. MoE detected via num_experts in HF config. Shared expert uses
shared_gate pattern matching Qwen3.5. No QK-norm, no sliding window.
"""
from __future__ import annotations

from transformers import Qwen2Config, Qwen2MoeConfig

from ..builders import (
    AttentionBuilder,
    DecoderLayerBuilder,
    DecoderLayerConfig,
    FfnBuilder,
    ModuleListBuilder,
    ModuleListConfig,
    MoeBuilder,
    TextModelBuilder,
    _act_type_id,
)
from ..text_model import TextModel
from .base import INPUT_MODELS
from .utils import (
    make_attention_config,
    make_ffn_config,
    make_model_weight_config,
    make_moe_config,
    reorder_rotary_emb,
)


@INPUT_MODELS.register_module(name='qwen2-moe')
@INPUT_MODELS.register_module(name='qwen2')
class Qwen2Model(TextModel):
    """Weight model for Qwen2 (dense) and Qwen2-MoE."""

    cfg: Qwen2Config | Qwen2MoeConfig

    def __init__(self, cfg: Qwen2Config | Qwen2MoeConfig, *, resolver):
        super().__init__(cfg, resolver=resolver)

        self._attn_cfg = make_attention_config(cfg)

        self._ffn_cfg = make_ffn_config(cfg,
                                        act_type=_act_type_id('silu'))

        self._n_experts = getattr(cfg, 'num_experts', 0)
        # ---- MoE template (only if MoE variant) ----
        if self._n_experts > 0:
            self._moe_cfg = make_moe_config(
                cfg,
                experts_per_token=cfg.num_experts_per_tok,
                norm_topk_prob=cfg.norm_topk_prob)
            self._moe_cfg.expert_num = self._n_experts

    # ------------------------------------------------------------------
    # model() — walks full hierarchy
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
        lm_pfx = (pfx + 'model.embed_tokens'
                  if self.cfg.tie_word_embeddings
                  else pfx + 'lm_head')
        builder.add_lm_head(self._linear(lm_pfx))
        builder.layers = self.layers(pfx + 'model.layers')
        builder.build()

    # ------------------------------------------------------------------
    # Attention / FFN / MoE factories
    # ------------------------------------------------------------------

    def attn(self, pfx):
        q, k, v, o = [self._linear(pfx + f'{x}_proj') for x in 'qkvo']

        cfg = self._attn_cfg.clone()

        def reorder(x):
            return reorder_rotary_emb(x, cfg.head_dim, cfg.rope.dim, resolver=self._resolver)

        q, k = [reorder(x) for x in (q, k)]

        # No QK-norm for Qwen2.
        m = AttentionBuilder(cfg, self._ctx, tp=self._attn_tp)

        m.add_qkv_proj(q, k, v)
        m.add_o_proj(o)

        return m.build()

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

        experts = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for e in range(self.cfg.num_experts):
            experts[e] = self.ffn(pfx + 'experts' + e,
                                  self.cfg.moe_intermediate_size,
                                  is_expert=True)
        m.experts = experts.build()

        m.add_gate('shared_gate', self._linear(pfx + 'shared_expert_gate'))
        shared = self.ffn(pfx + 'shared_expert',
                          self.cfg.shared_expert_intermediate_size)

        return m.build(), shared

    # ------------------------------------------------------------------
    # layers() — layer dispatch loop
    # ------------------------------------------------------------------

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            d.attention = self.attn(p + 'self_attn')
            if self._n_experts > 0:
                d.moe_ffn, d.feed_forward = self.moe(p + 'mlp')
            else:
                d.feed_forward = self.ffn(p + 'mlp', self.cfg.intermediate_size)
            d.attention_norm = self.norm(p + 'input_layernorm')
            d.ffn_norm = self.norm(p + 'post_attention_layernorm')
            layers[i] = d.build()
        return layers.build()
