# Copyright (c) OpenMMLab. All rights reserved.
"""Gpt-oss TextModel for the new pipeline."""
from __future__ import annotations

import re

from transformers import GptOssConfig

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
    read_packed_moe_expert,
    reorder_rotary_emb,
)


def map_experts(s: str) -> str:
    s = re.sub(r'(experts.*proj)$', r'\1.weight', s)
    s = re.sub(r'(experts.*proj)_bias$', r'\1.bias', s)
    s = re.sub(r'(experts.*proj)_blocks$', r'\1.blocks', s)
    s = re.sub(r'(experts.*proj)_scales$', r'\1.scales', s)
    return s


@INPUT_MODELS.register_module(name='gpt-oss')
class GptOssModel(TextModel):
    """Weight model for gpt-oss (MoE with packed experts)."""

    cfg: GptOssConfig

    _loader_mappings = [map_experts]

    def __init__(self, cfg: GptOssConfig, *, resolver):
        super().__init__(cfg, resolver=resolver)

        self._attn_cfg = make_attention_config(cfg)

        self._ffn_cfg = make_ffn_config(cfg,
                                        act_type=_act_type_id('gpt-oss'))
        self._ffn_cfg.inter_size = cfg.intermediate_size
        self._ffn_cfg.is_expert = True

        # ---- MoE template ----
        self._moe_cfg = make_moe_config(
            cfg,
            act_type=_act_type_id('gpt-oss'),
            experts_per_token=cfg.num_experts_per_tok)
        self._moe_cfg.expert_num = cfg.num_local_experts

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

    def attn(self, pfx, layer):
        q, k, v, o = [self._linear(pfx + f'{x}_proj') for x in 'qkvo']

        cfg = self._attn_cfg.clone()
        if self.cfg.layer_types[layer] == 'sliding_attention':
            cfg.window_size = self.cfg.sliding_window

        def reorder(x):
            return reorder_rotary_emb(x, cfg.head_dim, cfg.rope.dim, resolver=self._resolver)

        q, k = [reorder(x) for x in (q, k)]

        m = AttentionBuilder(cfg, self._ctx, tp=self._attn_tp)
        m.add_qkv_proj(q, k, v)
        m.add_o_proj(o)

        m.add_param('sinks', pfx.pop('sinks'))
        return m.build()

    def moe(self, pfx):
        cfg = self._moe_cfg.clone()
        m = MoeBuilder(cfg, self._ctx)
        m.add_gate('gate', self._linear(pfx + 'router'))
        experts_pfx = pfx + 'experts'
        experts = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for e in range(cfg.expert_num):
            experts[e] = self._packed_moe_ffn(experts_pfx, e)
        m.experts = experts.build()
        return m.build()

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            d.attention = self.attn(p + 'self_attn', i)
            d.moe_ffn = self.moe(p + 'mlp')
            d.attention_norm = self.norm(p + 'input_layernorm')
            d.ffn_norm = self.norm(p + 'post_attention_layernorm')
            layers[i] = d.build()
        return layers.build()

    def _packed_moe_ffn(self, experts_pfx, idx):
        w1, w2, w3 = read_packed_moe_expert(
            experts_pfx + 'gate_up_proj',
            experts_pfx + 'down_proj',
            idx,
            resolver=self._resolver,
            interleaved=True,
            trans=True,
        )
        cfg = self._ffn_cfg.clone()
        m = FfnBuilder(cfg, self._ctx, tp=self._mlp_tp)
        m.add_ffn(w1, w2, w3)
        return m.build()
