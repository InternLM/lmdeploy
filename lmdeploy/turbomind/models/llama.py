# Copyright (c) OpenMMLab. All rights reserved.
"""Llama TextModel for the new pipeline."""
from __future__ import annotations

from transformers import LlamaConfig

from ..builders import (
    AttentionBuilder,
    DecoderLayerBuilder,
    DecoderLayerConfig,
    FfnBuilder,
    ModuleListBuilder,
    ModuleListConfig,
    TextModelBuilder,
    _act_type_id,
)
from ..text_model import TextModel
from .base import INPUT_MODELS
from .utils import (
    layer_progress,
    make_attention_config,
    make_ffn_config,
    make_model_weight_config,
    reorder_rotary_emb,
)


@INPUT_MODELS.register_module(name='llama')
class LlamaModel(TextModel):
    """Weight model for Llama decoder-only variants."""

    cfg: LlamaConfig

    def __init__(self, cfg: LlamaConfig, *, resolver):
        super().__init__(cfg, resolver=resolver)

        self._attn_cfg = make_attention_config(cfg)

        self._ffn_cfg = make_ffn_config(cfg,
                                        act_type=_act_type_id('silu'))

    # ------------------------------------------------------------------
    # model() — walks full hierarchy
    # ------------------------------------------------------------------

    def model(self):
        embed_key = 'model.embed_tokens.weight'
        root_cfg = make_model_weight_config(self.cfg)
        root = TextModelBuilder(
            root_cfg, self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size)
        root.add_token_embeds(self._get(embed_key))
        root.norm = self.norm(self._get('model.norm.weight'))
        lm_key = embed_key if self.cfg.tie_word_embeddings else 'lm_head.weight'
        root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
        root.layers = self.layers('model.layers')
        root.build()

    # ------------------------------------------------------------------
    # Attention / FFN factories
    # ------------------------------------------------------------------

    def attn(self, pfx):
        q, k, v, o = [self._linear(f'{pfx}.{x}_proj') for x in 'qkvo']

        cfg = self._attn_cfg.clone()

        def reorder(x):
            return reorder_rotary_emb(x, cfg.head_dim, cfg.rope.dim, resolver=self._resolver)

        q, k = [reorder(x) for x in (q, k)]

        # No QK-norm for Llama.
        m = AttentionBuilder(cfg, self._ctx, tp=self._attn_tp)

        m.add_qkv_proj(q, k, v)
        m.add_o_proj(o)

        return m.build()

    def ffn(self, pfx):
        w1, w3, w2 = [self._linear(f'{pfx}.{x}_proj') for x in ('gate', 'up', 'down')]

        cfg = self._ffn_cfg.clone()

        m = FfnBuilder(cfg, self._ctx, tp=self._mlp_tp)
        m.add_ffn(w1, w2, w3)
        return m.build()

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i in layer_progress(self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            d.attention_norm = self.norm(
                self._get(f'{pfx}.{i}.input_layernorm.weight'))
            d.attention = self.attn(f'{pfx}.{i}.self_attn')
            d.ffn_norm = self.norm(
                self._get(f'{pfx}.{i}.post_attention_layernorm.weight'))
            d.feed_forward = self.ffn(f'{pfx}.{i}.mlp')
            layers[i] = d.build()
        return layers.build()
