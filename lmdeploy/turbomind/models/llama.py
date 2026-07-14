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
    # Attention / FFN factories
    # ------------------------------------------------------------------

    def attn(self, pfx):
        q, k, v, o = [self._linear(pfx + f'{x}_proj') for x in 'qkvo']

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
        w1, w3, w2 = [self._linear(pfx + f'{x}_proj') for x in ('gate', 'up', 'down')]

        cfg = self._ffn_cfg.clone()

        m = FfnBuilder(cfg, self._ctx, tp=self._mlp_tp)
        m.add_ffn(w1, w2, w3)
        return m.build()

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            d.attention_norm = self.norm(p + 'input_layernorm')
            d.attention = self.attn(p + 'self_attn')
            d.ffn_norm = self.norm(p + 'post_attention_layernorm')
            d.feed_forward = self.ffn(p + 'mlp')
            layers[i] = d.build()
        return layers.build()
