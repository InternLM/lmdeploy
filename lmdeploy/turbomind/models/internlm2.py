# Copyright (c) OpenMMLab. All rights reserved.
"""InternLM2 / InternLM2.5 TextModel for the new pipeline.

Handles InternLM2 and InternLM2.5 decoder variants.  The key difference from Llama is the GQA-interleaved fused wqkv
projection that must be deinterleaved into separate Q / K / V bundles before feeding to AttentionBuilder.
"""
from __future__ import annotations

from transformers import PretrainedConfig

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
from ..linear import transform_output_dim
from ..text_model import TextModel
from .base import INPUT_MODELS
from .utils import (
    make_attention_config,
    make_ffn_config,
    make_model_weight_config,
    reorder_rotary_emb,
)


@transform_output_dim
def _split_qkv_gqa(w_qkv, *, head_dim, q_heads, kv_heads):
    """Deinterleave a GQA-fused QKV tensor into separate Q, K, V tensors.

    InternLM2 layout: ``[Q0 Q1 Q2 Q3 K V]`` repeated per KV group.
    ``per_head_elems`` self-adapts (128 for weights, 1 for block scales).
    """
    groups = kv_heads
    q_per_group = q_heads // kv_heads
    slots = q_per_group + 2               # Q-slots + K + V
    total = groups * slots
    n = w_qkv.size(-1) // total           # elems per head-equivalent

    t = w_qkv.unflatten(-1, (groups, slots, n))
    q = t[..., :q_per_group, :].flatten(-3, -2)
    k = t[..., q_per_group, :].flatten(-2, -1)
    v = t[..., q_per_group + 1, :].flatten(-2, -1)
    return q.contiguous(), k.contiguous(), v.contiguous()


@INPUT_MODELS.register_module(name='internlm2')
class InternLM2Model(TextModel):
    """Weight model for InternLM2 / InternLM2.5 decoder-only variants."""

    cfg: PretrainedConfig

    def __init__(self, cfg: PretrainedConfig, *, resolver):
        super().__init__(cfg, resolver=resolver)

        self._attn_cfg = make_attention_config(cfg)

        self._ffn_cfg = make_ffn_config(cfg,
                                        act_type=_act_type_id('silu'))

    # ------------------------------------------------------------------
    # model() — full topology
    # ------------------------------------------------------------------

    def model(self, pfx):
        root_cfg = make_model_weight_config(self.cfg)
        builder = TextModelBuilder(
            root_cfg, self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size)
        builder.add_token_embeds(pfx.get('model.tok_embeddings.weight'))
        builder.norm = self.norm(pfx + 'model.norm')
        lm_pfx = (pfx + 'model.tok_embeddings'
                  if self.cfg.tie_word_embeddings
                  else pfx + 'output')
        builder.add_lm_head(self._linear(lm_pfx))
        builder.layers = self.layers(pfx + 'model.layers')
        builder.build()

    # ------------------------------------------------------------------
    # attn() — deinterleave fused wqkv then feed to AttentionBuilder
    # ------------------------------------------------------------------

    def attn(self, pfx):
        wqkv = self._linear(pfx + 'wqkv')
        cfg = self._attn_cfg.clone()
        q, k, v = _split_qkv_gqa(
            wqkv, head_dim=cfg.head_dim,
            q_heads=cfg.head_num, kv_heads=cfg.kv_head_num)
        o = self._linear(pfx + 'wo')

        def reorder(x):
            return reorder_rotary_emb(x, cfg.head_dim, cfg.rope.dim, resolver=self._resolver)

        q, k = [reorder(x) for x in (q, k)]

        m = AttentionBuilder(cfg, self._ctx, tp=self._attn_tp)
        m.add_qkv_proj(q, k, v)
        m.add_o_proj(o)

        return m.build()

    # ------------------------------------------------------------------
    # ffn() — InternLM2 uses w1 / w3 / w2 naming
    # ------------------------------------------------------------------

    def ffn(self, pfx):
        w1, w3, w2 = [self._linear(pfx + x) for x in ('w1', 'w3', 'w2')]

        cfg = self._ffn_cfg.clone()

        m = FfnBuilder(cfg, self._ctx, tp=self._mlp_tp)
        m.add_ffn(w1, w2, w3)
        return m.build()

    # ------------------------------------------------------------------
    # layers() — standard loop, InternLM2 norm names
    # ------------------------------------------------------------------

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            d.attention_norm = self.norm(p + 'attention_norm')
            d.attention = self.attn(p + 'attention')
            d.ffn_norm = self.norm(p + 'ffn_norm')
            d.feed_forward = self.ffn(p + 'feed_forward')
            layers[i] = d.build()
        return layers.build()
