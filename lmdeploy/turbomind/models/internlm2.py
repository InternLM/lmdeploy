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
    layer_progress,
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

    def model(self):
        embed_key = 'model.tok_embeddings.weight'
        root_cfg = make_model_weight_config(self.cfg)
        root = TextModelBuilder(
            root_cfg, self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size)
        root.add_token_embeds(self._get(embed_key))
        root.norm = self.norm(self._get('model.norm.weight'))
        lm_key = embed_key if self.cfg.tie_word_embeddings else 'output.weight'
        root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
        root.layers = self.layers('model.layers')
        root.build()

    # ------------------------------------------------------------------
    # attn() — deinterleave fused wqkv then feed to AttentionBuilder
    # ------------------------------------------------------------------

    def attn(self, pfx):
        wqkv = self._linear(f'{pfx}.wqkv')
        cfg = self._attn_cfg.clone()
        q, k, v = _split_qkv_gqa(
            wqkv, head_dim=cfg.head_dim,
            q_heads=cfg.head_num, kv_heads=cfg.kv_head_num)
        o = self._linear(f'{pfx}.wo')

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
        w1, w3, w2 = [self._linear(f'{pfx}.{x}') for x in ('w1', 'w3', 'w2')]

        cfg = self._ffn_cfg.clone()

        m = FfnBuilder(cfg, self._ctx, tp=self._mlp_tp)
        m.add_ffn(w1, w2, w3)
        return m.build()

    # ------------------------------------------------------------------
    # layers() — standard loop, InternLM2 norm names
    # ------------------------------------------------------------------

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i in layer_progress(self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            d.attention_norm = self.norm(
                self._get(f'{pfx}.{i}.attention_norm.weight'))
            d.attention = self.attn(f'{pfx}.{i}.attention')
            d.ffn_norm = self.norm(
                self._get(f'{pfx}.{i}.ffn_norm.weight'))
            d.feed_forward = self.ffn(f'{pfx}.{i}.feed_forward')
            layers[i] = d.build()
        return layers.build()
