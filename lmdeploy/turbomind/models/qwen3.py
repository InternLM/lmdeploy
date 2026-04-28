# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3 TextModel for the new pipeline.

Qwen3 is a standard Llama-like model with QK norm and optional MoE. No shared expert in the MoE variant, no linear
attention, no zero-centered norm.
"""
from __future__ import annotations

import _turbomind as _tm

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
from .utils import layer_progress, reorder_rotary_emb

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


@INPUT_MODELS.register_module(name='qwen3-moe')
@INPUT_MODELS.register_module(name='qwen3')
class Qwen3TextModel(TextModel):
    """Weight model for Qwen3 (dense) and Qwen3-MoE."""

    _layer_pattern = _LAYER_PATTERN

    def __init__(self, hf_cfg: dict, engine_cfg, *, resolver):
        super().__init__(hf_cfg, engine_cfg, resolver=resolver)

        # Fixed layer prefix for Qwen3
        self._layer_prefix = 'model.layers'
        self._embed_key = 'model.embed_tokens.weight'
        self._norm_key = 'model.norm.weight'

        self._n_experts = hf_cfg.get('num_experts', 0)

        # ---- Attention template ----
        dtype = self._dtype
        self._attn_cfg = _tm.AttentionConfig()
        self._attn_cfg.hidden_dim  = self._hidden_units
        self._attn_cfg.head_dim    = self._head_dim
        self._attn_cfg.head_num    = self._head_num
        self._attn_cfg.kv_head_num = self._kv_head_num
        self._attn_cfg.has_bias    = hf_cfg.get('attention_bias', 0)
        self._attn_cfg.qk_norm     = True
        self._apply_rope(self._attn_cfg.rope)
        self._attn_cfg.window_size = 0
        self._attn_cfg.tp_size     = engine_cfg.attn_tp_size
        self._attn_cfg.data_type   = dtype
        self._attn_cfg.softmax_scale = self._softmax_scale

        # ---- FFN template ----
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self._hidden_units
        self._ffn_cfg.has_bias   = False
        self._ffn_cfg.tp_size    = engine_cfg.mlp_tp_size
        self._ffn_cfg.data_type  = dtype
        self._ffn_cfg.act_type   = _act_type_id('silu')
        # fuse_silu / fused_moe / inter_size set per-call in ffn()/moe()

        # ---- MoE template (only if MoE variant) ----
        if self._n_experts > 0:
            self._moe_cfg = _tm.MoeConfig()
            self._moe_cfg.method            = 1  # kFused
            self._moe_cfg.experts_per_token = hf_cfg.get('num_experts_per_tok', 8)
            self._moe_cfg.norm_topk_prob    = hf_cfg.get('norm_topk_prob', False)
            self._moe_cfg.shared_gate       = False
            self._moe_cfg.routed_scale      = 1.0
            self._moe_cfg.router_bias       = False
            self._moe_cfg.topk_group        = 1
            self._moe_cfg.topk_method       = 'greedy'
            self._moe_cfg.n_group           = 1
            self._moe_cfg.scoring_func      = 'softmax'
            self._moe_cfg.router_n_groups   = 0
            self._moe_cfg.hidden_dim        = self._hidden_units
            self._moe_cfg.mlp_bias          = False
            self._moe_cfg.data_type         = dtype
            self._moe_cfg.tp_size           = engine_cfg.mlp_tp_size
            self._moe_cfg.act_type          = _act_type_id('silu')
            self._moe_cfg.fuse_silu         = True

            self._expert_inter_size = hf_cfg.get('moe_intermediate_size', 768)
        else:
            self._expert_inter_size = 0

        # ---- Per-layer inter_size (dense FFN) ----
        raw_inter = hf_cfg.get('intermediate_size', 0) if self._n_experts == 0 else 0
        self._inter_sizes = [raw_inter] * self._num_layer
        self._expert_nums = (
            [self._n_experts] * self._num_layer if self._n_experts > 0 else []
        )

    # ------------------------------------------------------------------
    # model() — walks full hierarchy (same as existing code)
    # ------------------------------------------------------------------

    def model(self):
        ec = self.engine_cfg
        cfg = _tm.ModelWeightConfig()
        cfg.tp_size = ec.attn_tp_size * ec.attn_cp_size
        cfg.data_type = self._dtype
        cfg.hidden_units = self._hidden_units
        root = TextModelBuilder(
            cfg, self._contexts,
            root_handles=self._root_handles,
            tp=ec.attn_tp_size * ec.attn_cp_size,
            ranks=self._model_tp_ranks,
            vocab_size=self._vocab_size)
        root.add_token_embeds(self._get(self._embed_key))
        root.norm = self.norm(self._get(self._norm_key))
        lm_key = self._embed_key if self._tie_embeddings else 'lm_head.weight'
        root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
        root.layers = self.layers(self._layer_prefix)
        root.build()

    # ------------------------------------------------------------------
    # Attention / FFN / MoE factories
    # ------------------------------------------------------------------

    def attn(self, pfx, layer):
        q = self._linear(f'{pfx}.q_proj')
        k = self._linear(f'{pfx}.k_proj')
        v = self._linear(f'{pfx}.v_proj')
        o = self._linear(f'{pfx}.o_proj')

        q = reorder_rotary_emb(q, self._head_dim, self._rope.dim,
                               resolver=self._resolver)
        k = reorder_rotary_emb(k, self._head_dim, self._rope.dim,
                               resolver=self._resolver)

        cfg = self._attn_cfg.clone()
        # No per-layer attention fields for Qwen3 (no sliding window).
        attn = AttentionBuilder(cfg, self._contexts,
                                tp=self.engine_cfg.attn_tp_size,
                                ranks=self._attn_ranks)

        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)

        attn.q_norm = self.qk_norm(self._get(f'{pfx}.q_norm.weight'),
                                   head_dim=self._head_dim, rope_dim=self._rope.dim)
        attn.k_norm = self.qk_norm(self._get(f'{pfx}.k_norm.weight'),
                                   head_dim=self._head_dim, rope_dim=self._rope.dim)

        return attn.build()

    def ffn(self, pfx, layer, inter_size=None, fused_moe=False):
        w1 = self._linear(f'{pfx}.gate_proj')
        w3 = self._linear(f'{pfx}.up_proj')
        w2 = self._linear(f'{pfx}.down_proj')

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = (inter_size if inter_size is not None
                          else self._inter_sizes[layer])
        cfg.fuse_silu  = False
        cfg.fused_moe  = fused_moe

        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m.build()

    def moe(self, pfx, layer):
        if self.num_experts(layer) <= 0:
            return None

        cfg = self._moe_cfg.clone()
        cfg.layer_id   = layer
        cfg.expert_num = self._expert_nums[layer]
        cfg.inter_size = self._expert_inter_size

        m = MoeBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)

        m.add_gate('gate', self._linear(f'{pfx}.gate'))

        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[e] = self.ffn(
                f'{pfx}.experts.{e}', layer,
                inter_size=self._expert_inter_size, fused_moe=True)
        m.experts = experts.build()
        return m.build()

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in layer_progress(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(
                self._get(f'{pfx}.{i}.input_layernorm.weight'))
            d.attention = self.attn(f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(
                self._get(f'{pfx}.{i}.post_attention_layernorm.weight'))
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', i)
            layers[i] = d.build()
        return layers.build()

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    def qk_norm(self, weight, *, head_dim, rope_dim):
        return self.norm(reorder_rotary_emb(weight, head_dim, rope_dim))
