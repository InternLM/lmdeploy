# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 TextModel for the new pipeline."""
from __future__ import annotations

import re

import _turbomind as _tm
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from ..builders import (
    AttentionBuilder,
    DecoderLayerBuilder,
    DecoderLayerConfig,
    DeltaNetBuilder,
    FfnBuilder,
    ModuleListBuilder,
    ModuleListConfig,
    MoeBuilder,
    TextModelBuilder,
    _act_type_id,
)
from ..builders.attention import split_output_gate
from ..text_model import TextModel
from .base import INPUT_MODELS
from .utils import (
    layer_progress,
    make_attention_config,
    make_ffn_config,
    make_model_weight_config,
    make_moe_config,
    read_packed_moe_expert,
    reorder_rotary_emb,
)


def map_packed_qwen35_experts(name: str) -> str:
    """Map packed expert names to weight names so parameter.py can classify."""
    return re.sub(r'(mlp\.experts\.(?:gate_up|down)_proj)$', r'\1.weight', name)


@INPUT_MODELS.register_module(name='qwen3_5-moe')
@INPUT_MODELS.register_module(name='qwen3_5')
class Qwen3_5Model(TextModel):
    """Weight model for Qwen3.5 (dense + linear-attn + optional MoE)."""

    _loader_mappings = [map_packed_qwen35_experts]
    cfg: Qwen3_5TextConfig | Qwen3_5MoeTextConfig

    def __init__(self, cfg: Qwen3_5TextConfig | Qwen3_5MoeTextConfig, *, resolver):
        super().__init__(cfg, resolver=resolver)

        self._attn_cfg = make_attention_config(cfg)
        self._attn_cfg.output_gate = True

        self._n_experts = getattr(cfg, 'num_experts', 0)

        # ---- DeltaNet template ----
        ln_key_heads = cfg.linear_num_key_heads
        ln_val_heads = cfg.linear_num_value_heads
        ln_key_dim   = cfg.linear_key_head_dim
        ln_val_dim   = cfg.linear_value_head_dim

        self._dn_cfg = _tm.DeltaNetConfig()
        self._dn_cfg.hidden_dim      = self.cfg.hidden_size
        self._dn_cfg.num_k_heads     = ln_key_heads
        self._dn_cfg.num_v_heads     = ln_val_heads
        self._dn_cfg.key_head_dim    = ln_key_dim
        self._dn_cfg.value_head_dim  = ln_val_dim
        self._dn_cfg.d_conv          = cfg.linear_conv_kernel_dim or 4
        q_dim = ln_key_heads * ln_key_dim
        v_dim = ln_val_heads * ln_val_dim
        self._linear_qkv_split = (q_dim, q_dim, v_dim)

        # ---- MoE template ----
        if self._n_experts > 0:
            self._moe_cfg = make_moe_config(
                cfg,
                experts_per_token=cfg.num_experts_per_tok)
            self._moe_cfg.expert_num = self._n_experts
            inter_size=cfg.moe_intermediate_size
        else:
            inter_size=cfg.intermediate_size

        # ---- FFN template ----
        self._ffn_cfg = make_ffn_config(
            cfg,
            act_type=_act_type_id('silu'), inter_size=inter_size)

    # ------------------------------------------------------------------
    # model() — same topology as old code
    # ------------------------------------------------------------------

    def model(self):
        root_cfg = make_model_weight_config(self.cfg)
        root = TextModelBuilder(
            root_cfg, self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size)
        embed_key = 'model.language_model.embed_tokens.weight'
        root.add_token_embeds(self._get(embed_key))
        root.norm = self.norm(1.0 + self._get('model.language_model.norm.weight'))
        lm_key = embed_key if self.cfg.tie_word_embeddings else 'lm_head.weight'
        root.add_lm_head(self._linear(lm_key.removesuffix('.weight')))
        root.layers = self.layers('model.language_model.layers')
        root.build()

    # ------------------------------------------------------------------
    # Attention / linear-attention factories
    # ------------------------------------------------------------------

    def attn(self, pfx):
        q, k, v, o = [self._linear(f'{pfx}.{x}_proj') for x in 'qkvo']

        cfg = self._attn_cfg.clone()
        q, gate = split_output_gate(q, head_num=cfg.head_num)

        def reorder(x):
            return reorder_rotary_emb(x, cfg.head_dim, cfg.rope.dim, resolver=self._resolver)

        q, k = [reorder(x) for x in (q, k)]

        m = AttentionBuilder(cfg, self._ctx, tp=self._attn_tp)

        m.add_qkv_proj(q, k, v, gate=gate)
        m.add_o_proj(o)

        q_norm, k_norm = [self._get(f'{pfx}.{x}_norm.weight') for x in 'qk']

        m.q_norm = self.norm(reorder(1.0 + q_norm.float()))
        m.k_norm = self.norm(reorder(1.0 + k_norm.float()))

        return m.build()

    def linear_attn(self, pfx):
        cfg = self._dn_cfg.clone()
        builder = DeltaNetBuilder(cfg, self._ctx,
                                  tp=self._attn_tp)

        builder.add_input_projections(
            in_proj_qkv=self._linear(f'{pfx}.in_proj_qkv'),
            in_proj_z=self._linear(f'{pfx}.in_proj_z'),
            in_proj_b=self._linear(f'{pfx}.in_proj_b'),
            in_proj_a=self._linear(f'{pfx}.in_proj_a'),
            out_proj=self._linear(f'{pfx}.out_proj'),
            qkv_split=self._linear_qkv_split)
        builder.add_scalar_params(
            a_log=self._get(f'{pfx}.A_log'),
            dt_bias=self._get(f'{pfx}.dt_bias'))
        builder.add_conv1d(
            self._get(f'{pfx}.conv1d.weight'),
            qkv_split=self._linear_qkv_split)
        builder.norm = self.norm(self._get(f'{pfx}.norm.weight'))  # ! not zero-centered
        return builder.build()

    # ------------------------------------------------------------------
    # FFN / MoE factories
    # ------------------------------------------------------------------

    def ffn(self, pfx, inter_size, is_expert=False):
        try:
            w1, w3, w2 = [self._linear(f'{pfx}.{x}_proj')
                          for x in ('gate', 'up', 'down')]
        except KeyError:
            return None

        cfg = self._ffn_cfg.clone()
        cfg.inter_size = inter_size
        cfg.is_expert  = is_expert

        m = FfnBuilder(cfg, self._ctx, tp=self._mlp_tp)
        m.add_ffn(w1, w2, w3)
        return m.build()

    def moe(self, pfx):
        cfg = self._moe_cfg.clone()

        m = MoeBuilder(cfg, self._ctx)

        m.add_gate('gate', self._linear(f'{pfx}.gate'))

        experts = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for e in range(self._n_experts):
            experts[e] = self._moe_expert_ffn(f'{pfx}.experts', e, self.cfg.moe_intermediate_size)
        m.experts = experts.build()

        m.add_gate('shared_gate', self._linear(f'{pfx}.shared_expert_gate'))
        shared = self.ffn(f'{pfx}.shared_expert', self.cfg.shared_expert_intermediate_size)

        return m.build(), shared

    def _packed_moe_ffn(self, pfx, expert_idx, inter_size):
        w1, w2, w3 = read_packed_moe_expert(
            self.params,
            f'{pfx}.gate_up_proj',
            f'{pfx}.down_proj',
            expert_idx,
            resolver=self._resolver,
        )
        cfg = self._ffn_cfg.clone()
        cfg.inter_size = inter_size
        cfg.is_expert  = True
        m = FfnBuilder(cfg, self._ctx, tp=self._mlp_tp)
        m.add_ffn(w1, w2, w3)
        return m.build()

    def _moe_expert_ffn(self, pfx, expert_idx, inter_size):
        expert_pfx = f'{pfx}.{expert_idx}'
        inter_size = self.cfg.moe_intermediate_size
        return (self.ffn(expert_pfx, inter_size, is_expert=True)
                or self._packed_moe_ffn(pfx, expert_idx, inter_size))

    # ------------------------------------------------------------------
    # layers() — dispatch by layer type
    # ------------------------------------------------------------------

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i in layer_progress(self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            if self.cfg.layer_types[i] == 'linear_attention':
                d.linear_attn = self.linear_attn(f'{pfx}.{i}.linear_attn')
            else:
                d.attention = self.attn(f'{pfx}.{i}.self_attn')
            if self._n_experts > 0:
                d.moe_ffn, d.feed_forward = self.moe(f'{pfx}.{i}.mlp')
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', self.cfg.intermediate_size)
            d.attention_norm = self.norm(1.0 + self._get(f'{pfx}.{i}.input_layernorm.weight').float())
            d.ffn_norm = self.norm(1.0 + self._get(f'{pfx}.{i}.post_attention_layernorm.weight').float())
            layers[i] = d.build()
        return layers.build()
