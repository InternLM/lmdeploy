# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 TextModel for the new pipeline."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import _turbomind as _tm

if TYPE_CHECKING:
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


def _center_norm(w):
    return 1.0 + w.float()


class Qwen3_5TextModel(TextModel):
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

    def model(self, pfx):
        root_cfg = make_model_weight_config(self.cfg)
        builder = TextModelBuilder(
            root_cfg, self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size)
        builder.add_token_embeds(pfx.get('model.language_model.embed_tokens.weight'))
        builder.norm = self.norm(
            pfx + 'model.language_model.norm', _center_norm)
        lm_pfx = (pfx + 'model.language_model.embed_tokens'
                  if self.cfg.tie_word_embeddings
                  else pfx + 'lm_head')
        builder.add_lm_head(self._linear(lm_pfx))
        builder.layers = self.layers(pfx + 'model.language_model.layers')
        builder.build()

    # ------------------------------------------------------------------
    # Attention / linear-attention factories
    # ------------------------------------------------------------------

    def attn(self, pfx):
        q, k, v, o = [self._linear(pfx + f'{x}_proj') for x in 'qkvo']

        cfg = self._attn_cfg.clone()
        q, gate = split_output_gate(q, head_num=cfg.head_num)

        def reorder(x):
            return reorder_rotary_emb(x, cfg.head_dim, cfg.rope.dim, resolver=self._resolver)

        q, k = [reorder(x) for x in (q, k)]

        m = AttentionBuilder(cfg, self._ctx, tp=self._attn_tp)

        m.add_qkv_proj(q, k, v, gate=gate)
        m.add_o_proj(o)

        m.q_norm = self.norm(pfx + 'q_norm',
                             lambda w: reorder(_center_norm(w)))
        m.k_norm = self.norm(pfx + 'k_norm',
                             lambda w: reorder(_center_norm(w)))

        return m.build()

    def linear_attn(self, pfx):
        cfg = self._dn_cfg.clone()
        builder = DeltaNetBuilder(cfg, self._ctx, tp=self._attn_tp)

        builder.add_input_projections(
            in_proj_qkv=self._linear(pfx + 'in_proj_qkv'),
            in_proj_z=self._linear(pfx + 'in_proj_z'),
            in_proj_b=self._linear(pfx + 'in_proj_b'),
            in_proj_a=self._linear(pfx + 'in_proj_a'),
            out_proj=self._linear(pfx + 'out_proj'))
        builder.add_scalar_params(
            a_log=pfx.pop('A_log'),
            dt_bias=pfx.pop('dt_bias'))
        builder.add_conv1d(
            pfx.pop('conv1d.weight'))
        builder.norm = self.norm(pfx + 'norm')  # not zero-centered
        return builder.build()

    # ------------------------------------------------------------------
    # FFN / MoE factories
    # ------------------------------------------------------------------

    def ffn(self, pfx, inter_size, is_expert=False):
        try:
            w1, w3, w2 = [self._linear(pfx + f'{x}_proj')
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

        m.add_gate('gate', self._linear(pfx + 'gate'))

        experts_pfx = pfx + 'experts'
        experts = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for e in range(self._n_experts):
            experts[e] = self._moe_expert_ffn(
                experts_pfx, e, self.cfg.moe_intermediate_size)
        m.experts = experts.build()

        m.add_gate('shared_gate', self._linear(pfx + 'shared_expert_gate'))
        shared = self.ffn(pfx + 'shared_expert', self.cfg.shared_expert_intermediate_size)

        return m.build(), shared

    def _packed_moe_ffn(self, experts_pfx, expert_idx, inter_size):
        w1, w2, w3 = read_packed_moe_expert(
            experts_pfx + 'gate_up_proj',
            experts_pfx + 'down_proj',
            expert_idx,
            resolver=self._resolver,
        )
        cfg = self._ffn_cfg.clone()
        cfg.inter_size = inter_size
        cfg.is_expert  = True
        m = FfnBuilder(cfg, self._ctx, tp=self._mlp_tp)
        m.add_ffn(w1, w2, w3)
        return m.build()

    def _moe_expert_ffn(self, experts_pfx, expert_idx, inter_size):
        expert_pfx = experts_pfx + expert_idx
        return (self.ffn(expert_pfx, inter_size, is_expert=True)
                or self._packed_moe_ffn(experts_pfx, expert_idx, inter_size))

    # ------------------------------------------------------------------
    # layers() — dispatch by layer type
    # ------------------------------------------------------------------

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            if self.cfg.layer_types[i] == 'linear_attention':
                d.linear_attn = self.linear_attn(p + 'linear_attn')
            else:
                d.attention = self.attn(p + 'self_attn')
            if self._n_experts > 0:
                d.moe_ffn, d.feed_forward = self.moe(p + 'mlp')
            else:
                d.feed_forward = self.ffn(p + 'mlp', self.cfg.intermediate_size)
            d.attention_norm = self.norm(p + 'input_layernorm', _center_norm)
            d.ffn_norm = self.norm(p + 'post_attention_layernorm', _center_norm)
            layers[i] = d.build()
        return layers.build()


@INPUT_MODELS.register_module(name='qwen3_5-moe')
@INPUT_MODELS.register_module(name='qwen3_5')
class Qwen3_5Model:
    """Aggregate source model for Qwen3.5 checkpoints."""

    def __init__(self, cfg, *, resolver):
        text_cfg = getattr(cfg, 'text_config', cfg)
        self.text_model = Qwen3_5TextModel(text_cfg, resolver=resolver)
        self.vision_model = None

    def bind_runtime(self, *, ctx, root_handles,
                     attn_tp, mlp_tp, model_tp):
        self.text_model.bind_runtime(
            ctx=ctx,
            root_handles=root_handles,
            attn_tp=attn_tp,
            mlp_tp=mlp_tp,
            model_tp=model_tp,
        )

    @property
    def _vocab_size(self):
        return self.text_model.cfg.vocab_size

    @property
    def _loader_mappings(self):
        return list(getattr(type(self.text_model), '_loader_mappings', []))

    def model(self, pfx):
        self.text_model.model(pfx)
