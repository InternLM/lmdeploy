# Copyright (c) OpenMMLab. All rights reserved.
"""Gpt-oss TextModelSpec for the new pipeline."""
from __future__ import annotations

import re

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
from ..spec import TextModelSpec
from .base import INPUT_MODELS
from .utils import layer_progress, read_packed_moe_expert, reorder_rotary_emb

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


def map_experts(s: str) -> str:
    s = re.sub(r'(experts.*proj)$', r'\1.weight', s)
    s = re.sub(r'(experts.*proj)_bias$', r'\1.bias', s)
    s = re.sub(r'(experts.*proj)_blocks$', r'\1.blocks', s)
    s = re.sub(r'(experts.*proj)_scales$', r'\1.scales', s)
    return s


@INPUT_MODELS.register_module(name='gpt-oss')
class GptOssSpec(TextModelSpec):
    """Weight spec for gpt-oss (MoE with packed experts)."""

    _layer_pattern = _LAYER_PATTERN
    _loader_mappings = [map_experts]

    def __init__(self, hf_cfg: dict, engine_cfg, *, resolver):
        super().__init__(hf_cfg, engine_cfg, resolver=resolver)

        self._layer_prefix = 'model.layers'
        self._embed_key = 'model.embed_tokens.weight'
        self._norm_key = 'model.norm.weight'

        self._n_experts = hf_cfg['num_local_experts']
        dtype = self._dtype

        # ---- Attention template (sliding window set per layer) ----
        self._attn_cfg = _tm.AttentionConfig()
        self._attn_cfg.hidden_dim  = self._hidden_units
        self._attn_cfg.head_dim    = self._head_dim
        self._attn_cfg.head_num    = self._head_num
        self._attn_cfg.kv_head_num = self._kv_head_num
        self._attn_cfg.has_bias    = int(hf_cfg['attention_bias'])
        self._attn_cfg.attn_sink   = True
        self._apply_rope(self._attn_cfg.rope)
        self._attn_cfg.window_size = 0
        self._attn_cfg.tp_size     = engine_cfg.attn_tp_size
        self._attn_cfg.data_type   = dtype
        self._attn_cfg.softmax_scale          = self._softmax_scale

        # ---- FFN template ----
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self._hidden_units
        self._ffn_cfg.has_bias   = True
        self._ffn_cfg.tp_size    = engine_cfg.mlp_tp_size
        self._ffn_cfg.data_type  = dtype
        self._ffn_cfg.act_type   = _act_type_id('gpt-oss')

        # ---- MoE template ----
        self._moe_cfg = _tm.MoeConfig()
        self._moe_cfg.method            = 1
        self._moe_cfg.experts_per_token = hf_cfg['experts_per_token']
        self._moe_cfg.norm_topk_prob    = True
        self._moe_cfg.shared_gate       = False
        self._moe_cfg.routed_scale      = 1.0
        self._moe_cfg.router_bias       = True
        self._moe_cfg.topk_group        = 1
        self._moe_cfg.topk_method       = 'greedy'
        self._moe_cfg.n_group           = 1
        self._moe_cfg.scoring_func      = 'softmax'
        self._moe_cfg.router_n_groups   = 0
        self._moe_cfg.hidden_dim        = self._hidden_units
        self._moe_cfg.mlp_bias          = True
        self._moe_cfg.data_type         = dtype
        self._moe_cfg.tp_size           = engine_cfg.mlp_tp_size
        self._moe_cfg.act_type          = _act_type_id('gpt-oss')
        self._moe_cfg.fuse_silu         = True

        self._expert_inter_size = hf_cfg['intermediate_size']

        # Per-layer window sizes from layer_types
        types = hf_cfg['layer_types']
        sliding = hf_cfg['sliding_window']
        self._window_sizes = [
            sliding if t == 'sliding_attention' else 0 for t in types
        ]

        # Inter-size list (zero; gpt-oss has no dense FFN layers)
        self._inter_sizes = [0] * self._num_layer
        self._expert_nums = [self._n_experts] * self._num_layer

    def num_experts(self, layer: int) -> int:
        return self._n_experts

    # ------------------------------------------------------------------
    # model() — same topology as old code
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
    # Attention factory — sets per-layer window_size on the clone
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
        cfg.window_size = self._window_sizes[layer]

        attn = AttentionBuilder(cfg, self._contexts,
                                tp=self.engine_cfg.attn_tp_size,
                                ranks=self._attn_ranks)
        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(o)

        attn.add_param('sinks', self._get(f'{pfx}.sinks'))
        return attn.build()

    # ------------------------------------------------------------------
    # FFN/MoE factories — packed-expert handling
    # ------------------------------------------------------------------

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

        m.add_gate('gate', self._linear(f'{pfx}.router'))

        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self.num_experts(layer)):
            experts[e] = self._packed_moe_ffn(
                pfx, e, self._expert_inter_size)
        m.experts = experts.build()
        return m.build()

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in layer_progress(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(self._get(f'{pfx}.{i}.input_layernorm.weight'))
            d.attention = self.attn(f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(self._get(f'{pfx}.{i}.post_attention_layernorm.weight'))
            if self.num_experts(i) > 0:
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            layers[i] = d.build()
        return layers.build()

    def _packed_moe_ffn(self, mlp_pfx, expert_idx, inter_size):
        w1, w2, w3 = read_packed_moe_expert(
            self.params,
            f'{mlp_pfx}.experts.gate_up_proj',
            f'{mlp_pfx}.experts.down_proj',
            expert_idx,
            resolver=self._resolver,
            interleaved=True,
            trans=True,
        )
        cfg = self._ffn_cfg.clone()
        cfg.inter_size = inter_size
        cfg.fuse_silu  = False
        cfg.fused_moe  = True
        m = FfnBuilder(cfg, self._contexts,
                       tp=self.engine_cfg.mlp_tp_size,
                       ranks=self._mlp_ranks)
        m.add_ffn(w1, w2, w3)
        return m.build()
