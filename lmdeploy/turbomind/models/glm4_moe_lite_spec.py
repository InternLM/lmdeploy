# Copyright (c) OpenMMLab. All rights reserved.
"""GLM-4 MoE Lite (GLM-4.7-Flash) TextModelSpec for the new pipeline."""
from __future__ import annotations

import _turbomind as _tm

from ..builders import (
    DecoderLayerBuilder,
    DecoderLayerConfig,
    FfnBuilder,
    MLABuilder,
    ModuleListBuilder,
    ModuleListConfig,
    MoeBuilder,
    TextModelBuilder,
    _act_type_id,
)
from ..spec import TextModelSpec
from .base import INPUT_MODELS
from .utils import get_yarn_params, layer_progress, parse_rope_param

_LAYER_PATTERN = r'model\.layers\.([0-9]+).'


@INPUT_MODELS.register_module(name='glm4-moe-lite')
class Glm4MoeLiteSpec(TextModelSpec):
    """Weight spec for GLM-4 MoE Lite (e.g. GLM-4.7-Flash)."""

    _layer_pattern = _LAYER_PATTERN

    def __init__(self, hf_cfg: dict, engine_cfg, *, resolver):
        super().__init__(hf_cfg, engine_cfg, resolver=resolver)

        self._layer_prefix = 'model.layers'
        self._embed_key = 'model.embed_tokens.weight'
        self._norm_key = 'model.norm.weight'

        self._n_experts = hf_cfg.get('n_routed_experts', 0)
        self._dense_layers = hf_cfg.get('first_k_dense_replace', 1)
        dtype = self._dtype

        # ---- MLA head geometry (recomputed; differs from _parse_base default) ----
        qk_nope_dim = hf_cfg['qk_nope_head_dim']
        qk_rope_dim = hf_cfg['qk_rope_head_dim']
        kv_lora_rank = hf_cfg['kv_lora_rank']
        q_head_dim = qk_nope_dim + qk_rope_dim
        size_per_head = q_head_dim
        v_head_dim = hf_cfg['v_head_dim']
        softmax_scale = 0.0
        if kv_lora_rank and kv_lora_rank != qk_nope_dim:
            size_per_head = kv_lora_rank + qk_rope_dim
            v_head_dim = kv_lora_rank
            softmax_scale = q_head_dim ** (-0.5)

        # Override _parse_base defaults for MLA geometry
        self._head_dim = size_per_head
        self._kv_head_num = 1
        # RoPE dim = qk_rope_dim for MLA (not head_dim)
        self._rope, self._max_position_embeddings = parse_rope_param(
            hf_cfg, qk_rope_dim)
        self._softmax_scale = softmax_scale
        self._qk_nope_dim = qk_nope_dim

        # YaRN for MLA (override attention_factor + softmax_scale)
        rope_scaling = (hf_cfg.get('rope_parameters') or
                        hf_cfg.get('rope_scaling'))
        if rope_scaling and rope_scaling.get('type') == 'yarn':
            attention_factor, yarn_scale = get_yarn_params(rope_scaling)
            yarn_scale *= q_head_dim ** (-0.5)
            self._rope.max_position_embeddings = \
                rope_scaling['original_max_position_embeddings']
            self._rope.attention_factor = attention_factor
            self._softmax_scale = yarn_scale

        # ---- Attention template (for MLA; uses _tm.AttentionConfig) ----
        self._attn_cfg = _tm.AttentionConfig()
        self._attn_cfg.hidden_dim      = self._hidden_units
        self._attn_cfg.head_dim        = size_per_head
        self._attn_cfg.head_num        = self._head_num
        self._attn_cfg.kv_head_num     = self._kv_head_num
        self._attn_cfg.kv_lora_rank    = kv_lora_rank
        self._attn_cfg.q_lora_rank     = hf_cfg.get('q_lora_rank') or 0
        self._attn_cfg.qk_rope_dim     = qk_rope_dim
        self._attn_cfg.qk_nope_dim     = qk_nope_dim
        self._attn_cfg.v_head_dim      = v_head_dim
        self._attn_cfg.has_bias        = False
        self._attn_cfg.qk_norm         = False
        self._attn_cfg.attn_sink       = False
        self._attn_cfg.attn_output_gate = False
        self._apply_rope(self._attn_cfg.rope)
        self._attn_cfg.window_size     = 0
        self._attn_cfg.tp_size         = engine_cfg.attn_tp_size
        self._attn_cfg.data_type       = dtype
        self._attn_cfg.softmax_scale          = self._softmax_scale

        # ---- FFN template ----
        self._ffn_cfg = _tm.FfnConfig()
        self._ffn_cfg.hidden_dim = self._hidden_units
        self._ffn_cfg.has_bias   = False
        self._ffn_cfg.tp_size    = engine_cfg.mlp_tp_size
        self._ffn_cfg.data_type  = dtype
        self._ffn_cfg.act_type   = _act_type_id('silu')

        # ---- MoE template (GLM-specific: noaux_tc + sigmoid) ----
        if self._n_experts > 0:
            self._moe_cfg = _tm.MoeConfig()
            self._moe_cfg.method            = 1
            self._moe_cfg.experts_per_token = hf_cfg['num_experts_per_tok']
            self._moe_cfg.norm_topk_prob    = hf_cfg.get('norm_topk_prob', True)
            self._moe_cfg.shared_gate       = False
            self._moe_cfg.routed_scale      = hf_cfg.get('routed_scaling_factor', 1.0)
            self._moe_cfg.router_bias       = False
            self._moe_cfg.topk_group        = hf_cfg.get('topk_group', 1)
            self._moe_cfg.topk_method       = 'noaux_tc'  # GLM-specific
            self._moe_cfg.n_group           = hf_cfg.get('n_group', 1)
            self._moe_cfg.scoring_func      = 'sigmoid'   # GLM-specific
            self._moe_cfg.router_n_groups   = hf_cfg.get('router_n_groups', 0)
            self._moe_cfg.hidden_dim        = self._hidden_units
            self._moe_cfg.mlp_bias          = False
            self._moe_cfg.data_type         = dtype
            self._moe_cfg.tp_size           = engine_cfg.mlp_tp_size
            self._moe_cfg.act_type          = _act_type_id('silu')
            self._moe_cfg.fuse_silu         = True

            self._expert_inter_size = hf_cfg['moe_intermediate_size']
        else:
            self._expert_inter_size = 0

        # Per-layer inter_size:
        #   - dense layers use intermediate_size
        #   - MoE layers use n_shared_experts * moe_intermediate_size
        n_shared_experts = hf_cfg.get('n_shared_experts', 1)
        expert_inter = hf_cfg['moe_intermediate_size']
        raw_inter = [n_shared_experts * expert_inter] * self._num_layer
        raw_inter[0] = hf_cfg.get('intermediate_size',
                                  n_shared_experts * expert_inter)
        self._inter_sizes = raw_inter
        # Per-layer expert count (0 for dense layers)
        self._expert_nums = [
            self._n_experts if i >= self._dense_layers else 0
            for i in range(self._num_layer)
        ]

        self._tune_layer_num = 2  # GLM-MoE recommends tuning 2 layers

    def num_experts(self, layer: int) -> int:
        if layer < self._dense_layers:
            return 0
        return self._n_experts

    # ------------------------------------------------------------------
    # model() — same as old code
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
        root.add_lm_head(self._linear('lm_head'))  # GLM: never tied
        root.layers = self.layers(self._layer_prefix)
        root.build()

    # ------------------------------------------------------------------
    # MLA attention (uses MLABuilder + self._attn_cfg clone)
    # ------------------------------------------------------------------

    def attn(self, pfx, layer):
        cfg = self._attn_cfg.clone()
        # MLA nudge: kv_head_num must be >= attn_tp_size for TP
        if cfg.kv_lora_rank > 0 and cfg.kv_head_num < self.engine_cfg.attn_tp_size:
            cfg.kv_head_num = self.engine_cfg.attn_tp_size
        builder = MLABuilder(cfg, self._contexts,
                             tp=self.engine_cfg.attn_tp_size,
                             ranks=self._attn_ranks)

        q_b = (self._linear(f'{pfx}.q_b_proj', optional=True) or
               self._linear(f'{pfx}.q_proj'))
        builder.add_projections(
            q_a_proj=self._linear(f'{pfx}.q_a_proj'),
            q_b_proj=q_b,
            kv_a_proj=self._linear(f'{pfx}.kv_a_proj_with_mqa'),
            kv_b_proj=self._linear(f'{pfx}.kv_b_proj'),
            wo=self._linear(f'{pfx}.o_proj'),
        )
        builder.q_a_layernorm  = self.norm(self._get(f'{pfx}.q_a_layernorm.weight'))
        builder.kv_a_layernorm = self.norm(self._get(f'{pfx}.kv_a_layernorm.weight'))
        return builder.build()

    # ------------------------------------------------------------------
    # FFN / MoE factories
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

        m.add_gate('gate', self._linear(f'{pfx}.gate'))

        correction = self._get(f'{pfx}.gate.e_score_correction_bias')
        m.add_param('score_correction_bias', correction)

        experts = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for e in range(self._n_experts):
            experts[e] = self.ffn(
                f'{pfx}.experts.{e}', layer,
                inter_size=self._expert_inter_size, fused_moe=True)
        m.experts = experts.build()
        return m.build()

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._contexts)
        for i in layer_progress(self._num_layer):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._contexts)
            d.attention_norm = self.norm(self._get(f'{pfx}.{i}.input_layernorm.weight'))
            d.attention = self.attn(f'{pfx}.{i}.self_attn', i)
            d.ffn_norm = self.norm(self._get(f'{pfx}.{i}.post_attention_layernorm.weight'))
            if i < self._dense_layers:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp', i)
            else:
                d.feed_forward = self.ffn(f'{pfx}.{i}.mlp.shared_experts', i)
                d.moe_ffn = self.moe(f'{pfx}.{i}.mlp', i)
            layers[i] = d.build()
        return layers.build()
