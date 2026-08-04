# Copyright (c) OpenMMLab. All rights reserved.
"""Intern-S2-Mobius source model for the TurboMind pipeline.

Mobius is a Qwen3.5-topology checkpoint whose MoE weights are stored as
meta-MoE packs shared across layer groups::

    ...meta_mlp.{g}.experts.{gate_up,down}_proj   (packed, no .weight suffix)
    ...meta_mlp.{g}.gate.weight

The names are translated onto the canonical meta-MoE layout
(``meta_experts.{g}.*`` / ``meta_experts_gate.{g}.weight``) consumed by the
loading helpers below. Each meta group's routed gate/expert device buffers
exist once (on the donor layer); non-donor layers alias them in C++ via
``alias_routed_moe``. Groups are strided (``layer_id % n_groups``) so each
pack aligns with the repeating layer-type cycle (3×DeltaNet + 1×Attn).
"""
from __future__ import annotations

import re

from ..builders import (
    DecoderLayerBuilder,
    DecoderLayerConfig,
    ModuleListBuilder,
    ModuleListConfig,
    MoeBuilder,
)
from .base import INPUT_MODELS
from .qwen3_5 import (
    Qwen3_5Model,
    Qwen3_5TextModel,
    Qwen3_5VisionModel,
)


def map_mobius_meta_moe_names(name: str) -> str:
    # On-disk: ...meta_mlp.{g}.experts.{gate_up,down}_proj / meta_mlp.{g}.gate.weight
    # Canonical: ...meta_experts.{g}.{gate_up,down}_proj / meta_experts_gate.{g}.weight
    name = re.sub(r'meta_mlp\.(\d+)\.experts\.((?:gate_up|down)_proj)$',
                  r'meta_experts.\1.\2', name)
    name = re.sub(r'meta_mlp\.(\d+)\.gate\.weight$',
                  r'meta_experts_gate.\1.weight', name)
    return name


def map_meta_expert_names(name: str) -> str:
    # On-disk: ...meta_experts.{i}.gate_up_proj
    # After mapping: ...gate_up_proj.weight  (TrivialFormat suffix)
    return re.sub(
        r'(meta_experts\.\d+\.(?:gate_up|down)_proj)$',
        r'\1.weight',
        name,
    )


def infer_meta_geometry(lm_pfx, num_hidden_layers: int):
    n_groups = 0
    while lm_pfx.has(f'meta_experts.{n_groups}.gate_up_proj.weight'):
        n_groups += 1
    assert n_groups > 0, (
        'InternS2Mobius expects meta-MoE packs '
        '(meta_mlp.{g}.experts.*) in the checkpoint')
    assert num_hidden_layers % n_groups == 0, (
        f'num_hidden_layers={num_hidden_layers} not divisible by '
        f'n_meta_groups={n_groups}')
    layers_per_group = num_hidden_layers // n_groups
    # Host shape only — never Prefix.get() (would .cuda() the full pack)
    expert_num = int(lm_pfx.shape('meta_experts.0.gate_up_proj.weight')[0])
    assert int(lm_pfx.shape('meta_experts_gate.0.weight')[0]) == expert_num
    for g in range(1, n_groups):
        assert int(lm_pfx.shape(f'meta_experts.{g}.gate_up_proj.weight')[0]) == expert_num
        assert int(lm_pfx.shape(f'meta_experts_gate.{g}.weight')[0]) == expert_num
    return n_groups, layers_per_group, expert_num


def build_meta_moe_layer(text_model, pfx, layer_id: int):
    # Stride by n_groups so each meta pack aligns with the repeating
    # layer-type cycle (3×DeltaNet MoE + 1×Attn MoE). Contiguous
    # // layers_per_group would mix Attn and DeltaNet layers onto the
    # same expert pack.
    n_groups = text_model._n_meta_groups
    g = layer_id % n_groups
    is_donor = layer_id < n_groups

    cfg = text_model._moe_cfg.clone()
    cfg.expert_num = text_model._n_experts
    cfg.meta_group = g
    cfg.is_meta_donor = is_donor

    m = MoeBuilder(cfg, text_model._ctx, ep=text_model._ep)
    if is_donor:
        # Prefix ...meta_experts_gate.{g} ; resolver reads .weight
        m.add_gate('gate', text_model._linear(
            text_model._lm_pfx + f'meta_experts_gate.{g}'))
        # Packed pack prefix ...meta_experts.{g} ; read_packed uses
        # gate_up_proj / down_proj + mapping → *.weight
        experts_pfx = text_model._lm_pfx + f'meta_experts.{g}'
        experts = ModuleListBuilder(ModuleListConfig(), text_model._ctx)
        for e in m.range(text_model._n_experts):
            experts[e] = text_model._moe_expert_ffn(
                experts_pfx, e, text_model.cfg.moe_intermediate_size)
        m.experts = experts.build()
    # non-donor: no gate/experts — C++ alias_routed_moe creates them

    m.add_gate('shared_gate', text_model._linear(pfx + 'shared_expert_gate'))
    shared = text_model.ffn(
        pfx + 'shared_expert',
        text_model.cfg.shared_expert_intermediate_size)
    return m.build(), shared


class InternS2MobiusTextModel(Qwen3_5TextModel):
    """Qwen3.5 text model with the Mobius meta-MoE checkpoint layout."""

    _loader_mappings = [
        map_mobius_meta_moe_names,
        map_meta_expert_names,
    ]

    def model(self, pfx):
        # Geometry must be known before layers() builds the MoE modules.
        self._lm_pfx = pfx + 'model.language_model'
        self._n_meta_groups, _, e = infer_meta_geometry(
            self._lm_pfx, int(self.cfg.num_hidden_layers))
        self._n_experts = e
        self._moe_cfg.expert_num = e
        super().model(pfx)

    def layers(self, pfx):
        # Mobius is always meta-MoE — no per-layer-experts fallback.
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self.cfg.num_hidden_layers):
            d = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            if self.cfg.layer_types[i] == 'linear_attention':
                d.linear_attn = self.linear_attn(p + 'linear_attn')
            else:
                d.attention = self.attn(p + 'self_attn')
            d.moe_ffn, d.feed_forward = build_meta_moe_layer(self, p + 'mlp', i)
            d.attention_norm = self.norm(p + 'input_layernorm', zero_centered=True)
            d.ffn_norm = self.norm(p + 'post_attention_layernorm', zero_centered=True)
            layers[i] = d.build()
        return layers.build()


@INPUT_MODELS.register_module(name='interns2_mobius')
class InternS2MobiusModel(Qwen3_5Model):
    """Intern-S2-Mobius aggregate (text + vision), qwen3_5 topology."""

    def __init__(self, cfg, *, resolver, vision_resolver=None,
                 language_model_only: bool = False):
        text_cfg = getattr(cfg, 'text_config', cfg)
        if text_cfg is None:
            raise ValueError(
                'InternS2MobiusModel requires a checkpoint with text_config.')
        self.text_model = InternS2MobiusTextModel(text_cfg, resolver=resolver)

        vision_cfg = getattr(cfg, 'vision_config', None)
        if language_model_only or vision_cfg is None:
            self.vision_model = None
        else:
            self.vision_model = Qwen3_5VisionModel(
                vision_cfg, resolver=vision_resolver or resolver)
