# Copyright (c) OpenMMLab. All rights reserved.
"""Intern-S2-Mobius source model for the TurboMind pipeline.

Mobius is a Qwen3.5-topology checkpoint whose MoE weights are stored as
meta-MoE packs shared across layer groups::

    ...meta_mlp.{g}.experts.{gate_up,down}_proj   (packed, no .weight suffix)
    ...meta_mlp.{g}.gate.weight

The names are translated onto the canonical meta-MoE layout
(``meta_experts.{g}.*`` / ``meta_experts_gate.{g}.weight``) consumed by the
loading helpers below. Each pack is loaded exactly once, as a model-level
``meta_experts`` child of ``ModelWeight``. Per-layer MoE weights carry only
their own ``shared_gate``/``shared_expert`` and are wired to their pack via
``MoeWeight::set_meta_pack``; ``MoeWeight::prepare()`` aliases the pack's
routed gate/expert tensors through that pointer. Groups are strided
(``layer_id % n_groups``) so each pack aligns with the repeating layer-type
cycle (3×DeltaNet + 1×Attn).
"""
from __future__ import annotations

import re

from ..builders import (
    DecoderLayerBuilder,
    DecoderLayerConfig,
    ModuleListBuilder,
    ModuleListConfig,
    MoeBuilder,
    TextModelBuilder,
)
from .base import INPUT_MODELS
from .qwen3_5 import (
    Qwen3_5Model,
    Qwen3_5TextModel,
    Qwen3_5VisionModel,
)
from .utils import make_model_weight_config


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


def infer_meta_groups(lm_pfx, num_hidden_layers: int) -> int:
    n_groups = 0
    while lm_pfx.has(f'meta_experts.{n_groups}.gate_up_proj.weight'):
        n_groups += 1
    assert n_groups > 0, (
        'InternS2Mobius expects meta-MoE packs '
        '(meta_mlp.{g}.experts.*) in the checkpoint')
    assert num_hidden_layers % n_groups == 0, (
        f'num_hidden_layers={num_hidden_layers} not divisible by '
        f'n_meta_groups={n_groups}')
    return n_groups


def build_meta_experts(text_model):
    """Build the shared meta-MoE packs once, as a model-level ModuleList.

    Pack ``g`` holds the routed gate (``meta_experts_gate.{g}``) and the
    EP-sharded packed experts (``meta_experts.{g}``); it is a full routed
    ``MoeWeight`` prepared exactly once by the default recursion. The
    per-group BuiltModules are stashed on the text model so
    ``build_meta_moe_layer`` can wire each layer weight to its pack.
    """
    packs = ModuleListBuilder(ModuleListConfig(), text_model._ctx)
    pack_modules = []
    for g in range(text_model._n_meta_groups):
        cfg = text_model._moe_cfg.clone()
        cfg.expert_num = text_model._n_experts
        m = MoeBuilder(cfg, text_model._ctx, ep=text_model._ep)
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
        pack_modules.append(m.build())
        packs[g] = pack_modules[-1]
    text_model._meta_pack_modules = pack_modules
    return packs.build()


def build_meta_moe_layer(text_model, pfx, layer_id: int):
    """Per-layer MoE weight: the layer's own shared gate/expert, wired to
    its shared meta pack. Routed gate/experts are not loaded here — the
    C++ ``MoeWeight::prepare()`` aliases them through the wired pointer.

    Groups stride by n_groups so each meta pack aligns with the repeating
    layer-type cycle (3×DeltaNet MoE + 1×Attn MoE). Contiguous
    // layers_per_group would mix Attn and DeltaNet layers onto the
    same expert pack.
    """
    cfg = text_model._moe_cfg.clone()
    cfg.expert_num = text_model._n_experts

    m = MoeBuilder(cfg, text_model._ctx, ep=text_model._ep)
    m.add_gate('shared_gate', text_model._linear(pfx + 'shared_expert_gate'))
    moe = m.build()

    pack = text_model._meta_pack_modules[layer_id % text_model._n_meta_groups]
    for i, (moe_h, pack_h) in enumerate(zip(moe.handles, pack.handles)):
        if moe_h is not None and pack_h is not None:
            with text_model._ctx.devices[i]:
                moe_h.set_meta_pack(pack_h)

    shared = text_model.ffn(
        pfx + 'shared_expert',
        text_model.cfg.shared_expert_intermediate_size)
    return moe, shared


class InternS2MobiusTextModel(Qwen3_5TextModel):
    """Qwen3.5 text model with the Mobius meta-MoE checkpoint layout."""

    _loader_mappings = [
        map_mobius_meta_moe_names,
        map_meta_expert_names,
    ]

    def model(self, pfx):
        # Group count must be known before packs/layers are built.
        # self._n_experts/_moe_cfg.expert_num come from cfg.num_experts
        # (base __init__).
        self._lm_pfx = pfx + 'model.language_model'
        self._n_meta_groups = infer_meta_groups(
            self._lm_pfx, int(self.cfg.num_hidden_layers))

        # Same topology as Qwen3_5TextModel.model(), plus the model-level
        # meta_experts packs that layer weights are wired to.
        root_cfg = make_model_weight_config(self.cfg)
        builder = TextModelBuilder(
            root_cfg, self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size)
        builder.add_token_embeds(pfx.get('model.language_model.embed_tokens.weight'))
        builder.norm = self.norm(
            pfx + 'model.language_model.norm',
            zero_centered=True,
        )
        lm_pfx = (pfx + 'model.language_model.embed_tokens'
                  if self.cfg.tie_word_embeddings
                  else pfx + 'lm_head')
        builder.add_lm_head(self._linear(lm_pfx))
        builder.meta_experts = build_meta_experts(self)
        builder.layers = self.layers(pfx + 'model.language_model.layers')
        builder.build()

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
