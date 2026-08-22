# Copyright (c) OpenMMLab. All rights reserved.
"""Muse-Glimmer source model for the TurboMind pipeline."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import _turbomind as _tm
import torch

from ..builders import (
    AttentionBuilder,
    Builder,
    DecoderLayerBuilder,
    DecoderLayerConfig,
    FfnBuilder,
    ModuleListBuilder,
    ModuleListConfig,
    NormBuilder,
    SplitSide,
    TextModelBuilder,
    VisionModelBuilder,
    _act_type_id,
    make_norm_config,
)
from ..builders._base import ParallelGroup
from ..text_model import TextModel
from ..vision_model import VisionModel
from .base import INPUT_MODELS
from .qwen3_5 import (
    Qwen3_5VisionModel,
    _padded_vit_head_dim,
)
from .utils import (
    make_attention_config,
    make_ffn_config,
    make_model_weight_config,
    reorder_rotary_emb,
)
from .vision_utils import pad_attn_head_dim


def _cfg_get(cfg, name, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _config_namespace(cfg):
    """Convert generic PretrainedConfig nested dictionaries to attributes."""
    if not isinstance(cfg, dict):
        return cfg
    if any(not isinstance(key, str) for key in cfg):
        return cfg
    return SimpleNamespace(**{
        key: _config_namespace(value) if isinstance(value, dict) else value
        for key, value in cfg.items()
    })


class MuseGlimmerTextModel(TextModel):
    """Dense Muse-Glimmer language model."""

    def __init__(self, cfg, *, resolver):
        super().__init__(cfg, resolver=resolver)
        self._attn_cfg = make_attention_config(cfg, head_dim=cfg.head_dim)
        self._attn_cfg.output_gate = True
        self._ffn_cfg = make_ffn_config(cfg, act_type=_act_type_id('silu'))

    def _constant_norm(self, reference, *, dim, value, norm_eps, zero_centered=False):
        weight = torch.full((dim,), value, dtype=reference.dtype, device=reference.device)
        cfg = make_norm_config(dim=dim, norm_eps=norm_eps, zero_centered=zero_centered)
        norm = NormBuilder(cfg, self._ctx)
        norm.set_weight(weight)
        return norm.build()

    def model(self, pfx):
        root_cfg = make_model_weight_config(self.cfg)
        root_cfg.logit_scale = float(self.cfg.output_multiplier)
        root_cfg.logit_softcap = float(self.cfg.final_logit_softcapping)
        builder = TextModelBuilder(
            root_cfg,
            self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp,
            vocab_size=self.cfg.vocab_size,
        )

        embeds = pfx.pop('model.language_model.embed_tokens.weight')
        builder.add_token_embeds(embeds)
        builder.embedding_norm = self._constant_norm(
            embeds,
            dim=self.cfg.hidden_size,
            value=1.0,
            norm_eps=self.cfg.rms_norm_eps,
        )
        builder.norm = self.norm(pfx + 'model.language_model.norm')
        builder.add_lm_head(self._linear(pfx + 'lm_head'))
        builder.layers = self.layers(pfx + 'model.language_model.layers')
        builder.build()

    def attn(self, pfx, layer_idx):
        q, k, v, o = [self._linear(pfx + f'{name}_proj') for name in 'qkvo']
        gate = self._linear(pfx + 'gate_proj')

        cfg = self._attn_cfg.clone()
        use_rope = bool(self.cfg.layer_rope_theta[layer_idx])
        if self.cfg.layer_types[layer_idx] == 'sliding_attention':
            cfg.window_size = self.cfg.sliding_window
        if use_rope:
            cfg.rope.base = float(self.cfg.layer_rope_theta[layer_idx])

            def reorder(x):
                return reorder_rotary_emb(x, cfg.head_dim, cfg.rope.dim, resolver=self._resolver)

            q, k = reorder(q), reorder(k)
        else:
            cfg.rope.type = 0

        attn = AttentionBuilder(cfg, self._ctx, tp=self._attn_tp)
        attn.add_qkv_proj(q, k, v, gate=gate)
        attn.add_o_proj(o)

        reference = q.tensors['weight']
        attn.q_norm = self._constant_norm(
            reference,
            dim=cfg.head_dim,
            value=float(self.cfg.qk_scale_factor),
            norm_eps=self.cfg.rms_norm_eps,
        )
        attn.k_norm = self._constant_norm(
            reference,
            dim=cfg.head_dim,
            value=1.0,
            norm_eps=self.cfg.rms_norm_eps,
        )
        return attn.build()

    def ffn(self, pfx):
        w1 = self._linear(pfx + 'gate_proj')
        w3 = self._linear(pfx + 'up_proj')
        w2 = self._linear(pfx + 'down_proj')
        ffn = FfnBuilder(self._ffn_cfg.clone(), self._ctx, tp=self._mlp_tp)
        ffn.add_ffn(w1, w2, w3)
        return ffn.build()

    def layers(self, pfx):
        layers = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, layer_pfx in pfx.slices(0, self.cfg.num_hidden_layers):
            layer = DecoderLayerBuilder(DecoderLayerConfig(), self._ctx)
            layer.attention = self.attn(layer_pfx + 'self_attn', i)
            layer.feed_forward = self.ffn(layer_pfx + 'mlp')
            layer.attention_norm = self.norm(
                layer_pfx + 'input_layernorm', zero_centered=True)
            layer.post_attention_norm = self.norm(
                layer_pfx + 'post_attention_layernorm',
                zero_centered=True,
                norm_eps=self.cfg.post_norm_eps,
            )
            layer.ffn_norm = self.norm(
                layer_pfx + 'pre_feedforward_layernorm', zero_centered=True)
            layer.post_ffn_norm = self.norm(
                layer_pfx + 'post_feedforward_layernorm',
                zero_centered=True,
                norm_eps=self.cfg.post_norm_eps,
            )
            layers[i] = layer.build()
        return layers.build()


class MuseGlimmerVisionModel(Qwen3_5VisionModel):
    """Muse-Glimmer ViT using the unified native Qwen-ViT runtime."""

    def __init__(self, vision_cfg, model_cfg, *, resolver):
        VisionModel.__init__(self, vision_cfg, resolver=resolver)
        self.model_cfg = model_cfg
        self._vis_depth = int(vision_cfg.num_hidden_layers)
        self._vis_hidden = int(vision_cfg.hidden_size)
        self._vis_inter = int(vision_cfg.intermediate_size)
        self._vis_heads = int(vision_cfg.num_attention_heads)
        self._vis_out_hidden = int(model_cfg.text_config.hidden_size)
        self._vis_in_chans = 3
        self._vis_patch = int(vision_cfg.patch_size)
        self._vis_temporal = int(vision_cfg.patch_temporal)
        self._vis_pos_n = int(vision_cfg.pos_emb_height * vision_cfg.pos_emb_width)
        self._vis_spatial_merge = 1
        self._vis_output_merge = int(vision_cfg.merge_size)
        self._vis_norm_eps = float(vision_cfg.layer_norm_eps)
        self._patch_in_dim = self._vis_in_chans * self._vis_temporal * self._vis_patch**2

    def model(self, pfx):
        model_pfx = pfx + 'model'
        pfx = model_pfx + 'vision_tower'
        cfg = self._make_vision_root_cfg()
        root = self._restore_dtype(VisionModelBuilder(
            cfg, self._ctx, root_handles=self._root_handles, tp=self._model_tp))

        root._add_tensor('pos_embed', (pfx + 'patch_embedder.position_embedding_table').pop('weight'))
        root._add_linear('patch_embed', self._patch_embed(pfx + 'patch_embedder.patch_embedding'))
        root.pre_norm = self._layer_norm(pfx + 'ln_pre', dim=self._vis_hidden)
        root.blocks = self.vit_blocks(pfx + 'layers')
        root.merger_norm = self._layer_norm(pfx + 'ln_post', dim=self._vis_hidden)
        root._add_linear('merger_fc1', self._linear(model_pfx + 'vision_adapter.fc1'), SplitSide.OUTPUT)
        root._add_linear('merger_fc2', self._linear(model_pfx + 'vision_adapter.fc2'), SplitSide.INPUT)
        projection = self._linear(model_pfx + 'vision_projection')
        root._add_linear('merger_fc3', projection)

        projection_weight = projection.tensors['weight']
        out_norm_cfg = make_norm_config(
            dim=self._vis_out_hidden,
            norm_eps=self.model_cfg.text_config.rms_norm_eps,
        )
        out_norm = NormBuilder(out_norm_cfg, self._ctx)
        out_norm.set_weight(torch.ones(
            self._vis_out_hidden, dtype=projection_weight.dtype, device=projection_weight.device))
        root.output_norm = out_norm.build()
        root.build()

    def _make_vision_root_cfg(self):
        cfg = _tm.QwenVitConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._vis_hidden
        cfg.out_hidden_dim = self._vis_out_hidden
        cfg.depth = self._vis_depth
        cfg.head_num = self._vis_heads
        cfg.intermediate_size = self._vis_inter
        cfg.patch_in_dim = self._patch_in_dim
        cfg.in_channels = self._vis_in_chans
        cfg.patch_size = self._vis_patch
        cfg.temporal_patch_size = self._vis_temporal
        cfg.num_position_embeddings = self._vis_pos_n
        cfg.spatial_merge_size = self._vis_spatial_merge
        cfg.output_spatial_merge_size = self._vis_output_merge
        cfg.window_size = int(self.cfg.pos_emb_height * self._vis_patch)
        cfg.use_window_attention = True
        cfg.fullatt_block_indexes = [
            i for i, layer_type in enumerate(self.cfg.layer_types)
            if layer_type == 'full_attention'
        ]
        cfg.zero_padded_pos_embed = True
        cfg.rope_axes_w_first = True
        cfg.rope_position_offset = 1
        cfg.rope_theta = float(_cfg_get(self.cfg.rope_parameters, 'rope_theta', 10000.0))
        cfg.pixel_shuffle = True
        cfg.merger_double_gelu = True
        cfg.norm_eps = self._vis_norm_eps
        return cfg

    def vit_block(self, pfx):
        cfg = _tm.QwenVitBlockConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._vis_hidden
        cfg.head_num = self._vis_heads
        cfg.intermediate_size = self._vis_inter
        cfg.norm_eps = self._vis_norm_eps
        block = self._restore_dtype(Builder(cfg, self._ctx))
        block.tp = self._model_tp
        block.norm1 = self._layer_norm(pfx + 'norm1', dim=self._vis_hidden)
        block.norm2 = self._layer_norm(pfx + 'norm2', dim=self._vis_hidden)
        block.attention = self.vit_attn(pfx + 'attn')
        block._add_linear('mlp_fc1', self._linear(pfx + 'mlp.fc1'), SplitSide.OUTPUT)
        block._add_linear('mlp_fc2', self._linear(pfx + 'mlp.fc2'), SplitSide.INPUT)
        return block.build()

    def vit_attn(self, pfx):
        real_hd = self._vis_hidden // self._vis_heads
        padded_hd = _padded_vit_head_dim(real_hd)
        cfg = _tm.AttentionConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._vis_hidden
        cfg.head_dim = padded_hd
        cfg.head_num = self._vis_heads
        cfg.kv_head_num = self._vis_heads
        cfg.causal = False
        cfg.softmax_scale = 1.0 / math.sqrt(real_hd) if padded_hd != real_hd else 0.0

        q, k, v = [self._linear(pfx + f'{name}_proj') for name in 'qkv']
        q = reorder_rotary_emb(q, real_hd, real_hd, resolver=self._resolver)
        k = reorder_rotary_emb(k, real_hd, real_hd, resolver=self._resolver)
        proj = self._linear(pfx + 'proj')
        q, k, v, proj = pad_attn_head_dim(
            q, k, v, proj,
            num_heads=self._vis_heads,
            src_head_dim=real_hd,
            dst_head_dim=padded_hd,
        )
        attn_tp = self._model_tp if self._vis_heads % self._model_tp.size == 0 else ParallelGroup(1, None)
        attn = self._restore_dtype(AttentionBuilder(cfg, self._ctx, tp=attn_tp))
        attn.add_qkv_proj(q, k, v)
        attn.add_o_proj(proj)
        return attn.build()


@INPUT_MODELS.register_module(name='muse_glimmer')
class MuseGlimmerModel:
    """Aggregate Muse-Glimmer checkpoint model."""

    _vision = True

    def __init__(self, cfg, *, resolver, vision_resolver=None, language_model_only=False):
        text_cfg = _config_namespace(_cfg_get(cfg, 'text_config'))
        vision_cfg = _config_namespace(_cfg_get(cfg, 'vision_config'))
        model_cfg = _config_namespace(cfg.to_dict() if hasattr(cfg, 'to_dict') else cfg)
        model_cfg.text_config = text_cfg
        model_cfg.vision_config = vision_cfg
        self.text_model = MuseGlimmerTextModel(text_cfg, resolver=resolver)
        self.vision_model = None if language_model_only else MuseGlimmerVisionModel(
            vision_cfg, model_cfg, resolver=vision_resolver or resolver)

    def bind_runtime(self, *, ctx, root_handles, attn_tp, mlp_tp, ep, model_tp):
        self.text_model.bind_runtime(
            ctx=ctx,
            root_handles=root_handles,
            attn_tp=attn_tp,
            mlp_tp=mlp_tp,
            ep=ep,
            model_tp=model_tp,
        )
        if self.vision_model is not None:
            self.vision_model.bind_runtime(ctx=ctx, root_handles=root_handles, model_tp=model_tp)

    @property
    def _vocab_size(self):
        return self.text_model.cfg.vocab_size

    @property
    def _loader_mappings(self):
        return []

    def to_turbomind_multimodal(self, multimodal: list[dict[str, Any]]):
        if self.vision_model is None:
            raise ValueError('Muse-Glimmer TurboMind vision encoder is not available.')
        return self.vision_model.to_turbomind_multimodal(multimodal)

    def model(self, pfx):
        self.text_model.model(pfx)
        if self.vision_model is not None:
            self.vision_model.model(pfx)
