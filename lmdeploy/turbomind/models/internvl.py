# Copyright (c) OpenMMLab. All rights reserved.
"""InternVL aggregate source model for TurboMind (legacy InternVLChatModel and
HF-style InternVL/InternS1)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import _turbomind as _tm
import torch
from transformers import PretrainedConfig

from lmdeploy.vl.constants import Modality

from ..builders import (
    AttentionBuilder,
    Builder,
    LayerNormBuilder,
    ModuleListBuilder,
    ModuleListConfig,
    NormBuilder,
    SplitSide,
    VisionModelBuilder,
    make_layer_norm_config,
    make_norm_config,
)
from ..builders._base import ParallelGroup
from ..linear import Linear, transform_output_dim
from ..supported_models import SUPPORTED_ARCHS
from ..text_model import TextModel
from ..weight_format import TrivialFormat
from .base import INPUT_MODELS


def _cfg_get(cfg, name: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _to_tm_norm_type(norm_type: str):
    if norm_type == 'layer_norm':
        return _tm.NormType.LAYER_NORM
    if norm_type == 'rms_norm':
        return _tm.NormType.RMS_NORM
    raise ValueError(f'Unsupported InternVit vision norm_type: {norm_type!r}')


def map_interns1_hf_keys(name: str) -> str:
    """Map Intern-S1 HF VLM checkpoint keys to the Qwen3 text loader layout."""
    language_model_prefix = 'model.language_model.'
    if name.startswith(language_model_prefix):
        suffix = name[len(language_model_prefix):]
        return f'language_model.model.{suffix}'
    if name.startswith('lm_head.'):
        return f'language_model.{name}'
    return name


def map_internvl_hf_keys(name: str) -> str:
    """Map InternVL HF vision keys to the InternVit loader layout."""
    if name.startswith('vision_tower.') or name.startswith('multi_modal_projector.'):
        return f'model.{name}'
    return name


def map_legacy_internvl_keys(name: str) -> str:
    """Map legacy InternVLChatModel ViT keys to the InternVit loader layout."""
    if name == 'vision_model.embeddings.class_embedding':
        return 'model.vision_tower.embeddings.cls_token'
    if name == 'vision_model.embeddings.position_embedding':
        return 'model.vision_tower.embeddings.position_embeddings'

    patch_embed = 'vision_model.embeddings.patch_embedding.'
    if name.startswith(patch_embed):
        suffix = name[len(patch_embed):]
        return f'model.vision_tower.embeddings.patch_embeddings.projection.{suffix}'

    block_prefix = 'vision_model.encoder.layers.'
    if name.startswith(block_prefix):
        rest = name[len(block_prefix):]
        if '.' not in rest:
            return name
        layer_id, rest = rest.split('.', 1)
        prefix = f'model.vision_tower.encoder.layer.{layer_id}.'

        if rest == 'ls1':
            return prefix + 'lambda_1'
        if rest == 'ls2':
            return prefix + 'lambda_2'
        if rest.startswith('norm1.'):
            return prefix + 'layernorm_before.' + rest[len('norm1.'):]
        if rest.startswith('norm2.'):
            return prefix + 'layernorm_after.' + rest[len('norm2.'):]
        if rest.startswith('attn.qkv.'):
            return prefix + 'attention.qkv.' + rest[len('attn.qkv.'):]
        if rest.startswith('attn.proj.'):
            return prefix + 'attention.projection_layer.' + rest[len('attn.proj.'):]
        if rest.startswith('attn.q_norm.'):
            return prefix + 'attention.q_norm.' + rest[len('attn.q_norm.'):]
        if rest.startswith('attn.k_norm.'):
            return prefix + 'attention.k_norm.' + rest[len('attn.k_norm.'):]
        if rest.startswith('mlp.'):
            return prefix + rest

    if name.startswith('mlp1.0.'):
        return 'model.multi_modal_projector.layer_norm.' + name[len('mlp1.0.'):]
    if name.startswith('mlp1.1.'):
        return 'model.multi_modal_projector.linear_1.' + name[len('mlp1.1.'):]
    if name.startswith('mlp1.3.'):
        return 'model.multi_modal_projector.linear_2.' + name[len('mlp1.3.'):]

    return name


def _validate_legacy_internvl_chat(cfg):
    cfg = _legacy_namespace(cfg)
    if getattr(cfg, 'ps_version', None) != 'v2':
        raise ValueError(
            f"InternVLChatModel TurboMind native ViT requires ps_version='v2', "
            f"got {getattr(cfg, 'ps_version', None)!r}.")


def _legacy_namespace(cfg):
    return SimpleNamespace(**cfg) if isinstance(cfg, dict) else cfg


def _legacy_square_size(value) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(f'legacy InternVit expects a square size, got {value!r}')
        value = value[0]
    return int(value)


@transform_output_dim
def _split_packed_vision_qkv(tensor: torch.Tensor):
    """Split packed vision QKV layout [Q | K | V] along output dim."""
    if tensor.shape[-1] % 3 != 0:
        raise ValueError(f'packed vision qkv output dim is not divisible by 3: {tuple(tensor.shape)}')
    return tuple(x.contiguous() for x in tensor.chunk(3, dim=-1))


class InternVitVisionModel(TextModel):
    """InternVit weight model rooted at ``ModelRoot.vision_model``."""

    def __init__(self, cfg: PretrainedConfig, *, resolver, parent_cfg: PretrainedConfig):
        super().__init__(cfg, resolver=resolver)

        self._hidden = int(cfg.hidden_size)
        self._heads = int(cfg.num_attention_heads)
        self._depth = int(cfg.num_hidden_layers)
        self._inter = int(cfg.intermediate_size)
        self._channels = int(cfg.num_channels)
        image_h, image_w = cfg.image_size
        patch_h, patch_w = cfg.patch_size
        self._image_h, self._image_w = int(image_h), int(image_w)
        self._patch_h, self._patch_w = int(patch_h), int(patch_w)
        self._norm_eps = float(cfg.layer_norm_eps)
        self._norm_type = _to_tm_norm_type(cfg.norm_type)
        self._use_qk_norm = bool(cfg.use_qk_norm)
        self._head_dim = self._hidden // self._heads
        self._patch_in_dim = self._channels * self._patch_h * self._patch_w
        self._num_patches = (self._image_h // self._patch_h) * (self._image_w // self._patch_w)
        self._downsample_ratio = float(parent_cfg.downsample_ratio)
        self._image_seq_length = int(parent_cfg.image_seq_length)
        self._out_hidden = int(parent_cfg.text_config.hidden_size)
        self._projector_scale = int(round(1.0 / self._downsample_ratio))
        self._projector_in_dim = self._hidden * self._projector_scale * self._projector_scale

    def to_turbomind_multimodal(self, multimodal: list[dict[str, Any]]):
        items = []
        for input_mm in multimodal:
            modality = input_mm.get('modality', Modality.IMAGE)
            if modality not in (Modality.IMAGE, Modality.IMAGE.value, 'image'):
                raise ValueError(f'InternVit TurboMind does not support modality {modality!r}')

            pixel_values = self._tm_tensor(input_mm['pixel_values'])
            token_begin = int(input_mm['offset'])
            token_end = token_begin + int(input_mm['image_tokens'])
            items.append(
                _tm.multimodal.InternVitItem(
                    modality=_tm.multimodal.Modality.IMAGE,
                    data=pixel_values,
                    token_begin=token_begin,
                    token_end=token_end,
                ))

        return _tm.multimodal.InternVitInput(items)

    def model(self, pfx):
        self._build_vision_model(pfx + 'model.vision_tower', pfx + 'model.multi_modal_projector')

    def _build_vision_model(self, vision_pfx, projector_pfx):
        cfg = self._make_root_cfg()
        root = self._restore_dtype(VisionModelBuilder(
            cfg, self._ctx, root_handles=self._root_handles, tp=self._model_tp))

        emb_pfx = vision_pfx + 'embeddings'
        root._add_tensor('cls_token', (emb_pfx + 'cls_token').pop())
        root._add_tensor('position_embeddings', (emb_pfx + 'position_embeddings').pop())
        root._add_linear('patch_embed', self._patch_embed(emb_pfx + 'patch_embeddings.projection'))
        root.blocks = self.vit_blocks(vision_pfx + 'encoder.layer')

        root.projector_norm = self._layer_norm(projector_pfx + 'layer_norm',
                                               dim=self._projector_in_dim,
                                               norm_eps=1e-5)
        root._add_linear('projector_fc1', self._linear(projector_pfx + 'linear_1'), SplitSide.OUTPUT)
        root._add_linear('projector_fc2', self._linear(projector_pfx + 'linear_2'), SplitSide.INPUT)
        root.build()

    def _make_root_cfg(self):
        cfg = _tm.InternVitConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._hidden
        cfg.depth = self._depth
        cfg.patch_in_dim = self._patch_in_dim
        cfg.in_channels = self._channels
        cfg.image_height = self._image_h
        cfg.image_width = self._image_w
        cfg.patch_height = self._patch_h
        cfg.patch_width = self._patch_w
        cfg.num_patches = self._num_patches
        cfg.image_seq_length = self._image_seq_length
        cfg.norm_type = self._norm_type
        return cfg

    def _make_block_cfg(self):
        cfg = _tm.InternVitBlockConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._hidden
        cfg.head_num = self._heads
        cfg.intermediate_size = self._inter
        cfg.norm_eps = self._norm_eps
        return cfg

    def _make_attn_cfg(self):
        cfg = _tm.AttentionConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._hidden
        cfg.head_dim = self._head_dim
        cfg.head_num = self._heads
        cfg.kv_head_num = self._heads
        cfg.window_size = 0
        cfg.causal = False
        return cfg

    def vit_blocks(self, pfx):
        blocks = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self._depth):
            blocks[i] = self.vit_block(p)
        return blocks.build()

    def vit_block(self, pfx):
        b = self._restore_dtype(Builder(self._make_block_cfg(), self._ctx))
        b.tp = self._model_tp

        b.norm1 = self._vision_norm(pfx + 'layernorm_before')
        b.norm2 = self._vision_norm(pfx + 'layernorm_after')
        b.attention = self.vit_attn(pfx + 'attention')
        b._add_linear('mlp_fc1', self._linear(pfx + 'mlp.fc1'), SplitSide.OUTPUT)
        b._add_linear('mlp_fc2', self._linear(pfx + 'mlp.fc2'), SplitSide.INPUT)
        b._add_tensor('lambda_1', (pfx + 'lambda_1').pop())
        b._add_tensor('lambda_2', (pfx + 'lambda_2').pop())
        return b.build()

    def vit_attn(self, pfx):
        q = self._linear(pfx + 'q_proj')
        k = self._linear(pfx + 'k_proj')
        v = self._linear(pfx + 'v_proj')
        o = self._linear(pfx + 'projection_layer')

        cfg = self._make_attn_cfg()
        attn_tp = self._model_tp if self._heads % self._model_tp.size == 0 else ParallelGroup(1, None)
        m = self._restore_dtype(AttentionBuilder(cfg, self._ctx, tp=attn_tp))
        m.add_qkv_proj(q, k, v)
        m.add_o_proj(o)
        if self._use_qk_norm and (pfx + 'q_norm').has('weight') and (pfx + 'k_norm').has('weight'):
            m.q_norm = self._rms_norm(pfx + 'q_norm', tp=attn_tp)
            m.k_norm = self._rms_norm(pfx + 'k_norm', tp=attn_tp)
        return m.build()

    def _tm_tensor(self, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'InternVit multimodal data should be a torch.Tensor, got {type(tensor).__name__}')
        return _tm.from_dlpack(tensor.contiguous())

    def _restore_dtype(self, builder):
        builder.config.data_type = self._resolver.data_type
        return builder

    def _patch_embed(self, pfx):
        weight = pfx.pop('weight')
        weight = weight.reshape(weight.shape[0], -1).t().contiguous()
        tensors = {'weight': weight}
        if pfx.has('bias'):
            tensors['bias'] = pfx.pop('bias')
        return Linear(tensors=tensors, weight_format=TrivialFormat())

    def _vision_norm(self, pfx):
        if self._norm_type == _tm.NormType.LAYER_NORM:
            return self._layer_norm(pfx, dim=self._hidden, norm_eps=self._norm_eps)
        elif self._norm_type == _tm.NormType.RMS_NORM:
            return self._rms_norm(pfx, tp=ParallelGroup(1, None))
        else:
            raise ValueError(f'Unsupported InternVit vision norm_type: {self._norm_type!r}')

    def _rms_norm(self, pfx, tp: ParallelGroup):
        weight = pfx.pop('weight')
        tp_size = tp.size
        dim = weight.shape[-1]
        if tp_size > 1:
            assert dim % tp_size == 0, (
                f'{pfx}.weight dim={dim} is not divisible by tp={tp_size}')
            dim //= tp_size
        cfg = make_norm_config(dim=dim, norm_eps=self._norm_eps)
        cfg.data_type = self._resolver.data_type
        m = self._restore_dtype(NormBuilder(cfg, self._ctx))
        if tp_size > 1:
            m.tp = tp
            m._add_tensor('weight', weight, SplitSide.OUTPUT)
        else:
            m.set_weight(weight)
        return m.build()

    def _layer_norm(self, pfx, *, dim: int, norm_eps: float):
        weight = pfx.pop('weight')
        bias = pfx.pop('bias') if pfx.has('bias') else None
        cfg = make_layer_norm_config(dim=dim, data_type=self._resolver.data_type, norm_eps=norm_eps)
        m = self._restore_dtype(LayerNormBuilder(cfg, self._ctx))
        m.set_weight(weight, bias=bias)
        return m.build()


class LegacyInternVitVisionModel(InternVitVisionModel):
    """Legacy InternVLChatModel ViT adapter for the canonical InternVit layout.

    Legacy InternVL stores attention as a single ``attn.qkv`` linear. Other
    weights are normalized through ``map_legacy_internvl_keys``.
    """

    def __init__(self, cfg: PretrainedConfig, *, resolver, parent_cfg: PretrainedConfig):
        cfg = _legacy_namespace(cfg)
        parent_cfg = _legacy_namespace(parent_cfg)
        llm_cfg = _legacy_namespace(parent_cfg.llm_config)
        image_size = _legacy_square_size(cfg.image_size)
        patch_size = _legacy_square_size(cfg.patch_size)
        downsample_ratio = float(parent_cfg.downsample_ratio)
        image_seq_length = int((image_size // patch_size)**2 * (downsample_ratio**2))

        normalized_cfg = SimpleNamespace(
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            num_hidden_layers=cfg.num_hidden_layers,
            intermediate_size=cfg.intermediate_size,
            num_channels=cfg.num_channels,
            image_size=(image_size, image_size),
            patch_size=(patch_size, patch_size),
            layer_norm_eps=cfg.layer_norm_eps,
            norm_type=cfg.norm_type,
            use_qk_norm=cfg.qk_normalization,
        )
        normalized_parent_cfg = SimpleNamespace(
            downsample_ratio=downsample_ratio,
            image_seq_length=image_seq_length,
            text_config=llm_cfg,
        )
        super().__init__(normalized_cfg, resolver=resolver, parent_cfg=normalized_parent_cfg)

    def vit_attn(self, pfx):
        q, k, v = _split_packed_vision_qkv(self._linear(pfx + 'qkv'))
        o = self._linear(pfx + 'projection_layer')

        cfg = self._make_attn_cfg()
        attn_tp = self._model_tp if self._heads % self._model_tp.size == 0 else ParallelGroup(1, None)
        m = self._restore_dtype(AttentionBuilder(cfg, self._ctx, tp=attn_tp))
        m.add_qkv_proj(q, k, v)
        m.add_o_proj(o)
        if self._use_qk_norm and (pfx + 'q_norm').has('weight') and (pfx + 'k_norm').has('weight'):
            m.q_norm = self._rms_norm(pfx + 'q_norm', tp=attn_tp)
            m.k_norm = self._rms_norm(pfx + 'k_norm', tp=attn_tp)
        return m.build()


@INPUT_MODELS.register_module(name='internvl')
class InternVLModel:
    """Aggregate source model for InternVL checkpoints with any registered text
    model."""

    _vision = True

    def __init__(self, cfg: PretrainedConfig, *, resolver, vision_resolver=None, disable_vision_encoder: bool = False):
        llm_cfg = _cfg_get(cfg, 'llm_config')
        if llm_cfg is None:
            llm_cfg = _cfg_get(cfg, 'text_config')
        if llm_cfg is None:
            raise ValueError(
                'InternVL TurboMind requires llm_config or text_config.')

        archs = _cfg_get(llm_cfg, 'architectures')
        if not archs:
            raise ValueError(
                'InternVL TurboMind requires architectures on llm_config or text_config.')

        text_model_arch = archs[0]
        text_model_registered_name = SUPPORTED_ARCHS.get(text_model_arch)
        if text_model_registered_name is None:
            raise ValueError(
                f'InternVL text model architecture {text_model_arch!r} '
                'is not supported by TurboMind.')

        text_model_cls = INPUT_MODELS.get(text_model_registered_name)
        self.text_model = text_model_cls(llm_cfg, resolver=resolver)
        archs = _cfg_get(cfg, 'architectures') or []
        self._checkpoint_mappings = []
        arch = archs[0] if archs else None
        if arch == 'InternS1ForConditionalGeneration':
            self._checkpoint_mappings.append(map_interns1_hf_keys)
        elif arch == 'InternVLForConditionalGeneration':
            self._checkpoint_mappings.append(map_internvl_hf_keys)
        elif arch == 'InternVLChatModel':
            self._checkpoint_mappings.append(map_legacy_internvl_keys)
        vision_cfg = cfg.vision_config if hasattr(cfg, 'vision_config') else None
        if not disable_vision_encoder and vision_cfg is not None:
            if arch == 'InternVLChatModel':
                _validate_legacy_internvl_chat(cfg)
                self.vision_model = LegacyInternVitVisionModel(vision_cfg,
                                                               resolver=vision_resolver or resolver,
                                                               parent_cfg=cfg)
            elif arch in ('InternS1ForConditionalGeneration', 'InternVLForConditionalGeneration'):
                self.vision_model = InternVitVisionModel(vision_cfg,
                                                         resolver=vision_resolver or resolver,
                                                         parent_cfg=cfg)
            else:
                raise ValueError(f'InternVL TurboMind vision architecture {arch!r} is not supported.')
        else:
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
        if self.vision_model is not None:
            self.vision_model.bind_runtime(
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
        return self._checkpoint_mappings + list(getattr(type(self.text_model), '_loader_mappings', []))

    def to_turbomind_multimodal(self, multimodal: list[dict[str, Any]]):
        if self.vision_model is None:
            raise ValueError('InternVL TurboMind vision encoder is not available.')
        return self.vision_model.to_turbomind_multimodal(multimodal)

    def model(self, pfx):
        self.text_model.model(pfx + 'language_model')
        if self.vision_model is not None:
            self.vision_model.model(pfx)
