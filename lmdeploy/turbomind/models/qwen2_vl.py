# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen2-VL / Qwen2.5-VL aggregate source model for TurboMind."""
from __future__ import annotations

import math
from typing import Any

import _turbomind as _tm
import torch

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
from ..linear import Linear, transform_input_dim, transform_output_dim
from ..text_model import TextModel
from ..weight_format import TrivialFormat
from .base import INPUT_MODELS
from .qwen2 import Qwen2Model
from .qwen3_5 import (
    _assert_trivial,
    _pad_head_dim_in,
    _pad_head_dim_out,
    _split_packed_vision_qkv,
)
from .utils import reorder_rotary_emb

_VIT_HEAD_DIM_PADDED = {
    64: 64,
    80: 128,
}


@transform_output_dim
def _pad_output_dim(t: torch.Tensor, *, dst_dim: int) -> torch.Tensor:
    pad = dst_dim - t.shape[-1]
    if pad <= 0:
        return t
    return torch.cat([t, t.new_zeros(t.shape[:-1] + (pad, ))], dim=-1).contiguous()


@transform_input_dim
def _pad_input_dim(t: torch.Tensor, *, dst_dim: int) -> torch.Tensor:
    pad = dst_dim - t.shape[0]
    if pad <= 0:
        return t
    return torch.cat([t, t.new_zeros((pad, ) + t.shape[1:])], dim=0).contiguous()


def _padded_vit_head_dim(real_hd: int) -> int:
    if real_hd not in _VIT_HEAD_DIM_PADDED:
        raise NotImplementedError(
            f'Qwen2 ViT head_dim={real_hd} is not supported; '
            f'known: {sorted(_VIT_HEAD_DIM_PADDED)}')
    return _VIT_HEAD_DIM_PADDED[real_hd]


def _to_tm_norm_type(norm_type: str):
    if norm_type == 'layer_norm':
        return _tm.NormType.LAYER_NORM
    if norm_type == 'rms_norm':
        return _tm.NormType.RMS_NORM
    raise ValueError(f'Unsupported Qwen2 ViT norm_type: {norm_type!r}')


class _BaseQwen2VisionModel(TextModel):
    """Common Qwen2-VL vision sub-tree rooted at ``ModelRoot.vision_model``."""

    _gated_mlp = False
    _norm_type = ''
    _use_window_attention = False

    def __init__(self, cfg, *, resolver):
        super().__init__(cfg, resolver=resolver)

        self._vis_hidden = self._vision_hidden_size(cfg)
        self._vis_out_hidden = int(getattr(cfg, 'out_hidden_size', getattr(cfg, 'hidden_size', self._vis_hidden)))
        self._vis_inter = self._vision_intermediate_size(cfg)
        self._vis_depth = int(cfg.depth)
        self._vis_heads = int(cfg.num_heads)
        self._vis_in_chans = int(getattr(cfg, 'in_channels', getattr(cfg, 'in_chans', 3)))
        self._vis_patch = int(cfg.patch_size)
        self._vis_temporal = int(cfg.temporal_patch_size)
        self._vis_spatial_merge = int(cfg.spatial_merge_size)
        self._vis_norm_eps = 1e-6
        self._window_size = self._vision_window_size(cfg)
        self._fullatt_block_indexes = self._vision_fullatt_block_indexes(cfg)

        self._patch_in_dim = (self._vis_in_chans
                              * self._vis_temporal
                              * self._vis_patch
                              * self._vis_patch)

    def _vision_hidden_size(self, cfg):
        return int(getattr(cfg, 'hidden_size', getattr(cfg, 'embed_dim', 0)))

    def _vision_intermediate_size(self, cfg):
        raise NotImplementedError

    def _vision_window_size(self, cfg):
        return 0

    def _vision_fullatt_block_indexes(self, cfg):
        return []

    def _torch_dtype(self):
        if self._resolver.data_type == _tm.DataType.TYPE_FP16:
            return torch.float16
        if self._resolver.data_type == _tm.DataType.TYPE_BF16:
            return torch.bfloat16
        if self._resolver.data_type == _tm.DataType.TYPE_FP32:
            return torch.float32
        return None

    def _tm_tensor(self, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'Qwen2 ViT multimodal data should be a torch.Tensor, got {type(tensor).__name__}')
        target_dtype = self._torch_dtype()
        if target_dtype is not None and tensor.is_floating_point() and tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)
        return _tm.from_dlpack(tensor.contiguous())

    @staticmethod
    def _grid_thw(grid_thw) -> tuple[int, int, int]:
        if isinstance(grid_thw, torch.Tensor):
            values = grid_thw.flatten().tolist()
        else:
            values = list(grid_thw)
        if len(values) != 3:
            raise ValueError(f'Qwen2 ViT grid_thw should contain 3 values, got {values!r}')
        return int(values[0]), int(values[1]), int(values[2])

    @staticmethod
    def _token_range(input_mm: dict[str, Any]) -> tuple[int, int]:
        offset = input_mm['offset']
        if isinstance(offset, torch.Tensor):
            values = offset.flatten().tolist()
        elif isinstance(offset, (list, tuple)):
            values = list(offset)
        else:
            values = [offset]
        if len(values) == 2:
            return int(values[0]), int(values[1])
        if len(values) != 1:
            raise ValueError(f'Qwen2 ViT offset should contain 1 or 2 values, got {values!r}')
        tokens = input_mm['image_tokens']
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.flatten()[0].item()
        return int(values[0]), int(values[0]) + int(tokens)

    def to_turbomind_multimodal(self, multimodal: list[dict[str, Any]]):
        items = []
        for input_mm in multimodal:
            modality = input_mm.get('modality', Modality.IMAGE)
            if modality not in (Modality.IMAGE, Modality.IMAGE.value, 'image'):
                raise ValueError(f'Qwen2 TurboMind native vision only supports image inputs, got {modality!r}')

            data = self._tm_tensor(input_mm['pixel_values'])
            grid_thw = self._grid_thw(input_mm['image_grid_thw'])
            token_begin, token_end = self._token_range(input_mm)
            items.append(
                _tm.multimodal.QwenVitItem(
                    modality=_tm.multimodal.Modality.IMAGE,
                    data=data,
                    token_begin=token_begin,
                    token_end=token_end,
                    grid_thw=grid_thw,
                ))

        return _tm.multimodal.QwenVitInput(items)

    def model(self, pfx):
        self._build_vision_model(pfx + 'visual')

    def _restore_dtype(self, builder):
        builder.config.data_type = self._resolver.data_type
        return builder

    def _make_vision_root_cfg(self):
        cfg = _tm.QwenVitConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._vis_hidden
        cfg.out_hidden_dim = self._vis_out_hidden
        cfg.depth = self._vis_depth
        cfg.head_num = self._vis_heads
        cfg.intermediate_size = self._padded_inter_size()
        cfg.patch_in_dim = self._patch_in_dim
        cfg.in_channels = self._vis_in_chans
        cfg.patch_size = self._vis_patch
        cfg.temporal_patch_size = self._vis_temporal
        cfg.spatial_merge_size = self._vis_spatial_merge
        cfg.window_size = self._window_size
        cfg.gated_mlp = self._gated_mlp
        cfg.use_window_attention = self._use_window_attention
        cfg.norm_type = _to_tm_norm_type(self._norm_type)
        cfg.fullatt_block_indexes = self._fullatt_block_indexes
        cfg.norm_eps = self._vis_norm_eps
        return cfg

    def _build_vision_model(self, pfx):
        cfg = self._make_vision_root_cfg()
        root = self._restore_dtype(VisionModelBuilder(
            cfg, self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp))

        root._add_linear('patch_embed', self._patch_embed(pfx + 'patch_embed.proj'))
        root.blocks = self.vit_blocks(pfx + 'blocks')
        root.merger_norm = self._vision_norm(pfx + 'merger.ln_q', dim=self._vis_hidden)
        root._add_linear('merger_fc1', self._linear(pfx + 'merger.mlp.0'), SplitSide.OUTPUT)
        root._add_linear('merger_fc2', self._linear(pfx + 'merger.mlp.2'), SplitSide.INPUT)
        root.build()

    def _patch_embed(self, pfx):
        weight = pfx.pop('weight')
        weight = weight.reshape(weight.shape[0], -1).t().contiguous()
        return Linear(tensors={'weight': weight}, weight_format=TrivialFormat())

    def vit_blocks(self, pfx):
        blocks = ModuleListBuilder(ModuleListConfig(), self._ctx)
        for i, p in pfx.slices(0, self._vis_depth):
            blocks[i] = self.vit_block(p)
        return blocks.build()

    def vit_block(self, pfx):
        inter_size = self._padded_inter_size()
        cfg = _tm.QwenVitBlockConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._vis_hidden
        cfg.head_num = self._vis_heads
        cfg.intermediate_size = inter_size
        cfg.norm_eps = self._vis_norm_eps

        b = self._restore_dtype(Builder(cfg, self._ctx))
        b.tp = self._model_tp
        b.norm1 = self._vision_norm(pfx + 'norm1', dim=self._vis_hidden)
        b.norm2 = self._vision_norm(pfx + 'norm2', dim=self._vis_hidden)
        b.attention = self.vit_attn(pfx + 'attn')
        self._add_mlp(b, pfx, inter_size=inter_size)
        return b.build()

    def _add_mlp(self, builder, pfx, *, inter_size: int):
        raise NotImplementedError

    def _padded_inter_size(self):
        # Bias/activation kernels vectorize half/bf16 in 8-element chunks.
        # Pad the global intermediate so each TP shard has aligned width.
        align = max(1, self._model_tp.size) * 8
        return ((self._vis_inter + align - 1) // align) * align

    def _pad_plain_mlp(self, fc1: Linear, fc2: Linear, *, inter_size: int):
        if inter_size == self._vis_inter:
            return fc1, fc2
        return _pad_output_dim(fc1, dst_dim=inter_size), _pad_input_dim(fc2, dst_dim=inter_size)

    def _pad_mlp(self, gate: Linear, up: Linear, down: Linear, *, inter_size: int):
        if inter_size == self._vis_inter:
            return gate, up, down
        return (_pad_output_dim(gate, dst_dim=inter_size),
                _pad_output_dim(up, dst_dim=inter_size),
                _pad_input_dim(down, dst_dim=inter_size))

    def _make_vision_attn_cfg(self):
        real_hd = self._vis_hidden // self._vis_heads
        padded_hd = _padded_vit_head_dim(real_hd)
        cfg = _tm.AttentionConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._vis_hidden
        cfg.head_dim = padded_hd
        cfg.head_num = self._vis_heads
        cfg.kv_head_num = self._vis_heads
        cfg.window_size = 0
        cfg.causal = False
        cfg.softmax_scale = (1.0 / math.sqrt(real_hd) if padded_hd != real_hd else 0.0)
        return cfg

    def vit_attn(self, pfx):
        cfg = self._make_vision_attn_cfg()
        real_hd = self._vis_hidden // self._vis_heads
        padded_hd = cfg.head_dim
        H = cfg.head_num

        q, k, v = _split_packed_vision_qkv(self._linear(pfx + 'qkv'))
        q = reorder_rotary_emb(q, real_hd, real_hd, resolver=self._resolver)
        k = reorder_rotary_emb(k, real_hd, real_hd, resolver=self._resolver)
        proj = self._linear(pfx + 'proj')

        if padded_hd != real_hd:
            for ln, name in [(q, 'q'), (k, 'k'), (v, 'v'), (proj, 'proj')]:
                _assert_trivial(ln, name)
            pad_kwargs = dict(num_heads=H, src_hd=real_hd, dst_hd=padded_hd)
            q = _pad_head_dim_out(q, **pad_kwargs)
            k = _pad_head_dim_out(k, **pad_kwargs)
            v = _pad_head_dim_out(v, **pad_kwargs)
            proj = _pad_head_dim_in(proj, **pad_kwargs)

        attn_tp = self._model_tp if self._vis_heads % self._model_tp.size == 0 else ParallelGroup(1, None)
        m = self._restore_dtype(AttentionBuilder(cfg, self._ctx, tp=attn_tp))
        m.add_qkv_proj(q, k, v)
        m.add_o_proj(proj)
        return m.build()

    def _vision_norm(self, pfx, *, dim: int):
        raise NotImplementedError

    def _layer_norm(self, pfx, *, dim: int):
        weight = pfx.pop('weight')
        bias = pfx.pop('bias') if pfx.has('bias') else None
        cfg = make_layer_norm_config(dim=dim,
                                     data_type=self._resolver.data_type,
                                     norm_eps=self._vis_norm_eps)
        m = self._restore_dtype(LayerNormBuilder(cfg, self._ctx))
        m.set_weight(weight, bias=bias)
        return m.build()

    def _rms_norm(self, pfx, *, dim: int):
        weight = pfx.pop('weight')
        cfg = make_norm_config(dim=dim, norm_eps=self._vis_norm_eps)
        cfg.data_type = self._resolver.data_type
        m = self._restore_dtype(NormBuilder(cfg, self._ctx))
        m.set_weight(weight)
        return m.build()


class Qwen2VisionModel(_BaseQwen2VisionModel):
    """Qwen2-VL vision tower."""

    _norm_type = 'layer_norm'

    def _vision_hidden_size(self, cfg):
        return int(cfg.embed_dim)

    def _vision_intermediate_size(self, cfg):
        return int(self._vis_hidden * cfg.mlp_ratio)

    def _vision_norm(self, pfx, *, dim: int):
        return self._layer_norm(pfx, dim=dim)

    def _add_mlp(self, builder, pfx, *, inter_size: int):
        fc1, fc2 = self._pad_plain_mlp(
            self._linear(pfx + 'mlp.fc1'),
            self._linear(pfx + 'mlp.fc2'),
            inter_size=inter_size,
        )
        builder._add_linear('mlp_fc1', fc1, SplitSide.OUTPUT)
        builder._add_linear('mlp_fc2', fc2, SplitSide.INPUT)


class Qwen2_5VisionModel(_BaseQwen2VisionModel):
    """Qwen2.5-VL vision tower."""

    _gated_mlp = True
    _norm_type = 'rms_norm'
    _use_window_attention = True

    def _vision_intermediate_size(self, cfg):
        return int(cfg.intermediate_size)

    def _vision_window_size(self, cfg):
        return int(cfg.window_size)

    def _vision_fullatt_block_indexes(self, cfg):
        return [int(x) for x in cfg.fullatt_block_indexes]

    def _vision_norm(self, pfx, *, dim: int):
        return self._rms_norm(pfx, dim=dim)

    def _add_mlp(self, builder, pfx, *, inter_size: int):
        gate, up, down = self._pad_mlp(
            self._linear(pfx + 'mlp.gate_proj'),
            self._linear(pfx + 'mlp.up_proj'),
            self._linear(pfx + 'mlp.down_proj'),
            inter_size=inter_size,
        )
        builder._add_linear('mlp_gate', gate, SplitSide.OUTPUT)
        builder._add_linear('mlp_fc1', up, SplitSide.OUTPUT)
        builder._add_linear('mlp_fc2', down, SplitSide.INPUT)


_VISION_MODEL_CLS = {
    'Qwen2VLForConditionalGeneration': Qwen2VisionModel,
    'Qwen2_5_VLForConditionalGeneration': Qwen2_5VisionModel,
}


@INPUT_MODELS.register_module(name='qwen2_vl')
class Qwen2VLModel:
    """Aggregate source model for Qwen2-VL and Qwen2.5-VL checkpoints."""

    _vision = True

    def __init__(self, cfg, *, resolver, vision_resolver=None,
                 language_model_only: bool = False):
        text_cfg = getattr(cfg, 'text_config', cfg)
        if text_cfg is None:
            raise ValueError('Qwen2VLModel requires a checkpoint with text_config.')
        if not hasattr(text_cfg, 'tie_word_embeddings'):
            text_cfg.tie_word_embeddings = getattr(cfg, 'tie_word_embeddings', False)
        self.text_model = Qwen2Model(text_cfg, resolver=resolver)

        archs = getattr(cfg, 'architectures', None) or []
        self._arch = archs[0] if archs else ''
        vision_cfg = getattr(cfg, 'vision_config', None)
        if language_model_only or vision_cfg is None:
            self.vision_model = None
        else:
            vision_cls = _VISION_MODEL_CLS.get(self._arch)
            if vision_cls is None:
                raise ValueError(f'Unsupported Qwen2-VL architecture: {self._arch!r}')
            self.vision_model = vision_cls(vision_cfg, resolver=vision_resolver or resolver)

    def bind_runtime(self, *, ctx, root_handles, attn_tp, mlp_tp, model_tp):
        for m in (self.text_model, self.vision_model):
            if m is not None:
                m.bind_runtime(
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

    def to_turbomind_multimodal(self, multimodal: list[dict[str, Any]]):
        if self.vision_model is None:
            raise ValueError('Qwen2 TurboMind vision encoder is not available.')
        return self.vision_model.to_turbomind_multimodal(multimodal)

    def model(self, pfx):
        self.text_model.model(pfx)
        if self.vision_model is not None:
            self.vision_model.model(pfx)
