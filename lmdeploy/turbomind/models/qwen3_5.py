# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 TextModel — text path inherited, visual path added.

Loads ``Qwen3_5ForConditionalGeneration`` checkpoints whose top-level
HF config carries a ``vision_config`` block. Reuses ``_Qwen3_5Model``
verbatim for the language model and adds a visual sub-tree rooted at
``ModelRoot.visual_model``.

The patcher and position embedding are replicated across TP ranks. Visual
transformer blocks and merger linears shard with the model TP group.
"""
from __future__ import annotations

import math

import _turbomind as _tm
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

from ..builders import (
    AttentionBuilder,
    Builder,
    LayerNormBuilder,
    ModuleListBuilder,
    ModuleListConfig,
    SplitSide,
    VisualModelBuilder,
    make_layer_norm_config,
)
from ..linear import Linear, transform_input_dim, transform_output_dim
from ..weight_format import TrivialFormat
from ._qwen3_5 import _Qwen3_5Model
from .base import INPUT_MODELS
from .utils import reorder_rotary_emb

# Explicit map: ViT head_dim -> kernel-supported head_dim. The attention
# kernel only ships {64,128,192,256,576} and impl_16816.h requires
# head_dim % 16 == 0, so head_dim=72 (Qwen3.5-27B ViT) is zero-padded
# per-head to 128. Other dims must be added here deliberately — we don't
# auto-round-up because the next-supported jump can be large.
_VIT_HEAD_DIM_PADDED = {
    64:  64,    # <=2B: native
    72:  128,   # >=9B: pad to nearest kernel-supported dim
}


def _padded_vit_head_dim(real_hd: int) -> int:
    if real_hd not in _VIT_HEAD_DIM_PADDED:
        raise NotImplementedError(
            f'Qwen3.5 ViT head_dim={real_hd} not supported; '
            f'known: {sorted(_VIT_HEAD_DIM_PADDED)}')
    return _VIT_HEAD_DIM_PADDED[real_hd]


def _assert_trivial(linear: Linear, name: str) -> None:
    """Padding only handles trivial weights for now.

    Callers must gate this on `padded_hd != real_hd` so quantized 2B-style paths stay untouched.
    """
    if not isinstance(linear.weight_format, TrivialFormat):
        raise NotImplementedError(
            f'ViT {name} weight is {type(linear.weight_format).__name__}; '
            f'head_dim padding currently supports TrivialFormat only '
            f'(dequant ViT before padding).')


@transform_output_dim
def _pad_head_dim_out(t: torch.Tensor, *, num_heads: int, src_hd: int,
                      dst_hd: int) -> torch.Tensor:
    """Pad each head's OUTPUT block from src_hd to dst_hd; num_heads unchanged.

    Applied to Q/K/V projections: weight (in_dim, num_heads*src_hd) becomes (in_dim, num_heads*dst_hd) with the new
    [src_hd, dst_hd) slice zeroed. The @transform_output_dim decorator handles bias by promoting it to 2-D and squeezing
    back, so the bias is padded the same way as weight.
    """
    rest = t.shape[:-1]
    t = t.reshape(rest + (num_heads, src_hd))
    pad = t.new_zeros(rest + (num_heads, dst_hd - src_hd))
    return torch.cat([t, pad], dim=-1).reshape(rest + (num_heads * dst_hd,))


@transform_input_dim
def _pad_head_dim_in(t: torch.Tensor, *, num_heads: int, src_hd: int,
                     dst_hd: int) -> torch.Tensor:
    """Pad each head's INPUT block from src_hd to dst_hd; num_heads unchanged.

    Applied to the wo projection: weight (num_heads*src_hd, out_dim) becomes
    (num_heads*dst_hd, out_dim). The @transform_input_dim decorator passes
    1-D tensors (wo bias, which lives on the OUTPUT axis) through unchanged.
    """
    rest = t.shape[1:]
    t = t.reshape((num_heads, src_hd) + rest)
    pad = t.new_zeros((num_heads, dst_hd - src_hd) + rest)
    return torch.cat([t, pad], dim=1).reshape((num_heads * dst_hd,) + rest)


@transform_output_dim
def _split_packed_visual_qkv(qkv):
    """Split HF visual QKV layout [Q | K | V] along output dim."""
    return tuple(x.contiguous() for x in qkv.chunk(3, dim=-1))


@INPUT_MODELS.register_module(name='qwen3_5')
@INPUT_MODELS.register_module(name='qwen3_5-moe')
class Qwen3_5Model(_Qwen3_5Model):
    """Weight model for Qwen3.5 VLM (text + vision)."""

    _vision = True

    def __init__(self, cfg: Qwen3_5Config | Qwen3_5MoeConfig, *, resolver):
        text_cfg = cfg.text_config
        if text_cfg is None:
            raise ValueError(
                'Qwen3_5Model requires a checkpoint with text_config.')

        vision_cfg = cfg.vision_config
        if vision_cfg is None:
            raise ValueError(
                'Qwen3_5Model requires a checkpoint with vision_config; '
                'got none. Set disable_vision_encoder=True for text-only checkpoints.')

        super().__init__(text_cfg, resolver=resolver)

        self._vis_depth = int(vision_cfg.depth)
        self._vis_hidden = int(vision_cfg.hidden_size)
        self._vis_inter = int(vision_cfg.intermediate_size)
        self._vis_heads = int(vision_cfg.num_heads)
        self._vis_out_hidden = int(vision_cfg.out_hidden_size)
        self._vis_in_chans = int(vision_cfg.in_channels)
        self._vis_patch = int(vision_cfg.patch_size)
        self._vis_temporal = int(vision_cfg.temporal_patch_size)
        self._vis_pos_n = int(vision_cfg.num_position_embeddings)
        self._vis_spatial_merge = int(vision_cfg.spatial_merge_size)
        self._vis_norm_eps = 1e-6

        # in_dim of the patcher when the Conv3d is reinterpreted as a
        # Linear over flattened patches: C * T * H * W.
        self._patch_in_dim = (self._vis_in_chans
                              * self._vis_temporal
                              * self._vis_patch
                              * self._vis_patch)

    # ------------------------------------------------------------------
    # model() — extend the parent text build with the visual sub-tree
    # ------------------------------------------------------------------

    def model(self, pfx):
        super().model(pfx)
        self._build_visual_model(pfx + 'model.visual')

    # ------------------------------------------------------------------
    # Visual sub-tree
    # ------------------------------------------------------------------

    def _build_visual_model(self, pfx):
        cfg = self._make_visual_root_cfg()
        root = VisualModelBuilder(
            cfg, self._ctx, root_handles=self._root_handles)
        root.tp = self._model_tp

        root._add_tensor('pos_embed', (pfx + 'pos_embed').pop('weight'))
        root._add_linear('patch_embed', self._patch_embed(pfx + 'patch_embed.proj'))

        root.blocks = self.vit_blocks(pfx + 'blocks')

        root._add_linear('merger_fc1', self._linear(pfx + 'merger.linear_fc1'), SplitSide.OUTPUT)
        root._add_linear('merger_fc2', self._linear(pfx + 'merger.linear_fc2'), SplitSide.INPUT)
        root.merger_norm = self._layer_norm(pfx + 'merger.norm', dim=self._vis_hidden)

        root.build()

    def _make_visual_root_cfg(self):
        cfg = _tm.Qwen3_5VitConfig()
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
        cfg.norm_eps = self._vis_norm_eps
        return cfg

    def _patch_embed(self, pfx):
        weight = pfx.pop('weight')
        if weight.dim() >= 2:
            weight = weight.reshape(weight.shape[0], -1).t().contiguous()
        tensors = {'weight': weight}
        if pfx.has('bias'):
            tensors['bias'] = pfx.pop('bias')
        return Linear(tensors=tensors, weight_format=TrivialFormat())

    def vit_blocks(self, pfx):
        blocks = ModuleListBuilder(ModuleListConfig(), self._ctx)

        for i, p in pfx.slices(0, self._vis_depth):
            blocks[i] = self.vit_block(p)

        return blocks.build()

    def vit_block(self, pfx):
        cfg = _tm.Qwen3_5VitBlockConfig()
        cfg.data_type = self._resolver.data_type
        cfg.hidden_dim = self._vis_hidden
        cfg.head_num = self._vis_heads
        cfg.intermediate_size = self._vis_inter
        cfg.norm_eps = self._vis_norm_eps

        b = Builder(cfg, self._ctx)
        b.tp = self._model_tp

        b.norm1 = self._layer_norm(pfx + 'norm1', dim=self._vis_hidden)
        b.norm2 = self._layer_norm(pfx + 'norm2', dim=self._vis_hidden)

        b.attention = self.vit_attn(pfx + 'attn')
        b._add_linear('mlp_fc1', self._linear(pfx + 'mlp.linear_fc1'), SplitSide.OUTPUT)
        b._add_linear('mlp_fc2', self._linear(pfx + 'mlp.linear_fc2'), SplitSide.INPUT)
        return b.build()

    def _make_visual_attn_cfg(self):
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
        # When we pad head_dim, the softmax scale must stay tied to the
        # real head_dim — the padded slice contributes zero to QK^T, so the
        # math is equivalent to head_dim=real_hd. Setting softmax_scale != 0
        # bypasses the runtime's `1/sqrt(attn.head_dim)` fallback.
        cfg.softmax_scale = (1.0 / math.sqrt(real_hd)
                             if padded_hd != real_hd else 0.0)
        return cfg

    def vit_attn(self, pfx):
        cfg = self._make_visual_attn_cfg()
        real_hd = self._vis_hidden // self._vis_heads
        padded_hd = cfg.head_dim
        H = cfg.head_num

        q, k, v = _split_packed_visual_qkv(self._linear(pfx + 'qkv'))

        # Qwen3.5 ViT applies RoPE before invoking the attention kernel.
        # Reorder Q/K once at export time so the runtime can use the same
        # adjacent-pair RoPE layout as TurboMind's attention kernels.
        # RoPE is computed at the real head_dim regardless of padding.
        q = reorder_rotary_emb(q, real_hd, real_hd, resolver=self._resolver)
        k = reorder_rotary_emb(k, real_hd, real_hd, resolver=self._resolver)

        proj = self._linear(pfx + 'proj')

        # Only force TrivialFormat on the padded path; the native-head_dim
        # path keeps the existing quantized/non-quantized behavior intact.
        if padded_hd != real_hd:
            for ln, name in [(q, 'q'), (k, 'k'), (v, 'v'), (proj, 'proj')]:
                _assert_trivial(ln, name)
            pad_kwargs = dict(num_heads=H, src_hd=real_hd, dst_hd=padded_hd)
            q = _pad_head_dim_out(q, **pad_kwargs)
            k = _pad_head_dim_out(k, **pad_kwargs)
            v = _pad_head_dim_out(v, **pad_kwargs)
            proj = _pad_head_dim_in(proj, **pad_kwargs)

        m = AttentionBuilder(cfg, self._ctx, tp=self._model_tp)
        m.add_qkv_proj(q, k, v)
        m.add_o_proj(proj)
        return m.build()

    # ------------------------------------------------------------------
    # Helper: build a LayerNorm child
    # ------------------------------------------------------------------

    def _layer_norm(self, pfx, *, dim: int):
        weight = pfx.pop('weight')
        bias = pfx.pop('bias') if pfx.has('bias') else None
        cfg = make_layer_norm_config(dim=dim,
                                     data_type=self._resolver.data_type,
                                     norm_eps=self._vis_norm_eps)
        m = LayerNormBuilder(cfg, self._ctx)
        m.set_weight(weight, bias=bias)
        return m.build()
