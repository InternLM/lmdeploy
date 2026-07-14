# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen3.5 source models for the TurboMind pipeline.

This module hosts the full Qwen3.5 family in one place using composition
instead of inheritance (mirrors ``InternVLModel``):

* ``Qwen3_5TextModel`` -- text-only weight model (dense + linear-attn +
  optional MoE). Not registered on its own; used directly by the aggregate.
* ``Qwen3_5VisionModel`` -- the vision sub-tree rooted at
  ``ModelRoot.vision_model``.
* ``Qwen3_5Model`` -- a thin aggregate holding a ``text_model`` and an
  optional ``vision_model``, registered as ``qwen3_5`` / ``qwen3_5-moe``. It
  delegates the two-phase ``__init__`` / ``bind_runtime`` / ``model(pfx)``
  lifecycle to its children, and skips the vision encoder when
  ``disable_vision_encoder`` is set.

The patcher and position embedding are replicated across TP ranks. Vision
transformer blocks and merger linears shard with the model TP group.
"""
from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import _turbomind as _tm
import torch

from lmdeploy.vl.constants import Modality

from ..builders import (
    AttentionBuilder,
    Builder,
    DecoderLayerBuilder,
    DecoderLayerConfig,
    DeltaNetBuilder,
    FfnBuilder,
    LayerNormBuilder,
    ModuleListBuilder,
    ModuleListConfig,
    MoeBuilder,
    SplitSide,
    TextModelBuilder,
    VisionModelBuilder,
    _act_type_id,
    make_layer_norm_config,
)
from ..builders.attention import split_output_gate
from ..linear import Linear, transform_input_dim, transform_output_dim
from ..text_model import TextModel
from ..weight_format import TrivialFormat
from .base import INPUT_MODELS
from .utils import (
    make_attention_config,
    make_ffn_config,
    make_model_weight_config,
    make_moe_config,
    read_packed_moe_expert,
    reorder_rotary_emb,
)

if TYPE_CHECKING:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig


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
        self._attn_cfg.rope.mrope_interleaved = True

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
        builder.norm = self.norm(pfx + 'model.language_model.norm', _center_norm)
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
def _split_packed_vision_qkv(qkv):
    """Split HF vision QKV layout [Q | K | V] along output dim."""
    return tuple(x.contiguous() for x in qkv.chunk(3, dim=-1))


class Qwen3_5VisionModel(TextModel):
    """Vision sub-tree for Qwen3.5 VLM, rooted at ModelRoot.vision_model.

    Subclasses ``TextModel`` purely to reuse the two-phase lifecycle
    (``__init__`` / ``bind_runtime``) and the ``_linear`` resolver helper;
    its ``cfg`` is the HF ``vision_config`` and ``_vocab_size`` is never used.
    """

    def __init__(self, vision_cfg, *, resolver):
        super().__init__(vision_cfg, resolver=resolver)

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

    @staticmethod
    def _offset_pair(offset) -> tuple[int, int]:
        if isinstance(offset, torch.Tensor):
            values = offset.flatten().tolist()
        else:
            values = list(offset)
        if len(values) != 2:
            raise ValueError(f'Qwen3.5 ViT offset should contain 2 values, got {values!r}')
        return int(values[0]), int(values[1])

    @staticmethod
    def _grid_thw(grid_thw) -> tuple[int, int, int]:
        if isinstance(grid_thw, torch.Tensor):
            values = grid_thw.flatten().tolist()
        else:
            values = list(grid_thw)
        if len(values) != 3:
            raise ValueError(f'Qwen3.5 ViT grid_thw should contain 3 values, got {values!r}')
        return int(values[0]), int(values[1]), int(values[2])

    @staticmethod
    def _tm_tensor(tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'Qwen3.5 ViT multimodal data should be a torch.Tensor, got {type(tensor).__name__}')
        return _tm.from_dlpack(tensor.contiguous())

    def to_turbomind_multimodal(self, multimodal: list[dict[str, Any]]):
        """Convert Qwen3.5 VL preprocessor outputs to typed TurboMind input."""
        items = []
        for input_mm in multimodal:
            modality = input_mm.get('modality')
            if modality == Modality.IMAGE or modality == Modality.IMAGE.value:
                data = self._tm_tensor(input_mm['pixel_values'])
                grid_thw = self._grid_thw(input_mm['image_grid_thw'])
                tm_modality = _tm.multimodal.Modality.IMAGE
            elif modality == Modality.VIDEO or modality == Modality.VIDEO.value:
                data = self._tm_tensor(input_mm['pixel_values_videos'])
                grid_thw = self._grid_thw(input_mm['video_grid_thw'])
                tm_modality = _tm.multimodal.Modality.VIDEO
            else:
                raise ValueError(f'Qwen3.5 TurboMind does not support modality {modality!r}')

            token_begin, token_end = self._offset_pair(input_mm['offset'])
            items.append(
                _tm.multimodal.Qwen3_5VitItem(
                    modality=tm_modality,
                    data=data,
                    token_begin=token_begin,
                    token_end=token_end,
                    grid_thw=grid_thw,
                ))

        return _tm.multimodal.Qwen3_5VitInput(items)

    # ------------------------------------------------------------------
    # model() — build the vision sub-tree
    # ------------------------------------------------------------------

    def model(self, pfx):
        self._build_vision_model(pfx + 'model.visual')

    def _restore_dtype(self, builder):
        """Builder.__init__ unconditionally overwrites cfg.data_type with the
        context's (text-engine) dtype.

        The cfg is held by reference, so re-pinning it here propagates to every downstream _add_linear call on this
        builder, keeping the vision sub-tree on its native dtype.
        """
        builder.config.data_type = self._resolver.data_type
        return builder

    def _build_vision_model(self, pfx):
        cfg = self._make_vision_root_cfg()
        root = self._restore_dtype(VisionModelBuilder(
            cfg, self._ctx,
            root_handles=self._root_handles,
            tp=self._model_tp))

        root._add_tensor('pos_embed', (pfx + 'pos_embed').pop('weight'))
        root._add_linear('patch_embed', self._patch_embed(pfx + 'patch_embed.proj'))

        root.blocks = self.vit_blocks(pfx + 'blocks')

        root._add_linear('merger_fc1', self._linear(pfx + 'merger.linear_fc1'), SplitSide.OUTPUT)
        root._add_linear('merger_fc2', self._linear(pfx + 'merger.linear_fc2'), SplitSide.INPUT)
        root.merger_norm = self._layer_norm(pfx + 'merger.norm', dim=self._vis_hidden)

        root.build()

    def _make_vision_root_cfg(self):
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

        b = self._restore_dtype(Builder(cfg, self._ctx))
        b.tp = self._model_tp

        b.norm1 = self._layer_norm(pfx + 'norm1', dim=self._vis_hidden)
        b.norm2 = self._layer_norm(pfx + 'norm2', dim=self._vis_hidden)

        b.attention = self.vit_attn(pfx + 'attn')
        b._add_linear('mlp_fc1', self._linear(pfx + 'mlp.linear_fc1'), SplitSide.OUTPUT)
        b._add_linear('mlp_fc2', self._linear(pfx + 'mlp.linear_fc2'), SplitSide.INPUT)
        return b.build()

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
        # When we pad head_dim, the softmax scale must stay tied to the
        # real head_dim — the padded slice contributes zero to QK^T, so the
        # math is equivalent to head_dim=real_hd. Setting softmax_scale != 0
        # bypasses the runtime's `1/sqrt(attn.head_dim)` fallback.
        cfg.softmax_scale = (1.0 / math.sqrt(real_hd)
                             if padded_hd != real_hd else 0.0)
        return cfg

    def vit_attn(self, pfx):
        cfg = self._make_vision_attn_cfg()
        real_hd = self._vis_hidden // self._vis_heads
        padded_hd = cfg.head_dim
        H = cfg.head_num

        q, k, v = _split_packed_vision_qkv(self._linear(pfx + 'qkv'))

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

        m = self._restore_dtype(
            AttentionBuilder(cfg, self._ctx, tp=self._model_tp))
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
        m = self._restore_dtype(LayerNormBuilder(cfg, self._ctx))
        m.set_weight(weight, bias=bias)
        return m.build()


@INPUT_MODELS.register_module(name='qwen3_5-moe')
@INPUT_MODELS.register_module(name='qwen3_5')
class Qwen3_5Model:
    """Aggregate source model for Qwen3.5 checkpoints (text + optional
    vision)."""

    _vision = True

    def __init__(self, cfg: Qwen3_5Config | Qwen3_5MoeConfig, *, resolver,
                 vision_resolver=None,
                 disable_vision_encoder: bool = False):
        text_cfg = getattr(cfg, 'text_config', cfg)
        if text_cfg is None:
            raise ValueError(
                'Qwen3_5Model requires a checkpoint with text_config.')
        self.text_model = Qwen3_5TextModel(text_cfg, resolver=resolver)

        vision_cfg = getattr(cfg, 'vision_config', None)
        if disable_vision_encoder or vision_cfg is None:
            self.vision_model = None
        else:
            self.vision_model = Qwen3_5VisionModel(
                vision_cfg, resolver=vision_resolver or resolver)

    def bind_runtime(self, *, ctx, root_handles,
                     attn_tp, mlp_tp, model_tp):
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
            raise ValueError('Qwen3.5 TurboMind vision encoder is not available.')
        return self.vision_model.to_turbomind_multimodal(multimodal)

    def model(self, pfx):
        # Text root child must be attached before the vision one, since both
        # use the shared root_handles.
        self.text_model.model(pfx)
        if self.vision_model is not None:
            self.vision_model.model(pfx)
