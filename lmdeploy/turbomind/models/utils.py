# Copyright (c) OpenMMLab. All rights reserved.
"""Shared utilities for source model input classes."""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from lmdeploy.archs import get_model_arch

from ..builders._base import _dequant_linear
from ..linear import Linear


def load_model_config(model_path: str) -> dict:
    """Load and normalise the HuggingFace model config to a plain dict.

    Handles nested configs (text_config, llm_config) and transformers AutoConfig objects that expose a to_dict() method.
    """
    _, model_config = get_model_arch(model_path)
    if hasattr(model_config, 'text_config'):
        model_config = model_config.text_config
    elif hasattr(model_config, 'llm_config'):
        model_config = model_config.llm_config
    if hasattr(model_config, 'to_dict'):
        return model_config.to_dict()
    return model_config


_ROPE_TYPE_MAP = {
    'default': 1,
    'linear': 2,
    'dynamic': 3,
    'yarn': 4,
    'llama3': 5,
    'mrope': 6,
}


def rope_type_to_int(type_str: str) -> int:
    return _ROPE_TYPE_MAP[type_str]


def parse_rope_param(cfg: dict, head_dim: int) -> tuple[SimpleNamespace, int]:
    """Parse RoPE configuration from a model config dict.

    Returns:
        rope_param: SimpleNamespace carrying rope fields (type, base, dim,
            factor, max_position_embeddings, attention_factor, beta_fast,
            beta_slow, low_freq_factor, high_freq_factor,
            original_max_position_embeddings, mrope_section)
        max_position_embeddings: int (0 if not present in config)
    """
    if 'rope_parameters' in cfg:
        # transformers v5.0.0 aggregates rope settings into rope_parameters
        rope_scaling = cfg['rope_parameters']
        rope_theta = float(rope_scaling.get('rope_theta', 10000.0))
    else:
        rope_theta = float(cfg.get('rope_theta', 10000.0))
        rope_scaling = cfg.get('rope_scaling', None)

    max_position_embeddings = int(cfg.get('max_position_embeddings', 0))
    rope_param = SimpleNamespace(
        type='default',
        base=rope_theta,
        dim=head_dim,
        factor=1.0,
        max_position_embeddings=None,
        attention_factor=1.0,
        beta_fast=32,
        beta_slow=1,
        low_freq_factor=None,
        high_freq_factor=None,
        original_max_position_embeddings=None,
        mrope_section=None,
    )

    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get('rope_type', '') or rope_scaling.get('type', '')
        if rope_scaling.get('mrope_section') is not None:
            rope_type = 'mrope'
        scaling_factor = rope_scaling.get('factor', 0.0)

        if rope_type == 'default':
            pass
        elif rope_type == 'dynamic':
            rope_param.type = 'dynamic'
            rope_param.factor = scaling_factor
            rope_param.max_position_embeddings = max_position_embeddings
        elif rope_type == 'linear':
            rope_param.type = 'linear'
            rope_param.factor = scaling_factor
        elif rope_type == 'llama3':
            low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
            high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
            original_max_position_embeddings = rope_scaling.get('original_max_position_embeddings', 0)
            rope_param.type = 'llama3'
            rope_param.factor = scaling_factor
            rope_param.low_freq_factor = low_freq_factor
            rope_param.high_freq_factor = high_freq_factor
            rope_param.original_max_position_embeddings = original_max_position_embeddings
        elif rope_type == 'yarn':
            attention_factor = rope_scaling.get('attention_factor', None)
            if attention_factor is None:
                attention_factor = 0.1 * math.log(scaling_factor) + 1.0
            beta_fast = rope_scaling.get('beta_fast', 32.0)
            beta_slow = rope_scaling.get('beta_slow', 1.0)
            rope_param.type = 'yarn'
            if 'original_max_position_embeddings' in rope_scaling:
                original_max_position_embeddings = rope_scaling['original_max_position_embeddings']
                scaling_factor = max_position_embeddings / original_max_position_embeddings
            else:
                original_max_position_embeddings = max_position_embeddings
            rope_param.factor = scaling_factor
            rope_param.max_position_embeddings = original_max_position_embeddings
            rope_param.attention_factor = attention_factor
            rope_param.beta_fast = beta_fast
            rope_param.beta_slow = beta_slow
        elif rope_type == 'mrope':
            mrope_section = rope_scaling.get('mrope_section')
            rope_param.type = 'mrope'
            rope_param.mrope_section = mrope_section
        else:
            raise RuntimeError(f'Unsupported rope type: {rope_type}')

    return rope_param, max_position_embeddings


def get_yarn_params(rope_scaling: dict) -> tuple[float, float]:
    """Compute DeepSeek2/MLA YaRN attention scale factors.

    Returns:
        attention_factor: mscale ratio used for attention scaling
        softmax_scale: pre-softmax scale (non-zero only when mscale_all_dim > 0)
    """
    scaling_factor = float(rope_scaling['factor'])
    mscale = rope_scaling['mscale']
    mscale_all_dim = rope_scaling['mscale_all_dim']

    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    attention_factor = float(
        yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim))

    softmax_scale = 0.0
    if mscale_all_dim:
        scale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        softmax_scale = scale * scale

    return attention_factor, softmax_scale


def _reorder_rotary_emb(x: torch.Tensor, head_dim: int, rope_dim: int):
    """Reorder rotary embedding layout for TurboMind's RoPE kernel."""
    if rope_dim < head_dim:
        output_dims = x.size(-1)
        head_num = output_dims // head_dim
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), head_num, head_dim)
        rotary = x[:, :, :rope_dim]
        passthrough = x[:, :, rope_dim:]
        rotary = rotary.view(x.size(0), head_num, 2, rope_dim // 2).transpose(2, 3).contiguous()
        rotary = rotary.view(x.size(0), head_num, rope_dim)
        x = torch.cat([rotary, passthrough], dim=-1)
        return x.reshape(orig_shape)
    else:
        output_dims = x.size(-1)
        head_num = output_dims // head_dim
        return x.view(-1, head_num, 2, head_dim // 2).transpose(2, 3).reshape(x.shape)


def reorder_rotary_emb(x, head_dim: int, rope_dim: int, *, resolver=None):
    """Apply RoPE layout permutation.

    Accepts either a ``Linear`` or a raw ``torch.Tensor``.

    For ``Linear`` inputs the permutation is applied to every tensor in the
    bundle with quantization awareness (block-alignment check, dequant
    fallback, block-level shuffling for scales/zeros). ``resolver`` is
    required and must not be ``None`` — it supplies the compute dtype
    threaded into ``_dequant_linear``.

    For ``torch.Tensor`` inputs the element-level interleave-transpose is
    applied directly. ``resolver`` is ignored.
    """
    from ..linear import Linear

    if isinstance(x, Linear):
        if resolver is None:
            raise TypeError(
                'resolver is required when passing a Linear to reorder_rotary_emb'
            )
        data_type = resolver.data_type
        wfmt = x.weight_format
        block_out = wfmt.block_out or 0

        # If blocks don't align with heads, dequant first
        if block_out and block_out % head_dim != 0:
            x = _dequant_linear(x, data_type=data_type)
            block_out = 0

        new_tensors = {}
        for kind, tensor in x.tensors.items():
            if kind in ('scales', 'zeros') and block_out > 0:
                # Block-level shuffle: reinterpret each block as a "head"
                # so _reorder_rotary_emb shuffles at block granularity.
                blocks_per_head = block_out // head_dim
                if blocks_per_head <= 1:
                    new_tensors[kind] = tensor
                else:
                    rope_dim_blocks = rope_dim * blocks_per_head // head_dim
                    new_tensors[kind] = _reorder_rotary_emb(tensor, blocks_per_head, rope_dim_blocks)
            elif tensor.size(-1) % head_dim == 0:
                new_tensors[kind] = _reorder_rotary_emb(tensor, head_dim, rope_dim)
            else:
                new_tensors[kind] = tensor

        return Linear(tensors=new_tensors, weight_format=x.weight_format)

    return _reorder_rotary_emb(x, head_dim, rope_dim)


# --- TP padding helpers ----------------------------------------------------

def _pad_inter_size(inter_size: int, group_size: int, tp: int) -> int:
    """Pad inter_size so it is divisible by group_size * tp."""
    group_size = max(1, group_size)
    group_num = (inter_size + group_size - 1) // group_size
    groups_per_rank = (group_num + tp - 1) // tp
    inter_size_padded = groups_per_rank * group_size * tp
    return inter_size_padded


def _pad_kv_head(kv_head_num: int, attn_tp: int) -> int:
    """Pad kv_head_num up to attn_tp when attn_tp is a multiple of kv_head_num.

    Rule:
      if attn_tp > kv_head_num and attn_tp % kv_head_num == 0:
          kv_head_num = attn_tp
    """
    if attn_tp > kv_head_num and attn_tp % kv_head_num == 0:
        return attn_tp
    return kv_head_num


def layer_progress(num_layers: int):
    """Tqdm iterable for spec.layers() per-layer conversion loops.

    Yields the layer indices 0..num_layers-1, displaying a single-line
    progress bar on stderr. ``leave=False`` clears the bar when the loop
    completes. Lazy-imports tqdm so importing utils.py stays cheap.
    """
    from tqdm import tqdm
    return tqdm(range(num_layers), desc='Loading', leave=False)


def read_packed_moe_expert(
    params: dict,
    gate_up_pfx: str,
    down_pfx: str,
    expert_idx: int,
    *,
    resolver,
    interleaved: bool = False,
    trans: bool = False,
) -> tuple[Linear, Linear, Linear]:
    """Read one packed MoE expert's fused gate_up + down and split into (w1,
    w2, w3) Linears in TM layout.

    ``gate_up_pfx`` and ``down_pfx`` are the full prefixes to the two
    packed tensors (e.g. ``'model.layers.5.mlp.experts.gate_up_proj'``).
    The caller composes these strings; this helper concatenates nothing.

    Parameters
    ----------
    interleaved : bool
        Split scheme for the fused gate_up output dim.
        ``False`` -> contiguous ``[..., :half]`` / ``[..., half:]`` (qwen3.5).
        ``True``  -> stride-2 interleaved ``[..., ::2]`` / ``[..., 1::2]`` (gpt-oss).
    trans : bool
        For trivial-format checkpoints that store the packed tensor in
        ``[n_experts, in, out]`` layout (gpt-oss), transposes the 2D
        ``weight`` tensor to undo the HF-to-TM transpose applied by
        ``TrivialFormat.normalize``. Only affects the ``weight`` kind on
        trivial-format linears; quantized formats use their own normalizers.
    """
    gate_up = resolver.resolve(params, gate_up_pfx, index=expert_idx)
    down    = resolver.resolve(params, down_pfx,    index=expert_idx)

    if trans:
        for lin in (gate_up, down):
            if lin.weight_format.name == 'trivial':
                w = lin.tensors.get('weight')
                if w is not None and w.dim() == 2:
                    lin.tensors['weight'] = w.t().contiguous()

    w1_t: dict[str, torch.Tensor] = {}
    w3_t: dict[str, torch.Tensor] = {}
    for kind, t in gate_up.tensors.items():
        if interleaved:
            w1_t[kind] = t[..., ::2].contiguous()
            w3_t[kind] = t[..., 1::2].contiguous()
        else:
            half = t.shape[-1] // 2
            w1_t[kind] = t[..., :half].contiguous()
            w3_t[kind] = t[..., half:].contiguous()
    w1 = Linear(tensors=w1_t, weight_format=gate_up.weight_format)
    w3 = Linear(tensors=w3_t, weight_format=gate_up.weight_format)
    return w1, down, w3
