# Copyright (c) OpenMMLab. All rights reserved.
"""Shared utilities for source model input classes."""
from __future__ import annotations

import math
from types import SimpleNamespace

import _turbomind as _tm
import torch

from lmdeploy.archs import get_model_arch

from ..builders import _act_type_id
from ..linear import Linear, _dequant_linear


def source_model_config(model_config):
    """Local config consumed by a TurboMind source model."""
    # VLM aggregates expose both text_config and vision_config; only text-only HF wrappers unwrap.
    if hasattr(model_config, 'text_config') and not hasattr(model_config, 'vision_config'):
        return model_config.text_config
    return model_config


def load_model_config(model_path: str):
    """Load the local Transformers config object for a source text model."""
    _, model_config = get_model_arch(model_path)
    return source_model_config(model_config)


def _optional_attr(cfg, name: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _param_get(params, name: str, default=None):
    if params is None:
        return default
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)


def _param_has(params, name: str) -> bool:
    if params is None:
        return False
    if isinstance(params, dict):
        return name in params
    return hasattr(params, name)


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


def _get_mscale(scale, mscale=1):
    """YaRN mscale helper.

    Shared by parse_rope_param and MLA softmax_scale.
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def parse_rope_param(cfg, head_dim: int) -> tuple[SimpleNamespace, int]:
    """Parse RoPE configuration from a model config dict or object.

    Returns:
        rope_param: SimpleNamespace carrying rope fields (type, base, dim,
            factor, max_position_embeddings, attention_factor, beta_fast,
            beta_slow, low_freq_factor, high_freq_factor,
            original_max_position_embeddings, mrope_section)
        max_position_embeddings: int (0 if not present in config)
    """
    rope_parameters = _optional_attr(cfg, 'rope_parameters', None)
    if rope_parameters is not None:
        # transformers v5.0.0 aggregates rope settings into rope_parameters
        rope_scaling = rope_parameters
        rope_theta = float(_param_get(rope_scaling, 'rope_theta', 10000.0))
    else:
        rope_theta = float(_optional_attr(cfg, 'rope_theta', 10000.0))
        rope_scaling = _optional_attr(cfg, 'rope_scaling', None)

    max_position_embeddings = int(_optional_attr(cfg, 'max_position_embeddings', 0))
    partial_rotary_factor = _param_get(rope_parameters, 'partial_rotary_factor', None)
    if partial_rotary_factor is None:
        partial_rotary_factor = float(_optional_attr(cfg, 'partial_rotary_factor', 1.0))
    rope_param = SimpleNamespace(
        type='default',
        base=rope_theta,
        dim=int(head_dim * partial_rotary_factor),
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

    if rope_scaling is not None:
        rope_type = _param_get(rope_scaling, 'rope_type', '') or _param_get(rope_scaling, 'type', '')
        if _param_get(rope_scaling, 'mrope_section') is not None:
            rope_type = 'mrope'
        scaling_factor = _param_get(rope_scaling, 'factor', 0.0)

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
            low_freq_factor = _param_get(rope_scaling, 'low_freq_factor', 1.0)
            high_freq_factor = _param_get(rope_scaling, 'high_freq_factor', 1.0)
            original_max_position_embeddings = _param_get(rope_scaling, 'original_max_position_embeddings', 0)
            rope_param.type = 'llama3'
            rope_param.factor = scaling_factor
            rope_param.low_freq_factor = low_freq_factor
            rope_param.high_freq_factor = high_freq_factor
            rope_param.original_max_position_embeddings = original_max_position_embeddings
        elif rope_type == 'yarn':
            attention_factor = _param_get(rope_scaling, 'attention_factor', None)
            if attention_factor is None:
                mscale = _param_get(rope_scaling, 'mscale')
                mscale_all_dim = _param_get(rope_scaling, 'mscale_all_dim')
                if mscale is not None and mscale_all_dim is not None:
                    attention_factor = float(
                        _get_mscale(scaling_factor, mscale) /
                        _get_mscale(scaling_factor, mscale_all_dim))
                else:
                    attention_factor = _get_mscale(scaling_factor)
            beta_fast = _param_get(rope_scaling, 'beta_fast', 32.0)
            beta_slow = _param_get(rope_scaling, 'beta_slow', 1.0)
            rope_param.type = 'yarn'
            if _param_has(rope_scaling, 'original_max_position_embeddings'):
                original_max_position_embeddings = _param_get(rope_scaling, 'original_max_position_embeddings')
                scaling_factor = max_position_embeddings / original_max_position_embeddings
            else:
                original_max_position_embeddings = max_position_embeddings
            rope_param.factor = scaling_factor
            rope_param.max_position_embeddings = original_max_position_embeddings
            rope_param.attention_factor = attention_factor
            rope_param.beta_fast = beta_fast
            rope_param.beta_slow = beta_slow
        elif rope_type == 'mrope':
            mrope_section = _param_get(rope_scaling, 'mrope_section')
            rope_param.type = 'mrope'
            rope_param.mrope_section = mrope_section
        else:
            raise RuntimeError(f'Unsupported rope type: {rope_type}')

    return rope_param, max_position_embeddings


def copy_rope_config(rope_cfg, rope_param, max_position_embeddings: int):
    """Copy parsed RoPE fields into a TurboMind C++ rope config object."""
    rope_cfg.type = rope_type_to_int(rope_param.type)
    rope_cfg.base = rope_param.base
    rope_cfg.dim = rope_param.dim
    rope_cfg.factor = rope_param.factor
    rope_cfg.max_position_embeddings = max_position_embeddings
    if rope_param.type == 'yarn':
        rope_cfg.yarn_attention_factor = rope_param.attention_factor
        rope_cfg.yarn_beta_fast = rope_param.beta_fast
        rope_cfg.yarn_beta_slow = rope_param.beta_slow
    elif rope_param.type == 'llama3':
        rope_cfg.llama3_low_freq_factor = rope_param.low_freq_factor
        rope_cfg.llama3_high_freq_factor = rope_param.high_freq_factor
        rope_cfg.llama3_original_max_position_embeddings = rope_param.original_max_position_embeddings
    elif rope_param.type == 'mrope':
        rope_cfg.mrope_section = rope_param.mrope_section


def make_model_weight_config(cfg):
    """Build the root ModelWeightConfig from root-module fields."""
    model_cfg = _tm.ModelWeightConfig()
    model_cfg.hidden_units = cfg.hidden_size
    return model_cfg


def make_attention_config(cfg, *, head_dim=None):
    """Build common AttentionConfig fields from attention-module geometry."""
    hidden_dim = cfg.hidden_size
    head_num = cfg.num_attention_heads
    head_dim = head_dim if head_dim is not None else getattr(cfg, 'head_dim', hidden_dim // head_num)
    kv_head_num = cfg.num_key_value_heads
    rope, max_position_embeddings = parse_rope_param(cfg, head_dim)
    attn_cfg = _tm.AttentionConfig()
    attn_cfg.hidden_dim = hidden_dim
    attn_cfg.head_dim = head_dim
    attn_cfg.head_num = head_num
    attn_cfg.kv_head_num = kv_head_num
    attn_cfg.window_size = 0
    attn_cfg.softmax_scale = 0.0
    copy_rope_config(attn_cfg.rope, rope, max_position_embeddings)
    return attn_cfg


def make_ffn_config(cfg, *, act_type, inter_size=None):
    """Build common FfnConfig fields from FFN-module shape."""
    ffn_cfg = _tm.FfnConfig()
    ffn_cfg.hidden_dim = cfg.hidden_size
    ffn_cfg.act_type = act_type
    ffn_cfg.inter_size = inter_size if inter_size is not None else cfg.intermediate_size
    return ffn_cfg


def make_moe_config(cfg, *,
                    experts_per_token,
                    act_type=None,
                    norm_topk_prob=True,
                    topk_method='greedy',
                    scoring_func='softmax',
                    routed_scale=1.0,
                    topk_group=1,
                    n_group=1,
                    router_n_groups=0):
    """Build a MoeConfig populated from HF config and per-model overrides."""
    if act_type is None:
        act_type = _act_type_id('silu')

    moe_cfg = _tm.MoeConfig()
    moe_cfg.experts_per_token = experts_per_token
    moe_cfg.norm_topk_prob = norm_topk_prob
    moe_cfg.routed_scale = routed_scale
    moe_cfg.topk_group = topk_group
    moe_cfg.topk_method = topk_method
    moe_cfg.n_group = n_group
    moe_cfg.scoring_func = scoring_func
    moe_cfg.router_n_groups = router_n_groups
    moe_cfg.act_type = act_type
    moe_cfg.fuse_silu = True
    return moe_cfg


def make_mla_config(cfg):
    """Build an AttentionConfig for MLA models.

    Computes MLA geometry, softmax scale (including YaRN mscale_all_dim),
    and populates all MLA-specific AttentionConfig fields.

    Returns:
        _tm.AttentionConfig populated with MLA fields.
    """
    qk_nope_dim = cfg.qk_nope_head_dim
    qk_rope_dim = cfg.qk_rope_head_dim
    kv_lora_rank = cfg.kv_lora_rank
    q_head_dim = qk_nope_dim + qk_rope_dim

    size_per_head = q_head_dim
    v_head_dim = cfg.v_head_dim
    softmax_scale = 0.0
    if kv_lora_rank and kv_lora_rank != qk_nope_dim:
        size_per_head = kv_lora_rank + qk_rope_dim
        v_head_dim = kv_lora_rank
        softmax_scale = q_head_dim ** (-0.5)

    rope, max_position_embeddings = parse_rope_param(cfg, qk_rope_dim)

    # MLA-specific YaRN mscale_all_dim softmax_scale adjustment
    rope_params = (getattr(cfg, 'rope_parameters', None)
                   or getattr(cfg, 'rope_scaling', None))
    if rope_params:
        rope_type = (_param_get(rope_params, 'rope_type', '')
                     or _param_get(rope_params, 'type', ''))
        if rope_type == 'yarn':
            mscale_all_dim = _param_get(rope_params, 'mscale_all_dim')
            if mscale_all_dim:
                scaling_factor = float(_param_get(rope_params, 'factor', 0.0))
                mscale = _get_mscale(scaling_factor, mscale_all_dim)
                softmax_scale = q_head_dim ** (-0.5) * mscale * mscale

    attn_cfg = _tm.AttentionConfig()
    attn_cfg.hidden_dim = cfg.hidden_size
    attn_cfg.head_dim = size_per_head
    attn_cfg.head_num = cfg.num_attention_heads
    attn_cfg.kv_head_num = 1
    attn_cfg.kv_lora_rank = kv_lora_rank
    attn_cfg.q_lora_rank = cfg.q_lora_rank or 0
    attn_cfg.qk_rope_dim = qk_rope_dim
    attn_cfg.qk_nope_dim = qk_nope_dim
    attn_cfg.v_head_dim = v_head_dim
    copy_rope_config(attn_cfg.rope, rope, max_position_embeddings)
    attn_cfg.softmax_scale = softmax_scale

    return attn_cfg


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



def read_packed_moe_expert(
    gate_up_pfx,
    down_pfx,
    expert_idx: int,
    *,
    resolver,
    interleaved: bool = False,
    trans: bool = False,
):
    """Read one packed MoE expert's fused gate_up + down and split into (w1,
    w2, w3) Linears in TM layout.

    ``gate_up_pfx`` and ``down_pfx`` are Prefix objects pointing to the
    two packed tensors (e.g. ``experts_pfx + 'gate_up_proj'``).  The
    caller composes these via Prefix arithmetic; this helper concatenates
    nothing.

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
    gate_up = resolver.resolve(gate_up_pfx, index=expert_idx)
    down    = resolver.resolve(down_pfx,    index=expert_idx)

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
