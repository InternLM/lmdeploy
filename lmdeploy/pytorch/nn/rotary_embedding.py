# Copyright (c) OpenMMLab. All rights reserved.

import math

import torch
from torch import Tensor, nn
from transformers import PretrainedConfig

from ..backends import OpType, get_backend
from ..backends.rotary_embedding import (FopeParameters, Llama3Parameters, LongRoPEScalingParameters, RopeType,
                                         YarnParameters)


def _get_default_rope_parameters(config: PretrainedConfig):
    """Get default rope parameters."""
    return dict(emb_type=RopeType.Default, scaling_factor=1.0)


def _get_linear_scaling_rope_parameters(config: PretrainedConfig):
    """Get linear rope parameters."""
    rope_scaling = config.rope_scaling
    scaling_factor = rope_scaling['factor']
    return dict(emb_type=RopeType.LinearScaling, scaling_factor=scaling_factor)


def _get_dynamic_ntk_parameters(config: PretrainedConfig):
    """Get dynamic ntk parameters."""
    rope_scaling = config.rope_scaling
    scaling_factor = rope_scaling['factor']
    return dict(emb_type=RopeType.DynamicNTKScaling, scaling_factor=scaling_factor)


def _get_yarn_parameters(config: PretrainedConfig):
    """Get yarn parameters."""

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    rope_scaling = config.rope_scaling
    factor = rope_scaling['factor']
    params = YarnParameters()
    params.beta_fast = rope_scaling.get('beta_fast', params.beta_fast)
    params.beta_slow = rope_scaling.get('beta_slow', params.beta_slow)
    mscale = rope_scaling.get('mscale', params.mscale)
    mscale_all_dim = rope_scaling.get('mscale_all_dim', params.mscale_all_dim)
    truncate = rope_scaling.get('truncate', params.truncate)

    if 'attention_factor' in rope_scaling:
        attention_factor = rope_scaling.get('attention_factor')
    else:
        if mscale_all_dim and mscale:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    params.attention_factor = attention_factor
    params.mscale = mscale
    params.mscale_all_dim = mscale_all_dim
    params.truncate = truncate

    ret = dict(emb_type=RopeType.Yarn, scaling_factor=factor, yarn_params=params)
    if 'original_max_position_embeddings' in rope_scaling:
        ret['max_position_embeddings'] = rope_scaling['original_max_position_embeddings']
    return ret


def _get_longrope_parameters(config: PretrainedConfig):
    """Get longrope parameters."""
    rope_scaling = config.rope_scaling
    scaling_factor = rope_scaling.get('factor', 1.0)
    long_factor = rope_scaling['long_factor']
    short_factor = rope_scaling['short_factor']
    original_max_position_embeddings = getattr(config, 'original_max_position_embeddings',
                                               config.max_position_embeddings)
    original_max_position_embeddings = rope_scaling.get('original_max_position_embeddings',
                                                        original_max_position_embeddings)
    params = LongRoPEScalingParameters(
        long_factor=long_factor,
        short_factor=short_factor,
        original_max_position_embeddings=original_max_position_embeddings,
    )
    return dict(emb_type=RopeType.LongRoPEScaling, scaling_factor=scaling_factor, longrope_params=params)


def _get_llama3_parameters(config: PretrainedConfig):
    """Get llama rope parameters."""
    rope_scaling = config.rope_scaling
    params = Llama3Parameters()
    scaling_factor = rope_scaling['factor']
    params.low_freq_factor = rope_scaling['low_freq_factor']
    params.high_freq_factor = rope_scaling['high_freq_factor']
    params.original_max_position_embeddings = rope_scaling.get('original_max_position_embeddings',
                                                               params.original_max_position_embeddings)
    return dict(emb_type=RopeType.Llama3, scaling_factor=scaling_factor, llama3_params=params)


def _get_fope_parameters(config: PretrainedConfig):
    """Get fope parameters."""
    # check if fope is used
    rope_scaling = getattr(config, 'rope_scaling', dict())
    fope_keys = ['fope_sep_head', 'fope_num_inv_freq']
    is_fope = any(key in rope_scaling for key in fope_keys)
    if not is_fope:
        return dict()

    params = FopeParameters()
    rope_scaling = config.rope_scaling
    params.num_inv_freq = rope_scaling.get('fope_num_inv_freq', rope_scaling.get('num_inv_freq', params.num_inv_freq))
    params.num_key_value_heads = config.num_key_value_heads
    params.fope_sep_head = rope_scaling['fope_sep_head']
    return dict(fope_params=params)


def build_rotary_params(config: PretrainedConfig):
    """Get scaling_factor rotary params, and emb_type."""
    params = dict(emb_type=RopeType.Default)
    # cannot access config.rope_scaling when the model is "Qwen/Qwen2-Math-RM-72B"
    rope_scaling = getattr(config, 'rope_scaling', None)
    if rope_scaling is not None:
        # BC: "rope_type" was originally "type"
        rope_type_str = config.rope_scaling.get('rope_type', config.rope_scaling.get('type', 'default'))
        if rope_type_str == 'fope':
            rope_type_str = 'default'
        build_funcs = dict(default=_get_default_rope_parameters,
                           linear=_get_linear_scaling_rope_parameters,
                           dynamic=_get_dynamic_ntk_parameters,
                           yarn=_get_yarn_parameters,
                           longrope=_get_longrope_parameters,
                           su=_get_longrope_parameters,
                           llama3=_get_llama3_parameters)
        params.update(build_funcs[rope_type_str](config))
        params.update(_get_fope_parameters(config))

    # update partial_rotary_factor
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, 'partial_rotary_factor') else None
    if partial_rotary_factor is not None:
        params['partial_rotary_factor'] = partial_rotary_factor

    return params


def build_rotary_embedding(dim: int,
                           max_position_embeddings: int = 2048,
                           base: int = 10000,
                           scaling_factor: float = 1.0,
                           yarn_params: YarnParameters = None,
                           longrope_params: LongRoPEScalingParameters = None,
                           llama3_params: Llama3Parameters = None,
                           fope_params: FopeParameters = None,
                           emb_type: RopeType = RopeType.Default,
                           partial_rotary_factor: float = None,
                           device: torch.device = None) -> nn.Module:
    """Build rotary embedding op."""
    backend = get_backend()

    builder = backend.get_layer_impl_builder(OpType.RotaryEmbedding)

    # update rope_dim
    if partial_rotary_factor is not None:
        dim = int(dim * partial_rotary_factor)
    impl = builder.build(dim,
                         max_position_embeddings,
                         base,
                         scaling_factor,
                         yarn_params=yarn_params,
                         longrope_params=longrope_params,
                         llama3_params=llama3_params,
                         emb_type=emb_type)

    if fope_params is not None:
        inv_freq = impl.inv_freq
        fope_params.inv_freq = inv_freq
        fope = FopeRotaryEmbedding(dim, max_position_embeddings, scaling_factor, fope_params, device)
        return fope

    return impl


def build_rotary_embedding_from_config(config: PretrainedConfig, device: torch.device = None) -> nn.Module:
    """Build rotary embedding op from config."""
    emb_type = RopeType.LinearScaling
    rope_dim = getattr(config, 'head_dim', None)
    if rope_dim is None:
        rope_dim = config.hidden_size // config.num_attention_heads
    rope_max_pos_emb = config.max_position_embeddings
    rope_base = config.rope_theta
    rope_params = dict(emb_type=emb_type, dim=rope_dim, max_position_embeddings=rope_max_pos_emb, base=rope_base)
    update_params = build_rotary_params(config)
    rope_params.update(update_params)
    return build_rotary_embedding(**rope_params, device=device)


class ApplyRotaryEmb(nn.Module):
    """Apply rotary embedding."""

    def __init__(self):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.ApplyRotaryEmb)
        self.impl = builder.build()

    def forward(self, query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, inplace: bool = True):
        """forward."""

        assert cos.dim() <= 3 and sin.dim() <= 3

        need_reshape = False
        if cos.dim() == 3:
            # for fope
            assert query.dim() == key.dim() == 3, 'Expected query key (seq_len, heads, head_dim)'
            need_reshape = True
            query_shape = query.shape
            key_shape = key.shape
            cos = cos.flatten(0, 1)
            sin = sin.flatten(0, 1)
            seq_len = cos.size(0)
            query = query.view(seq_len, -1, query.size(-1))
            key = key.view(seq_len, -1, key.size(-1))

        query, key = self.impl.forward(query, key, cos, sin, inplace)

        if need_reshape:
            query = query.view(query_shape)
            key = key.view(key_shape)
        return query, key


class FopeRotaryEmbedding(nn.Module):
    """Fope rotary embedding."""

    def __init__(self,
                 dim: int,
                 max_position_embeddings: int,
                 attention_scaling: float,
                 params: FopeParameters,
                 device: torch.device = None):
        super().__init__()

        num_key_value_heads, tp = self.update_num_kv_heads(params.num_key_value_heads)
        self.tp = tp
        params.num_key_value_heads = num_key_value_heads

        # build impl
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.RotaryEmbedding)
        self.impl = builder.build(dim,
                                  max_position_embeddings=max_position_embeddings,
                                  scaling_factor=attention_scaling,
                                  fope_params=params,
                                  emb_type=RopeType.Fope)

        # setup params
        inv_freq = self.impl.inv_freq
        self.input_dim = inv_freq.shape[-1]
        self.output_dim = inv_freq.shape[-1]
        self.cos_coef = nn.Parameter(torch.empty(num_key_value_heads, self.input_dim, self.output_dim, device=device),
                                     requires_grad=False)
        self.sin_coef = nn.Parameter(torch.empty(num_key_value_heads, self.input_dim, self.output_dim, device=device),
                                     requires_grad=False)
        if self.tp:
            self.cos_coef.weight_loader = self.weight_loader
            self.sin_coef.weight_loader = self.weight_loader

    @staticmethod
    def update_num_kv_heads(num_key_value_heads: int):
        """Update num_key_value_heads."""
        from lmdeploy.pytorch.distributed import get_dist_manager
        dist_mgr = get_dist_manager()
        dist_ctx = dist_mgr.current_context()
        tp = dist_ctx.dist_config.attn_tp
        # tp = dist_ctx.dist_config.attn_config.tp
        if tp > 1:
            num_key_value_heads = max(1, num_key_value_heads // tp)
        return num_key_value_heads, tp

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader."""
        from lmdeploy.pytorch.distributed import get_tp_world_rank
        world_size, rank = get_tp_world_rank()
        num_key_value_heads = loaded_weight.size(0)

        if num_key_value_heads < world_size:
            n_replicate = world_size // num_key_value_heads
            world_size = num_key_value_heads
            rank = rank // n_replicate

        loaded_weight = loaded_weight.chunk(world_size, dim=0)[rank]
        param.copy_(loaded_weight)

    def forward(self, x: Tensor, position_ids: Tensor):
        """forward."""
        return self.impl.forward(x, position_ids, sin_coef=self.sin_coef, cos_coef=self.cos_coef)
