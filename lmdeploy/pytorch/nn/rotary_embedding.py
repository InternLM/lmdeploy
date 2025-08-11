# Copyright (c) OpenMMLab. All rights reserved.

import math

from torch import Tensor, nn
from transformers import PretrainedConfig

from ..backends import OpType, get_backend
from ..backends.rotary_embedding import Llama3Parameters, LongRoPEScalingParameters, RopeType, YarnParameters


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

    ret = dict(emb_type=RopeType.Yarn, scaling_factor=factor, yarn_params=params)
    if 'original_max_position_embeddings' in rope_scaling:
        ret['max_position_embeddings'] = rope_scaling['original_max_position_embeddings']
    return ret


def _get_longrope_parameters(config: PretrainedConfig):
    """Get longrope parameters."""
    rope_scaling = config.rope_scaling
    scaling_factor = rope_scaling.get('factor', 1.0)
    long_factor = rope_scaling['long_factor']
    short_factor = rope_scaling['long_factor']
    original_max_position_embeddings = rope_scaling.get('original_max_position_embeddings',
                                                        config.max_position_embeddings)
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


def build_rotary_params(config: PretrainedConfig):
    """Get scaling_factor rotary params, and emb_type."""
    params = dict(emb_type=RopeType.Default)
    # cannot access config.rope_scaling when the model is "Qwen/Qwen2-Math-RM-72B"
    rope_scaling = getattr(config, 'rope_scaling', None)
    if rope_scaling is not None:
        # BC: "rope_type" was originally "type"
        rope_type_str = config.rope_scaling.get('rope_type', config.rope_scaling.get('type', 'default'))
        build_funcs = dict(default=_get_default_rope_parameters,
                           linear=_get_linear_scaling_rope_parameters,
                           dynamic=_get_dynamic_ntk_parameters,
                           yarn=_get_yarn_parameters,
                           longrope=_get_longrope_parameters,
                           su=_get_longrope_parameters,
                           llama3=_get_llama3_parameters)
        params.update(build_funcs[rope_type_str](config))

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
                           emb_type: RopeType = RopeType.Default,
                           partial_rotary_factor: float = None) -> nn.Module:
    """Build rotary embedding op."""
    backend = get_backend()

    builder = backend.get_layer_impl_builder(OpType.RotaryEmbedding)

    # update rope_dim
    if partial_rotary_factor is not None:
        dim = int(dim * partial_rotary_factor)
    return builder.build(dim,
                         max_position_embeddings,
                         base,
                         scaling_factor,
                         yarn_params=yarn_params,
                         longrope_params=longrope_params,
                         llama3_params=llama3_params,
                         emb_type=emb_type)


def build_rotary_embedding_from_config(config: PretrainedConfig) -> nn.Module:
    """Build rotary embedding op from config."""
    # import pdb; pdb.set_trace()
    emb_type = RopeType.LinearScaling
    rope_dim = getattr(config, 'head_dim', None)
    if rope_dim is None:
        rope_dim = config.hidden_size // config.num_attention_heads
    rope_max_pos_emb = config.max_position_embeddings
    rope_base = config.rope_theta
    rope_params = dict(emb_type=emb_type, dim=rope_dim, max_position_embeddings=rope_max_pos_emb, base=rope_base)
    update_params = build_rotary_params(config)
    rope_params.update(update_params)
    # import pdb; pdb.set_trace()  # noqa
    return build_rotary_embedding(**rope_params)


class ApplyRotaryEmb(nn.Module):
    """Apply rotary embedding."""

    def __init__(self):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.ApplyRotaryEmb)
        self.impl = builder.build()

    def forward(self, query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, inplace: bool = True):
        """forward."""
        return self.impl.forward(query, key, cos, sin, inplace)
