# Copyright (c) OpenMMLab. All rights reserved.

from torch import Tensor, nn
from transformers import PretrainedConfig

from ..backends import OpType, get_backend
from ..backends.rotary_embedding import Llama3Parameters, LongRoPEScalingParameters, RopeType, YarnParameters


def _get_default_rope_parameters(config: PretrainedConfig):
    """get default rope parameters."""
    return dict(emb_type=RopeType.Default, scaling_factor=1.0)


def _get_linear_scaling_rope_parameters(config: PretrainedConfig):
    """get linear rope parameters."""
    rope_scaling = config.rope_scaling
    scaling_factor = rope_scaling['factor']
    return dict(emb_type=RopeType.LinearScaling, scaling_factor=scaling_factor)


def _get_dynamic_ntk_parameters(config: PretrainedConfig):
    """get dynamic ntk parameters."""
    rope_scaling = config.rope_scaling
    scaling_factor = rope_scaling['factor']
    return dict(emb_type=RopeType.DynamicNTKScaling, scaling_factor=scaling_factor)


def _get_yarn_parameters(config: PretrainedConfig):
    """get yarn parameters."""
    rope_scaling = config.rope_scaling
    scaling_factor = rope_scaling['factor']
    params = YarnParameters()
    params.attention_factor = rope_scaling.get('attention_factor', params.attention_factor)
    params.beta_fast = rope_scaling.get('beta_fast', params.beta_fast)
    params.beta_slow = rope_scaling.get('beta_slow', params.beta_slow)
    return dict(emb_type=RopeType.Yarn, scaling_factor=scaling_factor, yarn_params=params)


def _get_longrope_parameters(config: PretrainedConfig):
    """get longrope parameters."""
    rope_scaling = config.rope_scaling
    params = LongRoPEScalingParameters()
    scaling_factor = rope_scaling['factor']
    params.long_factor = rope_scaling.long_factor
    params.short_factor = rope_scaling.long_factor
    params.original_max_position_embeddings = rope_scaling.get('original_max_position_embeddings',
                                                               config.max_position_embeddings)
    return dict(emb_type=RopeType.LongRoPEScaling, scaling_factor=scaling_factor, longrope_params=params)


def _get_llama3_parameters(config: PretrainedConfig):
    """get llama rope parameters."""
    rope_scaling = config.rope_scaling
    params = Llama3Parameters()
    scaling_factor = rope_scaling['factor']
    params.low_freq_factor = rope_scaling['low_freq_factor']
    params.high_freq_factor = rope_scaling['high_freq_factor']
    params.original_max_position_embeddings = rope_scaling.get('original_max_position_embeddings',
                                                               params.original_max_position_embeddings)
    return dict(emb_type=RopeType.Llama3, scaling_factor=scaling_factor, llama3_params=params)


def build_rotary_params(config: PretrainedConfig):
    """get scaling_factor rotary params, and emb_type."""
    params = dict(emb_type=RopeType.Default)
    # cannot access config.rope_scaling when the model is "Qwen/Qwen2-Math-RM-72B"
    rope_scaling = getattr(config, 'rope_scaling', None)
    if rope_scaling is not None:
        rope_type_str = config.rope_scaling.get('rope_type', 'default')
        build_funcs = dict(default=_get_default_rope_parameters,
                           linear=_get_linear_scaling_rope_parameters,
                           dynamic=_get_dynamic_ntk_parameters,
                           yarn=_get_yarn_parameters,
                           longrope=_get_longrope_parameters,
                           llama3=_get_llama3_parameters)
        params.update(build_funcs[rope_type_str](config))
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
    """build rotary embedding op."""
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


class ApplyRotaryEmb(nn.Module):
    """apply rotary embedding."""

    def __init__(self):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.ApplyRotaryEmb)
        self.impl = builder.build()

    def forward(self, query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, inplace: bool = True):
        """forward."""
        return self.impl.forward(query, key, cos, sin, inplace)
