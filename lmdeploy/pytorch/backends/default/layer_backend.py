# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from ..base import LayersBackend, LayerType


class DefaultLayersBackend(LayersBackend):

    @staticmethod
    def get_name() -> str:
        raise 'default'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: LayerType):
        """get builder of given layer type."""
        if layer_type == LayerType.Linear:
            from .linear import DefaultLinearBuilder
            return DefaultLinearBuilder
        elif layer_type == LayerType.RotaryEmbedding:
            from .rotary_embedding import DefaultRotaryEmbeddingBuilder
            return DefaultRotaryEmbeddingBuilder
        elif layer_type == LayerType.ApplyRotaryEmb:
            from .apply_rotary_emb import DefaultApplyRotaryEmbBuilder
            return DefaultApplyRotaryEmbBuilder
        elif layer_type == LayerType.SiluAndMul:
            from .activation import DefaultSiluAndMulBuilder
            return DefaultSiluAndMulBuilder
        elif layer_type == LayerType.RMSNorm:
            from .norm import DefaultRMSNormBuilder
            return DefaultRMSNormBuilder
        elif layer_type == LayerType.LayerNorm:
            from .norm import DefaultLayerNormBuilder
            return DefaultLayerNormBuilder
        elif layer_type == LayerType.MultinomialSampling:
            from .multinomial_sampling import DefaultMultinomialSamplingBuilder
            return DefaultMultinomialSamplingBuilder
        elif layer_type == LayerType.LinearW4A16:
            from .awq_modules import DefaultLinearW4A16Builder
            return DefaultLinearW4A16Builder
        elif layer_type == LayerType.SoftmaxTopK:
            from .moe import DefaultSoftmaxTopKBuilder
            return DefaultSoftmaxTopKBuilder
        else:
            raise RuntimeError(f'{layer_type} not supported.')

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """get block shape of k."""
        return (
            block_size,
            num_heads,
            head_size,
        )

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """get block shape of v."""
        return (
            block_size,
            num_heads,
            head_size,
        )
