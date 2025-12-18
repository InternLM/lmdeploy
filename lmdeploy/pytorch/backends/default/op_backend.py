# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from ..base import OpsBackend, OpType


class DefaultOpsBackend(OpsBackend):

    @staticmethod
    def get_name() -> str:
        return 'default'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """Get builder of given layer type."""
        if layer_type == OpType.Linear:
            from .linear import DefaultLinearBuilder
            return DefaultLinearBuilder
        elif layer_type == OpType.RotaryEmbedding:
            from .rotary_embedding import DefaultRotaryEmbeddingBuilder
            return DefaultRotaryEmbeddingBuilder
        elif layer_type == OpType.ApplyRotaryEmb:
            from .apply_rotary_emb import DefaultApplyRotaryEmbBuilder
            return DefaultApplyRotaryEmbBuilder
        elif layer_type == OpType.SiluAndMul:
            from .activation import DefaultSiluAndMulBuilder
            return DefaultSiluAndMulBuilder
        elif layer_type == OpType.GeluAndMul:
            from .activation import DefaultGeluAndMulBuilder
            return DefaultGeluAndMulBuilder
        elif layer_type == OpType.RMSNorm:
            from .norm import DefaultRMSNormBuilder
            return DefaultRMSNormBuilder
        elif layer_type == OpType.LayerNorm:
            from .norm import DefaultLayerNormBuilder
            return DefaultLayerNormBuilder
        elif layer_type == OpType.MultinomialSampling:
            from .multinomial_sampling import DefaultMultinomialSamplingBuilder
            return DefaultMultinomialSamplingBuilder
        elif layer_type == OpType.LinearW4A16:
            from .awq_modules import DefaultLinearW4A16Builder
            return DefaultLinearW4A16Builder
        elif layer_type == OpType.SoftmaxTopK:
            from .moe import DefaultSoftmaxTopKBuilder
            return DefaultSoftmaxTopKBuilder
        elif layer_type == OpType.Embedding:
            from .embedding import DefaultEmbeddingBuilder
            return DefaultEmbeddingBuilder
        else:
            raise RuntimeError(f'{layer_type} not supported.')

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """Get block shape of k."""
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
        """Get block shape of v."""
        return (
            block_size,
            num_heads,
            head_size,
        )

    @staticmethod
    def init():
        pass

    @staticmethod
    def ccl_backend() -> str:
        return 'nccl'
