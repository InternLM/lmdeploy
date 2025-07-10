# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.utils import get_logger

from ..base import OpType
from ..default import DefaultOpsBackend

logger = get_logger('lmdeploy')


class DlinferOpsBackend(DefaultOpsBackend):
    """Dlinfer layer backend."""

    @staticmethod
    def get_name() -> str:
        """Backend name."""
        return 'dlinfer'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """Get dlinfer layer builder."""
        if layer_type == OpType.PagedAttention:
            from .attention import DlinferAttentionBuilder
            return DlinferAttentionBuilder
        elif layer_type == OpType.FlashAttention:
            from .flash_attention import DlinferFlashAttentionBuilder
            return DlinferFlashAttentionBuilder
        elif layer_type == OpType.ApplyRotaryEmb:
            from .apply_rotary_emb import DlinferApplyRotaryEmbBuilder
            return DlinferApplyRotaryEmbBuilder
        elif layer_type == OpType.SiluAndMul:
            from .activation import DlinferSiluAndMulBuilder
            return DlinferSiluAndMulBuilder
        elif layer_type == OpType.RMSNorm:
            from .norm import DlinferRMSNormBuilder
            return DlinferRMSNormBuilder
        elif layer_type == OpType.LinearW8A8:
            from .qmodules import DlinferLinearW8A8Builder
            return DlinferLinearW8A8Builder
        elif layer_type == OpType.RMSNormW8A8:
            from .qmodules import DlinferRMSNormW8A8Builder
            return DlinferRMSNormW8A8Builder
        elif layer_type == OpType.SoftmaxTopK:
            from .moe import DlinferSoftmaxTopKBuilder
            return DlinferSoftmaxTopKBuilder
        elif layer_type == OpType.FusedMoE:
            from .moe import DlinferFusedMoEBuilder
            return DlinferFusedMoEBuilder
        elif layer_type == OpType.Linear:
            from .linear import DlinferLinearBuilder
            return DlinferLinearBuilder
        elif layer_type == OpType.LinearW4A16:
            from .awq_modules import AwqLinearW4A16Builder
            return AwqLinearW4A16Builder
        elif layer_type == OpType.RotaryEmbedding:
            from .rotary_embedding import DlinferRotaryEmbeddingBuilder
            return DlinferRotaryEmbeddingBuilder
        else:
            logger.debug(f'Op {layer_type} fallback to default implementation.')
            return super().get_layer_impl_builder(layer_type)

    @staticmethod
    def get_attention_metadata_cls():
        from .attention import DlinferAttentionMetadata
        return DlinferAttentionMetadata

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
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
        return (
            block_size,
            num_heads,
            head_size,
        )

    @classmethod
    def update_step_context(cls, step_context):
        """Update step context."""
        raise NotImplementedError
