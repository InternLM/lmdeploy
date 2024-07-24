# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.utils import get_logger

from ..base import LayerType
from ..default import DefaultLayersBackend

logger = get_logger('lmdeploy')


class CudaLayersBackend(DefaultLayersBackend):

    @staticmethod
    def get_name() -> str:
        raise 'cuda'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: LayerType):
        if layer_type == LayerType.Attention:
            from .attention import TritonAttentionBuilder
            return TritonAttentionBuilder
        elif layer_type == LayerType.ApplyRotaryEmb:
            from .apply_rotary_emb import TritonApplyRotaryEmbBuilder
            return TritonApplyRotaryEmbBuilder
        elif layer_type == LayerType.RMSNorm:
            from .norm import TritonRMSNormBuilder
            return TritonRMSNormBuilder
        elif layer_type == LayerType.SLoRA:
            from .slora import TritonSLoRABuilder
            return TritonSLoRABuilder
        elif layer_type == LayerType.LinearW8A8:
            from .qmodules import TritonLinearW8A8Builder
            return TritonLinearW8A8Builder
        elif layer_type == LayerType.RMSNormW8A8:
            from .qmodules import TritonRMSNormBuilder
            return TritonRMSNormBuilder
        elif layer_type == LayerType.MultinomialSampling:
            from .multinomial_sampling import TritonMultinomialSamplingBuilder
            return TritonMultinomialSamplingBuilder
        elif layer_type == LayerType.LinearW4A16:
            from awq.modules.linear.gemm import AWQ_INSTALLED
            if AWQ_INSTALLED:
                from .awq_modules import AwqLinearW4A16Builder
                return AwqLinearW4A16Builder
            else:
                logger.debug(
                    f'Op {layer_type} fallback to default implementation.')
                return super().get_layer_impl_builder(layer_type)
        else:
            logger.debug(
                f'Op {layer_type} fallback to default implementation.')
            return super().get_layer_impl_builder(layer_type)

    @staticmethod
    def get_attention_metadata_cls():
        from .attention import TritonAttentionMetadata
        return TritonAttentionMetadata

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
        """update step context."""
        attn_meta_cls = cls.get_attention_metadata_cls()
        q_seqlens = step_context.q_seqlens
        q_start_loc = q_seqlens.cumsum(0) - q_seqlens
        attn_meta = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_seqlens=step_context.kv_seqlens,
        )

        step_context.attn_meta = attn_meta
        return step_context
