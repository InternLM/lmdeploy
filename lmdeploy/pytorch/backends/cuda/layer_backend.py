# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from ..base import LayerType
from ..default import DefaultLayersBackend


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
        else:
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
        attn_meta = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
            q_start_loc=step_context.q_start_loc,
            q_seqlens=step_context.q_seq_length,
            kv_seqlens=step_context.kv_seq_length,
            max_q_seqlen=step_context.max_q_seq_length,
            max_kv_seqlen=step_context.max_kv_seq_length,
        )

        step_context.attn_meta = attn_meta
        return step_context
