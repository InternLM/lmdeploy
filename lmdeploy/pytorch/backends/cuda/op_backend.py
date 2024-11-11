# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ..base import OpType
from ..default import DefaultOpsBackend

logger = get_logger('lmdeploy')


class CudaOpsBackend(DefaultOpsBackend):
    """cuda layer backend."""

    @staticmethod
    def get_name() -> str:
        """backend name."""
        return 'cuda'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """get cuda layer builder."""
        if layer_type == OpType.Attention:
            from .attention import TritonAttentionBuilder
            return TritonAttentionBuilder
        elif layer_type == OpType.ApplyRotaryEmb:
            from .apply_rotary_emb import TritonApplyRotaryEmbBuilder
            return TritonApplyRotaryEmbBuilder
        elif layer_type == OpType.RMSNorm:
            from .norm import TritonRMSNormBuilder
            return TritonRMSNormBuilder
        elif layer_type == OpType.LoRA:
            from .lora import TritonLoRABuilder
            return TritonLoRABuilder
        elif layer_type == OpType.LinearW8A8:
            from .qmodules import TritonLinearW8A8Builder
            return TritonLinearW8A8Builder
        elif layer_type == OpType.RMSNormW8A8:
            from .qmodules import TritonRMSNormBuilder
            return TritonRMSNormBuilder
        elif layer_type == OpType.MultinomialSampling:
            from .multinomial_sampling import TritonMultinomialSamplingBuilder
            return TritonMultinomialSamplingBuilder
        elif layer_type == OpType.SiluAndMul:
            from .activation import TritonSiluAndMulBuilder
            return TritonSiluAndMulBuilder
        elif layer_type == OpType.LinearW4A16:
            from awq.modules.linear.gemm import AWQ_INSTALLED
            if AWQ_INSTALLED:
                from .awq_modules import AwqLinearW4A16Builder
                return AwqLinearW4A16Builder
            else:
                logger.debug(
                    f'Op {layer_type} fallback to default implementation.')
                return super().get_layer_impl_builder(layer_type)
        elif layer_type == OpType.FusedMoE:
            from .moe import TritonFusedMoEBuilder
            return TritonFusedMoEBuilder
        else:
            logger.debug(
                f'Op {layer_type} fallback to default implementation.')
            return super().get_layer_impl_builder(layer_type)

    @staticmethod
    def get_attention_metadata_cls():
        """get attention metadata class."""
        from .attention import TritonAttentionMetadata
        return TritonAttentionMetadata

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """get k block shape."""
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
        """get v block shape."""
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
        kv_seqlens = step_context.kv_seqlens
        kv_start_loc = None
        kv_flatten_size = None
        if not step_context.is_decoding:
            kv_start_loc = kv_seqlens.cumsum(0) - kv_seqlens
            kv_flatten_size = kv_seqlens.sum().item()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_start_loc=kv_start_loc,
            kv_seqlens=kv_seqlens,
            kv_flatten_size=kv_flatten_size,
            quant_policy=step_context.kv_quant_policy,
        )

        cross_attn_metadata = None
        fill_seqlens = None
        if step_context.cross_attention_states is not None:
            fill_seqlens = torch.zeros_like(q_seqlens)
            for idx, state in enumerate(step_context.cross_attention_states):
                if state is not None:
                    fill_seqlens[idx] = state.shape[-2]
        cross_kv_seqlens = step_context.cross_kv_seqlens
        cross_kv_start_loc = None
        cross_kv_flatten_size = None
        if not step_context.is_decoding and cross_kv_seqlens is not None:
            cross_kv_start_loc = cross_kv_seqlens.cumsum(0) - cross_kv_seqlens
            cross_kv_flatten_size = cross_kv_seqlens.sum().item()
        cross_attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
            q_start_loc=q_start_loc,
            q_seqlens=q_seqlens,
            kv_start_loc=cross_kv_start_loc,
            kv_seqlens=cross_kv_seqlens,
            kv_flatten_size=cross_kv_flatten_size,
            fill_seqlens=fill_seqlens,
            quant_policy=step_context.kv_quant_policy,
        )

        step_context.attn_metadata = attn_metadata
        step_context.cross_attn_metadata = cross_attn_metadata
        return step_context

    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig,
                           cache_config: CacheConfig,
                           backend_config: BackendConfig,
                           device: torch.device):
        """build graph runner."""
        from .graph_runner import CUDAGraphRunner
        return CUDAGraphRunner(model, model_config, cache_config,
                               backend_config, device)
