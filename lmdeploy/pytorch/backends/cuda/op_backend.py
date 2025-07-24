# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Tuple

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ..base import OpType
from ..default import DefaultOpsBackend

logger = get_logger('lmdeploy')


def _get_meta_flashmla(kv_seqlens, num_attention_heads):
    """Get meta for flashmla."""
    import flash_mla
    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(kv_seqlens.to(torch.int32), num_attention_heads, 1)
    return tile_scheduler_metadata, num_splits


class CudaOpsBackend(DefaultOpsBackend):
    """Cuda layer backend."""

    @staticmethod
    def get_name() -> str:
        """Backend name."""
        return 'cuda'

    @classmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """Get cuda layer builder."""
        if layer_type == OpType.PagedAttention:
            from .attention import TritonAttentionBuilder
            return TritonAttentionBuilder
        elif layer_type == OpType.FlashAttention:
            from .flash_attention import TritonFlashAttentionBuilder
            return TritonFlashAttentionBuilder
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
            from .awq_modules import AwqLinearW4A16Builder
            return AwqLinearW4A16Builder
        elif layer_type == OpType.FusedMoE:
            from .moe import TritonFusedMoEBuilder
            return TritonFusedMoEBuilder
        elif layer_type == OpType.FusedMoEW8A8:
            from .moe import TritonFusedMoEW8A8Builder
            return TritonFusedMoEW8A8Builder
        elif layer_type == OpType.FusedMoEBlockedF8:
            from .moe import TritonFusedMoEBlockedF8Builder
            return TritonFusedMoEBlockedF8Builder
        elif layer_type == OpType.LinearBlockedF8:
            from .blockedf8_modules import TritonLinearBlockedF8Builder
            return TritonLinearBlockedF8Builder
        else:
            logger.debug(f'Op {layer_type} fallback to default implementation.')
            return super().get_layer_impl_builder(layer_type)

    @staticmethod
    def get_attention_metadata_cls():
        """Get attention metadata class."""
        from .attention import TritonAttentionMetadata
        return TritonAttentionMetadata

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """Get k block shape."""
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
        """Get v block shape."""
        return (
            block_size,
            num_heads,
            head_size,
        )

    @classmethod
    def update_meta_flashmla(cls, attn_metadata, num_attention_heads):
        """Update meta for flashmla."""
        tile_scheduler_metadata, num_splits = _get_meta_flashmla(attn_metadata.kv_seqlens.to(torch.int32),
                                                                 num_attention_heads)
        attn_metadata.tile_scheduler_metadata = tile_scheduler_metadata
        attn_metadata.num_splits = num_splits

        if attn_metadata.block_offsets.dtype != torch.int32:
            attn_metadata.block_offsets = attn_metadata.block_offsets.to(torch.int32)

    @classmethod
    def update_step_context(cls, step_context):
        """Update step context."""
        attn_meta_cls = cls.get_attention_metadata_cls()
        q_seqlens = step_context.q_seqlens
        q_start_loc = q_seqlens.cumsum(0) - q_seqlens
        kv_seqlens = step_context.kv_seqlens
        kv_start_loc = None
        kv_flatten_size = None
        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(q_seqlens, dim=0, dtype=torch.int32), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(kv_seqlens, dim=0, dtype=torch.int32), (1, 0))
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
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
        )
        if getattr(step_context.model_config, 'use_flash_mla', False) is True:
            if step_context.is_decoding is True:
                cls.update_meta_flashmla(attn_metadata, step_context.model_config.num_attention_heads)

        cross_seqlens = step_context.cross_seqlens
        cross_kv_seqlens = step_context.cross_kv_seqlens
        cross_attn_metadata = None
        if cross_seqlens is not None:
            fill_seqlens = cross_seqlens
            if fill_seqlens.sum().item() == 0:
                fill_seqlens = None
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
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                           backend_config: BackendConfig, device: torch.device):
        """Build graph runner."""
        from lmdeploy.pytorch import envs

        from .graph_runner import CUDAGraphRunner
        from .warmup_manager import WarmupMeta, get_warmup_manager

        # warmup ops.
        warmup_meta = WarmupMeta(
            max_num_tokens=cache_config.max_prefill_token_num,
            max_batch_size=cache_config.max_batches,
            dtype=model_config.dtype,
        )
        get_warmup_manager().warmup(warmup_meta)

        # add custom triton cache manager
        if envs.triton_custom_cache_mgr_enable:
            os.environ['TRITON_CACHE_MANAGER'] = 'lmdeploy.pytorch.kernels.cuda.triton_utils:MPLockCacheManager'

        # make graph runner.
        return CUDAGraphRunner(model, model_config, cache_config, backend_config, device)

    @staticmethod
    def device_count():
        """Get num available devices."""
        return torch.cuda.device_count()

    @staticmethod
    def support_ray():
        """Support ray."""
        return True
