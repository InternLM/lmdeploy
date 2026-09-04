# Copyright (c) OpenMMLab. All rights reserved.

import contextlib

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
from lmdeploy.utils import get_logger

from ..base import OpType
from ..default import DefaultOpsBackend

logger = get_logger('lmdeploy')


class CudaOpsBackend(DefaultOpsBackend):
    """Cuda layer backend."""

    @staticmethod
    def get_name() -> str:
        """Backend name."""
        return 'cuda'

    @classmethod
    def get_cache_backend(cls):
        """Get CUDA cache layouts and local primitives."""
        from .cache import CudaCacheBackend
        return CudaCacheBackend

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
        elif layer_type == OpType.RejectionSampling:
            from .rejection_sampling import CudaRejectionSamplingBuilder
            return CudaRejectionSamplingBuilder
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
        elif layer_type == OpType.FusedMoEW4A16:
            from .moe import TritonFusedMoEW4A16Builder
            return TritonFusedMoEW4A16Builder
        elif layer_type == OpType.FusedMoEStaticF8:
            from .moe import TritonFusedMoEStaticF8Builder
            return TritonFusedMoEStaticF8Builder
        elif layer_type == OpType.FusedMoEBlockedF8:
            from .moe import TritonFusedMoEBlockedF8Builder
            return TritonFusedMoEBlockedF8Builder
        elif layer_type == OpType.FusedMoEV4FP4:
            from .moe import TritonFusedMoEV4FP4Builder
            return TritonFusedMoEV4FP4Builder
        elif layer_type == OpType.LinearStaticF8:
            from .static_fp8_modules import (
                TritonLinearStaticF8Builder,
            )
            return TritonLinearStaticF8Builder
        elif layer_type == OpType.LinearBlockedF8:
            from .blockedf8_modules import CudaLinearBlockedF8Builder
            return CudaLinearBlockedF8Builder
        elif layer_type == OpType.NSAIndexFP8:
            from .nsa import TritonNSAIndexFP8Builder
            return TritonNSAIndexFP8Builder
        elif layer_type == OpType.V4Attention:
            from .attention import TritonV4AttentionBuilder
            return TritonV4AttentionBuilder
        elif layer_type == OpType.V4Indexer:
            from .v4_indexer import TritonV4IndexerBuilder
            return TritonV4IndexerBuilder
        elif layer_type == OpType.V4Compressor:
            from .v4_compressor import TritonV4CompressorBuilder
            return TritonV4CompressorBuilder
        elif layer_type == OpType.HcPrePost:
            from .hc_prepost import TritonHcPrePostBuilder
            return TritonHcPrePostBuilder
        elif layer_type == OpType.RouterGemm:
            from .moe_router import CudaRouterGemmBuilder
            return CudaRouterGemmBuilder
        elif layer_type == OpType.RouterNoauxTC:
            from .moe_router import TritonRouterNoauxTCBuilder
            return TritonRouterNoauxTCBuilder
        elif layer_type == OpType.CausalConv1d:
            from .causal_conv1d import CausalConv1dCudaBuilder
            return CausalConv1dCudaBuilder
        elif layer_type == OpType.GatedDeltaRule:
            from .gated_delta_rule import CudaGatedDeltaRuleBuilder
            return CudaGatedDeltaRuleBuilder
        else:
            logger.debug(f'Op {layer_type} fallback to default implementation.')
            return super().get_layer_impl_builder(layer_type)

    @staticmethod
    def get_attention_metadata_cls():
        """Get attention metadata class."""
        from .attention import TritonAttentionMetadata
        return TritonAttentionMetadata

    @staticmethod
    def get_v4_attention_metadata_cls():
        """Get V4 attention metadata class."""
        from .attention.v4 import CudaV4AttentionMetadata
        return CudaV4AttentionMetadata

    @classmethod
    def build_communicator(cls, cpu_group, device_group, dist_config):
        """Build a CUDA communicator."""
        from .comm.communicator import build_cuda_communicator
        communicator = build_cuda_communicator(
            cpu_group=cpu_group,
            device_group=device_group,
            dist_config=dist_config,
        )
        if communicator is not None:
            return communicator
        return super().build_communicator(
            cpu_group=cpu_group,
            device_group=device_group,
            dist_config=dist_config,
        )

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> tuple[int, ...]:
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
    ) -> tuple[int, ...]:
        """Get v block shape."""
        return (
            block_size,
            num_heads,
            head_size,
        )

    @classmethod
    def update_meta_flashmla(cls, attn_metadata, model_config: ModelConfig, decoding_query_len: int):
        """Update meta for flashmla."""
        from .attention.mla import needs_flash_mla_scheduler, update_flash_mla_metadata

        is_fp8_kvcache = model_config.use_mla_fp8_cache
        index_topk = model_config.mla_index_topk
        # FlashMLA block tables and BF16 sparse physical-slot indices are int32.
        if attn_metadata.block_offsets.dtype != torch.int32:
            attn_metadata.block_offsets = attn_metadata.block_offsets.to(torch.int32)
        if not needs_flash_mla_scheduler(is_fp8_kvcache, index_topk):
            return

        num_attention_heads, _ = model_config.get_num_qkv_head_by_tp()
        update_flash_mla_metadata(
            attn_metadata,
            num_attention_heads=num_attention_heads,
            decoding_query_len=decoding_query_len,
            is_fp8_kvcache=is_fp8_kvcache,
            index_topk=index_topk,
        )

    @classmethod
    def update_meta_flashattn(cls, attn_metadata, step_context):
        from .attention.fa3 import update_fa3_metadata

        num_attention_heads, num_key_value_heads = step_context.model_config.get_num_qkv_head_by_tp()
        update_fa3_metadata(
            attn_metadata,
            step_context,
            num_heads_q=num_attention_heads,
            num_heads_kv=num_key_value_heads,
            head_size=step_context.model_config.head_dim,
            sliding_window=step_context.model_config.sliding_window,
        )
        return attn_metadata

    @classmethod
    def update_chunked_gated_delta_rule_meta(cls, attn_metadata):
        from .gated_delta_rule import prepare_chunked_gated_delta_rule

        prepare_chunked_gated_delta_rule(attn_metadata.cu_seqlens_q)

    @classmethod
    def _legacy_update_step_context(cls, step_context, attn_metadata):
        """Preserve the previous model-config-driven CUDA preparation path."""
        q_seqlens = step_context.q_seqlens
        use_flash_mla = step_context.model_config.use_flash_mla
        use_flash_attn3_decoding = step_context.model_config.model_paradigm == 'ar_spec'

        if step_context.is_decoding:
            if use_flash_mla:
                model_config = step_context.model_config
                decode_query_len = step_context.input_ids.size(1) // q_seqlens.size(0)
                cls.update_meta_flashmla(attn_metadata, model_config, decode_query_len)
            elif use_flash_attn3_decoding:
                from .attention import use_fa3
                if not use_fa3:
                    sm = torch.cuda.get_device_capability()
                    cuda_ver = torch.version.cuda or 'N/A'
                    raise RuntimeError(
                        f'Speculative decoding on CUDA requires FlashAttention-3 (FA3), '
                        f'which needs SM80+ (Ampere and above) with CUDA >= 12.3 and '
                        f'flash-attn installed. Detected: SM{sm[0]}.{sm[1]}, CUDA {cuda_ver}. '
                        f'Please ensure your GPU meets SM80+, CUDA >= 12.3, and flash-attn '
                        f'is installed, or disable speculative decoding.')
                cls.update_meta_flashattn(attn_metadata, step_context)

        if step_context.model_config.is_gated_delta and not step_context.is_decoding:
            cls.update_chunked_gated_delta_rule_meta(attn_metadata)

        return attn_metadata

    @staticmethod
    def _resolve_step_meta_plan():
        """Return the supported implementation-derived plan, if available."""
        from .step_metadata import CudaStepMetaPlan

        ctx_mgr = get_step_ctx_manager()
        plan = getattr(ctx_mgr, 'backend_step_meta_plan', None)
        if isinstance(plan, CudaStepMetaPlan) and plan.is_supported:
            return plan
        return None

    @classmethod
    def update_step_context(cls, step_context):
        """Update step context."""
        from .attention.default import build_triton_attention_metadata
        from .step_metadata import CudaSequenceMetadata

        sequence_metadata = CudaSequenceMetadata.from_step_context(step_context)
        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = build_triton_attention_metadata(attn_meta_cls, step_context, sequence_metadata)

        plan = cls._resolve_step_meta_plan()
        if plan is not None:
            plan.prepare(step_context, sequence_metadata, attn_metadata)
        else:
            cls._legacy_update_step_context(step_context, attn_metadata)

        step_context.attn_metadata = attn_metadata
        return step_context

    @staticmethod
    @contextlib.contextmanager
    def model_build_context(ctx_mgr):
        """Collect metadata contracts while constructing one CUDA model."""
        from .step_metadata import collect_step_metadata
        with collect_step_metadata(ctx_mgr):
            yield

        plan = ctx_mgr.backend_step_meta_plan
        if not plan.is_supported:
            logger.debug('Use legacy CUDA step metadata preparation: %s', plan.fallback_reason)

    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                           backend_config: BackendConfig, device: torch.device):
        """Build graph runner."""
        from .graph_runner import CUDAGraphRunner
        from .warmup_manager import WarmupMeta, get_warmup_manager

        # warmup ops.
        warmup_meta = WarmupMeta(
            max_num_tokens=cache_config.max_prefill_token_num,
            max_batch_size=cache_config.max_batches,
            dtype=model_config.dtype,
            model_config=model_config,
        )
        get_warmup_manager().warmup(warmup_meta)

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
