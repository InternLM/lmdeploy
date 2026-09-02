# Copyright (c) OpenMMLab. All rights reserved.

import contextlib
from typing import cast

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
from lmdeploy.utils import get_logger

from ..base import BuildSpec, ImplT
from ..default import DefaultOpsBackend

logger = get_logger('lmdeploy')


class CudaOpsBackend(DefaultOpsBackend):
    """Cuda layer backend."""

    @staticmethod
    def get_name() -> str:
        """Backend name."""
        return 'cuda'

    @classmethod
    def build_op(cls, spec: BuildSpec[ImplT], *, enable_deterministic: bool = False) -> ImplT:
        """Build a typed CUDA operator implementation."""
        from ..activation import SiluAndMulBuildSpec
        from ..apply_rotary_emb import ApplyRotaryEmbBuildSpec
        from ..attention import PagedAttentionBuildSpec, V4AttentionBuildSpec
        from ..awq_modules import LinearW4A16BuildSpec
        from ..blockedf8_modules import LinearBlockedF8BuildSpec
        from ..cache_block_copy import CacheBlockCopyBuildSpec
        from ..causal_conv1d import CausalConv1dBuildSpec
        from ..compressor import V4CompressorBuildSpec
        from ..flash_attention import FlashAttentionBuildSpec
        from ..gated_delta_rule import GatedDeltaRuleBuildSpec
        from ..hc_prepost import HCPrePostBuildSpec
        from ..indexer import V4IndexerBuildSpec
        from ..lora import LoRABuildSpec
        from ..moe import (
            FusedMoEBlockedF8BuildSpec,
            FusedMoEBuildSpec,
            FusedMoEStaticF8BuildSpec,
            FusedMoEV4FP4BuildSpec,
            FusedMoEW4A16BuildSpec,
            FusedMoEW8A8BuildSpec,
        )
        from ..moe_router import RouterGemmBuildSpec, RouterNoauxTCBuildSpec
        from ..multinomial_sampling import MultinomialSamplingBuildSpec
        from ..norm import RMSNormBuildSpec
        from ..nsa import NSAIndexFP8BuildSpec
        from ..qmodules import LinearW8A8BuildSpec, RMSNormW8A8BuildSpec
        from ..rejection_sampling import RejectionSamplingBuildSpec
        from ..static_fp8_modules import LinearStaticF8BuildSpec
        if isinstance(spec, SiluAndMulBuildSpec):
            from .activation import TritonSiluAndMulImpl
            return cast(ImplT, TritonSiluAndMulImpl(spec.inplace))
        if isinstance(spec, ApplyRotaryEmbBuildSpec):
            from .apply_rotary_emb import TritonApplyRotaryEmbImpl
            return cast(ImplT, TritonApplyRotaryEmbImpl())
        if isinstance(spec, RMSNormBuildSpec):
            from .norm import TritonRMSNormImpl
            return cast(ImplT, TritonRMSNormImpl(spec.hidden_size, spec.eps))
        if isinstance(spec, RMSNormW8A8BuildSpec):
            from .qmodules import TritonRMSNormW8A8Impl
            return cast(ImplT, TritonRMSNormW8A8Impl(spec.hidden_size, spec.eps, spec.quant_dtype))
        if isinstance(spec, MultinomialSamplingBuildSpec):
            from .multinomial_sampling import TritonMultinomialSamplingImpl
            return cast(ImplT, TritonMultinomialSamplingImpl())
        if isinstance(spec, RejectionSamplingBuildSpec):
            from .rejection_sampling import CudaRejectionSamplingImpl
            return cast(ImplT, CudaRejectionSamplingImpl())
        if isinstance(spec, NSAIndexFP8BuildSpec):
            from .nsa import TritonNSAIndexFP8Impl
            return cast(
                ImplT,
                TritonNSAIndexFP8Impl(
                    spec.top_k,
                    spec.softmax_scale,
                    spec.block_size,
                    spec.fill,
                    allow_short_prefill_scoring_skip=spec.allow_short_prefill_scoring_skip,
                ),
            )
        if isinstance(spec, V4AttentionBuildSpec):
            from .attention.v4 import TritonV4AttentionImpl
            return cast(
                ImplT,
                TritonV4AttentionImpl(spec.head_dim, spec.scale, spec.window_size, spec.compress_ratio),
            )
        if isinstance(spec, V4IndexerBuildSpec):
            from .v4_indexer import TritonV4IndexerImpl
            return cast(
                ImplT,
                TritonV4IndexerImpl(
                    spec.index_top_k,
                    spec.compress_ratio,
                    spec.num_heads,
                    spec.head_dim,
                ),
            )
        if isinstance(spec, V4CompressorBuildSpec):
            from .v4_compressor import TritonV4CompressorImpl
            return cast(ImplT, TritonV4CompressorImpl(spec.compress_ratio, spec.overlap, spec.head_dim))
        if isinstance(spec, HCPrePostBuildSpec):
            from .hc_prepost import TritonHCPrePostImpl
            return cast(ImplT, TritonHCPrePostImpl(spec.hc_mult, spec.sinkhorn_iters, spec.eps))
        if isinstance(spec, RouterGemmBuildSpec):
            from .moe_router import CudaRouterGemmImpl
            return cast(ImplT, CudaRouterGemmImpl(out_dtype=spec.output_dtype))
        if isinstance(spec, RouterNoauxTCBuildSpec):
            from .moe_router import TritonRouterNoauxTCImpl
            return cast(
                ImplT,
                TritonRouterNoauxTCImpl(
                    scoring_func=spec.scoring_func,
                    top_k=spec.top_k,
                    n_group=spec.n_group,
                    topk_group=spec.top_k_group,
                    n_routed_experts=spec.n_routed_experts,
                    routed_scaling_factor=spec.routed_scaling_factor,
                    renormalize=spec.renormalize,
                    router_n_groups=spec.router_n_groups,
                ),
            )
        if isinstance(spec, CausalConv1dBuildSpec):
            from .causal_conv1d import _build_causal_conv1d
            return cast(ImplT, _build_causal_conv1d())
        if isinstance(spec, GatedDeltaRuleBuildSpec):
            from .gated_delta_rule import CudaGatedDeltaRuleImpl
            return cast(ImplT, CudaGatedDeltaRuleImpl())
        if isinstance(spec, CacheBlockCopyBuildSpec):
            from .cache_block_copy import CudaCacheBlockCopyImpl
            return cast(ImplT, CudaCacheBlockCopyImpl(spec.packed_caches, spec.pages_per_block))
        if isinstance(spec, LinearW4A16BuildSpec):
            from .awq_modules import _build_linear_w4a16
            return cast(ImplT, _build_linear_w4a16(spec))
        if isinstance(spec, LinearW8A8BuildSpec):
            from .qmodules import TritonLinearW8A8Impl
            return cast(
                ImplT,
                TritonLinearW8A8Impl(
                    spec.in_features,
                    spec.out_features,
                    spec.output_dtype,
                    spec.quant_dtype,
                ),
            )
        if isinstance(spec, LinearBlockedF8BuildSpec):
            from .blockedf8_modules import _build_linear_blocked_f8
            return cast(ImplT, _build_linear_blocked_f8(spec))
        if isinstance(spec, LinearStaticF8BuildSpec):
            from .static_fp8_modules import TritonLinearStaticF8Impl
            return cast(
                ImplT,
                TritonLinearStaticF8Impl(
                    spec.in_features,
                    spec.out_features,
                    out_dtype=spec.output_dtype,
                ),
            )
        if isinstance(spec, LoRABuildSpec):
            from .lora import TritonLoRAImpl
            return cast(ImplT, TritonLoRAImpl())
        if isinstance(spec, FusedMoEBuildSpec):
            from .moe.default import _build_fused_moe
            return cast(ImplT, _build_fused_moe(spec))
        if isinstance(spec, FusedMoEW4A16BuildSpec):
            from .moe.compressed_tensors import _build_fused_moe_w4a16
            return cast(ImplT, _build_fused_moe_w4a16(spec))
        if isinstance(spec, FusedMoEW8A8BuildSpec):
            from .moe.w8a8 import _build_fused_moe_w8a8
            return cast(ImplT, _build_fused_moe_w8a8(spec))
        if isinstance(spec, FusedMoEStaticF8BuildSpec):
            from .moe.static_fp8 import _build_fused_moe_static_f8
            return cast(ImplT, _build_fused_moe_static_f8(spec))
        if isinstance(spec, FusedMoEBlockedF8BuildSpec):
            from .moe.blocked_fp8 import _build_fused_moe_blocked_f8
            return cast(ImplT, _build_fused_moe_blocked_f8(spec))
        if isinstance(spec, FusedMoEV4FP4BuildSpec):
            from .moe.v4_fp4 import _build_fused_moe_v4_fp4
            return cast(ImplT, _build_fused_moe_v4_fp4(spec))
        if isinstance(spec, PagedAttentionBuildSpec):
            from .attention import _build_paged_attention
            return cast(ImplT, _build_paged_attention(spec))
        if isinstance(spec, FlashAttentionBuildSpec):
            from .flash_attention import TritonFlashAttentionImpl
            return cast(
                ImplT,
                TritonFlashAttentionImpl(
                    num_heads=spec.num_heads,
                    head_dim=spec.head_dim,
                    scale=spec.scale,
                    num_kv_heads=spec.num_kv_heads,
                    v_head_dim=spec.v_head_dim,
                    causal=spec.causal,
                    sliding_window=spec.sliding_window,
                    logit_softcapping=spec.logit_softcapping,
                ),
            )
        return super().build_op(spec, enable_deterministic=enable_deterministic)

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
