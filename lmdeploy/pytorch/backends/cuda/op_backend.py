# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
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
        elif layer_type == OpType.NSAIndexFP8:
            from .nsa import TritonNSAIndexFP8Builder
            return TritonNSAIndexFP8Builder
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
        import flash_mla
        num_attention_heads = model_config.num_attention_heads * decoding_query_len
        is_fp8_kvcache = model_config.use_mla_fp8_cache
        index_topk = model_config.mla_index_topk
        num_heads_q = None if index_topk is None else num_attention_heads
        tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(attn_metadata.kv_seqlens.to(torch.int32),
                                                                         num_attention_heads,
                                                                         num_heads_k=1,
                                                                         num_heads_q=num_heads_q,
                                                                         is_fp8_kvcache=is_fp8_kvcache,
                                                                         topk=index_topk)
        attn_metadata.tile_scheduler_metadata = tile_scheduler_metadata
        attn_metadata.num_splits = num_splits

        if attn_metadata.block_offsets.dtype != torch.int32:
            attn_metadata.block_offsets = attn_metadata.block_offsets.to(torch.int32)

    @classmethod
    def update_meta_flashattn(cls, attn_metadata, step_context):
        from lmdeploy.pytorch.models.utils.cudagraph import _get_meta_flashattn
        batch_size = attn_metadata.q_seqlens.size(0)
        max_seqlen_q = step_context.input_ids.size(1) // batch_size
        window_size = (step_context.model_config.sliding_window, ) * 2
        # FA3's mha_fwd internally computes seqlen_k as
        # page_table.size(1) * page_size when cu_seqlens_k is None
        # (paged KV without varlen_k). get_scheduler_metadata must use
        # the same value to produce a correctly-sized metadata buffer.
        max_seqlen_k = attn_metadata.block_offsets.size(1) * step_context.model_config.block_size
        scheduler_metadata = _get_meta_flashattn(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads_q=step_context.model_config.num_attention_heads,
            num_heads_kv=step_context.model_config.num_key_value_heads,
            headdim=step_context.model_config.head_dim,
            cache_seqlens=attn_metadata.kv_seqlens.to(torch.int32),
            qkv_dtype=step_context.model_config.dtype,
            page_size=step_context.model_config.block_size,
            window_size=window_size,
        )
        attn_metadata.scheduler_metadata = scheduler_metadata
        attn_metadata.max_kv_seqlen = max_seqlen_k
        return attn_metadata

    @classmethod
    def update_chunked_gated_delta_rule_meta(cls, attn_metadata, step_context):
        try:
            from fla.ops.utils import prepare_chunk_indices
        except ImportError:
            logger.warning(
                'Failed to import fla.ops.utils.prepare_chunk_indices for gated delta rule. '
                'Please make sure the version of fla is installed, up to date and compatible with lmdeploy.'
            )
        else:
            # prepare_chunk_indices would force sync the stream
            # we better maintain a cpu cu_seqlens_q cache in the future to avoid this
            try:
                # 64 is the default value that hard code inside fla
                # so we have to hard code it here.
                cu_seqlens_q = attn_metadata.cu_seqlens_q
                chunk_size = 64
                try:
                    prepare_chunk_indices(cu_seqlens_q, chunk_size, cu_seqlens_cpu=None)
                except TypeError:
                    prepare_chunk_indices(cu_seqlens_q, chunk_size)
            except Exception as exc:
                logger.exception(
                    'Unexpected error while preparing chunk indices for gated delta rule. '
                    'Please make sure the version of fla is up to date and compatible with lmdeploy.'
                )
                raise RuntimeError(
                    'Failed to prepare chunk indices for gated delta rule; see logs for details.'
                ) from exc

    @classmethod
    def update_step_context(cls, step_context):
        """Update step context."""
        attn_meta_cls = cls.get_attention_metadata_cls()
        q_seqlens = step_context.q_seqlens
        kv_seqlens = step_context.kv_seqlens
        kv_start_loc = None
        kv_flatten_size = None
        use_flash_mla = step_context.model_config.use_flash_mla
        use_flash_attn3_decoding = step_context.model_config.model_paradigm == 'ar_spec'

        # pad and cumsum requires 4 kernels, so we fuse seqlens cumsum into one kernel
        seqlens = torch.stack([q_seqlens, kv_seqlens], dim=0)
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=1, dtype=torch.int32), (1, 0))
        cu_seqlens_q = cu_seqlens[0]
        cu_seqlens_k = cu_seqlens[1]
        q_start_loc = step_context.q_start_loc
        if not step_context.is_decoding:
            kv_start_loc = cu_seqlens_k[:-1].to(kv_seqlens.dtype)
            kv_flatten_size = step_context.sum_kv_seqlen

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
            max_kv_seqlen=step_context.max_kv_seqlen,
        )
        if step_context.is_decoding:
            if use_flash_mla:
                model_config = step_context.model_config
                decode_query_len = step_context.input_ids.size(1) // q_seqlens.size(0)
                cls.update_meta_flashmla(attn_metadata, model_config, decode_query_len)
            elif use_flash_attn3_decoding:
                attn_metadata = cls.update_meta_flashattn(attn_metadata, step_context)

        # update chunk gated delta indices
        is_gated_delta = step_context.model_config.is_gated_delta
        if is_gated_delta and not step_context.is_decoding:
            cls.update_chunked_gated_delta_rule_meta(attn_metadata, step_context)

        step_context.attn_metadata = attn_metadata
        return step_context

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
