# Copyright (c) OpenMMLab. All rights reserved.

from typing import cast

import torch

from ..base import BuildSpec, ImplT
from ..default import DefaultOpsBackend


class DlinferOpsBackend(DefaultOpsBackend):
    """Dlinfer layer backend."""

    @staticmethod
    def get_name() -> str:
        """Backend name."""
        return 'dlinfer'

    @classmethod
    def build_op(cls, spec: BuildSpec[ImplT], *, enable_deterministic: bool = False) -> ImplT:
        """Build a typed dlinfer operator implementation."""
        from ..activation import SiluAndMulBuildSpec
        from ..apply_rotary_emb import ApplyRotaryEmbBuildSpec
        from ..attention import PagedAttentionBuildSpec
        from ..awq_modules import LinearW4A16BuildSpec
        from ..flash_attention import FlashAttentionBuildSpec
        from ..linear import LinearBuildSpec
        from ..moe import FusedMoEBuildSpec, SoftmaxTopKBuildSpec
        from ..norm import RMSNormBuildSpec
        from ..qmodules import LinearW8A8BuildSpec, RMSNormW8A8BuildSpec
        from ..rotary_embedding import RotaryEmbeddingBuildSpec
        if isinstance(spec, SiluAndMulBuildSpec):
            from .activation import DlinferSiluAndMulImpl
            return cast(ImplT, DlinferSiluAndMulImpl())
        if isinstance(spec, ApplyRotaryEmbBuildSpec):
            from .apply_rotary_emb import DlinferApplyRotaryEmbImpl
            return cast(ImplT, DlinferApplyRotaryEmbImpl())
        if isinstance(spec, RMSNormBuildSpec):
            from .norm import DlinferRMSNormImpl
            return cast(ImplT, DlinferRMSNormImpl(spec.hidden_size, spec.eps))
        if isinstance(spec, RMSNormW8A8BuildSpec):
            from .qmodules import DlinferRMSNormW8A8Impl
            return cast(ImplT, DlinferRMSNormW8A8Impl(spec.hidden_size, spec.eps, spec.quant_dtype))
        if isinstance(spec, SoftmaxTopKBuildSpec):
            from .moe import DlinferSoftmaxTopKImpl
            return cast(ImplT, DlinferSoftmaxTopKImpl(spec.top_k, spec.dim, spec.n_groups))
        if isinstance(spec, RotaryEmbeddingBuildSpec):
            from .rotary_embedding import _build_rotary_embedding
            return cast(ImplT, _build_rotary_embedding(spec))
        if isinstance(spec, LinearW4A16BuildSpec):
            from .awq_modules import AwqLinearW4A16Impl
            return cast(
                ImplT,
                AwqLinearW4A16Impl(
                    spec.in_features,
                    spec.out_features,
                    spec.w_bit,
                    spec.group_size,
                ),
            )
        if isinstance(spec, LinearW8A8BuildSpec):
            from .qmodules import DlinferLinearW8A8Impl
            return cast(
                ImplT,
                DlinferLinearW8A8Impl(
                    spec.in_features,
                    spec.out_features,
                    spec.output_dtype,
                    spec.quant_dtype,
                ),
            )
        if isinstance(spec, PagedAttentionBuildSpec):
            from .attention import DlinferAttentionImpl
            return cast(
                ImplT,
                DlinferAttentionImpl(
                    num_heads=spec.num_heads,
                    head_size=spec.head_dim,
                    scale=spec.scale,
                    num_kv_heads=spec.num_kv_heads,
                    v_head_size=spec.v_head_dim,
                    alibi=spec.alibi,
                    sliding_window=spec.sliding_window,
                    logit_softcapping=spec.logit_softcapping,
                    causal=spec.causal,
                    use_flash_mla=spec.use_flash_mla,
                ),
            )
        if isinstance(spec, FlashAttentionBuildSpec):
            from .flash_attention import DlinferFlashAttentionImpl
            return cast(
                ImplT,
                DlinferFlashAttentionImpl(
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
        if isinstance(spec, LinearBuildSpec):
            from .linear import DlinferLinearImpl
            return cast(ImplT, DlinferLinearImpl())
        if isinstance(spec, FusedMoEBuildSpec):
            from .moe import _build_fused_moe
            return cast(ImplT, _build_fused_moe(spec))
        return super().build_op(spec, enable_deterministic=enable_deterministic)

    @classmethod
    def build_communicator(cls, cpu_group, device_group, dist_config):
        """Build a DLInfer communicator."""
        from lmdeploy.pytorch import envs
        cuda_communicator_enabled = envs.enable_flashinfer_allreduce or envs.enable_symm_mem_allreduce
        assert not cuda_communicator_enabled, 'CUDA communicators are not supported by DLInfer.'
        return super().build_communicator(
            cpu_group=cpu_group,
            device_group=device_group,
            dist_config=dist_config,
        )

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
    ) -> tuple[int, ...]:
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
        return (
            block_size,
            num_heads,
            head_size,
        )

    @classmethod
    def update_step_context(cls, step_context):
        """Update step context."""
        raise NotImplementedError
