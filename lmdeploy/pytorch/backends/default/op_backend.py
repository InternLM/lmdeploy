# Copyright (c) OpenMMLab. All rights reserved.

from typing import cast

import torch

from ..base import BuildSpec, ImplT, OpsBackend


class DefaultOpsBackend(OpsBackend):

    @staticmethod
    def get_name() -> str:
        return 'default'

    @classmethod
    def build_op(cls, spec: BuildSpec[ImplT], *, enable_deterministic: bool = False) -> ImplT:
        """Build a typed operator implementation."""
        from ..activation import GeluAndMulBuildSpec, SiluAndMulBuildSpec
        from ..apply_rotary_emb import ApplyRotaryEmbBuildSpec
        from ..awq_modules import LinearW4A16BuildSpec
        from ..cache_block_copy import CacheBlockCopyBuildSpec
        from ..embedding import EmbeddingBuildSpec
        from ..linear import LinearBuildSpec
        from ..moe import SoftmaxTopKBuildSpec
        from ..moe_router import RouterNoauxTCBuildSpec
        from ..multinomial_sampling import MultinomialSamplingBuildSpec
        from ..norm import LayerNormBuildSpec, RMSNormBuildSpec
        from ..rotary_embedding import RotaryEmbeddingBuildSpec
        if isinstance(spec, SiluAndMulBuildSpec):
            from .activation import DefaultSiluAndMulImpl
            return cast(ImplT, DefaultSiluAndMulImpl(spec.inplace))
        if isinstance(spec, GeluAndMulBuildSpec):
            from .activation import DefaultGeluAndMulImpl
            return cast(ImplT, DefaultGeluAndMulImpl(spec.approximate))
        if isinstance(spec, RotaryEmbeddingBuildSpec):
            from .rotary_embedding import _build_rotary_embedding
            return cast(ImplT, _build_rotary_embedding(spec))
        if isinstance(spec, ApplyRotaryEmbBuildSpec):
            from .apply_rotary_emb import DefaultApplyRotaryEmbImpl
            return cast(ImplT, DefaultApplyRotaryEmbImpl())
        if isinstance(spec, RMSNormBuildSpec):
            from .norm import DefaultRMSNormImpl
            return cast(ImplT, DefaultRMSNormImpl(spec.hidden_size, spec.eps))
        if isinstance(spec, LayerNormBuildSpec):
            from .norm import DefaultLayerNormImpl
            return cast(ImplT, DefaultLayerNormImpl(spec.normalized_shape, spec.eps))
        if isinstance(spec, MultinomialSamplingBuildSpec):
            from .multinomial_sampling import DefaultMultinomialSamplingImpl
            return cast(ImplT, DefaultMultinomialSamplingImpl())
        if isinstance(spec, SoftmaxTopKBuildSpec):
            from .moe import DefaultSoftmaxTopKImpl
            return cast(ImplT, DefaultSoftmaxTopKImpl(spec.top_k, spec.dim, n_groups=spec.n_groups))
        if isinstance(spec, EmbeddingBuildSpec):
            from .embedding import DefaultEmbeddingImpl
            return cast(ImplT, DefaultEmbeddingImpl(spec.start_index, spec.end_index))
        if isinstance(spec, CacheBlockCopyBuildSpec):
            from .cache_block_copy import _build_cache_block_copy
            return cast(ImplT, _build_cache_block_copy(spec))
        if isinstance(spec, RouterNoauxTCBuildSpec):
            from .moe_router import DefaultRouterNoauxTCImpl
            return cast(
                ImplT,
                DefaultRouterNoauxTCImpl(
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
        if isinstance(spec, LinearW4A16BuildSpec):
            from .awq_modules import DefaultLinearW4A16Impl
            return cast(
                ImplT,
                DefaultLinearW4A16Impl(
                    spec.in_features,
                    spec.out_features,
                    spec.w_bit,
                    spec.group_size,
                ),
            )
        if isinstance(spec, LinearBuildSpec):
            from .linear import DefaultLinearImpl
            return cast(ImplT, DefaultLinearImpl())
        spec_name = type(spec).__name__
        raise RuntimeError(f'Build spec {spec_name} is not supported by {cls.get_name()} backend.')

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> tuple[int, ...]:
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
    ) -> tuple[int, ...]:
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
