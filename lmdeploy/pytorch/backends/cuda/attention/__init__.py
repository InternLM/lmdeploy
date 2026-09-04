# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.backends.attention import PagedAttentionBuildSpec
from lmdeploy.utils import get_logger

<<<<<<< HEAD
from .default import TritonAttentionImpl
from .default import TritonAttentionMetadata as TritonAttentionMetadata
from .fa3_capabilities import fa3_build_supports_head_dims
from .v4 import TritonV4AttentionBuilder  # noqa: F401
=======
from .default import TritonAttentionImpl, TritonAttentionMetadata
from .fa3_capabilities import fa3_build_supports_head_dims
from .v4 import TritonV4AttentionBuilder  # noqa: F401
>>>>>>> b9d967e0 (feat(pytorch): support MiMo-V2-Flash inference)

logger = get_logger('lmdeploy')

use_fa3 = False
try:
    # flash-attention supports FA3 for sm80+ (Ampere and above) && cuda >= 12.3
    _cuda_ver = tuple(int(x) for x in torch.version.cuda.split('.')[:2]) if torch.version.cuda else (0, 0)
    if (torch.cuda.get_device_capability()[0] >= 8) and _cuda_ver >= (12, 3):
        import lmdeploy.pytorch.third_party.flash_attn_interface  # noqa: F401
        assert torch.ops.flash_attn_3 is not None
        use_fa3 = True
except Exception:
    logger.debug('For higher performance, please install FlashAttention-3 '
                 'https://github.com/Dao-AILab/flash-attention')


def require_fa3_for_speculative_decoding() -> None:
    """Require the FA3 backend used by multi-token speculative decode."""
    if use_fa3:
        return

    sm = torch.cuda.get_device_capability()
    cuda_ver = torch.version.cuda or 'N/A'
    raise RuntimeError(
        f'Speculative decoding on CUDA requires FlashAttention-3 (FA3), '
        f'which needs SM80+ (Ampere and above) with CUDA >= 12.3 and '
        f'flash-attn installed. Detected: SM{sm[0]}.{sm[1]}, CUDA {cuda_ver}. '
        f'Please ensure your GPU meets SM80+, CUDA >= 12.3, and flash-attn '
        f'is installed, or disable speculative decoding.')


@functools.lru_cache
def use_fa3_warning():
    if use_fa3:
        return True
    logger.warning('For higher performance, please install FlashAttention-3 '
                   'https://github.com/Dao-AILab/flash-attention')
    return False


@functools.lru_cache
def _enable_fa3(alibi: bool,
                learnable_sink: bool,
                block_sparse_size: int,
                head_size: int,
                v_head_size: int | None = None) -> bool:
    """Check if FA3 should be enabled.

    FA3 is enabled when:
    - No alibi
    - No learnable sink
    - block_sparse_size == 1
    - FA3 is available (checked by use_fa3_warning)

    Returns:
        True if FA3 should be enabled, False otherwise.
    """
    enable = (not alibi and not learnable_sink and block_sparse_size == 1
              and fa3_build_supports_head_dims(head_size, v_head_size))
    if enable and not use_fa3_warning():
        enable = False
    return enable


def _normalize_sliding_window(sliding_window):
    """Normalize sliding window to tuple format.

    Args:
        sliding_window: None, int, or tuple of (left, right).

    Returns:
        Tuple of (left, right) or (-1, -1) if None.
    """
    if sliding_window is None:
        return (-1, -1)
    if isinstance(sliding_window, int):
        return (sliding_window, sliding_window)
    return sliding_window


def _build_paged_attention(spec: PagedAttentionBuildSpec) -> TritonAttentionImpl:
    """Build the selected CUDA paged-attention implementation.

    Selection order:
    1. use_flash_mla: Use dense or sparse FlashMLA for MLA models
    2. enable_fa3: Use FA3Impl if FA3 is available and supported
    3. Default: Use TritonAttentionImpl as fallback
    """
    sliding_window = _normalize_sliding_window(spec.sliding_window)
    common_args = dict(
        num_heads=spec.num_heads,
        head_size=spec.head_dim,
        scale=spec.scale,
        num_kv_heads=spec.num_kv_heads,
        v_head_size=spec.v_head_dim,
        alibi=spec.alibi,
        sliding_window=sliding_window,
        logit_softcapping=spec.logit_softcapping,
        causal=spec.causal,
    )
    enable_fa3 = spec.enable_fa3 and _enable_fa3(spec.alibi, spec.learnable_sink, spec.block_sparse_size, spec.head_dim, spec.v_head_dim)

<<<<<<< HEAD
    if spec.use_flash_mla is True:
        if spec.mla_index_topk is not None:
            if _envs.sparse_mla_backend == 'tilelang':
                logger.debug('Build TileLangSparseMLAImpl Attention')
                from .sparse_mla import TileLangSparseMLAImpl
                return TileLangSparseMLAImpl(
                    mla_index_topk=spec.mla_index_topk,
                    use_fa3=use_fa3,
                    **common_args,
                )
            logger.debug('Build FlashMLASparseImpl Attention')
            from .sparse_mla import FlashMLASparseImpl
            return FlashMLASparseImpl(
                mla_index_topk=spec.mla_index_topk,
                use_fa3=use_fa3,
                **common_args,
            )
        logger.debug('Build FlashMLAImpl Attention')
        from .mla import FlashMLAImpl
        return FlashMLAImpl(use_fa3=use_fa3, **common_args)
    elif enable_fa3:
        logger.debug('Build FA3Impl Attention')
        from .fa3 import FA3Impl
        return FA3Impl(**common_args)
    else:
        logger.debug('Build TritonAttentionImpl Attention')
        return TritonAttentionImpl(block_sparse_size=spec.block_sparse_size, **common_args)
=======
    @staticmethod
    def build(
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logit_softcapping: float = 0.0,
        causal: bool = True,
        use_flash_mla: bool = False,
        mla_index_topk: int | None = None,
        learnable_sink: bool = False,
        block_sparse_size: int = 1,
        enable_fa3: bool = True,
        **kwargs,
    ) -> TritonAttentionImpl:
        """Build appropriate attention implementation.

        Args:
            num_heads: Number of attention heads.
            head_size: Size of each attention head.
            scale: Scaling factor for attention scores.
            num_kv_heads: Number of key-value heads (for GQA).
            v_head_size: Size of value head (for MLA).
            alibi: Whether to use ALiBi positional encoding.
            sliding_window: Sliding window size for local attention.
            logit_softcapping: Logit softcapping value (for Gemma 2).
            causal: Whether to use causal attention.
            use_flash_mla: Whether to use Flash MLA implementation.
            mla_index_topk: Sparse MLA top-k width, or ``None`` for dense MLA.
            learnable_sink: Whether to use learnable sink tokens.
            block_sparse_size: Block sparse attention size.
            enable_fa3: Whether this attention configuration may use FA3.
            **kwargs: Additional arguments.

        Returns:
            Appropriate AttentionImpl instance.
        """
        # Normalize sliding window format
        sliding_window = _normalize_sliding_window(sliding_window)

        # Common arguments for all implementations
        common_args = dict(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            v_head_size=v_head_size,
            alibi=alibi,
            sliding_window=sliding_window,
            logit_softcapping=logit_softcapping,
            causal=causal,
            **kwargs,
        )
        enable_fa3 = enable_fa3 and _enable_fa3(
            alibi, learnable_sink, block_sparse_size, head_size, v_head_size)

        if use_flash_mla is True:
            if mla_index_topk is not None:
                logger.debug('Build FlashMLASparseImpl Attention')
                from .sparse_mla import FlashMLASparseImpl
                return FlashMLASparseImpl(mla_index_topk=mla_index_topk,
                                          use_fa3=use_fa3,
                                          **common_args)
            logger.debug('Build FlashMLAImpl Attention')
            from .mla import FlashMLAImpl
            return FlashMLAImpl(use_fa3=use_fa3, **common_args)
        elif enable_fa3:
            logger.debug('Build FA3Impl Attention')
            from .fa3 import FA3Impl
            return FA3Impl(**common_args)
        else:
            logger.debug('Build TritonAttentionImpl Attention')
            return TritonAttentionImpl(block_sparse_size=block_sparse_size, **common_args)
>>>>>>> b9d967e0 (feat(pytorch): support MiMo-V2-Flash inference)
