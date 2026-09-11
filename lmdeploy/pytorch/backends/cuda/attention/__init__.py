# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.backends.attention import PagedAttentionBuildSpec
from lmdeploy.utils import get_logger

from .default import TritonAttentionImpl
from .default import TritonAttentionMetadata as TritonAttentionMetadata
from .fa3_capabilities import fa3_build_supports_operation

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


def _enable_fa3(alibi: bool,
                learnable_sink: bool,
                block_sparse_size: int,
                head_size: int,
                v_head_size: int | None = None,
                sliding_window: tuple[int, int] | None = None,
                logit_softcapping: float = 0.0,
                dtype: torch.dtype | None = None) -> bool:
    """Check if FA3 should be enabled.

    FA3 is enabled when the attention features, current GPU architecture,
    and installed wheel build flags all support the complete implementation.
    Do not cache this decision: one process may construct models on devices
    with different compute capabilities.

    Returns:
        True if FA3 should be enabled, False otherwise.
    """
    if alibi or learnable_sink or block_sparse_size != 1:
        return False
    try:
        device_capability = torch.cuda.get_device_capability()
    except Exception:
        return False

    sliding_window = _normalize_sliding_window(sliding_window)
    supported = fa3_build_supports_operation(
        head_size,
        v_head_size,
        device_capability=device_capability,
        dtype=dtype,
        # FA3Impl serves varlen prefill and paged-KV decoding, so a selected
        # wheel must contain both operation families.
        paged_kv=True,
        varlen=True,
        local=sliding_window != (-1, -1),
        softcap=logit_softcapping > 0.0,
    )
    return supported and use_fa3_warning()


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
    2. enable_fa3: Select FA3Impl if FA3 is available and supported
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
    enable_fa3 = _enable_fa3(
        spec.alibi,
        spec.learnable_sink,
        spec.block_sparse_size,
        spec.head_dim,
        spec.v_head_dim,
        sliding_window,
        spec.logit_softcapping,
        spec.dtype,
    )
    # FlashMLA's FA3 prefill path uses its own split dimensions (rope/nope),
    # so the paged-attention head-shape capability check above does not apply.
    use_fa3_for_mla = use_fa3

    if spec.use_flash_mla is True:
        if spec.mla_index_topk is not None:
            if _envs.sparse_mla_backend == 'tilelang':
                logger.debug('Build TileLangSparseMLAImpl Attention')
                from .sparse_mla import TileLangSparseMLAImpl
                return TileLangSparseMLAImpl(
                    mla_index_topk=spec.mla_index_topk,
                    use_fa3=use_fa3_for_mla,
                    **common_args,
                )
            logger.debug('Build FlashMLASparseImpl Attention')
            from .sparse_mla import FlashMLASparseImpl
            return FlashMLASparseImpl(
                mla_index_topk=spec.mla_index_topk,
                use_fa3=use_fa3_for_mla,
                **common_args,
            )
        logger.debug('Build FlashMLAImpl Attention')
        from .mla import FlashMLAImpl
        return FlashMLAImpl(use_fa3=use_fa3_for_mla, **common_args)
    elif enable_fa3:
        logger.debug('Build FA3Impl Attention')
        from .fa3 import FA3Impl
        return FA3Impl(**common_args)
    else:
        logger.debug('Build TritonAttentionImpl Attention')
        return TritonAttentionImpl(block_sparse_size=spec.block_sparse_size, **common_args)
