# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch

from lmdeploy.pytorch.backends.attention import PagedAttentionBuildSpec
from lmdeploy.utils import get_logger

from .default import TritonAttentionImpl
from .default import TritonAttentionMetadata as TritonAttentionMetadata

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


@functools.lru_cache
def use_fa3_warning():
    if use_fa3:
        return True
    logger.warning('For higher performance, please install FlashAttention-3 '
                   'https://github.com/Dao-AILab/flash-attention')
    return False


@functools.lru_cache
def _enable_fa3(alibi: bool, learnable_sink: bool, block_sparse_size: int, head_size: int) -> bool:
    """Check if FA3 should be enabled.

    FA3 is enabled when:
    - No alibi
    - No learnable sink
    - block_sparse_size == 1
    - FA3 is available (checked by use_fa3_warning)

    Returns:
        True if FA3 should be enabled, False otherwise.
    """
    enable = not alibi and not learnable_sink and block_sparse_size == 1 and head_size <= 256
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


def build_paged_attention(spec: PagedAttentionBuildSpec) -> TritonAttentionImpl:
    """Build the selected CUDA paged-attention implementation.

    Selection order:
    1. use_flash_mla: Use FlashMLAImpl for MLA models
    2. enable_fa3: Use FA3Impl if FA3 is available and supported
    3. Default: Use TritonAttentionImpl as fallback
    """
    sliding_window = _normalize_sliding_window(spec.sliding_window)
    common_args = dict(
        num_heads=spec.num_heads,
        head_size=spec.head_size,
        scale=spec.scale,
        num_kv_heads=spec.num_kv_heads,
        v_head_size=spec.v_head_size,
        alibi=spec.alibi,
        sliding_window=sliding_window,
        logit_softcapping=spec.logit_softcapping,
        causal=spec.causal,
    )
    enable_fa3 = _enable_fa3(spec.alibi, spec.learnable_sink, spec.block_sparse_size, spec.head_size)

    if spec.use_flash_mla is True:
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
