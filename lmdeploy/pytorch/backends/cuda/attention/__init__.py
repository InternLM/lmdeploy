# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch

from lmdeploy.pytorch.backends.attention import AttentionBuilder
from lmdeploy.utils import get_logger

from .default import TritonAttentionImpl, TritonAttentionMetadata
from .v4 import TritonV4AttentionBuilder  # noqa: F401

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


def _fa3_build_supports_head_dims(head_size: int, v_head_size: int | None = None) -> bool:
    """Return whether the installed FA3 wheel contains a required head shape.

    FlashAttention-3 wheels may omit template instantiations to reduce build
    time and binary size.  In particular, MiMo-V2-Flash needs the asymmetric
    ``Q/K=192, V=128`` instantiation, which is controlled by both HDIM192 and
    HDIMDIFF192 build flags.

    A wheel without build metadata is accepted only for symmetric head shapes,
    preserving compatibility with older FA3 packages while avoiding unsafe
    asymmetric dispatch.
    """
    if v_head_size is None:
        v_head_size = head_size

    try:
        from flash_attn_config import CONFIG
        flags = CONFIG['build_flags']
    except (ImportError, KeyError, TypeError):
        return head_size == v_head_size

    def _disabled(name: str) -> bool:
        # Older generated configs used FLASH_ATTENTION for HDIMDIFF while the
        # other flags use FLASHATTENTION.  Accept both spellings.
        aliases = (name, name.replace('FLASHATTENTION_DISABLE_HDIMDIFF',
                                      'FLASH_ATTENTION_DISABLE_HDIMDIFF'))
        return any(bool(flags.get(alias, False)) for alias in aliases)

    if head_size <= 64:
        if _disabled('FLASHATTENTION_DISABLE_HDIM64'):
            return False
        return v_head_size <= 64 or (
            v_head_size <= 512 and not _disabled('FLASHATTENTION_DISABLE_HDIMDIFF64'))
    if head_size <= 96:
        return v_head_size <= 96 and not _disabled('FLASHATTENTION_DISABLE_HDIM96')
    if head_size <= 128:
        return v_head_size <= 128 and not _disabled('FLASHATTENTION_DISABLE_HDIM128')
    if head_size <= 192:
        if _disabled('FLASHATTENTION_DISABLE_HDIM192'):
            return False
        if v_head_size <= 128:
            return not _disabled('FLASHATTENTION_DISABLE_HDIMDIFF192')
        return v_head_size <= 192
    if head_size <= 256:
        return v_head_size <= 256 and not _disabled('FLASHATTENTION_DISABLE_HDIM256')
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
              and _fa3_build_supports_head_dims(head_size, v_head_size))
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


class TritonAttentionBuilder(AttentionBuilder[TritonAttentionMetadata]):
    """Triton attention builder.

    This builder selects the appropriate attention implementation based on:
    1. use_flash_mla: Use FlashMLAImpl for MLA models
    2. enable_fa3: Use FA3Impl if FA3 is available and supported
    3. Default: Use TritonAttentionImpl as fallback
    """

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
