# Copyright (c) OpenMMLab. All rights reserved.
import torch
import functools

from lmdeploy.utils import get_logger

from .default import TritonAttentionMetadata, TritonAttentionImpl
from lmdeploy.pytorch.backends.attention import AttentionBuilder, AttentionImpl, AttentionMetadata

logger = get_logger('lmdeploy')

use_fa3 = False
try:
    # Now flash-attention only support FA3 for sm90a && cuda >= 12.3
    if (torch.cuda.get_device_capability()[0] == 9) and (torch.version.cuda >= '12.3'):
        import flash_attn_interface  # noqa: F401
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
def _enable_fa3(alibi: bool, learnable_sink: bool, block_sparse_size: int):
    enable = not alibi and not learnable_sink and block_sparse_size == 1
    if enable and not use_fa3_warning():
        enable = False
    return enable


class TritonAttentionBuilder(AttentionBuilder[TritonAttentionMetadata]):
    """Triton attention builder."""

    @staticmethod
    def build(
        num_heads: int,
        head_size: int,
        scale: float = None,
        num_kv_heads: int = None,
        v_head_size: int = None,
        alibi: bool = False,
        sliding_window: int = None,
        logical_softcapping: float = None,
        causal: bool = True,
        use_flash_mla: bool = False,
        learnable_sink: bool = False,
        block_sparse_size: int = 1,
        **kwargs,
    ) -> TritonAttentionImpl:
        """build."""
        enable_fa3 = _enable_fa3(alibi, learnable_sink, block_sparse_size)
        if use_flash_mla is True:
            logger.debug('Build FlashMLAImpl Attention')
            from .mla import FlashMLAImpl
            return FlashMLAImpl(num_heads,
                                head_size,
                                scale=scale,
                                num_kv_heads=num_kv_heads,
                                v_head_size=v_head_size,
                                alibi=alibi,
                                sliding_window=sliding_window,
                                logical_softcapping=logical_softcapping,
                                causal=causal,
                                **kwargs)
        elif enable_fa3:
            logger.debug('Build FA3Impl Attention')
            from .fa3 import FA3Impl
            return FA3Impl(num_heads,
                           head_size,
                           scale=scale,
                           num_kv_heads=num_kv_heads,
                           v_head_size=v_head_size,
                           alibi=alibi,
                           sliding_window=sliding_window,
                           logical_softcapping=logical_softcapping,
                           causal=causal,
                           **kwargs)
        else:
            logger.debug('Build TritonAttentionImpl Attention')
            return TritonAttentionImpl(num_heads,
                                       head_size,
                                       scale=scale,
                                       num_kv_heads=num_kv_heads,
                                       v_head_size=v_head_size,
                                       alibi=alibi,
                                       sliding_window=sliding_window,
                                       logical_softcapping=logical_softcapping,
                                       causal=causal,
                                       block_sparse_size=block_sparse_size,
                                       **kwargs)
