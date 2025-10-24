# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def flash_mla_available():
    """Check if flash mla is available."""
    # use flash_mla by default if it is installed
    use_flash_mla = False
    try:
        # torch_npu device_properties doesn't have 'major' attribute
        device_properties = torch.cuda.get_device_properties(0)
        if hasattr(device_properties, 'major') and device_properties.major >= 9:
            import flash_mla  # noqa
            use_flash_mla = True
    except ImportError:
        logger.warning('For higher performance, please install flash_mla https://github.com/deepseek-ai/FlashMLA')
    return use_flash_mla


def flash_attn_v3_available():
    """Check if flash attn v3 is available."""
    use_fa3 = False
    try:
        # Now flash-attention only support FA3 for sm90a && cuda >= 12.3
        if (torch.cuda.get_device_capability()[0] == 9) and (torch.version.cuda >= '12.3'):
            import flash_attn_interface  # noqa: F401
            assert torch.ops.flash_attn_3 is not None
            use_fa3 = True
    except Exception:
        logger.warning('For higher performance, please install FlashAttention-3 '
                       'https://github.com/Dao-AILab/flash-attention')
    return use_fa3
