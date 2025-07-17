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
