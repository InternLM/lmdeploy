# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def flash_mla_available():
    """Check if flash mla is available."""
    # use flash_mla by default if it is installed
    use_flash_mla = False
    try:
        if torch.cuda.get_device_properties(0).major >= 9:
            import flash_mla_cuda  # noqa
            use_flash_mla = True
    except ImportError:
        logger.warning('For higher performance, please install flash_mla https://github.com/deepseek-ai/FlashMLA')
    return use_flash_mla
