# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def flash_mla_available():
    """Check if flash mla is available."""
    # use flash_mla by default if it is installed
    use_flash_mla = False
    try:
        """In some torch_npu versions, device_properties doesn't have 'major'
        attribute; In other torch_npu versions, the value of major is None."""
        device_properties = torch.cuda.get_device_properties(0)
        major = getattr(device_properties, 'major', None)
        if isinstance(major, int) and major >= 9:
            import flash_mla  # noqa
            use_flash_mla = True
    except ImportError:
        logger.warning('For higher performance, please install flash_mla https://github.com/deepseek-ai/FlashMLA')
    return use_flash_mla
