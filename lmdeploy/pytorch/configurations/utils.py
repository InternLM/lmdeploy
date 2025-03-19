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


def deep_gemm_available(hf_config):
    """Whether to use deep gemm.

    Only to return True if the device is SM90 or higher, meanwhile the model config should be the deepseekv3 or R1
    config and of course deep_gemm was installed.
    """
    if torch.cuda.get_device_properties(0).major < 9:
        return False
    quant_config = getattr(hf_config, 'quantization_config', None)
    if quant_config is None:
        return False
    deepseek_quanti_config = {'quant_method': 'fp8', 'weight_block_size': [128, 128]}
    # deepseek_quanti_config should be the minimal requirement
    if not (deepseek_quanti_config.items() <= quant_config.items()):
        return False
    try:
        import deep_gemm  # noqa
    except:  # noqa
        return False
    return True
