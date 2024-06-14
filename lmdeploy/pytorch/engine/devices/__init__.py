# Copyright (c) OpenMMLab. All rights reserved.
import importlib

from lmdeploy.pytorch.devices import get_device_manager
from lmdeploy.utils import get_logger

from .base_device_utils import BaseDeviceUtils

logger = get_logger('lmdeploy')

CURRENT_DEVICE_UTILS = None


def _device_utils_callback(*args, **kwargs):
    """callback."""
    global CURRENT_DEVICE_UTILS
    CURRENT_DEVICE_UTILS = None


get_device_manager().register_context_callback(_device_utils_callback)


def get_current_device_utils() -> BaseDeviceUtils:
    """get device utils."""
    global CURRENT_DEVICE_UTILS
    if CURRENT_DEVICE_UTILS is not None:
        return CURRENT_DEVICE_UTILS

    current_context = get_device_manager().current_context()
    device_type = current_context.device_type
    loaded_utils = BaseDeviceUtils._sub_classes
    if device_type not in loaded_utils:
        try:
            importlib.import_module(f'{__name__}.{device_type}')
            assert device_type in loaded_utils
        except ImportError:
            logger.debug('Failed to import device utils for '
                         f'device: {device_type}. ')
            importlib.import_module(f'{__name__}.cuda')
            loaded_utils[device_type] = loaded_utils['cuda']

    CURRENT_DEVICE_UTILS = loaded_utils[device_type]
    return CURRENT_DEVICE_UTILS
