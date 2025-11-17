# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.logger import get_logger

logger = get_logger('lmdeploy')

try:
    logger.debug('Registering DLSlime Backend')
    from .dlslime import DLSlimeBackend
except ImportError:
    logger.debug('Disable DLSlime Backend')

try:
    logger.debug('Registering Mooncake Backend')
    from .mooncake import MooncakeBackend
except ImportError:
    logger.warning('Disable Mooncake Backend')

__all__ = ['DLSlimeBackend', 'MooncakeBackend']
