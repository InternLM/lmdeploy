# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def build_eviction_helper(scheduler, eviction_type: str):
    """Build eviction helper."""
    if eviction_type == 'copy':
        logger.warning('`copy` eviction has been deprecated, '
                       'use `recompute` instead.')
        eviction_type = 'recompute'
    if eviction_type == 'recompute':
        from .recompute_eviction_helper import RecomputeEvictionHelper
        return RecomputeEvictionHelper(scheduler)
    else:
        raise TypeError(f'Unknown eviction type: {eviction_type}')
