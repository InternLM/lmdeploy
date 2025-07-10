# Copyright (c) OpenMMLab. All rights reserved.

import enum
import os

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

LATENCY_DEQUE_LEN = 15
AIOHTTP_TIMEOUT = os.getenv('AIOHTTP_TIMEOUT', None)
if AIOHTTP_TIMEOUT is not None:
    AIOHTTP_TIMEOUT = int(AIOHTTP_TIMEOUT)
logger.info(f'AIOHTTP_TIMEOUT set to {AIOHTTP_TIMEOUT}. It can be modified before launching the proxy server '
            'through env variable AIOHTTP_TIMEOUT')


class RoutingStrategy(enum.Enum):
    """Strategy to dispatch requests to nodes."""
    RANDOM = enum.auto()
    MIN_EXPECTED_LATENCY = enum.auto()
    MIN_OBSERVED_LATENCY = enum.auto()

    @classmethod
    def from_str(cls, name):
        """Get strategy from string."""
        if name == 'random':
            return cls.RANDOM
        elif name == 'min_expected_latency':
            return cls.MIN_EXPECTED_LATENCY
        elif name == 'min_observed_latency':
            return cls.MIN_OBSERVED_LATENCY
        else:
            raise ValueError(f'Invalid strategy: {name}. Supported: random, '
                             f'min_expected_latency, min_observed_latency.')


class ErrorCodes(enum.Enum):
    """Error codes."""
    MODEL_NOT_FOUND = 10400
    SERVICE_UNAVAILABLE = 10401
    API_TIMEOUT = 10402


err_msg = {
    ErrorCodes.MODEL_NOT_FOUND: 'The request model name does not exist in the model list.',
    ErrorCodes.SERVICE_UNAVAILABLE: 'The service is unavailable now. May retry later.',
    ErrorCodes.API_TIMEOUT: 'Failed to get response after a period of time'
}
