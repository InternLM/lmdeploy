# Copyright (c) OpenMMLab. All rights reserved.

import enum

LATENCY_DEEQUE_LEN = 15
API_TIMEOUT_LEN = 100


class Strategy(enum.Enum):
    RANDOM = enum.auto()
    MIN_EXPECTED_LATENCY = enum.auto()
    MIN_OBSERVED_LATENCY = enum.auto()

    @classmethod
    def from_str(cls, name):
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
    MODEL_NOT_FOUND = 10400
    SERVICE_UNAVAILABLE = 10401
    API_TIMEOUT = 10402


err_msg = {
    ErrorCodes.MODEL_NOT_FOUND:
    'The request model name does not exist in the model list.',
    ErrorCodes.SERVICE_UNAVAILABLE:
    'The service is unavailable now. May retry later.',
    ErrorCodes.API_TIMEOUT: 'Failed to get response after a period of time'
}
