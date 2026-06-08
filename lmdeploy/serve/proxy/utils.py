# Copyright (c) OpenMMLab. All rights reserved.

import enum

LATENCY_DEQUE_LEN = 15


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


class APIServerException(Exception):

    def __init__(self, status_code: int, body: bytes, headers: dict | None = None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        if 'content-type' not in self.headers:
            self.headers['content-type'] = 'application/json'
