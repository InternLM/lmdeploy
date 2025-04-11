# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


@dataclass
class WarmupMeta:
    """warmup meta."""
    max_num_tokens: int
    max_batch_size: int
    dtype: torch.dtype


class WarmupManager:

    def __init__(self):
        self._warmup_calls = dict()

    def __contains__(self, key: str):
        """contain key."""
        return key in self._warmup_calls

    def __getitem__(self, key: str):
        """get item."""
        return self._warmup_calls.get(key, None)

    def __setitem__(self, key: str, val):
        """set item."""
        self._warmup_calls[key] = val

    def warmup(self, warmup_meta: WarmupMeta):
        """warmup meta."""
        if len(self._warmup_calls) == 0:
            return

        logger.info('Warming up ops.')
        for key, func in self._warmup_calls.items():
            func(warmup_meta)


_WARMUP_MANAGER = None


def get_warmup_manager():
    """get warmup manager."""
    global _WARMUP_MANAGER
    if _WARMUP_MANAGER is None:
        _WARMUP_MANAGER = WarmupManager()

    return _WARMUP_MANAGER
