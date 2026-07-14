# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass

import torch

from lmdeploy.pytorch.utils import singleton
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


@dataclass
class WarmupMeta:
    """Warmup meta."""
    max_num_tokens: int
    max_batch_size: int
    dtype: torch.dtype


@singleton
class WarmupManager:

    def __init__(self):
        self._warmup_calls = dict()

    def __contains__(self, key: str):
        """Contain key."""
        return key in self._warmup_calls

    def __getitem__(self, key: str):
        """Get item."""
        return self._warmup_calls.get(key, None)

    def __setitem__(self, key: str, val):
        """Set item."""
        self._warmup_calls[key] = val

    def warmup(self, warmup_meta: WarmupMeta):
        """Warmup meta."""
        if len(self._warmup_calls) == 0:
            return
        import random
        logger.info('Warming up ops.')
        funcs = list(self._warmup_calls.values())
        random.shuffle(funcs)
        for func in funcs:
            func(warmup_meta)


def get_warmup_manager():
    """Get warmup manager."""
    return WarmupManager()
