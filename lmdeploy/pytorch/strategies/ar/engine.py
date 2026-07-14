# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig

from ..base.engine import EngineStrategy


class AREngineStrategy(EngineStrategy):
    """AR Engine Strategy."""

    def __init__(self, scheduler_config: SchedulerConfig, cache_config: CacheConfig) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

    def get_prealloc_size(self, is_decoding: bool):
        """Get prealloc_size."""
        return self.scheduler_config.prefill_interval if is_decoding else 0

    def get_num_loops(self, is_decoding: bool) -> int:
        """Get num_loops."""
        return self.scheduler_config.prefill_interval if is_decoding else 1

    def get_num_decode_tokens(self) -> int:
        """Get num_decode_tokens."""
        return 1
