# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.utils import get_logger

from ..base.engine import EngineStrategy

logger = get_logger('lmdeploy')


class DLLMEngineStrategy(EngineStrategy):
    """DLLM Engine Strategy."""

    def __init__(self, scheduler_config: SchedulerConfig, cache_config: CacheConfig, dllm_block_length: int) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.dllm_block_length = dllm_block_length

        self._check()

    def _check(self):
        """check."""
        max_prefill_token_num = self.cache_config.max_prefill_token_num
        max_batches = self.cache_config.max_batches
        if self.dllm_block_length * max_batches > max_prefill_token_num:
            logger.warning(f'dllm_block_length({self.dllm_block_length}) * max_batch_size ({max_batches}) '
                           f'> max_prefill_token_num ({max_prefill_token_num}). '
                           'This may lead to OOM. Consider to reduce max_batch_size or dllm_block_length.')

    @lru_cache(maxsize=2)
    def get_prealloc_size(self, is_decoding: bool) -> int:
        """Get prealloc_size."""
        if not is_decoding:
            return 0
        block_size = self.cache_config.block_size
        dllm_block_length = self.dllm_block_length
        num_blocks = min(self.scheduler_config.prefill_interval // 2, block_size // dllm_block_length)
        return num_blocks * dllm_block_length

    @lru_cache(maxsize=2)
    def get_num_loops(self, is_decoding: bool) -> int:
        """Get num_loops."""
        if not is_decoding:
            return 1
        block_size = self.cache_config.block_size
        dllm_block_length = self.dllm_block_length
        max_num_loops = block_size // dllm_block_length * 2
        num_loops = min(self.scheduler_config.prefill_interval, max_num_loops)
        return num_loops

    def get_num_decode_tokens(self) -> int:
        """Get num_decode_tokens."""
        return self.dllm_block_length
