# Copyright (c) OpenMMLab. All rights reserved.
# Inspired by vLLM: https://github.com/vllm-project/vllm
import asyncio
from typing import Any, Dict

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class ExecutorBase:
    """Executor base class."""

    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 backend_config: BackendConfig,
                 tokenizer: Any,
                 dp: int,
                 tp: int,
                 adapters: Dict[str, str] = None,
                 device_type: str = 'cuda'):
        """initialize Executor."""
        cache_config.window_size = model_config.sliding_window
        self.model_config = model_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.tokenizer = tokenizer
        self.dp = dp
        self.tp = tp
        self.device_type = device_type

    def download_models(self):
        """download model."""
        raise NotImplementedError('Not Implemented.')

    def build_model(self):
        """build model."""
        raise NotImplementedError('Not Implemented.')

    def gather_free_mem(self):
        """gather available memory."""
        raise NotImplementedError('Not Implemented.')

    def set_cache_config(self, cache_config: CacheConfig):
        """set all cache config."""
        raise NotImplementedError('Not Implemented.')

    def set_model_config(self, model_config: ModelConfig):
        """set all model config."""
        raise NotImplementedError('Not Implemented.')

    def build_graph_runner(self):
        """build graph runner."""
        raise NotImplementedError('Not Implemented.')

    def build_cache_engine(self):
        """build cache engine."""
        raise NotImplementedError('Not Implemented.')

    def get_input_processor(self):
        """get input processor."""
        raise NotImplementedError('Not Implemented.')

    def start(self, forward_event: asyncio.Event):
        """start engine loop."""
        raise NotImplementedError('Not Implemented.')

    def stop(self):
        """stop engine loop."""
        raise NotImplementedError('Not Implemented.')

    def release(self):
        """release resources."""
        raise NotImplementedError('Not Implemented.')

    async def forward_async(self, inputs):
        """start forward."""
        raise NotImplementedError('Not Implemented')

    async def get_output_async(self):
        """get output async."""
        raise NotImplementedError('Not Implemented')

    def _get_runtime_size(self, num_free_gpu_mem: int, cache_block_size: int, vocal_size: int):
        """find best prefill num."""
        cache_max_entry_count = self.cache_config.cache_max_entry_count
        max_prefill_token_num = self.cache_config.max_prefill_token_num
        runtime_cache_size = 0
        while max_prefill_token_num > 0:
            # lm_head output(2) + to float(4) + estimated misc(1) = 7
            runtime_cache_size = int(max_prefill_token_num * vocal_size * 7)
            num_available = (num_free_gpu_mem - runtime_cache_size) * cache_max_entry_count
            if int(num_available) // cache_block_size >= 16:
                break
            max_prefill_token_num = max_prefill_token_num // 2
        return runtime_cache_size, max_prefill_token_num

    def _adjust_block_size(self):
        """adjust block_size."""
        # TODO: support kernel with both large head dim and large block size.
        if self.model_config.k_head_dim >= 512 and self.cache_config.block_size > 32:
            self.cache_config.block_size = 32
            logger.warning(f'Update `block_size={self.cache_config.block_size}`'
                           f' for large `head_dim={self.model_config.k_head_dim}`.')

    def update_configs(self):
        """update cache config."""
        self._adjust_block_size()
        cache_config = self.cache_config
        model_config = self.model_config
        free_mems = self.gather_free_mem()
        free_mem = min(free_mems)
        logger.debug(f'minimal free gpu memory: {free_mem>>20} mb')
        vocal_size = self.model_config.vocab_size

        cache_block_size = CacheEngine.get_cache_block_size(cache_config.block_size, model_config, self.tp,
                                                            cache_config.quant_policy)
        runtime_mem, max_prefill_token_num = self._get_runtime_size(free_mem, cache_block_size, vocal_size)
        if cache_config.max_prefill_token_num != max_prefill_token_num:
            if max_prefill_token_num <= 0:
                raise RuntimeError('No enough gpu memory for runtime.')
            cache_config.max_prefill_token_num = max_prefill_token_num
            logger.warning(f'No enough memory. Update max_prefill_token_num={max_prefill_token_num}')
        free_mem -= runtime_mem
        logger.debug(f'estimated max runtime memory: {runtime_mem>>20} mb')
        available_mem = free_mem * cache_config.cache_max_entry_count

        if cache_config.num_gpu_blocks == 0:
            cache_config.num_gpu_blocks = int(available_mem / cache_block_size)
            if cache_config.num_gpu_blocks <= 0:
                raise RuntimeError('No enough gpu memory for kv cache.')
        self.set_cache_config(cache_config)
        self.set_model_config(model_config)

    def init(self):
        """init."""
        logger.info('Building Model.')
        self.build_model()
        logger.info('Updating configs.')
        self.update_configs()
        logger.info('Building GraphRunner.')
        self.build_graph_runner()
        logger.info(f'Building CacheEngine with config:\n{self.cache_config}.')
        self.build_cache_engine()
