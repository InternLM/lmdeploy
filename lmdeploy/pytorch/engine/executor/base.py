# Copyright (c) OpenMMLab. All rights reserved.
# Inspired by vLLM: https://github.com/vllm-project/vllm
import asyncio
import contextlib
from typing import Any, Dict, List, Optional

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig
from lmdeploy.pytorch.disagg.conn.protocol import DistServeInitRequest, DistServeKVTransferEndpointInfo
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
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
                 dist_config: DistConfig,
                 misc_config: MiscConfig,
                 adapters: Dict[str, str] = None,
                 device_type: str = 'cuda'):
        """Initialize Executor."""
        cache_config.window_size = model_config.sliding_window
        if cache_config.window_size is not None and cache_config.window_size > 0:
            # do not support sliding window prefix caching
            logger.warning('Sliding window prefix caching is not supported.')
            cache_config.enable_prefix_caching = False
        self.model_config = model_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.dist_config = dist_config
        self.misc_config = misc_config
        self.dp = dist_config.dp
        self.tp = dist_config.tp
        self.world_size = dist_config.world_size
        self.device_type = device_type

    def download_models(self):
        """Download model."""
        raise NotImplementedError('Not Implemented.')

    def build_model(self):
        """Build model."""
        raise NotImplementedError('Not Implemented.')

    def gather_free_mem(self):
        """Gather available memory."""
        raise NotImplementedError('Not Implemented.')

    def set_cache_config(self, cache_config: CacheConfig):
        """Set all cache config."""
        raise NotImplementedError('Not Implemented.')

    def set_model_config(self, model_config: ModelConfig):
        """Set all model config."""
        raise NotImplementedError('Not Implemented.')

    def build_graph_runner(self):
        """Build graph runner."""
        raise NotImplementedError('Not Implemented.')

    def build_cache_engine(self):
        """Build cache engine."""
        raise NotImplementedError('Not Implemented.')

    def warmup(self):
        """warmup."""
        raise NotImplementedError('Not Implemented.')

    def sleep(self, level: int = 1):
        """Sleep."""
        raise NotImplementedError('Not Implemented.')

    def wakeup(self, tags: Optional[List[str]] = None):
        """Wakeup."""
        raise NotImplementedError('Not Implemented.')

    def update_params(self, request: Any):
        """Update params."""
        raise NotImplementedError('Not Implemented.')

    def get_input_processor(self):
        """Get input processor."""
        raise NotImplementedError('Not Implemented.')

    def start(self, forward_event: asyncio.Event):
        """Start engine loop."""
        raise NotImplementedError('Not Implemented.')

    def stop(self):
        """Stop engine loop."""
        raise NotImplementedError('Not Implemented.')

    def release(self):
        """Release resources."""
        raise NotImplementedError('Not Implemented.')

    async def forward_async(self, inputs):
        """Start forward."""
        raise NotImplementedError('Not Implemented')

    async def get_output_async(self):
        """Get output async."""
        raise NotImplementedError('Not Implemented')

    """ PD Disaggregation API Begin """

    def p2p_initialize(self, remote_engine_config: DistServeInitRequest):
        """Init rdma link."""
        raise NotImplementedError('Not implemented')

    def p2p_connect(self, conn_request: List[DistServeKVTransferEndpointInfo]):
        """rdma_connect."""
        raise NotImplementedError('Not Implemented')

    async def migrate(self, batch: MigrationExecutionBatch):
        """KV Cache Migration."""
        raise NotImplementedError('Not Implemented')

    """ PD Disaggregation API End """

    def _get_runtime_size(self, num_free_gpu_mem: int, cache_block_size: int, vocal_size: int):
        """Find best prefill num."""
        cache_max_entry_count = self.cache_config.cache_max_entry_count
        max_prefill_token_num = self.cache_config.max_prefill_token_num
        max_batches = self.cache_config.max_batches
        runtime_cache_size = 0
        while max_prefill_token_num > 0:
            # estimate runtime mem size
            runtime_cache_size = int((max_prefill_token_num + max_batches * 2) * vocal_size * 2)
            num_available = (num_free_gpu_mem - runtime_cache_size) * cache_max_entry_count
            if cache_block_size == 0 or int(num_available) // cache_block_size >= 16:
                break
            max_prefill_token_num = max_prefill_token_num // 2
        return runtime_cache_size, max_prefill_token_num

    def _adjust_block_size(self):
        """Adjust block_size."""
        if self.model_config.use_flash_mla is True:
            if self.cache_config.block_size != 64:
                raise ValueError('Please set block_size to 64 for flash_mla.')
            return
        # TODO: support kernel with both large head dim and large block size.
        if self.model_config.k_head_dim >= 512 and self.cache_config.block_size > 32:
            self.cache_config.block_size = 32
            logger.warning(
                f'Update `block_size={self.cache_config.block_size}` for large `head_dim={self.model_config.k_head_dim}`.'  # noqa
            )

    def _get_state_cache_mem(self):
        """Get state cache mem usage."""
        cache_config = self.cache_config
        if len(cache_config.states_shapes) == 0:
            return 0

        from lmdeploy.pytorch.engine.cache_engine import StateCacheEngine

        num_state_caches = cache_config.num_state_caches
        if num_state_caches is None:
            # add more caches for eviction
            # TODO: Share memory between state cache and pageable cache
            num_state_caches = int(cache_config.max_batches + 8)
            cache_config.num_state_caches = num_state_caches

        mems = StateCacheEngine.get_cache_state_size(cache_config.states_shapes)
        mems *= num_state_caches

        if cache_config.enable_prefix_caching:
            cache_config.enable_prefix_caching = False
            logger.warning('Prefix caching has not been support for state space model.')

        return mems

    def update_configs(self):
        """Update cache config."""
        self._adjust_block_size()
        cache_config = self.cache_config
        model_config = self.model_config
        cache_config.states_shapes = model_config.states_shapes

        # get free mems
        free_mems = self.gather_free_mem()
        free_mem = min(free_mems)
        logger.debug(f'minimal free gpu memory: {free_mem >> 20} mb')

        # get state cache size
        state_cache_mem = self._get_state_cache_mem()
        free_mem = free_mem - state_cache_mem
        assert free_mem > 0, 'No enough gpu memory for state cache. Please reduce max_batch_size.'

        vocal_size = self.model_config.vocab_size
        tp = self.dist_config.attn_config.tp
        cache_block_size = CacheEngine.get_cache_block_size(cache_config.block_size, model_config, tp,
                                                            cache_config.quant_policy)
        runtime_mem, max_prefill_token_num = self._get_runtime_size(free_mem, cache_block_size, vocal_size)
        if cache_config.max_prefill_token_num != max_prefill_token_num:
            if max_prefill_token_num <= 0:
                raise RuntimeError('No enough gpu memory for runtime.')
            cache_config.max_prefill_token_num = max_prefill_token_num
            logger.warning(f'No enough memory. Update max_prefill_token_num={max_prefill_token_num}')
        free_mem -= runtime_mem
        logger.debug(f'estimated max runtime memory: {runtime_mem >> 20} mb')
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
        logger.info('Building GraphRunner and warmup ops, please waiting.')
        self.build_graph_runner()
        logger.info(f'Building CacheEngine with config: \n{self.cache_config}.')
        self.build_cache_engine()
        logger.info('Warming up model.')
        self.warmup()

    @contextlib.contextmanager
    def remote_log(self, msg: str):
        """Send log for debugging.

        Do not use it in production.
        """
        # Different executor may have different log sending logic.
        yield
