# Copyright (c) OpenMMLab. All rights reserved.
# Inspired by vLLM: https://github.com/vllm-project/vllm
import asyncio
import contextlib
from typing import Any, NamedTuple

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig, SpecDecodeConfig
from lmdeploy.pytorch.disagg.conn.protocol import DistServeInitRequest, DistServeKVTransferEndpointInfo
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class _CacheBlockSize(NamedTuple):
    """Memory size of one logical cache block."""

    target: int
    spec: int = 0

    @property
    def total(self) -> int:
        """Total cache block size when target and spec caches coexist."""
        return self.target + self.spec


class ExecutorBase:
    """Executor base class."""

    def __init__(self,
                 model_path: str,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 backend_config: BackendConfig,
                 dist_config: DistConfig,
                 misc_config: MiscConfig,
                 adapters: dict[str, str] = None,
                 specdecode_config: SpecDecodeConfig = None,
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
        self.world_size = dist_config.world_size
        self.device_type = device_type
        self.specdecode_config = specdecode_config

    def download_models(self):
        """Download model."""
        raise NotImplementedError('Not Implemented.')

    def build_model(self):
        """Build model."""
        raise NotImplementedError('Not Implemented.')

    def gather_free_mem(self):
        """Gather available memory."""
        raise NotImplementedError('Not Implemented.')

    def set_cache_config(self, cache_config: CacheConfig, spec_cache_config: CacheConfig = None):
        """Set all cache config."""
        raise NotImplementedError('Not Implemented.')

    def set_model_config(self, model_config: ModelConfig, spec_model_config: ModelConfig = None):
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

    async def sleep(self, level: int = 1):
        """Sleep."""
        raise NotImplementedError('Not Implemented.')

    def wakeup(self, tags: list[str] | None = None):
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

    async def wait_tasks(self):
        """Wait tasks."""
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

    def p2p_connect(self, conn_request: list[DistServeKVTransferEndpointInfo]):
        """rdma_connect."""
        raise NotImplementedError('Not Implemented')

    async def migrate(self, batch: MigrationExecutionBatch):
        """KV Cache Migration."""
        raise NotImplementedError('Not Implemented')

    """ PD Disaggregation API End """

    @staticmethod
    def _get_num_gpu_blocks(available_mem: int, cache_block_size: int, spec_cache_block_size: int = 0) -> int:
        """Get the number of GPU blocks fitting in available memory."""
        total_cache_block_size = cache_block_size + spec_cache_block_size
        if total_cache_block_size <= 0:
            raise RuntimeError('No enough gpu memory for kv cache.')
        # `available_mem` is already an integer byte budget. Keep the division
        # integral as well so cache sizing never depends on float rounding.
        return available_mem // total_cache_block_size

    @staticmethod
    def _get_min_num_gpu_blocks(available_mems: list[int], cache_block_sizes: list[int]) -> int:
        """Get the minimum GPU blocks fitting on all ranks."""
        # All ranks must use the same logical num_gpu_blocks, even if their
        # per-rank cache footprint differs. The smallest rank capacity wins.
        num_gpu_blocks = [
            ExecutorBase._get_num_gpu_blocks(available_mem, cache_block_size)
            for available_mem, cache_block_size in zip(available_mems, cache_block_sizes)
        ]
        return min(num_gpu_blocks)

    def _get_rank_cache_block_sizes(self, num_ranks: int, cache_block_size: _CacheBlockSize) -> list[int]:
        """Get per-rank KV cache block sizes."""
        if cache_block_size.spec == 0:
            return [cache_block_size.target] * num_ranks

        attn_tp = self.dist_config.attn_tp
        # Spec decoding only builds the draft/spec cache on one rank in each
        # attention-TP group. Other ranks can use the memory that would have
        # gone to spec cache for additional target KV blocks.
        return [
            cache_block_size.total if rank % attn_tp == 0 else cache_block_size.target
            for rank in range(num_ranks)
        ]

    def _get_runtime_size(self, free_mems: list[int], cache_block_size: _CacheBlockSize,
                          vocab_size: int) -> tuple[int, int]:
        """Find best prefill num."""
        cache_max_entry_count = self.cache_config.cache_max_entry_count
        max_prefill_token_num = self.cache_config.max_prefill_token_num
        max_batches = self.cache_config.max_batches
        rank_cache_block_sizes = self._get_rank_cache_block_sizes(len(free_mems), cache_block_size)
        runtime_cache_size = 0
        while max_prefill_token_num > 0:
            # Runtime buffers scale mostly with the prefill token budget and
            # logits/vocab size. They are not pageable KV cache, so reserve
            # them before applying the KV cache memory ratio.
            runtime_cache_size = int((max_prefill_token_num + max_batches * 2) * vocab_size * 2)
            available_mems = [int((free_mem - runtime_cache_size) * cache_max_entry_count) for free_mem in free_mems]
            # Keep at least a small number of KV blocks after runtime reserve.
            # If not possible, reduce the prefill token budget and try again.
            if self._get_min_num_gpu_blocks(available_mems, rank_cache_block_sizes) >= 16:
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
            self.cache_config.kernel_block_size = 32
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
            num_state_caches = int(cache_config.max_batches + 1)
            cache_config.num_state_caches = num_state_caches

        mems = StateCacheEngine.get_cache_state_size(cache_config.states_shapes)
        mems *= num_state_caches

        if cache_config.enable_prefix_caching:
            cache_config.enable_prefix_caching = False
            logger.warning('Prefix caching has not been support for state space model.')

        return mems

    def _sync_spec_cache_block_size(self) -> None:
        """Keep spec cache block sizes aligned with target cache."""
        if self.specdecode_config and self.specdecode_config.cache_config:
            # The executor may adjust target block sizes after engine config
            # construction. Keep spec cache layout compatible with that final
            # target layout before estimating or allocating caches.
            spec_cache_config = self.specdecode_config.cache_config
            spec_cache_config.block_size = self.cache_config.block_size
            spec_cache_config.kernel_block_size = self.cache_config.kernel_block_size

    def _get_free_gpu_mems(self) -> list[int]:
        """Get free GPU memory across workers."""
        free_mems = self.gather_free_mem()
        logger.debug(f'minimal free gpu memory: {min(free_mems) >> 20} mb')
        return free_mems

    def _reserve_state_cache_mem(self, free_mems: list[int]) -> list[int]:
        """Reserve non-pageable state cache memory from free memory."""
        state_cache_mem = self._get_state_cache_mem()
        # State cache is allocated as a separate pool and is not governed by
        # cache_max_entry_count, so subtract it from every rank first.
        free_mems = [free_mem - state_cache_mem for free_mem in free_mems]
        assert min(free_mems) > 0, 'No enough gpu memory for state cache. Please reduce max_batch_size.'
        return free_mems

    def _get_spec_configs(self) -> tuple[CacheConfig | None, ModelConfig | None]:
        """Get spec model and cache configs if enabled."""
        if self.specdecode_config is None:
            return None, None
        return self.specdecode_config.cache_config, self.specdecode_config.model_config

    def _get_cache_block_sizes(self, spec_cache_config: CacheConfig | None,
                               spec_model_config: ModelConfig | None) -> _CacheBlockSize:
        """Get per-block KV cache memory for target and spec models."""
        cache_block_size = CacheEngine.get_cache_block_size(self.cache_config, self.model_config,
                                                            self.dist_config.attn_tp)

        spec_cache_block_size = 0
        if spec_cache_config is not None:
            # Draft/spec cache is not tensor-parallelized with the target
            # attention group here, so its block size is measured at world_size=1.
            spec_cache_block_size = CacheEngine.get_cache_block_size(spec_cache_config, spec_model_config, 1)

        return _CacheBlockSize(target=cache_block_size, spec=spec_cache_block_size)

    def _reserve_runtime_mem(self, free_mems: list[int], cache_block_size: _CacheBlockSize,
                             spec_cache_config: CacheConfig | None) -> list[int]:
        """Reserve runtime memory and update prefill token limit if needed."""
        runtime_mem, max_prefill_token_num = self._get_runtime_size(free_mems, cache_block_size,
                                                                    self.model_config.vocab_size)
        if self.cache_config.max_prefill_token_num != max_prefill_token_num:
            if max_prefill_token_num <= 0:
                raise RuntimeError('No enough gpu memory for runtime.')
            self.cache_config.max_prefill_token_num = max_prefill_token_num
            logger.warning(f'No enough memory. Update max_prefill_token_num={max_prefill_token_num}')

        if spec_cache_config is not None:
            spec_cache_config.max_prefill_token_num = max_prefill_token_num

        free_mems = [free_mem - runtime_mem for free_mem in free_mems]
        logger.debug(f'estimated max runtime memory: {runtime_mem >> 20} mb')
        return free_mems

    def _update_num_gpu_blocks(self, free_mems: list[int], cache_block_size: _CacheBlockSize,
                               spec_cache_config: CacheConfig | None) -> None:
        """Update target and spec GPU block counts from remaining memory."""
        if self.cache_config.num_gpu_blocks != 0:
            # User supplied an explicit block count. Do not resize it from the
            # current free-memory snapshot.
            if spec_cache_config is not None:
                spec_cache_config.num_gpu_blocks = self.cache_config.num_gpu_blocks
            return

        available_mems = [int(free_mem * self.cache_config.cache_max_entry_count) for free_mem in free_mems]
        rank_cache_block_sizes = self._get_rank_cache_block_sizes(len(free_mems), cache_block_size)
        self.cache_config.num_gpu_blocks = self._get_min_num_gpu_blocks(available_mems, rank_cache_block_sizes)
        if self.cache_config.num_gpu_blocks <= 0:
            raise RuntimeError('No enough gpu memory for kv cache.')
        if spec_cache_config is not None:
            spec_cache_config.num_gpu_blocks = self.cache_config.num_gpu_blocks

    def update_configs(self) -> None:
        """Update cache config."""
        self._adjust_block_size()
        self._sync_spec_cache_block_size()
        self.cache_config.states_shapes = self.model_config.states_shapes

        spec_cache_config, spec_model_config = self._get_spec_configs()
        cache_block_size = self._get_cache_block_sizes(spec_cache_config, spec_model_config)

        free_mems = self._get_free_gpu_mems()
        free_mems = self._reserve_state_cache_mem(free_mems)
        free_mems = self._reserve_runtime_mem(free_mems, cache_block_size, spec_cache_config)
        self._update_num_gpu_blocks(free_mems, cache_block_size, spec_cache_config)

        self.set_cache_config(self.cache_config, spec_cache_config)
        self.set_model_config(self.model_config, spec_model_config)

    def init(self):
        """init."""
        logger.info('Building Model.')
        self.build_model()
        logger.info('Updating configs.')
        self.update_configs()
        logger.info('Building GraphRunner and warmup ops, please waiting.')
        self.build_graph_runner()
        logger.info(f'Building CacheEngine with config: \n{self.cache_config}.')
        if self.specdecode_config:
            if spec_cache_config := self.specdecode_config.cache_config:
                logger.info(f'Building Spec CacheEngine with config: \n{spec_cache_config}.')
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
