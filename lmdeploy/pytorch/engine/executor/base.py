# Copyright (c) OpenMMLab. All rights reserved.
# Inspired by vLLM: https://github.com/vllm-project/vllm
import asyncio
import contextlib
from typing import Any, NamedTuple

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, DistConfig, MiscConfig, ModelConfig, SpecDecodeConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import DistServeInitRequest, DistServeKVTransferEndpointInfo
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class _WorkerCachePlanSizes(NamedTuple):
    """Per-plan bytes for one logical cache block on one worker."""

    target: int
    spec: int = 0
    memory: int = 0

    @property
    def total(self) -> int:
        """Total cache block size."""
        return self.target + self.spec + self.memory


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
                 device_type: str = 'cuda',
                 trust_remote_code: bool = False):
        """Initialize Executor."""
        cache_config.window_size = model_config.sliding_window
        self.model_config = model_config
        self.cache_config = cache_config
        self.backend_config = backend_config
        self.dist_config = dist_config
        self.misc_config = misc_config
        self.dp = dist_config.dp
        self.world_size = dist_config.world_size
        self.device_type = device_type
        self.specdecode_config = specdecode_config
        self._maybe_disable_unsupported_prefix_caching(check_window=not self._has_cache_update_hook())

    def _has_cache_update_hook(self):
        """Return whether the model may normalize cache config later."""
        return getattr(self.model_config, 'update_cache_config_func', None) is not None

    def _maybe_disable_unsupported_prefix_caching(self, *, check_window: bool = True):
        """Disable prefix caching for unsupported executor/cache modes."""
        if not getattr(self.cache_config, 'enable_prefix_caching', False):
            return
        if check_window and self.cache_config.window_size is not None and self.cache_config.window_size > 0:
            # do not support generic sliding window prefix caching
            logger.warning('Sliding window prefix caching is not supported.')
            self.cache_config.enable_prefix_caching = False
            return
        if self.cache_config.role != EngineRole.Hybrid:
            logger.warning('PD prefix caching is not supported.')
            self.cache_config.enable_prefix_caching = False

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

    def init_weights_update_group(self, request: Any):
        """Init disaggregated weights-update process group."""
        raise NotImplementedError('Not Implemented.')

    def update_weights_from_distributed(self, request: Any):
        """Receive weights through the disaggregated process group."""
        raise NotImplementedError('Not Implemented.')

    def destroy_weights_update_group(self, request: Any):
        """Tear down a previously initialized weights-update process group."""
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
        num_gpu_blocks = available_mem // total_cache_block_size
        if num_gpu_blocks <= 2:
            raise RuntimeError('No enough gpu memory for kv cache.')
        return num_gpu_blocks

    @staticmethod
    def _get_min_num_gpu_blocks(available_mems: list[int], cache_block_sizes: list[int]) -> int:
        """Get the minimum GPU blocks fitting on all ranks."""
        if len(available_mems) != len(cache_block_sizes):
            raise ValueError('Free-memory and cache-plan results must contain the same worker ranks.')
        # All ranks must use the same logical num_gpu_blocks, even if their
        # per-rank cache footprint differs. The smallest rank capacity wins.
        num_gpu_blocks = [
            ExecutorBase._get_num_gpu_blocks(available_mem, cache_block_size)
            for available_mem, cache_block_size in zip(available_mems, cache_block_sizes)
        ]
        return min(num_gpu_blocks)

    @staticmethod
    def _get_rank_cache_block_sizes(cache_block_sizes: list[_WorkerCachePlanSizes]) -> list[int]:
        """Get per-rank KV cache block sizes."""
        return [cache_block_size.total for cache_block_size in cache_block_sizes]

    def _get_dsa_score_workspace_size(self) -> int:
        """Return the bounded sparse-indexer score workspace in bytes."""
        if getattr(self.model_config, 'mla_index_topk', None) is None:
            return 0
        return _envs.dsa_indexer_max_logits_mb * (1 << 20)

    def _get_runtime_size(self, free_mems: list[int], cache_block_sizes: list[_WorkerCachePlanSizes],
                          vocab_size: int) -> tuple[int, int]:
        """Find best prefill num."""
        cache_max_entry_count = self.cache_config.cache_max_entry_count
        max_prefill_token_num = self.cache_config.max_prefill_token_num
        max_batches = self.cache_config.max_batches
        rank_cache_block_sizes = self._get_rank_cache_block_sizes(cache_block_sizes)
        dsa_score_workspace = self._get_dsa_score_workspace_size()
        runtime_cache_size = 0
        while max_prefill_token_num > 0:
            # Runtime buffers scale mostly with the prefill token budget and
            # logits/vocab size. They are not pageable KV cache, so reserve
            # them before applying the KV cache memory ratio.
            runtime_cache_size = int((max_prefill_token_num + max_batches * 2) * vocab_size * 2)
            runtime_cache_size += dsa_score_workspace
            available_mems = [int((free_mem - runtime_cache_size) * cache_max_entry_count) for free_mem in free_mems]
            # Keep at least a small number of KV blocks after runtime reserve.
            # If not possible, reduce the prefill token budget and try again.
            if self._get_min_num_gpu_blocks(available_mems, rank_cache_block_sizes) >= 16:
                break
            max_prefill_token_num = max_prefill_token_num // 2
        return runtime_cache_size, max_prefill_token_num

    def _adjust_block_size(self):
        """Adjust block_size."""
        if self.model_config.update_cache_config_func is not None:
            self.model_config.update_cache_config_func(self.cache_config)
            # TODO: Remove this mirror after graph and warmup metadata consume
            # CacheConfig.block_size directly.
            self.model_config.block_size = self.cache_config.block_size
            return
        if self.model_config.use_flash_mla is True:
            if self.cache_config.block_size != 64:
                raise ValueError('Please set block_size to 64 for flash_mla.')
            return
        # head_dim=256 requires block_size=128 on ascend.
        # Other models keep the user-provided block size.
        if (self.cache_config.device_type == 'ascend' and self.model_config.k_head_dim == 256 and
                (self.cache_config.block_size != 128 or self.cache_config.kernel_block_size != 128)):
            logger.warning(
                'Force `block_size=128` and `kernel_block_size=128` '
                f'(was block_size={self.cache_config.block_size}, '
                f'kernel_block_size={self.cache_config.kernel_block_size}) '
                'for head_dim=256 on ascend.')
            self.cache_config.block_size = 128
            self.cache_config.kernel_block_size = 128
            return
        # TODO: support kernel with both large head dim and large block size.
        if self.model_config.k_head_dim >= 512 and self.cache_config.block_size > 32:
            self.cache_config.block_size = 32
            self.cache_config.kernel_block_size = 32
            logger.warning(
                f'Update `block_size={self.cache_config.block_size}` for large `head_dim={self.model_config.k_head_dim}`.'  # noqa
            )

    def _get_state_cache_mem(self, states_shapes=None, cache_config=None, model_config=None):
        """Get state cache mem usage."""
        cache_config = cache_config or self.cache_config
        states_shapes = states_shapes if states_shapes is not None else cache_config.states_shapes
        if len(states_shapes) == 0:
            return 0

        from lmdeploy.pytorch.engine.cache_engine import StateCacheEngine

        num_state_caches = cache_config.num_state_caches
        if num_state_caches is None:
            # One state slot is reserved for system use. Active sequences need
            # max_batches runtime slots plus one spare for rolling prefill;
            # prefix-cache checkpoints use an explicitly configured extra budget.
            # TODO: Share memory between state cache and pageable cache
            num_state_caches = int(cache_config.max_batches + 2 + cache_config.prefix_cache_state_budget)
            cache_config.num_state_caches = num_state_caches

        if model_config is None:
            model_config = getattr(self, 'model_config', None)
        state_specs = getattr(model_config, 'state_cache_specs', None)
        mems = StateCacheEngine.get_state_slot_nbytes(states_shapes, state_specs=state_specs)
        mems *= num_state_caches

        return mems

    def _get_mem_state_cache_mem(self) -> int:
        """Get memory-model state cache mem usage for memdecode."""
        memdecode_config = self.misc_config.memdecode_config
        if memdecode_config is None:
            return 0
        memory_model_config = memdecode_config.memory_model_config
        if len(memory_model_config.states_shapes) == 0:
            return 0
        return self._get_state_cache_mem(memory_model_config.states_shapes, self.cache_config, memory_model_config)

    def _validate_memdecode_configs(self):
        """Validate MemDecode config compatibility."""
        memdecode_config = self.misc_config.memdecode_config
        if memdecode_config is None:
            return
        memory_model_config = memdecode_config.memory_model_config

        if self.specdecode_config is not None:
            raise ValueError('MemDecode and speculative decoding cannot be enabled together.')

        base_has_states = bool(self.model_config.states_shapes)
        memory_has_states = bool(memory_model_config.states_shapes)
        if base_has_states != memory_has_states:
            raise ValueError('Base and memory model must both use SSM state caches or both not use them.')

        base_vocab_size = self.model_config.vocab_size
        memory_vocab_size = memory_model_config.vocab_size
        if memory_vocab_size != base_vocab_size:
            logger.warning(
                f'Memory model vocab_size ({memory_vocab_size}) differs from base vocab_size ({base_vocab_size}); '
                'fusion logits will be aligned to the base vocab before sampling.'
            )

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
        state_cache_mem = self._get_state_cache_mem() + self._get_mem_state_cache_mem()
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

    def _prepare_worker_cache_plans(self, cache_config: CacheConfig,
                                    spec_cache_config: CacheConfig | None = None) -> list[_WorkerCachePlanSizes]:
        """Ask each worker to retain its cache plans and return byte sizes."""
        raise NotImplementedError('Not Implemented.')

    def _reserve_runtime_mem(self, free_mems: list[int], cache_block_sizes: list[_WorkerCachePlanSizes],
                             spec_cache_config: CacheConfig | None) -> list[int]:
        """Reserve runtime memory and update prefill token limit if needed."""
        dsa_score_workspace = self._get_dsa_score_workspace_size()
        if dsa_score_workspace > 0:
            logger.info('Reserve %d MiB for DSA prefill score workspace.',
                        dsa_score_workspace >> 20)
        runtime_mem, max_prefill_token_num = self._get_runtime_size(free_mems, cache_block_sizes,
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

    def _update_num_gpu_blocks(self, free_mems: list[int], cache_block_sizes: list[_WorkerCachePlanSizes],
                               spec_cache_config: CacheConfig | None) -> None:
        """Update target and spec GPU block counts from remaining memory."""
        if self.cache_config.num_gpu_blocks != 0:
            # User supplied an explicit block count. Do not resize it from the
            # current free-memory snapshot.
            if spec_cache_config is not None:
                spec_cache_config.num_gpu_blocks = self.cache_config.num_gpu_blocks
            return

        available_mems = [int(free_mem * self.cache_config.cache_max_entry_count) for free_mem in free_mems]
        rank_cache_block_sizes = self._get_rank_cache_block_sizes(cache_block_sizes)
        self.cache_config.num_gpu_blocks = self._get_min_num_gpu_blocks(available_mems, rank_cache_block_sizes)
        if self.cache_config.num_gpu_blocks <= 2:
            raise RuntimeError('No enough gpu memory for kv cache.')
        if spec_cache_config is not None:
            spec_cache_config.num_gpu_blocks = self.cache_config.num_gpu_blocks

    def update_configs(self) -> None:
        """Update cache config."""
        self._adjust_block_size()
        self._maybe_disable_unsupported_prefix_caching()
        self._sync_spec_cache_block_size()
        self._validate_memdecode_configs()
        self.cache_config.states_shapes = self.model_config.states_shapes

        spec_cache_config, spec_model_config = self._get_spec_configs()
        cache_block_sizes = self._prepare_worker_cache_plans(self.cache_config, spec_cache_config)

        free_mems = self._get_free_gpu_mems()
        free_mems = self._reserve_state_cache_mem(free_mems)
        free_mems = self._reserve_runtime_mem(free_mems, cache_block_sizes, spec_cache_config)
        self._update_num_gpu_blocks(free_mems, cache_block_sizes, spec_cache_config)

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
        if self.misc_config.memdecode_config is not None:
            logger.info('Building MemDecode memory KV/state cache engines.')
        self.build_cache_engine()
        if self.misc_config.empty_init:
            logger.info('Skip warming up model during empty init.')
            return
        logger.info('Warming up model.')
        self.warmup()

    @contextlib.contextmanager
    def remote_log(self, msg: str):
        """Send log for debugging.

        Do not use it in production.
        """
        # Different executor may have different log sending logic.
        yield
