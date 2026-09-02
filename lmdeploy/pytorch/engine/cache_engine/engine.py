# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeCachePoolInfo,
    DistServeInitRequest,
    DistServeKVTransferEndpointInfo,
)
from lmdeploy.pytorch.disagg.messages import (
    DistServeRegisterMRMessage,
    MigrationAssignment,
    MigrationExecutionBatch,
)
from lmdeploy.utils import get_logger

from ....messages import QuantPolicy
from ...config import CacheConfig
from .layout import CachePool
from .migration import (
    build_cache_pool_assignments,
    describe_cache_pools,
    infer_remote_pool_without_metadata,
    validate_cache_pool_layouts,
)
from .plan import BlockCachePlan
from .view import NamedCacheView

logger = get_logger('lmdeploy')


_KV_CACHE_QUANT_POLICY_DESCS = {
    QuantPolicy.FP8: 'fp8_e4m3 KV cache',
    QuantPolicy.FP8_E5M2: 'fp8_e5m2 KV cache',
    QuantPolicy.INT4: 'int4 KV cache',
    QuantPolicy.INT8: 'int8 KV cache',
    QuantPolicy.TURBO_QUANT: 'TurboQuant KV cache',
}


class CacheEngine:
    """Own block-cache allocations and runtime movement.

    Args:
        cache_config (CacheConfig): config of the cache information.
        rank (int): distribution rank, 0 on non-distributed environment.
        tp_rank (int): rank within the attention tensor-parallel group.
        cache_stream (torch.cuda.Stream): the stream used for cache engine swap,
            if set to None, it's created in CacheEngine.
        block_cache_plan (BlockCachePlan): finalized worker-local allocation
            recipe shared by sizing, CPU allocation, and device allocation.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        rank: int = 0,
        tp_rank: int = 0,
        cache_stream: torch.cuda.Stream | None = None,
        *,
        block_cache_plan: BlockCachePlan,
    ) -> None:
        self.rank = rank
        self.tp_rank = tp_rank
        self.cache_config = cache_config

        quant_desc = _KV_CACHE_QUANT_POLICY_DESCS.get(cache_config.quant_policy)
        if quant_desc is not None:
            logger.info('Using %s.', quant_desc)

        self.block_cache_plan = block_cache_plan

        # Initialize the cache.
        self.local_gpu_cache = self.allocate_gpu_cache()
        self.local_cpu_cache = self.allocate_cpu_cache()
        self._build_swap_pairs()
        self._build_block_copy()

        self.migration_backend_impl: MigrationBackendImpl | None = None
        self._pd_cache_pool_infos: tuple[DistServeCachePoolInfo, ...] | None = None
        self._remote_pd_cache_pool_infos: dict[str, tuple[DistServeCachePoolInfo, ...]] = {}

        # Initialize the stream for caching operations.
        # Non-CUDA device integrations currently provide CUDA-compatible torch
        # APIs in their backend layer, so the cache engine keeps this path.
        self.cache_stream = cache_stream or torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.swap_event = torch.cuda.Event()

        logger.debug(f'Initialize cache engine with {cache_config.num_gpu_blocks}'
                     f' gpu blocks and {cache_config.num_cpu_blocks} cpu blocks.')

    @property
    def cpu_cache(self):
        """CPU cache tensors in per-layer model order."""
        return self.local_cpu_cache

    @property
    def gpu_cache(self):
        """Device cache tensors in per-layer model order."""
        return self.local_gpu_cache

    def _build_model_layer_cache(self, caches: Sequence[torch.Tensor]):
        """Build the per-layer model cache without scoped named tensors."""
        caches = [caches[index] for index in self.block_cache_plan.model_cache_indices]
        return list(zip(*caches))

    def allocate_gpu_cache(self):
        """Allocate caches on GPU."""
        # Non-CUDA device integrations patch the canonical "cuda" device path
        # before reaching this layer, so keep using it here.
        self.gpu_allocation = self.block_cache_plan.allocate(
            num_logical_blocks=self.cache_config.num_gpu_blocks,
            device='cuda',
        )
        caches = self.gpu_allocation.tensor_views
        self._block_caches = self._build_block_cache_view(caches)
        return self._build_model_layer_cache(caches)

    def allocate_cpu_cache(self):
        """Allocate caches on Host."""
        self.cpu_allocation = self.block_cache_plan.allocate(
            num_logical_blocks=self.cache_config.num_cpu_blocks,
            device='cpu',
        )
        caches = self.cpu_allocation.tensor_views
        return self._build_model_layer_cache(caches)

    def _build_block_cache_view(self, caches: Sequence[torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """Build the model-facing view once for this device allocation."""
        tensor_specs = self.block_cache_plan.tensor_specs
        if any(spec.consumer_rows is not None for spec in tensor_specs):
            return NamedCacheView(tensor_specs, caches)
        return {
            spec.name: cache
            for spec, cache in zip(tensor_specs, caches)
        }

    @property
    def block_caches(self) -> Mapping[str, torch.Tensor]:
        """Return all standard and operator-requested caches by name."""
        return self._block_caches

    @property
    def connector_kv_caches(self) -> Mapping[str, torch.Tensor]:
        """Return physical cache-pool rows for an external KV connector.

        Each value is one owning-pool row containing all physical kernel pages for that row. Pool and row indices form
        stable log-only keys; transfer identity comes from rank and content hashes. Both endpoints must realize the same
        ordered layout.

        Current connector registration requires row-major block pools whose movable kernel-page axis is 1. Empty pools
        are omitted. The returned mapping is structurally read-only, and callers must not retain its temporary row views
        beyond the cache engine's lifetime.
        """
        allocation = self.gpu_allocation

        connector_caches: dict[str, torch.Tensor] = {}
        for pool_index, pool in enumerate(allocation.pools):
            if pool.nbytes == 0:
                continue
            if pool.entry_axis != 1:
                raise ValueError(
                    f'External KV connectors require cache pool {pool_index} '
                    f'to use entry_axis=1, got {pool.entry_axis}.')
            for row_index, row in enumerate(pool.tensor):
                key = f'cache_pool.{pool_index}.row.{row_index}'
                connector_caches[key] = row

        return MappingProxyType(connector_caches)

    def _build_swap_pairs(self):
        """Resolve compatible CPU-to-device cache entries once at build
        time."""
        cpu_entries = [(pool.tensor, pool.entry_axis) for pool in self.cpu_allocation.pools]
        gpu_entries = [(pool.tensor, pool.entry_axis) for pool in self.gpu_allocation.pools]

        if len(cpu_entries) != len(gpu_entries):
            raise RuntimeError('CPU and device cache layouts must contain the same number of entries.')

        swap_in_pairs = []
        for idx, ((cpu_cache, cpu_axis), (gpu_cache, gpu_axis)) in enumerate(zip(cpu_entries, gpu_entries)):
            if cpu_axis != gpu_axis:
                raise RuntimeError(f'CPU and device cache entry axes differ for pool {idx}.')
            if cpu_cache.dtype != gpu_cache.dtype:
                raise RuntimeError(f'CPU and device cache dtypes differ for pool {idx}.')
            cpu_payload_shape = cpu_cache.shape[:cpu_axis] + cpu_cache.shape[cpu_axis + 1:]
            gpu_payload_shape = gpu_cache.shape[:gpu_axis] + gpu_cache.shape[gpu_axis + 1:]
            if cpu_payload_shape != gpu_payload_shape:
                raise RuntimeError(f'CPU and device cache payload shapes differ for pool {idx}.')
            swap_in_pairs.append((cpu_cache, gpu_cache, cpu_axis))

        self._swap_in_pairs = tuple(swap_in_pairs)
        self._swap_out_pairs = tuple((dst, src, axis) for src, dst, axis in swap_in_pairs)

    def _build_block_copy(self):
        """Build local logical-block copy from the device allocation."""
        pages_per_block = self.block_cache_plan.kernel_blocks_per_logical_block
        cache_backend = get_backend().get_cache_backend()
        self._block_copy = cache_backend.build_block_copy(
            self.gpu_allocation,
            num_logical_blocks=self.cache_config.num_gpu_blocks,
            pages_per_block=pages_per_block,
        )

    @torch.inference_mode()
    def copy_logical_blocks(self, copy_plan: torch.Tensor) -> None:
        """Copy complete scheduler-sized blocks on the current stream.

        ``copy_plan`` contains physical block-table offsets with shape
        ``[2, num_pairs]``. The host-side plan owner validates relationships
        and lifetimes before dispatch; this device path never reads index
        values back to the host.
        """
        if not isinstance(copy_plan, torch.Tensor):
            raise TypeError('copy_plan must be a torch.Tensor.')
        if copy_plan.dim() != 2 or copy_plan.size(0) != 2:
            raise ValueError('copy_plan must have shape [2, num_pairs].')
        if copy_plan.dtype != torch.long:
            raise TypeError('copy_plan must use torch.long indices.')

        block_copy = self._block_copy
        if copy_plan.device != block_copy.device:
            raise ValueError('copy_plan must be on the block-cache allocation device.')
        if copy_plan.size(1) == 0:
            return
        block_copy.copy(copy_plan[0], copy_plan[1])

    @torch.inference_mode()
    def _swap(self, cache_pairs: tuple[tuple[torch.Tensor, torch.Tensor, int], ...],
              src_to_dst: dict[int, int]):
        """Move caches from src memory to dst memory.

        Args:
            cache_pairs: Source cache, destination cache, and entry axis.
            src_to_dst: Map between source and destination scheduler-block
                offsets.
        """
        if not cache_pairs or not src_to_dst:
            return

        LOGICAL_BLOCKS_PER_COPY = 2
        pages_per_block = self.block_cache_plan.kernel_blocks_per_logical_block
        src_blocks, dst_blocks = list(zip(*src_to_dst.items()))
        src_entries = [
            block * pages_per_block + page for block in src_blocks for page in range(pages_per_block)
        ]
        dst_entries = [
            block * pages_per_block + page for block in dst_blocks for page in range(pages_per_block)
        ]
        src_idx = torch.tensor(src_entries, device=cache_pairs[0][0].device)
        dst_idx = torch.tensor(dst_entries, device=cache_pairs[0][1].device)

        entries_per_copy = LOGICAL_BLOCKS_PER_COPY * pages_per_block
        num_entries = src_idx.numel()
        with torch.cuda.stream(self.cache_stream):
            for scache, dcache, entry_axis in cache_pairs:
                for idx in range(0, num_entries, entries_per_copy):
                    sidx = src_idx[idx:idx + entries_per_copy]
                    didx = dst_idx[idx:idx + entries_per_copy]
                    sdata = scache.index_select(entry_axis, sidx)
                    dcache.index_copy_(entry_axis, didx, sdata.to(dcache.device))
            self.swap_event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: dict[int, int]) -> None:
        """Move cache from Host to Device.

        Args:
            src_to_dst (dict[int, int]): Map between src and dst.
        """
        self._swap(self._swap_in_pairs, src_to_dst)

    def swap_out(self, src_to_dst: dict[int, int]) -> None:
        """Move cache from Device to Host.

        Args:
            src_to_dst (dict[int, int]): Map between src and dst.
        """
        self._swap(self._swap_out_pairs, src_to_dst)

    # PD disaggregation.

    def _resolve_pd_cache_pools(self) -> tuple[CachePool, ...]:
        """Return owning pools with the metadata required by PD migration."""
        if self.cache_config.block_size != self.cache_config.kernel_block_size:
            raise RuntimeError('PD migration does not support block_size != kernel_block_size.')

        return self.gpu_allocation.pools

    def _get_pd_cache_pool_infos(self) -> tuple[DistServeCachePoolInfo, ...]:
        """Describe the stable local allocation once for every PD link."""
        pool_infos = self._pd_cache_pool_infos
        if pool_infos is None:
            pool_infos = describe_cache_pools(self._resolve_pd_cache_pools(), self.cache_config.num_gpu_blocks)
            self._pd_cache_pool_infos = pool_infos
        return pool_infos

    def p2p_initialize(self, migration_init_request: DistServeInitRequest) -> DistServeKVTransferEndpointInfo:
        pools = self._resolve_pd_cache_pools()
        pool_infos = describe_cache_pools(pools, self.cache_config.num_gpu_blocks)
        self._pd_cache_pool_infos = pool_infos
        if self.migration_backend_impl is None:
            self.migration_backend_impl = MIGRATION_BACKENDS.module_dict[self.cache_config.migration_backend.name]()
        migration_init_request.rank = self.rank
        self.migration_backend_impl.p2p_initialize(migration_init_request)
        for mr_key, pool in enumerate(pools):
            tensor = pool.tensor
            if tensor.numel() == 0:
                continue
            register_mr_request = DistServeRegisterMRMessage(protocol=migration_init_request.protocol,
                                                             remote_engine_id=migration_init_request.remote_engine_id,
                                                             mr_key=mr_key,
                                                             addr=tensor.data_ptr(),
                                                             offset=tensor.storage_offset(),
                                                             length=tensor.numel() * tensor.itemsize)
            self.migration_backend_impl.register_memory_region(register_mr_request)
        return DistServeKVTransferEndpointInfo(protocol=migration_init_request.protocol,
                                               endpoint_info=json.dumps(
                                                   self.migration_backend_impl.endpoint_info(
                                                       migration_init_request.remote_engine_id,
                                                       migration_init_request.protocol)),
                                               cache_pools=pool_infos)

    def p2p_connect(self, remote_engine_id: str, migration_conn_request: list[DistServeKVTransferEndpointInfo]):
        conn_request = migration_conn_request[self.tp_rank]
        local_pools = self._get_pd_cache_pool_infos()
        remote_num_blocks = self.migration_backend_impl.links[
            remote_engine_id].remote_engine_config.num_gpu_blocks
        remote_pools = conn_request.cache_pools
        if remote_pools is None:
            remote_pools = infer_remote_pool_without_metadata(local_pools, remote_num_blocks)
        validate_cache_pool_layouts(local_pools, remote_pools, self.cache_config.num_gpu_blocks, remote_num_blocks)
        self._remote_pd_cache_pool_infos[remote_engine_id] = tuple(remote_pools)
        self.migration_backend_impl.p2p_connect(remote_engine_id, conn_request)

    async def migrate(self, migration_execution_inputs: MigrationExecutionBatch):
        local_pools = self._get_pd_cache_pool_infos()
        blocks_by_remote: dict[str, list[tuple[int, int]]] = {}
        for remote_engine_id, block_pairs in migration_execution_inputs.requests:
            blocks_by_remote.setdefault(remote_engine_id, []).extend(block_pairs)

        for remote_engine_id, block_pairs in blocks_by_remote.items():
            remote_pools = self._remote_pd_cache_pool_infos.get(remote_engine_id)
            if remote_pools is None:
                raise RuntimeError(f'PD cache-pool metadata is unavailable for remote engine {remote_engine_id}.')
            assignment_batch = build_cache_pool_assignments(local_pools, remote_pools, block_pairs)
            await self.migration_backend_impl.p2p_migrate(
                MigrationAssignment(
                    protocol=migration_execution_inputs.protocol,
                    remote_engine_id=remote_engine_id,
                    batch=assignment_batch,
                ))
