# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.conn.protocol import DistServeInitRequest, DistServeKVTransferEndpointInfo
from lmdeploy.pytorch.disagg.messages import (AssignmentInstruct, DistServeRegisterMRMessage, MigrationAssignment,
                                              MigrationExecutionBatch)
from lmdeploy.utils import get_logger

from ..config import CacheConfig, ModelConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]

logger = get_logger('lmdeploy')


def round_up(x: int, alignment: int) -> int:
    """Round up x to the nearest multiple of alignment."""
    return ((x + alignment - 1) // alignment) * alignment


@dataclass
class CacheDesc:
    """Cache description."""
    shape: List[int]
    dtype: torch.dtype
    alignment: int = 256

    def __post_init__(self):
        self.size = math.prod(self.shape) * self.dtype.itemsize
        self.aligned_size = round_up(self.size, self.alignment)


class CacheEngine:
    """Host and Device memory maintainer.

    Args:
        cache_config (CacheConfig): config of the cache information.
        model_config (ModelConfig): config of the model.
        rank (int): distribution rank, 0 on non-distributed environment.
        world_size (int): distribution world size, 1 on non-distributed
            environment.
        cache_stream (torch.cuda.Stream): the stream used for cache engine swap,
            if set to None, it's created in CacheEngine.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        rank: int = 0,
        tp_rank: int = 0,
        world_size: int = 1,
        cache_stream: torch.cuda.Stream = None,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.tp_rank = tp_rank
        self.cache_config = cache_config
        self.model_config = model_config

        self.block_size = cache_config.block_size
        self.num_layers = model_config.num_layers
        self.kv_cache_dtype = model_config.dtype
        if cache_config.quant_policy > 0:
            if self.cache_config.device_type in ['cuda']:
                self.kv_cache_dtype = torch.uint8
            elif self.cache_config.device_type in ['ascend', 'npu']:
                self.kv_cache_dtype = torch.int8
            else:
                raise ValueError(f'unsupported device_type {self.cache_config.device_type}')

        # Initialize the cache.
        self.local_gpu_cache = self.allocate_gpu_cache()
        self.local_cpu_cache = self.allocate_cpu_cache()

        self.migration_backend_impl: Optional[MigrationBackendImpl] = None

        # Initialize the stream for caching operations.
        self.cache_stream = cache_stream or torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = torch.cuda.Event()

        logger.debug(f'Initialize cache engine with {cache_config.num_gpu_blocks}'
                     f' gpu blocks and {cache_config.num_cpu_blocks} cpu blocks.')

    @property
    def cpu_cache(self):
        """Gpu cache."""
        return self.local_cpu_cache

    @property
    def gpu_cache(self):
        """Gpu cache."""
        return self.local_gpu_cache

    @property
    def num_gpu_blocks(self):
        """Num gpu blocks."""
        return self.cache_config.num_gpu_blocks

    @property
    def num_cpu_blocks(self):
        """Num gpu blocks."""
        return self.cache_config.num_cpu_blocks

    @classmethod
    def _get_key_block_shape_impl(cls,
                                  model_config: ModelConfig,
                                  block_size: int,
                                  head_size: int,
                                  world_size: int = 1,
                                  quant_policy: Literal[0, 4, 8] = 0,
                                  local: bool = True):
        """Get single block shape."""
        attn_backend = get_backend()
        dtype = model_config.dtype
        num_heads = model_config.num_key_value_heads
        if local:
            assert num_heads % world_size == 0, \
                f'num_heads: {num_heads}, world_size: {world_size}'
            num_heads = num_heads // world_size
        if quant_policy == 4:  # pack head_dim to uint8
            assert head_size % 2 == 0, \
                f'head_size: {head_size}, quant_policy: {quant_policy}'
            head_size = head_size // 2
        return attn_backend.get_k_block_shape(block_size, num_heads, head_size, dtype)

    @classmethod
    def _get_value_block_shape_impl(cls,
                                    model_config: ModelConfig,
                                    block_size: int,
                                    head_size: int,
                                    world_size: int = 1,
                                    quant_policy: Literal[0, 4, 8] = 0,
                                    local: bool = True):
        """Get single block shape."""
        attn_backend = get_backend()
        dtype = model_config.dtype
        num_heads = model_config.num_key_value_heads
        if local:
            assert num_heads % world_size == 0, \
                f'num_heads: {num_heads}, world_size: {world_size}'
            num_heads = num_heads // world_size
        if quant_policy == 4:  # pack head_dim to uint8
            assert head_size % 2 == 0, \
                f'head_size: {head_size}, quant_policy: {quant_policy}'
            head_size = head_size // 2

        return attn_backend.get_v_block_shape(block_size, num_heads, head_size, dtype)

    def get_key_block_shape(self, local: bool = False) -> Tuple[int, int, int]:
        """Get shape of key block."""
        head_size = self.model_config.k_head_dim
        if head_size is None:
            head_size = self.model_config.head_dim
        return self._get_key_block_shape_impl(
            self.model_config,
            block_size=self.block_size,
            head_size=head_size,
            world_size=self.world_size,
            quant_policy=self.cache_config.quant_policy,
            local=local,
        )

    def get_value_block_shape(self, local: bool = False) -> Tuple[int, int, int]:
        """Get shape of value block."""
        head_size = self.model_config.v_head_dim
        if head_size is None:
            head_size = self.model_config.head_dim
        return self._get_value_block_shape_impl(
            self.model_config,
            block_size=self.block_size,
            head_size=head_size,
            world_size=self.world_size,
            quant_policy=self.cache_config.quant_policy,
            local=local,
        )

    def _allocate_cache(self, num_blocks: int, device: torch.device):
        """Allocate cache implement."""
        key_block_shape = self.get_key_block_shape(local=True)
        value_block_shape = self.get_value_block_shape(local=True)

        num_layers = self.num_layers
        kv_cache_dtype = self.kv_cache_dtype

        key_cache = torch.empty(
            size=(num_layers, num_blocks, *key_block_shape),
            dtype=kv_cache_dtype,
            device=device,
        )
        value_cache = torch.empty(
            size=(num_layers, num_blocks, *value_block_shape),
            dtype=kv_cache_dtype,
            device=device,
        )

        output = (key_cache, value_cache)

        if self.cache_config.quant_policy in (4, 8):
            dtype = self.model_config.dtype
            key_sz_cache = torch.empty(
                size=(num_layers, num_blocks, *key_block_shape[:-1], 2),
                dtype=dtype,
                device=device,
            )
            val_sz_cache = torch.empty(
                size=(num_layers, num_blocks, *value_block_shape[:-1], 2),
                dtype=dtype,
                device=device,
            )
            output = output + (key_sz_cache, val_sz_cache)

        return output

    def allocate_gpu_cache(self):
        """Allocate caches on GPU."""
        caches = self._allocate_cache(self.num_gpu_blocks, 'cuda')
        self.full_gpu_cache = caches
        self.local_gpu_cache = list(zip(*caches))
        return self.local_gpu_cache

    def allocate_cpu_cache(self):
        """Allocate caches on Host."""
        caches = self._allocate_cache(self.num_cpu_blocks, 'cpu')

        self.full_cpu_cache = caches
        self.local_cpu_cache = list(zip(*caches))
        return self.local_cpu_cache

    @torch.inference_mode()
    def _swap(self, src: List[torch.Tensor], dst: List[torch.Tensor], src_to_dst: Dict[int, int]):
        """Move caches from src memory to dst memory.

        Args:
            src (List[KVCache]): Source cache.
            dst (List[KVCache]): Destination cache.
            src_to_dst (Dict[int, int]): Map between src and dst.
        """
        BLOCKS_PER_COPY = 2
        num_copy = len(src_to_dst)
        src_idx, dst_idx = list(zip(*src_to_dst.items()))
        src_idx = torch.tensor(src_idx, device=src[0].device)
        dst_idx = torch.tensor(dst_idx, device=dst[0].device)
        with torch.cuda.stream(self.cache_stream):
            for scache, dcache in zip(src, dst):
                for idx in range(0, num_copy, BLOCKS_PER_COPY):
                    sidx = src_idx[idx:idx + BLOCKS_PER_COPY]
                    didx = dst_idx[idx:idx + BLOCKS_PER_COPY]
                    sdata = scache[:, sidx]
                    dcache.index_copy_(1, didx, sdata.to(dcache.device))
            self.events.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        """Move cache from Host to Device.

        Args:
            src_to_dst (Dict[int, int]): Map between src and dst.
        """
        self._swap(self.full_cpu_cache, self.full_gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        """Move cache from Device to Host.

        Args:
            src_to_dst (Dict[int, int]): Map between src and dst.
        """
        self._swap(self.full_gpu_cache, self.full_cpu_cache, src_to_dst)

    @classmethod
    def get_cache_block_size(cls,
                             block_size: int,
                             model_config: ModelConfig,
                             world_size: int = 1,
                             quant_policy: int = 0) -> int:
        """Get the required cache size of the model.

        Args:
            block_size (int): The token numbers of the block.
            model_config (ModelConfig): The config of the model.

        Return:
            int: Required memory size in bytes.
        """
        num_layers = model_config.num_layers
        key_head_size = model_config.k_head_dim
        value_head_size = model_config.v_head_dim
        if key_head_size is None:
            key_head_size = model_config.head_dim
        if value_head_size is None:
            value_head_size = model_config.head_dim
        key_shape = cls._get_key_block_shape_impl(
            model_config,
            block_size=block_size,
            head_size=key_head_size,
            world_size=world_size,
            local=True,
            quant_policy=quant_policy,
        )
        value_shape = cls._get_value_block_shape_impl(
            model_config,
            block_size=block_size,
            head_size=value_head_size,
            world_size=world_size,
            quant_policy=quant_policy,
            local=True,
        )
        if quant_policy == 0:
            dtype = model_config.dtype
            key_block = torch.empty(key_shape, dtype=dtype, device='meta')
            value_block = torch.empty(value_shape, dtype=dtype, device='meta')
            mem_key_block = key_block.numel() * key_block.element_size()
            mem_value_block = value_block.numel() * value_block.element_size()
        elif quant_policy in (4, 8):
            key_block = torch.empty(key_shape, dtype=torch.uint8, device='meta')
            value_block = torch.empty(value_shape, dtype=torch.uint8, device='meta')
            key_scale_zero_block = torch.empty((*key_shape[:-1], 2), dtype=model_config.dtype, device='meta')
            value_scale_zero_block = torch.empty((*value_shape[:-1], 2), dtype=model_config.dtype, device='meta')
            mem_key_block = key_block.numel() * key_block.element_size() + key_scale_zero_block.numel(
            ) * key_scale_zero_block.element_size()
            mem_value_block = value_block.numel() * value_block.element_size() + value_scale_zero_block.numel(
            ) * value_scale_zero_block.element_size()
        else:
            raise ValueError(f'unsupported quant_policy {quant_policy}')

        total = num_layers * (mem_key_block + mem_value_block)
        return total

    """ Metheds for PD Disaggregation Begin. """

    def p2p_initialize(self, migration_init_request: DistServeInitRequest) -> DistServeKVTransferEndpointInfo:
        if not self.migration_backend_impl:
            self.migration_backend_impl = MIGRATION_BACKENDS.module_dict[self.cache_config.migration_backend.name]()
        migration_init_request.rank = self.rank
        self.migration_backend_impl.p2p_initialize(migration_init_request)
        for i, t in enumerate(self.full_gpu_cache):
            if t.numel() == 0:
                continue
            register_mr_request = DistServeRegisterMRMessage(protocol=migration_init_request.protocol,
                                                             remote_engine_id=migration_init_request.remote_engine_id,
                                                             mr_key=str(i),
                                                             addr=t.data_ptr(),
                                                             offset=t.storage_offset(),
                                                             length=t.numel() * t.itemsize)
            self.migration_backend_impl.register_memory_region(register_mr_request)
        return DistServeKVTransferEndpointInfo(protocol=migration_init_request.protocol,
                                               endpoint_info=json.dumps(
                                                   self.migration_backend_impl.endpoint_info(
                                                       migration_init_request.remote_engine_id,
                                                       migration_init_request.protocol)))

    def p2p_connect(self, remote_engine_id: str, migration_conn_request: List[DistServeKVTransferEndpointInfo]):
        self.migration_backend_impl.p2p_connect(remote_engine_id, migration_conn_request[self.tp_rank])

    async def migrate(self, migration_execution_inputs: MigrationExecutionBatch):

        def get_assignment_len():
            head_dim = self.model_config.get_head_size()
            num_heads = self.model_config.num_key_value_heads // self.world_size
            block_size = self.cache_config.block_size
            return head_dim * num_heads * block_size * self.model_config.dtype.itemsize

        assignment_len = get_assignment_len()
        layer_stride = self.cache_config.num_gpu_blocks * assignment_len

        def get_assignment_batch(mr_key, block_ids, assignment_len, layer_stride, remote_layer_stride):
            return [
                AssignmentInstruct(mr_key=mr_key,
                                   target_offset=block_id[0] * assignment_len + layer * remote_layer_stride,
                                   source_offset=block_id[1] * assignment_len + layer * layer_stride,
                                   length=assignment_len) for layer in range(self.model_config.num_layers)
                for block_id in block_ids
            ]

        assignment_batch: List[Tuple[str, int, int, int]] = []  # mr_key, target, source, offset
        for migration_exe_req in migration_execution_inputs.requests:
            remote_engine_id = migration_exe_req[0]
            blocks_to_migration = migration_exe_req[1]
            remote_layer_stride = self.migration_backend_impl.links[
                remote_engine_id].remote_engine_config.num_gpu_blocks * assignment_len

            for i, t in enumerate(self.full_gpu_cache):
                if t.numel() == 0:
                    continue
                assignment_batch.extend(
                    get_assignment_batch(str(i), blocks_to_migration, assignment_len, layer_stride,
                                         remote_layer_stride))
        await self.migration_backend_impl.p2p_migrate(
            MigrationAssignment(
                protocol=migration_execution_inputs.protocol,
                remote_engine_id=remote_engine_id,
                batch=assignment_batch,
            ))

    """ Metheds for PD Disaggregation End. """


class StateCacheEngine:
    """Cache engine for state cache."""

    def __init__(self, cache_config: CacheConfig):
        self.cache_config = cache_config
        self.mem_pool, self._state_caches = self.allocate_caches(num_caches=cache_config.num_state_caches,
                                                                 state_shapes=cache_config.states_shapes,
                                                                 device='cuda')

    @staticmethod
    def allocate_caches(num_caches: int, state_shapes: List[Tuple[Tuple[int], torch.dtype]], device: torch.device):
        """Allocate cache implement."""

        if len(state_shapes) == 0 or num_caches == 0:
            return torch.empty((0, 0), dtype=torch.uint8, device=device), []

        cache_descs = [CacheDesc(shape, dtype) for shape, dtype in state_shapes]

        # get mempool size
        mem_pool_size = 0
        for desc in cache_descs:
            mem_pool_size += desc.aligned_size

        # create pool
        mem_pool = torch.zeros((num_caches, mem_pool_size), dtype=torch.uint8, device=device)

        # slice caches
        caches = []
        remain_pool = mem_pool
        for desc in cache_descs:
            cache = remain_pool[:, :desc.size].view(desc.dtype).view((num_caches, *desc.shape))
            remain_pool = remain_pool[:, desc.aligned_size:]
            caches.append(cache)
        return mem_pool, caches

    @staticmethod
    def get_cache_state_size(state_shapes: List[Tuple[Tuple[int], torch.dtype]]) -> int:
        """Get the required cache size of the state cache.

        Args:
            state_shapes (List[Tuple[Tuple[int], torch.dtype]]): The shapes and dtypes of the states.

        Return:
            int: Required memory size in bytes.
        """
        mem_pool, _ = StateCacheEngine.allocate_caches(num_caches=1, state_shapes=state_shapes, device='meta')
        return mem_pool.numel() * mem_pool.element_size()

    @property
    def state_caches(self):
        """State caches."""
        return self._state_caches

    def init_caches(self, idx: torch.Tensor, mask: torch.Tensor):
        """Initialize state caches.

        idx: indices of caches to be initialized.
        mask: mask to indicate which idx to be initialized.
        """
        if idx is None:
            return

        if len(self._state_caches) <= 0:
            return

        num_caches = self.cache_config.num_state_caches

        # get mask of all caches so we can perform inplace mask fill
        cache_masks = torch.zeros((num_caches, ), dtype=torch.bool, device=idx.device)
        cache_masks.index_copy_(0, idx, mask)
        reshaped_mask = cache_masks.view((-1, ) + (1, ) * (self.mem_pool.dim() - 1))
        self.mem_pool.masked_fill_(reshaped_mask, 0)
