# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import json
from collections.abc import Callable, Mapping, Sequence
from operator import index as as_index

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.disagg.backend.backend import MIGRATION_BACKENDS
from lmdeploy.pytorch.disagg.backend.base import MigrationBackendImpl
from lmdeploy.pytorch.disagg.conn.protocol import DistServeInitRequest, DistServeKVTransferEndpointInfo
from lmdeploy.pytorch.disagg.messages import (
    AssignmentInstruct,
    DistServeRegisterMRMessage,
    MigrationAssignment,
    MigrationExecutionBatch,
)
from lmdeploy.utils import get_logger

from ....messages import QuantPolicy
from ...config import CacheConfig, ModelConfig, StateCacheSpec
from .layout import CacheAllocation
from .plan import BlockCachePlan
from .schema import (
    BlockCacheGeometry,
    CacheDesc,
    CacheResource,
    ScopedBlockCacheRequest,
    build_block_cache_resources,
    build_block_cache_resources_from_requests,
    build_state_cache_resources,
    layer_maps_from_resources,
)

KVCache = tuple[torch.Tensor, torch.Tensor]

logger = get_logger('lmdeploy')


def _unpack_cache_allocation(result):
    """Return the native owner and temporary compatibility views."""
    if isinstance(result, CacheAllocation):
        mem_pool, caches = result.as_legacy()
        return result, mem_pool, caches
    mem_pool, caches = result
    return None, mem_pool, caches


class NamedCacheView(Mapping[str, torch.Tensor]):
    """Dict-like named cache view with optional layer-scoped rows."""

    def __init__(self, caches: dict[str, torch.Tensor], layer_maps: dict[str, dict[int, int]] | None = None):
        self._caches = caches
        self._layer_maps = layer_maps or {}

    def __getitem__(self, name: str):
        return self._caches[name]

    def __contains__(self, name: str):
        return name in self._caches

    def __iter__(self):
        return iter(self._caches)

    def __len__(self):
        return len(self._caches)

    def layer(self, name: str, layer_id: int):
        """Return a named cache row for a global layer id."""
        layer_map = self._layer_maps.get(name)
        cache_row = layer_id
        if layer_map is not None:
            try:
                cache_row = layer_map[layer_id]
            except KeyError as e:
                raise RuntimeError(f'Layer {layer_id} does not own cache {name}.') from e
        return self._caches[name][cache_row]


def _get_kv_cache_dtype(model_config: ModelConfig):
    kv_cache_dtype = model_config.dtype
    if model_config.use_mla_fp8_cache:
        kv_cache_dtype = torch.float8_e4m3fn
    elif model_config.mla_kv_cache_dtype == 'bfloat16':
        kv_cache_dtype = torch.bfloat16
    return kv_cache_dtype


def _update_mla_kv_cache_dtype(model_config: ModelConfig, cache_config: CacheConfig):
    """Apply an explicit sparse MLA cache policy to the model config."""
    if model_config.mla_index_topk is None or cache_config.quant_policy == QuantPolicy.NONE:
        return
    if cache_config.quant_policy == QuantPolicy.FP8:
        model_config.mla_kv_cache_dtype = 'fp8_ds_mla'
        return
    raise ValueError(f'Sparse MLA does not support quant_policy={cache_config.quant_policy}. '
                     'Use none/0 for BF16 or fp8/16 for FP8.')


_FP8_CACHE_DTYPES = {
    QuantPolicy.FP8: torch.float8_e4m3fn,
    QuantPolicy.FP8_E5M2: torch.float8_e5m2,
}

_KV_CACHE_QUANT_POLICY_DESCS = {
    QuantPolicy.FP8: 'fp8_e4m3 KV cache',
    QuantPolicy.FP8_E5M2: 'fp8_e5m2 KV cache',
    QuantPolicy.INT4: 'int4 KV cache',
    QuantPolicy.INT8: 'int8 KV cache',
    QuantPolicy.TURBO_QUANT: 'TurboQuant KV cache',
}


def _is_fp8_quant_policy(quant_policy: QuantPolicy):
    """Return whether quant policy stores KV payload as torch FP8."""
    return quant_policy in _FP8_CACHE_DTYPES


def _get_fp8_cache_dtype(quant_policy: QuantPolicy):
    """Get the cache tensor dtype for an FP8 KV-cache quant policy."""
    try:
        return _FP8_CACHE_DTYPES[quant_policy]
    except KeyError as e:
        raise ValueError(f'Not an FP8 quant policy: {quant_policy}') from e


def _describe_kv_cache_quant_policy(quant_policy: QuantPolicy):
    """Describe the active KV-cache quantization policy for logs."""
    return _KV_CACHE_QUANT_POLICY_DESCS.get(quant_policy)


# 512*1 + 4*4 + 64*2 = 656
MLA_FP8_HEAD_DIM = 656


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
        block_cache_plan: BlockCachePlan | None = None,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.tp_rank = tp_rank
        self.cache_config = cache_config
        self.model_config = model_config
        _update_mla_kv_cache_dtype(model_config, cache_config)

        self.block_size = cache_config.kernel_block_size
        self.num_layers = model_config.num_layers
        self.kv_cache_dtype = _get_kv_cache_dtype(self.model_config)

        if self.model_config.mla_index_topk is not None:
            cache_config.quant_policy = 0

        if _is_fp8_quant_policy(cache_config.quant_policy):
            self.kv_cache_dtype = _get_fp8_cache_dtype(cache_config.quant_policy)
            assert self.cache_config.device_type in ['cuda'], \
                f'FP8 quantization is only supported on CUDA device, but got {self.cache_config.device_type}.'
        elif cache_config.quant_policy > 0:
            if self.cache_config.device_type in ['cuda']:
                self.kv_cache_dtype = torch.uint8
            elif self.cache_config.device_type in ['ascend', 'npu']:
                self.kv_cache_dtype = torch.int8
            else:
                raise ValueError(f'unsupported device_type {self.cache_config.device_type}')

        quant_desc = _describe_kv_cache_quant_policy(cache_config.quant_policy)
        if quant_desc is not None:
            logger.info('Using %s.', quant_desc)

        if block_cache_plan is None:
            block_cache_plan = self.build_cache_plan(model_config, cache_config, world_size)
        self.block_cache_plan = block_cache_plan

        # Initialize the cache.
        self.local_gpu_cache = self.allocate_gpu_cache()
        self.local_cpu_cache = self.allocate_cpu_cache()
        self._build_swap_pairs()

        self.migration_backend_impl: MigrationBackendImpl | None = None

        # Initialize the stream for caching operations.
        # Non-CUDA device integrations currently provide CUDA-compatible torch
        # APIs in their backend layer, so the cache engine keeps this path.
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
                                  quant_policy: QuantPolicy = QuantPolicy.NONE):
        """Get single block shape."""
        attn_backend = get_backend()
        dtype = model_config.dtype
        num_heads = model_config.num_key_value_heads

        # split heads by tp
        assert num_heads % world_size == 0, \
            f'num_heads: {num_heads}, world_size: {world_size}'
        num_heads = num_heads // world_size

        # patch for flash mla
        if model_config.use_mla_fp8_cache:
            return (block_size, num_heads, MLA_FP8_HEAD_DIM)

        # pack head_dim to uint8 (4-bit)
        if quant_policy == QuantPolicy.INT4 or quant_policy == QuantPolicy.TURBO_QUANT:
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
                                    quant_policy: QuantPolicy = QuantPolicy.NONE):
        """Get single block shape."""
        attn_backend = get_backend()
        dtype = model_config.dtype
        num_heads = model_config.num_key_value_heads

        # split heads by tp
        assert num_heads % world_size == 0, \
            f'num_heads: {num_heads}, world_size: {world_size}'
        num_heads = num_heads // world_size

        # patch for flash mla
        if model_config.use_mla_fp8_cache:
            # flash mla shared key and value
            return (block_size, num_heads, 0)

        if quant_policy == QuantPolicy.TURBO_QUANT:  # pack head_dim to uint8 (2-bit for V cache)
            assert head_size % 4 == 0, \
                f'head_size: {head_size}, quant_policy: {quant_policy}'
            head_size = head_size // 4
        elif quant_policy == QuantPolicy.INT4:  # pack head_dim to uint8 (4-bit)
            assert head_size % 2 == 0, \
                f'head_size: {head_size}, quant_policy: {quant_policy}'
            head_size = head_size // 2

        return attn_backend.get_v_block_shape(block_size, num_heads, head_size, dtype)

    @classmethod
    def get_k_cache_desc(cls, model_config: ModelConfig, cache_config: CacheConfig, world_size: int = 1) -> CacheDesc:
        """Get key cache description."""
        head_size = model_config.k_head_dim
        if head_size is None:
            head_size = model_config.head_dim
        shape = cls._get_key_block_shape_impl(
            model_config,
            block_size=cache_config.kernel_block_size,
            head_size=head_size,
            world_size=world_size,
            quant_policy=cache_config.quant_policy,
        )
        shape = list(shape)
        dtype = _get_kv_cache_dtype(model_config)
        if _is_fp8_quant_policy(cache_config.quant_policy):
            dtype = _get_fp8_cache_dtype(cache_config.quant_policy)
        elif cache_config.quant_policy in (QuantPolicy.INT4, QuantPolicy.INT8, QuantPolicy.TURBO_QUANT):
            dtype = torch.uint8
        return CacheDesc(shape=shape, dtype=dtype)

    @classmethod
    def get_v_cache_desc(cls, model_config: ModelConfig, cache_config: CacheConfig, world_size: int = 1) -> CacheDesc:
        """Get value cache description."""
        head_size = model_config.v_head_dim
        if head_size is None:
            head_size = model_config.head_dim
        shape = cls._get_value_block_shape_impl(
            model_config,
            block_size=cache_config.kernel_block_size,
            head_size=head_size,
            world_size=world_size,
            quant_policy=cache_config.quant_policy,
        )
        shape = list(shape)
        dtype = _get_kv_cache_dtype(model_config)
        if _is_fp8_quant_policy(cache_config.quant_policy):
            dtype = _get_fp8_cache_dtype(cache_config.quant_policy)
        elif cache_config.quant_policy in (QuantPolicy.INT4, QuantPolicy.INT8, QuantPolicy.TURBO_QUANT):
            dtype = torch.uint8
        return CacheDesc(shape=shape, dtype=dtype)

    @classmethod
    def get_quant_cache_descs(cls, k_cache_desc: CacheDesc, v_cache_desc: CacheDesc, model_config: ModelConfig,
                              cache_config: CacheConfig):
        """Get quant cache descs."""
        if cache_config.quant_policy == QuantPolicy.NONE:
            return []
        if _is_fp8_quant_policy(cache_config.quant_policy):
            # Regular FP8 KV cache uses fixed scalar scales from Attention, not
            # per-token scale/zero cache tensors.
            return []

        dtype = model_config.dtype
        # For quant_policy==QuantPolicy.TURBO_QUANT, K uses 4-bit quantization (has MSE norm and QJL norm),
        # V uses 2-bit quantization (only has MSE norm)
        if cache_config.quant_policy == QuantPolicy.TURBO_QUANT:
            key_scale_zero_shape = k_cache_desc.shape[:-1] + [2]
            val_scale_zero_shape = v_cache_desc.shape[:-1] + [1]
        else:
            key_scale_zero_shape = k_cache_desc.shape[:-1] + [2]
            val_scale_zero_shape = v_cache_desc.shape[:-1] + [2]
        key_scale_zero_desc = CacheDesc(shape=key_scale_zero_shape, dtype=dtype)
        val_scale_zero_desc = CacheDesc(shape=val_scale_zero_shape, dtype=dtype)
        return [key_scale_zero_desc, val_scale_zero_desc]

    @classmethod
    def _get_cache_resources(cls,
                             model_config: ModelConfig,
                             cache_config: CacheConfig,
                             world_size: int,
                             block_requests: Sequence[ScopedBlockCacheRequest]
                             | None = None) -> tuple[CacheResource, ...]:
        """Build the ordered resources consumed by the physical layout."""
        resources = []
        use_std = model_config.use_standard_kv_cache

        if use_std:
            k_cache_desc = cls.get_k_cache_desc(model_config, cache_config, world_size)
            v_cache_desc = cls.get_v_cache_desc(model_config, cache_config, world_size)
            quant_cache_descs = cls.get_quant_cache_descs(k_cache_desc, v_cache_desc, model_config, cache_config)
            resources.append(CacheResource(name='k_cache', desc=k_cache_desc))
            resources.append(CacheResource(name='v_cache', desc=v_cache_desc))
            for idx, desc in enumerate(quant_cache_descs):
                resources.append(CacheResource(name=f'quant_{idx}', desc=desc))

        if block_requests is not None:
            if use_std and len(block_requests) > 0:
                raise ValueError('Operator block cache requests cannot coexist with standard KV until mixed cache '
                                 'access metadata is implemented.')
            resources.extend(build_block_cache_resources_from_requests(block_requests))
        # named block cache specs (shape without block_size, same as cache_shapes)
        elif len(model_config.block_cache_specs) > 0:
            resources.extend(build_block_cache_resources(model_config.block_cache_specs))
        else:
            # legacy anonymous cache_shapes (shape without block_size)
            custom_descs = cls.get_custom_cache_descs(model_config, cache_config)
            for idx, desc in enumerate(custom_descs):
                resources.append(CacheResource(name=f'custom_{idx}', desc=desc))

        names = [resource.name for resource in resources]
        if len(names) != len(set(names)):
            raise ValueError('Block cache resource names must be unique after provider and fallback collection.')
        return tuple(resources)

    @classmethod
    def _uses_layer_scoped_block_caches(cls, model_config: ModelConfig):
        """Whether model-facing named caches use declared layer rows."""
        return (model_config is not None and not model_config.use_standard_kv_cache
                and len(model_config.block_cache_specs) > 0)

    @staticmethod
    def _get_block_cache_layer_maps(model_config: ModelConfig) -> dict[str, dict[int, int]]:
        """Build global-layer-id to local-row maps for named block caches."""
        if not CacheEngine._uses_layer_scoped_block_caches(model_config):
            return {}
        resources = build_block_cache_resources(model_config.block_cache_specs)
        return layer_maps_from_resources(resources)

    @classmethod
    def get_custom_cache_descs(cls, model_config: ModelConfig, cache_config: CacheConfig) -> list[CacheDesc]:
        """Get custom cache descs."""
        descs = []
        block_size = cache_config.kernel_block_size
        # named block cache specs (shape without block_size, same convention as cache_shapes)
        if len(model_config.block_cache_specs) > 0:
            for resource in build_block_cache_resources(model_config.block_cache_specs):
                descs.append(resource.desc)
            return descs
        # legacy cache_shapes
        if len(model_config.cache_shapes) > 0:
            for shape, dtype in model_config.cache_shapes:
                custom_shape = (block_size, *shape)
                descs.append(CacheDesc(shape=custom_shape, dtype=dtype))
        return descs

    @classmethod
    def build_cache_plan(
        cls,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        world_size: int,
        request_provider: Callable[[BlockCacheGeometry], Sequence[ScopedBlockCacheRequest]] | None = None,
    ) -> BlockCachePlan:
        """Finalize block geometry, resources, and backend layout."""
        geometry = BlockCacheGeometry(block_size=cache_config.block_size,
                                      kernel_block_size=cache_config.kernel_block_size)
        _update_mla_kv_cache_dtype(model_config, cache_config)
        block_requests = None
        if request_provider is not None:
            allocator = cls.allocate_caches
            allocator_func = getattr(allocator, '__func__', allocator)
            if allocator_func is not _NATIVE_BLOCK_ALLOCATOR:
                raise RuntimeError('Built-model cache providers require the native CacheEngine allocator.')
            block_requests = tuple(request_provider(geometry))
        num_layers = model_config.num_layers
        resources = cls._get_cache_resources(model_config, cache_config, world_size, block_requests=block_requests)
        cache_backend = get_backend().get_cache_backend()
        layout = cache_backend.build_block_layout(resources, num_layers=num_layers)
        return BlockCachePlan(
            resources=resources,
            layout=layout,
            kernel_blocks_per_logical_block=geometry.kernel_blocks_per_logical_block,
        )

    @classmethod
    def allocate_caches(cls, num_blocks: int, model_config: ModelConfig, cache_config: CacheConfig, world_size: int,
                        device: str) -> CacheAllocation:
        """Compatibility facade that builds and realizes one cache plan."""
        plan = cls.build_cache_plan(model_config, cache_config, world_size)
        return plan.allocate(num_logical_blocks=num_blocks, device=device)

    def _allocate_runtime_caches(self, num_blocks: int, device: str):
        """Realize the retained plan or use a legacy patched allocator."""
        plan = getattr(self, 'block_cache_plan', None)
        allocator = self.allocate_caches
        class_allocator = type(self).allocate_caches
        allocator_func = getattr(class_allocator, '__func__', class_allocator)
        if plan is not None and allocator_func is _NATIVE_BLOCK_ALLOCATOR:
            return plan.allocate(num_logical_blocks=num_blocks, device=device)
        return allocator(
            num_blocks=num_blocks,
            model_config=self.model_config,
            cache_config=self.cache_config,
            world_size=self.world_size,
            device=device,
        )

    def allocate_gpu_cache(self):
        """Allocate caches on GPU."""
        # Non-CUDA device integrations patch the canonical "cuda" device path
        # before reaching this layer, so keep using it here.
        result = self._allocate_runtime_caches(
            num_blocks=self.num_gpu_blocks,
            device='cuda',
        )
        self.gpu_allocation, mem_pool, caches = _unpack_cache_allocation(result)
        self.full_gpu_cache = mem_pool
        self._gpu_cache_list = caches
        plan = getattr(self, 'block_cache_plan', None)
        uses_layer_rows = plan.uses_layer_rows if plan is not None else self._uses_layer_scoped_block_caches(
            self.model_config)
        if uses_layer_rows:
            self.local_gpu_cache = []
        else:
            self.local_gpu_cache = list(zip(*caches))
        if plan is None:
            resources = self._get_cache_resources(self.model_config, self.cache_config, self.world_size)
            self._cache_names = [resource.name for resource in resources]
            self._block_cache_layer_maps = self._get_block_cache_layer_maps(self.model_config)
        else:
            self._cache_names = list(plan.cache_names)
            self._block_cache_layer_maps = plan.layer_maps
        self._cache_list = self._gpu_cache_list
        return self.local_gpu_cache

    def allocate_cpu_cache(self):
        """Allocate caches on Host."""
        result = self._allocate_runtime_caches(
            num_blocks=self.num_cpu_blocks,
            device='cpu',
        )
        self.cpu_allocation, mem_pool, caches = _unpack_cache_allocation(result)
        self.full_cpu_cache = mem_pool
        self._cpu_cache_list = caches
        plan = getattr(self, 'block_cache_plan', None)
        uses_layer_rows = plan.uses_layer_rows if plan is not None else self._uses_layer_scoped_block_caches(
            self.model_config)
        if uses_layer_rows:
            self.local_cpu_cache = []
        else:
            self.local_cpu_cache = list(zip(*caches))
        return self.local_cpu_cache

    @property
    def block_caches(self) -> Mapping[str, torch.Tensor]:
        """Return all caches (including k/v and custom) as a dict keyed by
        name."""
        if not hasattr(self, '_cache_names') or not hasattr(self, '_cache_list'):
            return {}
        caches = {
            name: cache
            for name, cache in zip(self._cache_names, self._cache_list)
        }
        layer_maps = getattr(self, '_block_cache_layer_maps', {})
        if not layer_maps:
            return caches
        return NamedCacheView(caches, layer_maps)

    @staticmethod
    def get_custom_cache_shape_impl(num_layers: int, num_blocks: int, block_size: int, shape: list[int]):
        """Get single block shape."""
        return (num_layers, num_blocks, block_size, *shape)

    @staticmethod
    def _allocate_single_custom_cache(shape: Sequence[int], dtype: torch.dtype, device: str):
        """Allocate custom cache."""
        return torch.empty(shape, dtype=dtype, device=device)

    def allocate_custom_cache(self, device: str):
        """Allocate custom caches on GPU."""
        num_layers = self.model_config.num_layers
        custom_caches = []
        for shape, dtype in self.model_config.cache_shapes:
            custom_shape = self.get_custom_cache_shape_impl(
                num_layers=num_layers,
                num_blocks=self.num_gpu_blocks,
                block_size=self.block_size,
                shape=shape,
            )
            custom_cache = self._allocate_single_custom_cache(shape=custom_shape, dtype=dtype, device=device)
            custom_caches.append(custom_cache)
        return custom_caches

    @staticmethod
    def _as_mem_pools(mem_pool: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        """Normalize one or many allocation pools."""
        if isinstance(mem_pool, torch.Tensor):
            return [mem_pool]
        return mem_pool

    @staticmethod
    def _mem_pool_nbytes(mem_pool: torch.Tensor | list[torch.Tensor]) -> int:
        """Return memory size for one or many allocation pools."""
        return sum(pool.numel() * pool.element_size() for pool in CacheEngine._as_mem_pools(mem_pool))

    def _build_swap_pairs(self):
        """Resolve compatible CPU-to-device cache entries once at build
        time."""
        cpu_allocation = self.cpu_allocation
        gpu_allocation = self.gpu_allocation
        if (cpu_allocation is None) != (gpu_allocation is None):
            raise RuntimeError('CPU and device caches must use the same allocation contract.')

        if cpu_allocation is not None:
            cpu_entries = [(pool.tensor, pool.entry_axis) for pool in cpu_allocation.pools]
            gpu_entries = [(pool.tensor, pool.entry_axis) for pool in gpu_allocation.pools]
        else:
            # Existing dlinfer patches return raw owning envelopes whose axes do
            # not describe cache blocks. Their typed resource views retain the
            # legacy [layer, block, ...] contract, so use those views directly.
            cpu_entries = [(cache, 1) for cache in self._cpu_cache_list]
            gpu_entries = [(cache, 1) for cache in self._gpu_cache_list]

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

    @torch.inference_mode()
    def _swap(self, cache_pairs: tuple[tuple[torch.Tensor, torch.Tensor, int], ...],
              src_to_dst: dict[int, int]):
        """Move caches from src memory to dst memory.

        Args:
            cache_pairs: Source cache, destination cache, and entry axis.
            src_to_dst (dict[int, int]): Map between src and dst.
        """
        if not cache_pairs or not src_to_dst:
            return

        BLOCKS_PER_COPY = 2
        num_copy = len(src_to_dst)
        src_idx, dst_idx = list(zip(*src_to_dst.items()))
        src_idx = torch.tensor(src_idx, device=cache_pairs[0][0].device)
        dst_idx = torch.tensor(dst_idx, device=cache_pairs[0][1].device)
        with torch.cuda.stream(self.cache_stream):
            for scache, dcache, entry_axis in cache_pairs:
                for idx in range(0, num_copy, BLOCKS_PER_COPY):
                    sidx = src_idx[idx:idx + BLOCKS_PER_COPY]
                    didx = dst_idx[idx:idx + BLOCKS_PER_COPY]
                    sdata = scache.index_select(entry_axis, sidx)
                    dcache.index_copy_(entry_axis, didx, sdata.to(dcache.device))
            self.events.record(stream=self.cache_stream)

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

    @classmethod
    def get_cache_block_size(cls,
                             cache_config: CacheConfig,
                             model_config: ModelConfig,
                             world_size: int = 1,
                             block_cache_plan: BlockCachePlan | None = None) -> int:
        """Get the required cache size of the model.

        Args:
            block_size (int): The token numbers of the block.
            model_config (ModelConfig): The config of the model.

        Return:
            int: Required memory size in bytes.
        """
        allocator = cls.allocate_caches
        allocator_func = getattr(allocator, '__func__', allocator)
        if allocator_func is _NATIVE_BLOCK_ALLOCATOR:
            if block_cache_plan is None:
                # Preserve direct sizing callers that validate cache policy
                # without constructing complete model/cache configs.
                _update_mla_kv_cache_dtype(model_config, cache_config)
                block_cache_plan = cls.build_cache_plan(model_config, cache_config, world_size)
            return block_cache_plan.logical_block_nbytes

        # Existing patched allocators derive their layout from ModelConfig.
        _update_mla_kv_cache_dtype(model_config, cache_config)
        result = allocator(
            num_blocks=1,
            model_config=model_config,
            cache_config=cache_config,
            world_size=world_size,
            device='meta',
        )
        allocation, mem_pool, _ = _unpack_cache_allocation(result)
        if allocation is not None:
            return allocation.nbytes
        return cls._mem_pool_nbytes(mem_pool)

    """ Metheds for PD Disaggregation Begin. """

    def p2p_initialize(self, migration_init_request: DistServeInitRequest) -> DistServeKVTransferEndpointInfo:
        if isinstance(getattr(self, 'full_gpu_cache', None), list):
            raise RuntimeError('PD migration does not support packed named block caches.')
        if not self.migration_backend_impl:
            self.migration_backend_impl = MIGRATION_BACKENDS.module_dict[self.cache_config.migration_backend.name]()
        migration_init_request.rank = self.rank
        self.migration_backend_impl.p2p_initialize(migration_init_request)
        for i, t in enumerate([self.full_gpu_cache]):
            if t.numel() == 0:
                continue
            register_mr_request = DistServeRegisterMRMessage(protocol=migration_init_request.protocol,
                                                             remote_engine_id=migration_init_request.remote_engine_id,
                                                             mr_key=i,
                                                             addr=t.data_ptr(),
                                                             offset=t.storage_offset(),
                                                             length=t.numel() * t.itemsize)
            self.migration_backend_impl.register_memory_region(register_mr_request)
        return DistServeKVTransferEndpointInfo(protocol=migration_init_request.protocol,
                                               endpoint_info=json.dumps(
                                                   self.migration_backend_impl.endpoint_info(
                                                       migration_init_request.remote_engine_id,
                                                       migration_init_request.protocol)))

    def p2p_connect(self, remote_engine_id: str, migration_conn_request: list[DistServeKVTransferEndpointInfo]):
        self.migration_backend_impl.p2p_connect(remote_engine_id, migration_conn_request[self.tp_rank])

    async def migrate(self, migration_execution_inputs: MigrationExecutionBatch):
        if isinstance(getattr(self, 'full_gpu_cache', None), list):
            raise RuntimeError('PD migration does not support packed named block caches.')
        if self.cache_config.block_size != self.cache_config.kernel_block_size:
            raise RuntimeError('PD migration does not support block_size != kernel_block_size.')

        assignment_len = self.full_gpu_cache.element_size() * self.full_gpu_cache.size(-1)
        layer_stride = self.cache_config.num_gpu_blocks * assignment_len

        def get_assignment_batch(mr_key, block_ids, assignment_len, layer_stride, remote_layer_stride):
            return [
                AssignmentInstruct(mr_key=mr_key,
                                   target_offset=block_id[0] * assignment_len + layer * remote_layer_stride,
                                   source_offset=block_id[1] * assignment_len + layer * layer_stride,
                                   length=assignment_len) for layer in range(self.model_config.num_layers)
                for block_id in block_ids
            ]

        assignment_batch: list[tuple[str, int, int, int]] = []  # mr_key, target, source, offset
        for migration_exe_req in migration_execution_inputs.requests:
            remote_engine_id = migration_exe_req[0]
            blocks_to_migration = migration_exe_req[1]
            remote_layer_stride = self.migration_backend_impl.links[
                remote_engine_id].remote_engine_config.num_gpu_blocks * assignment_len

            for i, t in enumerate([self.full_gpu_cache]):
                if t.numel() == 0:
                    continue
                assignment_batch.extend(
                    get_assignment_batch(i, blocks_to_migration, assignment_len, layer_stride, remote_layer_stride))
        await self.migration_backend_impl.p2p_migrate(
            MigrationAssignment(
                protocol=migration_execution_inputs.protocol,
                remote_engine_id=remote_engine_id,
                batch=assignment_batch,
            ))

    """ Metheds for PD Disaggregation End. """


# Existing dlinfer releases replace this class method after importing LMDeploy.
# Keep its original function identity so runtime and sizing can retain that
# compatibility path until the external feature check is deployed.
_NATIVE_BLOCK_ALLOCATOR = CacheEngine.allocate_caches.__func__


class StateCacheEngine:
    """Cache engine for state cache."""

    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig | None = None):
        self.cache_config = cache_config
        self.model_config = model_config
        state_specs = None
        if model_config is not None and len(model_config.state_cache_specs) > 0:
            state_specs = model_config.state_cache_specs
        resources = build_state_cache_resources(cache_config.states_shapes, state_specs=state_specs)
        self._state_cache_names = [resource.name for resource in resources]
        self._state_cache_layer_maps = layer_maps_from_resources(resources)
        # Non-CUDA device integrations patch the canonical "cuda" device path
        # before reaching this layer, so keep using it here.
        allocate_kwargs = dict(num_caches=cache_config.num_state_caches,
                               state_shapes=cache_config.states_shapes,
                               device='cuda')
        if state_specs is not None:
            allocate_kwargs['state_specs'] = state_specs
        result = self.allocate_caches(**allocate_kwargs)
        self.allocation, self.mem_pool, self._state_caches = _unpack_cache_allocation(result)
        self._state_entries = self._build_state_entries(self.allocation, self._state_caches)

    @staticmethod
    def allocate_caches(num_caches: int,
                        state_shapes: list[tuple[tuple[int], torch.dtype]],
                        device: torch.device,
                        state_specs: list[StateCacheSpec] | None = None) -> CacheAllocation:
        """Allocate cache implement."""

        resources = build_state_cache_resources(state_shapes, state_specs=state_specs)
        cache_backend = get_backend().get_cache_backend()
        layout = cache_backend.build_state_layout(resources)
        return layout.allocate(num_caches=num_caches, device=device)

    @staticmethod
    def _build_state_entries(allocation: CacheAllocation | None,
                             state_caches: Sequence[torch.Tensor]) -> tuple[tuple[torch.Tensor, int], ...]:
        """Resolve tensors and state-slot axes used by runtime operations."""
        if allocation is not None:
            return tuple((pool.tensor, pool.entry_axis) for pool in allocation.pools)

        # The pinned dlinfer tuple contract allocates every state resource as a
        # contiguous [state_slot, ...] tensor. Keep this explicit compatibility
        # path separate from native owning-pool metadata.
        return tuple((cache, 0) for cache in state_caches)

    @staticmethod
    def _get_state_cache_layer_maps(state_specs: list[StateCacheSpec]) -> dict[str, dict[int, int]]:
        """Build global-layer-id to local-row maps for named state caches."""
        resources = build_state_cache_resources((), state_specs=state_specs)
        return layer_maps_from_resources(resources)

    @staticmethod
    def get_cache_state_size(state_shapes: list[tuple[tuple[int], torch.dtype]],
                             state_specs: list[StateCacheSpec] | None = None) -> int:
        """Get the required cache size of the state cache.

        Args:
            state_shapes (list[tuple[tuple[int], torch.dtype]]): The shapes and dtypes of the states.

        Return:
            int: Required memory size in bytes.
        """
        allocate_kwargs = dict(num_caches=1, state_shapes=state_shapes, device='meta')
        if state_specs is not None:
            allocate_kwargs['state_specs'] = state_specs
        result = StateCacheEngine.allocate_caches(**allocate_kwargs)
        allocation, mem_pool, _ = _unpack_cache_allocation(result)
        if allocation is not None:
            return allocation.nbytes
        return mem_pool.numel() * mem_pool.element_size()

    @property
    def state_caches(self):
        """State caches."""
        return self._state_caches

    @property
    def named_state_caches(self) -> Mapping[str, torch.Tensor]:
        """State caches keyed by name."""
        if not self._state_cache_names or not self._state_caches:
            return {}
        caches = {
            name: cache
            for name, cache in zip(self._state_cache_names, self._state_caches)
        }
        layer_maps = getattr(self, '_state_cache_layer_maps', {})
        if not layer_maps:
            return caches
        return NamedCacheView(caches, layer_maps)

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
        for state_cache, entry_axis in self._state_entries:
            mask_shape = [1] * state_cache.dim()
            mask_shape[entry_axis] = num_caches
            state_cache.masked_fill_(cache_masks.view(mask_shape), 0)

    @staticmethod
    def _index_list(idx: int | Sequence[int]):
        """Normalize host-side cache indices."""
        if isinstance(idx, torch.Tensor):
            raise TypeError('State cache copy indices must be host integers, not torch.Tensor.')
        if isinstance(idx, (str, bytes)):
            raise TypeError('State cache copy indices must be an int or a sequence of ints.')
        try:
            return [as_index(idx)]
        except TypeError:
            pass
        if not isinstance(idx, Sequence):
            raise TypeError('State cache copy indices must be an int or a sequence of ints.')
        if any(isinstance(item, torch.Tensor) for item in idx):
            raise TypeError('State cache copy indices must be host integers, not torch.Tensor.')
        return [as_index(item) for item in idx]

    @staticmethod
    def _validate_index_bounds(indices: Sequence[int], num_caches: int):
        """Check normalized cache indices are valid state slots."""
        for idx in indices:
            if idx < 0 or idx >= num_caches:
                raise ValueError(f'State cache index {idx} is out of range [0, {num_caches}).')

    @staticmethod
    def _copy_ranges(src_list: list[int], dst_list: list[int]):
        """Yield contiguous copy ranges as (src_start, dst_start, length)."""
        pairs = sorted(zip(src_list, dst_list))
        if len(pairs) == 0:
            return
        start_src = prev_src = pairs[0][0]
        start_dst = prev_dst = pairs[0][1]
        length = 1
        for src, dst in pairs[1:]:
            if src == prev_src + 1 and dst == prev_dst + 1:
                prev_src = src
                prev_dst = dst
                length += 1
                continue
            yield start_src, start_dst, length
            start_src = prev_src = src
            start_dst = prev_dst = dst
            length = 1
        yield start_src, start_dst, length

    def copy_caches(self, src_idx: int | Sequence[int], dst_idx: int | Sequence[int]):
        """Copy state cache slots.

        This is the low-level primitive needed by SSM prefix caching: a frozen
        state checkpoint can be copied into a newly allocated runtime slot
        before the next forward.
        """
        if len(self._state_caches) <= 0:
            return

        src_list = self._index_list(src_idx)
        dst_list = self._index_list(dst_idx)
        if len(src_list) != len(dst_list):
            raise ValueError('src_idx and dst_idx must have the same number of elements.')
        if len(src_list) == 0:
            return
        num_caches = self.cache_config.num_state_caches
        self._validate_index_bounds(src_list, num_caches)
        self._validate_index_bounds(dst_list, num_caches)
        dst_set = set(dst_list)
        if len(dst_set) != len(dst_list):
            raise ValueError('dst_idx must not contain duplicate entries.')
        if not set(src_list).isdisjoint(dst_set):
            raise ValueError('src_idx and dst_idx must not overlap for stream-ordered state copies.')

        copy_ranges = tuple(self._copy_ranges(src_list, dst_list))
        for state_cache, entry_axis in self._state_entries:
            for src, dst, length in copy_ranges:
                src_cache = state_cache.narrow(entry_axis, src, length)
                dst_cache = state_cache.narrow(entry_axis, dst, length)
                dst_cache.copy_(src_cache, non_blocking=True)
