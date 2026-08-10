# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import json
from collections.abc import Callable, Mapping, Sequence

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
from ...config import CacheConfig, ModelConfig
from .layout import CacheAllocation, CachePool, _unpack_cache_allocation
from .migration import (
    build_cache_pool_assignments,
    describe_cache_pools,
    describe_legacy_remote_pool,
    validate_cache_pool_layouts,
)
from .plan import BlockCachePlan
from .schema import (
    BlockCacheGeometry,
    BlockCacheRequest,
    CacheDesc,
    CacheTensorSpec,
    build_block_cache_tensor_specs,
    build_block_cache_tensor_specs_from_requests,
    layer_maps_from_specs,
)
from .view import NamedCacheView

KVCache = tuple[torch.Tensor, torch.Tensor]

logger = get_logger('lmdeploy')


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
    """Own block-cache allocations and runtime movement.

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

        self.kernel_block_size = cache_config.kernel_block_size
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
        """CPU cache tensors in legacy model order."""
        return self.local_cpu_cache

    @property
    def gpu_cache(self):
        """Device cache tensors in legacy model order."""
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
    def _get_cache_tensor_specs(
            cls,
            model_config: ModelConfig,
            cache_config: CacheConfig,
            world_size: int,
            block_requests: Sequence[BlockCacheRequest] | None = None) -> tuple[CacheTensorSpec, ...]:
        """Build the ordered tensor specs consumed by the physical layout."""
        tensor_specs = []
        use_std = model_config.use_standard_kv_cache

        if use_std:
            k_cache_desc = cls.get_k_cache_desc(model_config, cache_config, world_size)
            v_cache_desc = cls.get_v_cache_desc(model_config, cache_config, world_size)
            quant_cache_descs = cls.get_quant_cache_descs(k_cache_desc, v_cache_desc, model_config, cache_config)
            tensor_specs.append(CacheTensorSpec(name='k_cache', desc=k_cache_desc))
            tensor_specs.append(CacheTensorSpec(name='v_cache', desc=v_cache_desc))
            for idx, desc in enumerate(quant_cache_descs):
                tensor_specs.append(CacheTensorSpec(name=f'quant_{idx}', desc=desc))

        if block_requests is not None:
            tensor_specs.extend(build_block_cache_tensor_specs_from_requests(block_requests))
        # named block cache specs (shape without block_size, same as cache_shapes)
        elif len(model_config.block_cache_specs) > 0:
            tensor_specs.extend(build_block_cache_tensor_specs(model_config.block_cache_specs))
        else:
            # legacy anonymous cache_shapes (shape without block_size)
            custom_descs = cls.get_custom_cache_descs(model_config, cache_config)
            for idx, desc in enumerate(custom_descs):
                tensor_specs.append(CacheTensorSpec(name=f'custom_{idx}', desc=desc))

        return tuple(tensor_specs)

    @classmethod
    def _uses_layer_scoped_block_caches(cls, model_config: ModelConfig):
        """Whether model-facing named caches use declared layer rows."""
        return (model_config is not None and not model_config.use_standard_kv_cache
                and len(model_config.block_cache_specs) > 0)

    @staticmethod
    def _get_block_rows_by_layer(model_config: ModelConfig) -> dict[str, dict[int, int]]:
        """Build global-layer-id to local-row maps for named block caches."""
        if not CacheEngine._uses_layer_scoped_block_caches(model_config):
            return {}
        tensor_specs = build_block_cache_tensor_specs(model_config.block_cache_specs)
        return layer_maps_from_specs(tensor_specs)

    @classmethod
    def get_custom_cache_descs(cls, model_config: ModelConfig, cache_config: CacheConfig) -> list[CacheDesc]:
        """Get custom cache descs."""
        descs = []
        block_size = cache_config.kernel_block_size
        # named block cache specs (shape without block_size, same convention as cache_shapes)
        if len(model_config.block_cache_specs) > 0:
            for spec in build_block_cache_tensor_specs(model_config.block_cache_specs):
                descs.append(spec.desc)
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
        request_collector: Callable[[BlockCacheGeometry], Sequence[BlockCacheRequest] | None] | None = None,
    ) -> BlockCachePlan:
        """Finalize block geometry, tensor specs, and backend layout."""
        geometry = BlockCacheGeometry(block_size=cache_config.block_size,
                                      kernel_block_size=cache_config.kernel_block_size)
        _update_mla_kv_cache_dtype(model_config, cache_config)
        block_requests = None
        if request_collector is not None:
            collected_requests = request_collector(geometry)
            if collected_requests is not None:
                allocator = cls.allocate_caches
                allocator_func = getattr(allocator, '__func__', allocator)
                if allocator_func is not _NATIVE_BLOCK_ALLOCATOR:
                    raise RuntimeError(
                        'Built-operator cache request collection requires the native CacheEngine allocator.')
                block_requests = tuple(collected_requests)
        num_layers = model_config.num_layers
        tensor_specs = cls._get_cache_tensor_specs(model_config,
                                                   cache_config,
                                                   world_size,
                                                   block_requests=block_requests)
        cache_backend = get_backend().get_cache_backend()
        layout = cache_backend.build_block_layout(tensor_specs, num_layers=num_layers)
        return BlockCachePlan(
            tensor_specs=tensor_specs,
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

    def _build_legacy_layer_cache(self, caches: Sequence[torch.Tensor]):
        """Build the compatibility tuple without scoped named tensors."""
        plan = getattr(self, 'block_cache_plan', None)
        if plan is not None:
            caches = [caches[index] for index in plan.legacy_cache_indices]
            return list(zip(*caches)) if caches else []
        if self._uses_layer_scoped_block_caches(self.model_config):
            return []
        return list(zip(*caches))

    def allocate_gpu_cache(self):
        """Allocate caches on GPU."""
        # Non-CUDA device integrations patch the canonical "cuda" device path
        # before reaching this layer, so keep using it here.
        result = self._allocate_runtime_caches(
            num_blocks=self.num_gpu_blocks,
            device='cuda',
        )
        self.gpu_allocation, mem_pool, caches = _unpack_cache_allocation(result)
        self._legacy_gpu_cache_pool = mem_pool if self.gpu_allocation is None else None
        # Retain the raw tensor-or-list facade for downstream compatibility.
        self.full_gpu_cache = mem_pool
        self._gpu_cache_list = caches
        self.local_gpu_cache = self._build_legacy_layer_cache(caches)
        plan = getattr(self, 'block_cache_plan', None)
        if plan is None:
            tensor_specs = self._get_cache_tensor_specs(self.model_config, self.cache_config, self.world_size)
            self._block_cache_names = [spec.name for spec in tensor_specs]
            self._cache_tensor_specs = None
            self._block_rows_by_layer = self._get_block_rows_by_layer(self.model_config)
        else:
            self._block_cache_names = list(plan.cache_names)
            self._cache_tensor_specs = plan.tensor_specs
            self._block_rows_by_layer = {}
        self._block_caches = self._build_block_cache_view()
        return self.local_gpu_cache

    def allocate_cpu_cache(self):
        """Allocate caches on Host."""
        result = self._allocate_runtime_caches(
            num_blocks=self.num_cpu_blocks,
            device='cpu',
        )
        self.cpu_allocation, mem_pool, caches = _unpack_cache_allocation(result)
        # Retain the raw tensor-or-list facade for downstream compatibility.
        self.full_cpu_cache = mem_pool
        self._cpu_cache_list = caches
        self.local_cpu_cache = self._build_legacy_layer_cache(caches)
        return self.local_cpu_cache

    def _build_block_cache_view(self) -> Mapping[str, torch.Tensor]:
        """Build the model-facing view once for this device allocation."""
        if not hasattr(self, '_block_cache_names') or not hasattr(self, '_gpu_cache_list'):
            return {}
        tensor_specs = getattr(self, '_cache_tensor_specs', None)
        if tensor_specs is not None and any(spec.has_rows for spec in tensor_specs):
            return NamedCacheView.from_specs(tensor_specs, self._gpu_cache_list)
        caches = {
            name: cache
            for name, cache in zip(self._block_cache_names, self._gpu_cache_list)
        }
        rows_by_layer = getattr(self, '_block_rows_by_layer', {})
        if not rows_by_layer:
            return caches
        return NamedCacheView(caches, rows_by_layer)

    @property
    def block_caches(self) -> Mapping[str, torch.Tensor]:
        """Return all caches (including k/v and custom) by name."""
        if hasattr(self, '_block_caches'):
            return self._block_caches
        return self._build_block_cache_view()

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
                block_size=self.kernel_block_size,
                shape=shape,
            )
            custom_cache = self._allocate_single_custom_cache(shape=custom_shape, dtype=dtype, device=device)
            custom_caches.append(custom_cache)
        return custom_caches

    @staticmethod
    def _legacy_mem_pool_nbytes(mem_pool: torch.Tensor | list[torch.Tensor]) -> int:
        """Size the tensor-or-list result of an external patched allocator."""
        pools = [mem_pool] if isinstance(mem_pool, torch.Tensor) else mem_pool
        return sum(pool.numel() * pool.element_size() for pool in pools)

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
            # not describe cache blocks. Their typed cache views retain the
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

    def _build_block_copy(self):
        """Build local logical-block copy from the device allocation."""
        self._block_copy = None
        allocation = getattr(self, 'gpu_allocation', None)
        if allocation is None:
            return

        pages_per_block = self.block_cache_plan.kernel_blocks_per_logical_block
        cache_backend = get_backend().get_cache_backend()
        self._block_copy = cache_backend.build_block_copy(
            allocation,
            num_logical_blocks=self.num_gpu_blocks,
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

        block_copy = getattr(self, '_block_copy', None)
        if block_copy is None:
            raise RuntimeError('Logical block copy requires a native cache allocation.')
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

    @classmethod
    def get_logical_block_nbytes(cls,
                                 cache_config: CacheConfig,
                                 model_config: ModelConfig,
                                 world_size: int = 1,
                                 block_cache_plan: BlockCachePlan | None = None) -> int:
        """Return owning storage bytes required by one logical block."""
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
        return cls._legacy_mem_pool_nbytes(mem_pool)

    # PD disaggregation.

    def _resolve_pd_cache_pools(self) -> tuple[CachePool, ...]:
        """Return owning pools with the metadata required by PD migration."""
        if self.cache_config.block_size != self.cache_config.kernel_block_size:
            raise RuntimeError('PD migration does not support block_size != kernel_block_size.')

        allocation = getattr(self, 'gpu_allocation', None)
        if allocation is None:
            pool = getattr(self, '_legacy_gpu_cache_pool', None)
            if not isinstance(pool, torch.Tensor):
                raise RuntimeError('PD migration of multiple pools requires native CacheAllocation metadata.')
            return (CachePool(pool, entry_axis=1), )
        return allocation.pools

    def _get_pd_cache_pool_infos(self) -> tuple[DistServeCachePoolInfo, ...]:
        """Describe the stable local allocation once for every PD link."""
        pool_infos = getattr(self, '_pd_cache_pool_infos', None)
        if pool_infos is None:
            pool_infos = describe_cache_pools(self._resolve_pd_cache_pools(), self.num_gpu_blocks)
            self._pd_cache_pool_infos = pool_infos
        return pool_infos

    def p2p_initialize(self, migration_init_request: DistServeInitRequest) -> DistServeKVTransferEndpointInfo:
        pools = self._resolve_pd_cache_pools()
        pool_infos = describe_cache_pools(pools, self.num_gpu_blocks)
        self._pd_cache_pool_infos = pool_infos
        if not self.migration_backend_impl:
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
            remote_pools = describe_legacy_remote_pool(local_pools, remote_num_blocks)
        validate_cache_pool_layouts(local_pools, remote_pools, self.num_gpu_blocks, remote_num_blocks)
        self._remote_pd_cache_pool_infos[remote_engine_id] = tuple(remote_pools)
        self.migration_backend_impl.p2p_connect(remote_engine_id, conn_request)

    async def migrate(self, migration_execution_inputs: MigrationExecutionBatch):
        local_pools = self._get_pd_cache_pool_infos()
        blocks_by_remote: dict[str, list[tuple[int, int]]] = {}
        for remote_engine_id, block_pairs in migration_execution_inputs.requests:
            blocks_by_remote.setdefault(remote_engine_id, []).extend(block_pairs)

        for remote_engine_id, block_pairs in blocks_by_remote.items():
            remote_pools = getattr(self, '_remote_pd_cache_pool_infos', {}).get(remote_engine_id)
            if remote_pools is None:
                raise RuntimeError(f'PD cache-pool metadata is unavailable for remote engine {remote_engine_id}.')
            assignment_batch = build_cache_pool_assignments(local_pools, remote_pools, block_pairs)
            await self.migration_backend_impl.p2p_migrate(
                MigrationAssignment(
                    protocol=migration_execution_inputs.protocol,
                    remote_engine_id=remote_engine_id,
                    batch=assignment_batch,
                ))

# Existing dlinfer releases replace this class method after importing LMDeploy.
# Keep its original function identity so runtime and sizing can retain that
# compatibility path until the external feature check is deployed.
_NATIVE_BLOCK_ALLOCATOR = CacheEngine.allocate_caches.__func__
