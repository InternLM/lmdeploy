# Copyright (c) OpenMMLab. All rights reserved.
"""Cache payload descriptions, tensor specifications, and row bindings."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from operator import index as as_index

import torch

from lmdeploy.pytorch.backends import get_backend

from ....messages import QuantPolicy
from ...config import CacheConfig, ModelConfig, StateCacheSpec

_FP8_CACHE_DTYPES = {
    QuantPolicy.FP8: torch.float8_e4m3fn,
    QuantPolicy.FP8_E5M2: torch.float8_e5m2,
}

# 512*1 + 4*4 + 64*2 = 656
_MLA_FP8_HEAD_DIM = 656

# Cache payload descriptions and policy.


def _round_up(x: int, alignment: int) -> int:
    """Round up x to the nearest multiple of alignment."""
    return ((x + alignment - 1) // alignment) * alignment


@dataclass
class CacheDesc:
    """Describe one cache payload and its aligned storage size."""
    shape: list[int]
    dtype: torch.dtype
    alignment: int = 256

    def __post_init__(self):
        self.numel = math.prod(self.shape)
        self.size = self.numel * self.dtype.itemsize
        self.aligned_size = _round_up(self.size, self.alignment)


def _resolve_kv_cache_dtype(model_config: ModelConfig, quant_policy: QuantPolicy) -> torch.dtype:
    """Resolve the storage dtype for standard key and value caches."""
    if quant_policy in _FP8_CACHE_DTYPES:
        return _FP8_CACHE_DTYPES[quant_policy]
    if quant_policy in (QuantPolicy.INT4, QuantPolicy.INT8, QuantPolicy.TURBO_QUANT):
        return torch.uint8
    if model_config.use_mla_fp8_cache:
        return torch.float8_e4m3fn
    if model_config.mla_kv_cache_dtype == 'bfloat16':
        return torch.bfloat16
    return model_config.dtype


def _resolve_key_block_shape(model_config: ModelConfig,
                             block_size: int,
                             head_size: int,
                             world_size: int = 1,
                             quant_policy: QuantPolicy = QuantPolicy.NONE) -> tuple[int, ...]:
    """Resolve one backend-specific key block shape."""
    attn_backend = get_backend()
    dtype = model_config.dtype
    num_heads = model_config.num_key_value_heads

    assert num_heads % world_size == 0, f'num_heads: {num_heads}, world_size: {world_size}'
    num_heads = num_heads // world_size

    if model_config.use_mla_fp8_cache:
        return (block_size, num_heads, _MLA_FP8_HEAD_DIM)

    if quant_policy in (QuantPolicy.INT4, QuantPolicy.TURBO_QUANT):
        assert head_size % 2 == 0, f'head_size: {head_size}, quant_policy: {quant_policy}'
        head_size = head_size // 2
    return attn_backend.get_k_block_shape(block_size, num_heads, head_size, dtype)


def _resolve_value_block_shape(model_config: ModelConfig,
                               block_size: int,
                               head_size: int,
                               world_size: int = 1,
                               quant_policy: QuantPolicy = QuantPolicy.NONE) -> tuple[int, ...]:
    """Resolve one backend-specific value block shape."""
    attn_backend = get_backend()
    dtype = model_config.dtype
    num_heads = model_config.num_key_value_heads

    assert num_heads % world_size == 0, f'num_heads: {num_heads}, world_size: {world_size}'
    num_heads = num_heads // world_size

    if model_config.use_mla_fp8_cache:
        # FlashMLA shares key and value storage.
        return (block_size, num_heads, 0)

    if quant_policy == QuantPolicy.TURBO_QUANT:
        assert head_size % 4 == 0, f'head_size: {head_size}, quant_policy: {quant_policy}'
        head_size = head_size // 4
    elif quant_policy == QuantPolicy.INT4:
        assert head_size % 2 == 0, f'head_size: {head_size}, quant_policy: {quant_policy}'
        head_size = head_size // 2
    return attn_backend.get_v_block_shape(block_size, num_heads, head_size, dtype)


def build_k_cache_desc(model_config: ModelConfig, cache_config: CacheConfig, world_size: int = 1) -> CacheDesc:
    """Build the standard key-cache payload description."""
    head_size = model_config.k_head_dim
    if head_size is None:
        head_size = model_config.head_dim
    shape = _resolve_key_block_shape(
        model_config,
        block_size=cache_config.kernel_block_size,
        head_size=head_size,
        world_size=world_size,
        quant_policy=cache_config.quant_policy,
    )
    dtype = _resolve_kv_cache_dtype(model_config, cache_config.quant_policy)
    return CacheDesc(shape=list(shape), dtype=dtype)


def build_v_cache_desc(model_config: ModelConfig, cache_config: CacheConfig, world_size: int = 1) -> CacheDesc:
    """Build the standard value-cache payload description."""
    head_size = model_config.v_head_dim
    if head_size is None:
        head_size = model_config.head_dim
    shape = _resolve_value_block_shape(
        model_config,
        block_size=cache_config.kernel_block_size,
        head_size=head_size,
        world_size=world_size,
        quant_policy=cache_config.quant_policy,
    )
    dtype = _resolve_kv_cache_dtype(model_config, cache_config.quant_policy)
    return CacheDesc(shape=list(shape), dtype=dtype)


def build_quant_cache_descs(k_cache_desc: CacheDesc, v_cache_desc: CacheDesc, model_config: ModelConfig,
                            cache_config: CacheConfig) -> list[CacheDesc]:
    """Build auxiliary scale/zero cache descriptions when required."""
    if cache_config.quant_policy == QuantPolicy.NONE or cache_config.quant_policy in _FP8_CACHE_DTYPES:
        return []

    dtype = model_config.dtype
    if cache_config.quant_policy == QuantPolicy.TURBO_QUANT:
        key_scale_zero_shape = k_cache_desc.shape[:-1] + [2]
        val_scale_zero_shape = v_cache_desc.shape[:-1] + [1]
    else:
        key_scale_zero_shape = k_cache_desc.shape[:-1] + [2]
        val_scale_zero_shape = v_cache_desc.shape[:-1] + [2]
    return [
        CacheDesc(shape=key_scale_zero_shape, dtype=dtype),
        CacheDesc(shape=val_scale_zero_shape, dtype=dtype),
    ]

# Normalized requests and row bindings.


@dataclass(frozen=True)
class BlockCacheGeometry:
    """Relate one logical scheduler block to physical kernel cache blocks."""

    logical_block_size: int
    kernel_block_size: int

    def __post_init__(self):
        if self.logical_block_size <= 0:
            raise ValueError('logical_block_size must be positive.')
        if self.kernel_block_size <= 0:
            raise ValueError('kernel_block_size must be positive.')
        if self.logical_block_size < self.kernel_block_size:
            raise ValueError(
                f'logical_block_size {self.logical_block_size} must be greater than or equal to '
                f'kernel_block_size {self.kernel_block_size}.')
        if self.logical_block_size % self.kernel_block_size != 0:
            raise ValueError(
                f'logical_block_size {self.logical_block_size} must be divisible by '
                f'kernel_block_size {self.kernel_block_size}.')

    @property
    def kernel_blocks_per_logical_block(self) -> int:
        """Return physical kernel blocks in one scheduler block."""
        return self.logical_block_size // self.kernel_block_size


@dataclass(frozen=True)
class BlockCacheRequestContext:
    """Carry worker-finalized inputs into built-operator cache requests.

    Model and backend facts already owned by the built operator do not belong here.
    """

    geometry: BlockCacheGeometry


@dataclass(frozen=True)
class BlockCacheRequest:
    """Describe one block-cache payload requested by a built operator."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    alignment: int = 256
    per_row_contiguous: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Block cache request name must not be empty.')
        alignment = as_index(self.alignment)
        shape = tuple(as_index(dim) for dim in self.shape)
        if alignment <= 0:
            raise ValueError(f'{self.name} alignment must be positive.')
        if any(dim < 0 for dim in shape):
            raise ValueError(f'{self.name} shape dimensions must be non-negative.')
        object.__setattr__(self, 'alignment', alignment)
        object.__setattr__(self, 'shape', shape)


@dataclass(frozen=True)
class BlockCacheBinding:
    """Identify one built operator's logical row in a named block cache."""

    cache_name: str
    consumer_row: int


@dataclass(frozen=True)
class LayerRowMap:
    """Map global layer ids to compact cache-tensor rows."""

    layer_ids: tuple[int, ...]

    @classmethod
    def build(cls, cache_name: str, layer_ids: Sequence[int]):
        normalized_layer_ids = []
        seen_layer_ids = set()
        for layer_id in layer_ids:
            layer_id = as_index(layer_id)
            if layer_id < 0:
                raise ValueError(f'{cache_name} layer id {layer_id} must be non-negative.')
            if layer_id in seen_layer_ids:
                raise ValueError(f'{cache_name} layer id {layer_id} is duplicated.')
            seen_layer_ids.add(layer_id)
            normalized_layer_ids.append(layer_id)
        if not normalized_layer_ids:
            raise ValueError(f'{cache_name} layer_ids must not be empty.')

        layer_ids = tuple(normalized_layer_ids)
        return cls(layer_ids=layer_ids)

    @property
    def num_rows(self) -> int:
        """Number of compact cache rows required by this layout."""
        return len(self.layer_ids)


@dataclass(frozen=True)
class CacheTensorSpec:
    """Describe one named cache tensor without owning its storage."""

    name: str
    desc: CacheDesc
    layer_rows: LayerRowMap | None = None
    consumer_rows: tuple[int, ...] | None = None
    per_row_contiguous: bool = False

    def __post_init__(self):
        if self.consumer_rows is None:
            return
        if self.layer_rows is not None:
            raise ValueError(f'{self.name} cannot define both consumer_rows and layer_rows.')

        consumer_rows = tuple(as_index(row) for row in self.consumer_rows)
        if not consumer_rows:
            raise ValueError(f'{self.name} consumer_rows must not be empty.')
        if any(row < 0 for row in consumer_rows):
            raise ValueError(f'{self.name} consumer rows must be non-negative.')
        if len(consumer_rows) != len(set(consumer_rows)):
            raise ValueError(f'{self.name} consumer rows must be unique within one tensor spec.')
        object.__setattr__(self, 'consumer_rows', consumer_rows)

    @property
    def num_rows(self) -> int:
        """Return compact rows for an operator- or layer-scoped tensor."""
        if self.consumer_rows is not None:
            return len(self.consumer_rows)
        assert self.layer_rows is not None
        return self.layer_rows.num_rows

# Tensor-spec assembly.


def build_block_cache_tensor_specs_from_requests(
        requests: Sequence[BlockCacheRequest]) -> tuple[CacheTensorSpec, ...]:
    """Group equal contracts while retaining each consumer's logical row."""
    rows_by_request: dict[BlockCacheRequest, list[int]] = {}
    next_row_by_name: dict[str, int] = {}

    for request in requests:
        row = next_row_by_name.get(request.name, 0)
        next_row_by_name[request.name] = row + 1
        rows_by_request.setdefault(request, []).append(row)

    tensor_specs = []
    for request, consumer_rows in rows_by_request.items():
        desc = CacheDesc(shape=list(request.shape), dtype=request.dtype, alignment=request.alignment)
        tensor_specs.append(
            CacheTensorSpec(name=request.name,
                            desc=desc,
                            consumer_rows=tuple(consumer_rows),
                            per_row_contiguous=request.per_row_contiguous))
    return tuple(tensor_specs)


def build_state_cache_tensor_specs(
        state_shapes: Sequence[tuple[tuple[int, ...], torch.dtype]],
        state_specs: Sequence[StateCacheSpec] | None = None) -> tuple[CacheTensorSpec, ...]:
    """Normalize named or anonymous state-cache declarations."""
    state_specs = state_specs or ()
    if len(state_specs) > 0:
        tensor_specs = []
        for spec in state_specs:
            layer_rows = None
            shape = spec.shape
            if spec.layer_ids is not None:
                layer_rows = LayerRowMap.build(spec.name, spec.layer_ids)
                shape = (layer_rows.num_rows, *shape)
            desc = CacheDesc(shape=shape, dtype=spec.dtype, alignment=spec.alignment)
            tensor_specs.append(CacheTensorSpec(name=spec.name, desc=desc, layer_rows=layer_rows))
        return tuple(tensor_specs)

    return tuple(
        CacheTensorSpec(name=f'state_{idx}', desc=CacheDesc(shape=shape, dtype=dtype))
        for idx, (shape, dtype) in enumerate(state_shapes))


def build_model_block_cache_tensor_specs(
        model_config: ModelConfig,
        cache_config: CacheConfig,
        world_size: int,
        block_requests: Sequence[BlockCacheRequest] = ()) -> tuple[CacheTensorSpec, ...]:
    """Build ordered block-cache tensor specs from model and operator
    inputs."""
    tensor_specs = []
    if model_config.use_standard_kv_cache:
        k_cache_desc = build_k_cache_desc(model_config, cache_config, world_size)
        v_cache_desc = build_v_cache_desc(model_config, cache_config, world_size)
        quant_cache_descs = build_quant_cache_descs(k_cache_desc, v_cache_desc, model_config, cache_config)
        tensor_specs.append(CacheTensorSpec(name='k_cache', desc=k_cache_desc))
        tensor_specs.append(CacheTensorSpec(name='v_cache', desc=v_cache_desc))
        tensor_specs.extend(
            CacheTensorSpec(name=f'quant_{index}', desc=desc)
            for index, desc in enumerate(quant_cache_descs))

    tensor_specs.extend(build_block_cache_tensor_specs_from_requests(block_requests))
    return tuple(tensor_specs)
