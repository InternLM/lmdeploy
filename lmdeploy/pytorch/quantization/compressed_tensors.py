# Copyright (c) OpenMMLab. All rights reserved.
"""Validation and reference helpers for compressed-tensors checkpoints.

Production execution stays in the PyTorch backend and reads the packed layout
directly. The unpack/dequantize functions here are correctness oracles for
small fixtures and must not materialize every expert in a real model.
"""

import hashlib
import json
import math
import os.path as osp
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


_QUANT_SUFFIXES = ('.weight_packed', '.weight_scale', '.weight_shape')
_UNSUPPORTED_QUANT_SUFFIXES = ('.weight_zero_point', '.weight_g_idx', '.qzeros', '.g_idx')
_TOP_LEVEL_FIELDS = {
    'config_groups',
    'format',
    'ignore',
    'kv_cache_scheme',
    'quant_method',
    'quantization_status',
}
_GROUP_FIELDS = {'input_activations', 'output_activations', 'targets', 'weights'}
_WEIGHT_FIELDS = {
    'actorder',
    'block_structure',
    'dynamic',
    'group_size',
    'num_bits',
    'observer',
    'observer_kwargs',
    'strategy',
    'symmetric',
    'type',
}


def _config_error(path: str, message: str):
    raise ValueError(f'Invalid compressed-tensors config at `{path}`: {message}')


def _required(mapping: Mapping[str, Any], key: str, path: str):
    if not isinstance(mapping, Mapping):
        _config_error(path, f'expected an object, got {type(mapping).__name__}')
    if key not in mapping:
        _config_error(f'{path}.{key}', 'missing required field')
    return mapping[key]


def _reject_unknown_fields(mapping: Mapping[str, Any], allowed_fields: set[str], path: str):
    unknown_fields = set(mapping) - allowed_fields
    if unknown_fields:
        unknown = sorted(repr(field) for field in unknown_fields)
        _config_error(path, f'unknown fields: {unknown}')


def _expect_value(mapping: Mapping[str, Any], key: str, expected: Any, path: str):
    value = _required(mapping, key, path)
    if expected is None:
        is_expected = value is None
    elif isinstance(expected, bool):
        is_expected = isinstance(value, bool) and value is expected
    else:
        is_expected = isinstance(value, type(expected)) and value == expected
    if not is_expected:
        _config_error(f'{path}.{key}', f'expected {expected!r}, got {value!r}')
    return value


def _expect_int(mapping: Mapping[str, Any], key: str, expected: int, path: str):
    value = _required(mapping, key, path)
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        _config_error(f'{path}.{key}', f'expected integer {expected}, got {value!r}')
    return value


@dataclass(frozen=True)
class CompressedTensorsW4A16Config:
    """Validated compressed-tensors profile supported by the Kimi MVP."""

    format: str
    targets: tuple[str, ...]
    num_bits: int
    group_size: int
    strategy: str
    symmetric: bool
    dynamic: bool
    weight_type: str
    observer: str | None
    observer_kwargs: tuple[tuple[str, Any], ...]
    ignore: tuple[str, ...]
    quantization_status: str

    @classmethod
    def from_dict(cls, quant_config: Mapping[str, Any]):
        """Parse the supported W4A16 profile and reject unknown semantics."""
        path = 'quantization_config'
        if not isinstance(quant_config, Mapping):
            _config_error(path, f'expected an object, got {type(quant_config).__name__}')
        _reject_unknown_fields(quant_config, _TOP_LEVEL_FIELDS, path)

        _expect_value(quant_config, 'quant_method', 'compressed-tensors', path)
        quant_format = _expect_value(quant_config, 'format', 'pack-quantized', path)
        quantization_status = _expect_value(quant_config, 'quantization_status', 'compressed', path)

        groups = _required(quant_config, 'config_groups', path)
        if not isinstance(groups, Mapping):
            _config_error(f'{path}.config_groups', f'expected an object, got {type(groups).__name__}')
        if set(groups) != {'group_0'}:
            _config_error(f'{path}.config_groups', f'expected exactly `group_0`, got {sorted(groups)}')

        group_path = f'{path}.config_groups.group_0'
        group = groups['group_0']
        if not isinstance(group, Mapping):
            _config_error(group_path, f'expected an object, got {type(group).__name__}')
        _reject_unknown_fields(group, _GROUP_FIELDS, group_path)

        targets = _required(group, 'targets', group_path)
        if not isinstance(targets, list) or targets != ['Linear']:
            _config_error(f'{group_path}.targets', f"expected ['Linear'], got {targets!r}")
        _expect_value(group, 'input_activations', None, group_path)
        _expect_value(group, 'output_activations', None, group_path)

        weights_path = f'{group_path}.weights'
        weights = _required(group, 'weights', group_path)
        if not isinstance(weights, Mapping):
            _config_error(weights_path, f'expected an object, got {type(weights).__name__}')
        _reject_unknown_fields(weights, _WEIGHT_FIELDS, weights_path)
        num_bits = _expect_int(weights, 'num_bits', 4, weights_path)
        group_size = _expect_int(weights, 'group_size', 32, weights_path)
        strategy = _expect_value(weights, 'strategy', 'group', weights_path)
        symmetric = _expect_value(weights, 'symmetric', True, weights_path)
        dynamic = _expect_value(weights, 'dynamic', False, weights_path)
        weight_type = _expect_value(weights, 'type', 'int', weights_path)
        _expect_value(weights, 'actorder', None, weights_path)
        _expect_value(weights, 'block_structure', None, weights_path)

        has_observer = 'observer' in weights
        has_observer_kwargs = 'observer_kwargs' in weights
        if has_observer != has_observer_kwargs:
            _config_error(weights_path, '`observer` and `observer_kwargs` must be specified together')
        observer = None
        observer_kwargs = ()
        if has_observer:
            observer = _expect_value(weights, 'observer', 'minmax', weights_path)
            raw_observer_kwargs = _required(weights, 'observer_kwargs', weights_path)
            if not isinstance(raw_observer_kwargs, Mapping) or raw_observer_kwargs:
                _config_error(f'{weights_path}.observer_kwargs', 'expected an empty object')
            observer_kwargs = tuple(sorted(raw_observer_kwargs.items()))

        _expect_value(quant_config, 'kv_cache_scheme', None, path)

        ignore = _required(quant_config, 'ignore', path)
        if not isinstance(ignore, list) or not ignore or not all(isinstance(rule, str) and rule for rule in ignore):
            _config_error(f'{path}.ignore', 'expected a non-empty list of non-empty strings')
        if len(set(ignore)) != len(ignore):
            _config_error(f'{path}.ignore', 'duplicate rules are not allowed')
        for index, rule in enumerate(ignore):
            if not rule.startswith('re:'):
                continue
            if not rule[3:]:
                _config_error(f'{path}.ignore[{index}]', 'regular expression must not be empty')
            try:
                re.compile(rule[3:])
            except re.error as exc:
                _config_error(f'{path}.ignore[{index}]', f'invalid regular expression: {exc}')

        return cls(
            format=quant_format,
            targets=tuple(targets),
            num_bits=num_bits,
            group_size=group_size,
            strategy=strategy,
            symmetric=symmetric,
            dynamic=dynamic,
            weight_type=weight_type,
            observer=observer,
            observer_kwargs=observer_kwargs,
            ignore=tuple(ignore),
            quantization_status=quantization_status,
        )

    def is_ignored(self, module_fqn: str) -> bool:
        """Match ignore rules using compressed-tensors canonical semantics."""
        if not isinstance(module_fqn, str) or not module_fqn:
            raise ValueError('compressed-tensors matching requires a non-empty canonical module FQN')
        return any(_matches_ignore(rule, module_fqn) for rule in self.ignore)

    @staticmethod
    def is_routed_expert(module_fqn: str) -> bool:
        """Return whether an LMDeploy fused-MoE prefix owns routed experts."""
        return re.search(r'(^|\.)mlp\.experts$', module_fqn) is not None

    @staticmethod
    def is_routed_expert_projection(module_fqn: str) -> bool:
        """Return whether a Linear FQN is a routed expert projection."""
        pattern = r'(^|\.)mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$'
        return re.search(pattern, module_fqn) is not None


@dataclass(frozen=True)
class CompressedTensorsW4A16Shard:
    """One TP-local view of a logical compressed-tensors weight."""

    logical_shape: tuple[int, int]
    weight_packed: torch.Tensor
    weight_scale: torch.Tensor


def _normalize_logical_shape(logical_shape: torch.Tensor | tuple[int, int]) -> tuple[int, int]:
    if isinstance(logical_shape, torch.Tensor):
        if logical_shape.dtype != torch.int32 or logical_shape.shape != (2, ):
            raise ValueError('logical_shape tensor must have dtype int32 and shape [2]')
        logical_shape = tuple(int(dim) for dim in logical_shape.tolist())
    elif isinstance(logical_shape, (list, tuple)):
        logical_shape = tuple(logical_shape)
    else:
        raise TypeError('logical_shape must be an int32 tensor or a two-element sequence')

    if (len(logical_shape) != 2
            or not all(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in logical_shape)):
        raise ValueError('logical_shape must contain two positive integers')
    return logical_shape


def _validate_w4a16_runtime_layout(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_shape: torch.Tensor | tuple[int, int],
    config: CompressedTensorsW4A16Config,
) -> tuple[int, int]:
    if not isinstance(config, CompressedTensorsW4A16Config):
        raise TypeError('config must be a CompressedTensorsW4A16Config')
    if weight_packed.dtype != torch.int32:
        raise ValueError(f'weight_packed must use int32 storage, got {weight_packed.dtype}')
    if weight_scale.dtype != torch.bfloat16:
        raise ValueError(f'weight_scale must use bfloat16, got {weight_scale.dtype}')

    logical_shape = _normalize_logical_shape(logical_shape)
    out_features, in_features = logical_shape
    pack_factor = 32 // config.num_bits
    expected_packed_shape = (out_features, math.ceil(in_features / pack_factor))
    expected_scale_shape = (out_features, math.ceil(in_features / config.group_size))
    if tuple(weight_packed.shape) != expected_packed_shape:
        raise ValueError(
            f'weight_packed shape mismatch: expected {expected_packed_shape}, got {tuple(weight_packed.shape)}')
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f'weight_scale shape mismatch: expected {expected_scale_shape}, got {tuple(weight_scale.shape)}')
    if weight_packed.device != weight_scale.device:
        raise ValueError('weight_packed and weight_scale must be on the same device')
    return logical_shape


def unpack_compressed_tensors_w4a16(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_shape: torch.Tensor | tuple[int, int],
    config: CompressedTensorsW4A16Config,
) -> torch.Tensor:
    """Unpack pack-quantized INT4 codes into signed int8 reference values.

    compressed-tensors stores ``q + 8`` in little-nibble order along K: q0
    occupies bits 0..3 and q7 occupies bits 28..31 of each int32 word.
    This function is a correctness oracle and must not be used to materialize
    every expert of a production checkpoint.
    """
    _, in_features = _validate_w4a16_runtime_layout(weight_packed, weight_scale, logical_shape, config)
    pack_factor = 32 // config.num_bits
    shifts = torch.arange(pack_factor, dtype=torch.int32, device=weight_packed.device) * config.num_bits
    mask = (1 << config.num_bits) - 1
    unsigned_codes = (weight_packed.unsqueeze(-1) >> shifts) & mask
    unsigned_codes = unsigned_codes.flatten(-2)[..., :in_features]
    signed_offset = 1 << (config.num_bits - 1)
    return (unsigned_codes - signed_offset).to(torch.int8)


def dequantize_compressed_tensors_w4a16(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_shape: torch.Tensor | tuple[int, int],
    config: CompressedTensorsW4A16Config,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Explicitly dequantize one W4A16 weight for tests and tiny fixtures."""
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError('reference dequantization dtype must be a floating-point torch.dtype')
    logical_shape = _validate_w4a16_runtime_layout(weight_packed, weight_scale, logical_shape, config)
    signed_codes = unpack_compressed_tensors_w4a16(weight_packed, weight_scale, logical_shape, config)
    in_features = logical_shape[1]
    expanded_scale = weight_scale.repeat_interleave(config.group_size, dim=-1)[..., :in_features]
    return signed_codes.to(dtype) * expanded_scale.to(dtype)


def shard_compressed_tensors_w4a16(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_shape: torch.Tensor | tuple[int, int],
    config: CompressedTensorsW4A16Config,
    world_size: int,
    rank: int,
    colwise: bool,
) -> CompressedTensorsW4A16Shard:
    """Return an exact TP shard without unpacking the INT4 payload.

    Column parallelism slices logical N. Row parallelism slices logical K and
    therefore requires every local K range to preserve both int32 pack and
    quantization-group boundaries.
    """
    logical_shape = _validate_w4a16_runtime_layout(weight_packed, weight_scale, logical_shape, config)
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError('world_size must be a positive integer')
    if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
        raise ValueError(f'rank must be in [0, {world_size}), got {rank!r}')

    out_features, in_features = logical_shape
    pack_factor = 32 // config.num_bits
    if colwise:
        if out_features % world_size != 0:
            raise ValueError(f'logical N={out_features} is not divisible by TP world_size={world_size}')
        local_out = out_features // world_size
        start = rank * local_out
        packed_shard = weight_packed.narrow(0, start, local_out).contiguous()
        scale_shard = weight_scale.narrow(0, start, local_out).contiguous()
        local_shape = (local_out, in_features)
    else:
        if in_features % world_size != 0:
            raise ValueError(f'logical K={in_features} is not divisible by TP world_size={world_size}')
        local_in = in_features // world_size
        if local_in % pack_factor != 0:
            raise ValueError(f'TP-local K={local_in} splits an INT4 storage word of {pack_factor} values')
        if local_in % config.group_size != 0:
            raise ValueError(f'TP-local K={local_in} splits a quantization group of {config.group_size} values')
        packed_per_rank = local_in // pack_factor
        scales_per_rank = local_in // config.group_size
        packed_shard = weight_packed.narrow(1, rank * packed_per_rank, packed_per_rank).contiguous()
        scale_shard = weight_scale.narrow(1, rank * scales_per_rank, scales_per_rank).contiguous()
        local_shape = (out_features, local_in)

    return CompressedTensorsW4A16Shard(
        logical_shape=local_shape,
        weight_packed=packed_shard,
        weight_scale=scale_shard,
    )


@dataclass(frozen=True)
class CompressedTensorEntry:
    """The three checkpoint tensors that represent one logical weight."""

    module_name: str
    shard: str
    packed_name: str
    scale_name: str
    shape_name: str


@dataclass(frozen=True)
class CompressedTensorsCheckpointManifest:
    """Index-only manifest; constructing it never reads tensor payloads."""

    model_path: str
    index_path: str
    index_sha256: str
    quant_config: CompressedTensorsW4A16Config
    expected_module_pattern: str
    expected_module_shapes: tuple[tuple[str, tuple[int, int]], ...]
    total_size: int
    tensor_count: int
    shards: tuple[str, ...]
    entries: tuple[CompressedTensorEntry, ...]
    ignored_rule_tensor_counts: tuple[tuple[str, int], ...]

    @property
    def quantized_module_count(self) -> int:
        return len(self.entries)

    @property
    def quantized_tensor_count(self) -> int:
        return len(self.entries) * len(_QUANT_SUFFIXES)


@dataclass(frozen=True)
class CompressedTensorLayout:
    """Observed logical and physical layout shared by checkpoint weights."""

    logical_shape: tuple[int, ...]
    packed_dtype: str
    packed_shape: tuple[int, ...]
    scale_dtype: str
    scale_shape: tuple[int, ...]
    shape_dtype: str
    shape_shape: tuple[int, ...]
    count: int


@dataclass(frozen=True)
class CompressedTensorsHeaderAudit:
    """Result of a header audit that reads only tiny weight_shape payloads."""

    tensor_count: int
    quantized_module_count: int
    header_bytes: int
    shape_payload_bytes: int
    layouts: tuple[CompressedTensorLayout, ...]
    ignored_rule_dtype_counts: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]


def _reject_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f'duplicate JSON key: {key}')
        obj[key] = value
    return obj


def _load_index(index_path: str):
    try:
        with open(index_path, 'rb') as file:
            raw_index = file.read()
        index_sha256 = hashlib.sha256(raw_index).hexdigest()
        index = json.loads(raw_index, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f'Invalid compressed-tensors checkpoint index `{index_path}`: {exc}') from exc

    if not isinstance(index, dict):
        raise ValueError(f'Invalid compressed-tensors checkpoint index `{index_path}`: expected an object')
    metadata = index.get('metadata')
    weight_map = index.get('weight_map')
    if not isinstance(metadata, dict):
        raise ValueError('Invalid compressed-tensors checkpoint index at `metadata`: expected an object')
    total_size = metadata.get('total_size')
    if not isinstance(total_size, int) or isinstance(total_size, bool) or total_size < 0:
        raise ValueError('Invalid compressed-tensors checkpoint index at `metadata.total_size`: '
                         'expected a non-negative integer')
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError('Invalid compressed-tensors checkpoint index at `weight_map`: expected a non-empty object')
    if not all(isinstance(name, str) and name and isinstance(shard, str) and shard
               for name, shard in weight_map.items()):
        raise ValueError('Invalid compressed-tensors checkpoint index at `weight_map`: '
                         'tensor names and shard names must be non-empty strings')
    return index_sha256, total_size, weight_map


def _validate_shard(model_path: str, shard: str):
    if osp.isabs(shard) or osp.basename(shard) != shard or shard in {'.', '..'}:
        raise ValueError(f'Invalid checkpoint shard path `{shard}`: only a file name is allowed')
    shard_path = osp.join(model_path, shard)
    if not osp.isfile(shard_path):
        raise ValueError(f'Checkpoint shard does not exist: `{shard_path}`')


def _canonical_module_name(tensor_name: str):
    for suffix in ('.weight', '.bias'):
        if tensor_name.endswith(suffix):
            return tensor_name[:-len(suffix)]
    return tensor_name.rsplit('.', 1)[0] if '.' in tensor_name else tensor_name


def _matches_ignore(rule: str, module_name: str):
    if rule.startswith('re:'):
        return re.match(rule[3:], module_name) is not None
    return rule == module_name


def _compile_expected_module_pattern(expected_module_pattern: str):
    if not isinstance(expected_module_pattern, str) or not expected_module_pattern:
        raise ValueError('A non-empty expected module pattern is required for target-scope validation')
    try:
        return re.compile(expected_module_pattern)
    except re.error as exc:
        raise ValueError(f'Invalid expected module pattern: {exc}') from exc


def _validate_expected_module_shapes(
    expected_module_shapes: Mapping[str, tuple[int, int]],
    expected_re,
):
    if not isinstance(expected_module_shapes, Mapping) or not expected_module_shapes:
        raise ValueError('expected_module_shapes must be a non-empty mapping')

    normalized_shapes = {}
    for module_name, logical_shape in expected_module_shapes.items():
        if not isinstance(module_name, str) or not module_name:
            raise ValueError('expected_module_shapes keys must be non-empty strings')
        if expected_re.fullmatch(module_name) is None:
            raise ValueError(f'Expected module `{module_name}` is outside the expected target scope')
        if (not isinstance(logical_shape, (list, tuple)) or len(logical_shape) != 2
                or not all(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in logical_shape)):
            raise ValueError(f'Expected logical shape for `{module_name}` must contain two positive integers')
        normalized_shapes[module_name] = tuple(logical_shape)
    return normalized_shapes


def _collect_compressed_entries(
    weight_map: Mapping[str, str],
    config: CompressedTensorsW4A16Config,
):
    """Rebuild compressed triplets from an index without reading payloads."""
    module_parts = defaultdict(dict)
    for tensor_name, shard in weight_map.items():
        if tensor_name.endswith(_UNSUPPORTED_QUANT_SUFFIXES):
            raise ValueError(f'Unsupported compressed-tensors auxiliary tensor: `{tensor_name}`')
        for suffix in _QUANT_SUFFIXES:
            if not tensor_name.endswith(suffix):
                continue
            module_name = tensor_name[:-len(suffix)]
            module_parts[module_name][suffix] = (tensor_name, shard)
            break

    if not module_parts:
        raise ValueError('Checkpoint index contains no compressed-tensors weight triplets')

    allowed_companions = {
        module_name: {tensor_name for tensor_name, _ in parts.values()}
        for module_name, parts in module_parts.items()
    }
    for tensor_name in weight_map:
        if '.weight_' not in tensor_name:
            continue
        module_name = tensor_name.rsplit('.weight_', 1)[0]
        if module_name in allowed_companions and tensor_name not in allowed_companions[module_name]:
            raise ValueError(f'Unsupported compressed-tensors sibling tensor: `{tensor_name}`')

    entries = []
    for module_name in sorted(module_parts):
        parts = module_parts[module_name]
        missing = [suffix for suffix in _QUANT_SUFFIXES if suffix not in parts]
        if missing:
            raise ValueError(f'Incomplete compressed-tensors triplet for `{module_name}`: missing {missing}')
        part_shards = {parts[suffix][1] for suffix in _QUANT_SUFFIXES}
        if len(part_shards) != 1:
            raise ValueError(f'Compressed-tensors companions for `{module_name}` span shards: {sorted(part_shards)}')
        if f'{module_name}.weight' in weight_map:
            raise ValueError(f'Compressed module `{module_name}` also contains an uncompressed `.weight` tensor')
        if config.is_ignored(module_name):
            raise ValueError(f'Compressed module `{module_name}` matches an ignore rule')

        entries.append(
            CompressedTensorEntry(
                module_name=module_name,
                shard=next(iter(part_shards)),
                packed_name=parts['.weight_packed'][0],
                scale_name=parts['.weight_scale'][0],
                shape_name=parts['.weight_shape'][0],
            ))
    return tuple(entries)


def _validate_target_scope(
    weight_map: Mapping[str, str],
    entries: tuple[CompressedTensorEntry, ...],
    config: CompressedTensorsW4A16Config,
    expected_re,
    expected_module_names: set[str],
):
    actual_module_names = {entry.module_name for entry in entries}
    for entry in entries:
        if expected_re.fullmatch(entry.module_name) is None:
            raise ValueError(f'Compressed module `{entry.module_name}` is outside the expected target scope')

    for tensor_name in weight_map:
        if not tensor_name.endswith('.weight'):
            continue
        module_name = tensor_name[:-len('.weight')]
        if (expected_re.fullmatch(module_name) is not None and not config.is_ignored(module_name)
                and module_name not in actual_module_names):
            raise ValueError(f'Expected compressed target `{module_name}` has an uncompressed `.weight` tensor '
                             'without a compressed triplet')

    if len(actual_module_names) != len(expected_module_names):
        raise ValueError(f'Compressed module count mismatch: expected {len(expected_module_names)}, '
                         f'got {len(actual_module_names)}')
    if actual_module_names != expected_module_names:
        missing = sorted(expected_module_names - actual_module_names)[:3]
        extra = sorted(actual_module_names - expected_module_names)[:3]
        raise ValueError(f'Compressed module set mismatch: missing={missing}, extra={extra}')


def build_compressed_tensors_manifest(
    model_path: str,
    config: CompressedTensorsW4A16Config,
    expected_module_pattern: str,
    expected_module_shapes: Mapping[str, tuple[int, int]],
    expected_index_sha256: str = None,
):
    """Validate a sharded index and return its compressed weight manifest."""
    if not isinstance(config, CompressedTensorsW4A16Config):
        raise TypeError('config must be a CompressedTensorsW4A16Config')
    model_path = osp.abspath(model_path)
    index_path = osp.join(model_path, 'model.safetensors.index.json')
    if not osp.isfile(index_path):
        raise ValueError(f'Checkpoint index does not exist: `{index_path}`')

    index_sha256, total_size, weight_map = _load_index(index_path)
    if expected_index_sha256 is not None and index_sha256 != expected_index_sha256:
        raise ValueError(f'Checkpoint index SHA256 mismatch: expected {expected_index_sha256}, got {index_sha256}')

    shards = tuple(sorted(set(weight_map.values())))
    for shard in shards:
        _validate_shard(model_path, shard)

    expected_re = _compile_expected_module_pattern(expected_module_pattern)
    expected_module_shapes = _validate_expected_module_shapes(expected_module_shapes, expected_re)
    expected_module_names = set(expected_module_shapes)
    entries = _collect_compressed_entries(weight_map, config)
    _validate_target_scope(weight_map, entries, config, expected_re, expected_module_names)

    ignore_counts = Counter({rule: 0 for rule in config.ignore})
    quantized_names = {name for entry in entries for name in (entry.packed_name, entry.scale_name, entry.shape_name)}
    for tensor_name in weight_map:
        if tensor_name in quantized_names:
            continue
        module_name = _canonical_module_name(tensor_name)
        for rule in config.ignore:
            if _matches_ignore(rule, module_name):
                ignore_counts[rule] += 1

    return CompressedTensorsCheckpointManifest(
        model_path=model_path,
        index_path=index_path,
        index_sha256=index_sha256,
        quant_config=config,
        expected_module_pattern=expected_module_pattern,
        expected_module_shapes=tuple(sorted(expected_module_shapes.items())),
        total_size=total_size,
        tensor_count=len(weight_map),
        shards=shards,
        entries=entries,
        ignored_rule_tensor_counts=tuple((rule, ignore_counts[rule]) for rule in config.ignore),
    )


def _header_size(path: str):
    with open(path, 'rb') as file:
        raw_size = file.read(8)
    if len(raw_size) != 8:
        raise ValueError(f'Invalid safetensors file `{path}`: missing header length')
    return 8 + int.from_bytes(raw_size, byteorder='little', signed=False)


def audit_compressed_tensors_headers(
    manifest: CompressedTensorsCheckpointManifest,
    config: CompressedTensorsW4A16Config,
):
    """Audit all safetensors headers without loading packed/scale payloads."""
    from safetensors import safe_open

    if not isinstance(manifest, CompressedTensorsCheckpointManifest):
        raise TypeError('manifest must be a CompressedTensorsCheckpointManifest')
    if not isinstance(config, CompressedTensorsW4A16Config):
        raise TypeError('config must be a CompressedTensorsW4A16Config')
    if config != manifest.quant_config:
        raise ValueError('Header audit config does not match the manifest quantization config')

    index_sha256, total_size, weight_map = _load_index(manifest.index_path)
    if index_sha256 != manifest.index_sha256:
        raise ValueError('Checkpoint index changed after the manifest was built')
    if total_size != manifest.total_size or len(weight_map) != manifest.tensor_count:
        raise ValueError('Checkpoint index metadata changed after the manifest was built')
    current_shards = tuple(sorted(set(weight_map.values())))
    if current_shards != manifest.shards:
        raise ValueError('Checkpoint shard set changed after the manifest was built')

    expected_re = _compile_expected_module_pattern(manifest.expected_module_pattern)
    stored_module_shapes = manifest.expected_module_shapes
    if not isinstance(stored_module_shapes, tuple):
        raise ValueError('Checkpoint manifest has invalid expected module shapes')
    try:
        stored_shapes_mapping = dict(stored_module_shapes)
    except (TypeError, ValueError) as exc:
        raise ValueError('Checkpoint manifest has invalid expected module shapes') from exc
    if len(stored_shapes_mapping) != len(stored_module_shapes):
        raise ValueError('Checkpoint manifest has duplicate expected module shapes')
    expected_module_shapes = _validate_expected_module_shapes(stored_shapes_mapping, expected_re)
    expected_module_names = set(expected_module_shapes)
    current_entries = _collect_compressed_entries(weight_map, config)
    _validate_target_scope(weight_map, current_entries, config, expected_re, expected_module_names)
    if current_entries != manifest.entries:
        raise ValueError('Checkpoint manifest entries do not match the current checkpoint index')

    expected_by_shard = defaultdict(set)
    entries_by_shard = defaultdict(list)
    for tensor_name, shard in weight_map.items():
        expected_by_shard[shard].add(tensor_name)
    for entry in current_entries:
        entries_by_shard[entry.shard].append(entry)

    layout_counts = Counter()
    ignored_dtype_counts = {rule: Counter() for rule in config.ignore}
    quantized_names = {
        name for entry in current_entries for name in (entry.packed_name, entry.scale_name, entry.shape_name)
    }
    header_bytes = 0
    checkpoint_payload_bytes = 0
    shape_payload_bytes = 0
    for shard in manifest.shards:
        shard_path = osp.join(manifest.model_path, shard)
        shard_header_bytes = _header_size(shard_path)
        shard_size = osp.getsize(shard_path)
        if shard_size < shard_header_bytes:
            raise ValueError(f'Invalid safetensors file `{shard_path}`: header exceeds file size')
        header_bytes += shard_header_bytes
        checkpoint_payload_bytes += shard_size - shard_header_bytes
        with safe_open(shard_path, framework='pt', device='cpu') as file:
            actual_names = set(file.keys())
            if actual_names != expected_by_shard[shard]:
                missing = sorted(expected_by_shard[shard] - actual_names)[:3]
                extra = sorted(actual_names - expected_by_shard[shard])[:3]
                raise ValueError(f'Safetensors header/index mismatch for `{shard}`: missing={missing}, extra={extra}')

            for tensor_name in expected_by_shard[shard] - quantized_names:
                module_name = _canonical_module_name(tensor_name)
                matched_rules = [rule for rule in config.ignore if _matches_ignore(rule, module_name)]
                if not matched_rules:
                    continue
                tensor_dtype = file.get_slice(tensor_name).get_dtype()
                if tensor_dtype != 'BF16':
                    raise ValueError(f'Ignored tensor `{tensor_name}` must remain BF16, got {tensor_dtype}')
                for rule in matched_rules:
                    ignored_dtype_counts[rule][tensor_dtype] += 1

            for entry in entries_by_shard[shard]:
                packed = file.get_slice(entry.packed_name)
                scale = file.get_slice(entry.scale_name)
                shape = file.get_slice(entry.shape_name)
                packed_dtype = packed.get_dtype()
                scale_dtype = scale.get_dtype()
                shape_dtype = shape.get_dtype()
                packed_shape = tuple(packed.get_shape())
                scale_shape = tuple(scale.get_shape())
                shape_shape = tuple(shape.get_shape())

                if shape_dtype != 'I32' or shape_shape != (2, ):
                    raise ValueError(f'Invalid logical shape tensor for `{entry.module_name}`: '
                                     f'dtype={shape_dtype}, shape={shape_shape}')
                if packed_dtype != 'I32':
                    raise ValueError(f'Packed tensor for `{entry.module_name}` must use I32 storage, '
                                     f'got {packed_dtype}')
                if scale_dtype != 'BF16':
                    raise ValueError(f'Scale tensor for `{entry.module_name}` must use BF16, got {scale_dtype}')
                logical_shape_tensor = file.get_tensor(entry.shape_name)
                logical_shape = tuple(int(value) for value in logical_shape_tensor.tolist())
                shape_payload_bytes += logical_shape_tensor.numel() * logical_shape_tensor.element_size()
                if len(logical_shape) != 2 or any(value <= 0 for value in logical_shape):
                    raise ValueError(f'Invalid logical shape value for `{entry.module_name}`: {logical_shape}')
                expected_logical_shape = expected_module_shapes[entry.module_name]
                if logical_shape != expected_logical_shape:
                    raise ValueError(f'Logical shape mismatch for `{entry.module_name}`: '
                                     f'expected {expected_logical_shape}, got {logical_shape}')

                pack_factor = 32 // config.num_bits
                if logical_shape[-1] % pack_factor != 0:
                    raise ValueError(f'Logical K dimension for `{entry.module_name}` is not divisible by '
                                     f'pack factor {pack_factor}: {logical_shape[-1]}')
                expected_packed_shape = logical_shape[:-1] + (logical_shape[-1] // pack_factor, )
                if packed_shape != expected_packed_shape:
                    raise ValueError(f'Packed shape mismatch for `{entry.module_name}`: '
                                     f'expected {expected_packed_shape}, got {packed_shape}')
                if logical_shape[-1] % config.group_size != 0:
                    raise ValueError(f'Logical K dimension for `{entry.module_name}` is not divisible by '
                                     f'group size {config.group_size}: {logical_shape[-1]}')
                expected_scale_shape = logical_shape[:-1] + (logical_shape[-1] // config.group_size, )
                if scale_shape != expected_scale_shape:
                    raise ValueError(f'Scale shape mismatch for `{entry.module_name}`: '
                                     f'expected {expected_scale_shape}, got {scale_shape}')

                layout_counts[(logical_shape, packed_dtype, packed_shape, scale_dtype, scale_shape, shape_dtype,
                               shape_shape)] += 1

    if checkpoint_payload_bytes != manifest.total_size:
        raise ValueError(f'Checkpoint payload size mismatch: index={manifest.total_size}, '
                         f'safetensors={checkpoint_payload_bytes}')

    manifest_ignore_counts = dict(manifest.ignored_rule_tensor_counts)
    for rule, dtype_counts in ignored_dtype_counts.items():
        if sum(dtype_counts.values()) != manifest_ignore_counts[rule]:
            raise ValueError(f'Ignored tensor count changed for rule `{rule}`')

    layouts = tuple(
        CompressedTensorLayout(
            logical_shape=key[0],
            packed_dtype=key[1],
            packed_shape=key[2],
            scale_dtype=key[3],
            scale_shape=key[4],
            shape_dtype=key[5],
            shape_shape=key[6],
            count=count,
        ) for key, count in sorted(layout_counts.items()))
    return CompressedTensorsHeaderAudit(
        tensor_count=manifest.tensor_count,
        quantized_module_count=len(current_entries),
        header_bytes=header_bytes,
        shape_payload_bytes=shape_payload_bytes,
        layouts=layouts,
        ignored_rule_dtype_counts=tuple(
            (rule, tuple(sorted(ignored_dtype_counts[rule].items()))) for rule in config.ignore),
    )
