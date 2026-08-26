# Copyright (c) OpenMMLab. All rights reserved.
"""Shared data structures for the Mooncake Store connector."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np

from lmdeploy.pytorch.kv_connector.base import KVConnectorMetadata, KVSaveBlockLease

DEFAULT_GLOBAL_SEGMENT_SIZE = 4 * 1024 * 1024 * 1024
DEFAULT_LOCAL_BUFFER_SIZE = 4 * 1024 * 1024 * 1024
MOONCAKE_CONFIG_PATH_ENV = 'MOONCAKE_CONFIG_PATH'
MOONCAKE_BLOCK_HASH_BYTES = hashlib.sha256().digest_size

_BLOCK_HASH_SCHEMA = b'lmdeploy-mooncake-prefix-block-v1\x00'

MooncakeMode = Literal['embedded']


def validate_kv_head_replica_num(
    kv_head_replica_num: int,
    tp_size: int,
) -> int:
    """Validate and return the number of TP replicas per KV-head shard."""
    if tp_size % kv_head_replica_num != 0:
        raise ValueError(
            f'tp_size ({tp_size}) must be divisible by kv_head_replica_num '
            f'({kv_head_replica_num})')
    return kv_head_replica_num


def _identity_bytes(extra_identity: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(extra_identity, str):
        return extra_identity.encode('utf-8')
    if isinstance(extra_identity, (bytes, bytearray, memoryview)):
        return bytes(extra_identity)
    raise TypeError('extra_identity must be a string or bytes-like value')


@lru_cache(maxsize=16)
def _token_block_struct(block_size: int) -> struct.Struct:
    """Return a cached encoder for one block of unsigned token IDs."""
    return struct.Struct(f'>{block_size}Q')


def _pack_python_token_block(
    token_ids: Sequence[int],
    block_struct: struct.Struct,
) -> bytes:
    """Validate and encode one generic token block."""
    values = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id < 0 or token_id >= 2**64:
            raise ValueError('token IDs must fit in an unsigned 64-bit integer')
        values.append(token_id)
    return block_struct.pack(*values)


def _pack_numpy_token_suffix(
    token_ids: np.ndarray,
    start: int,
    end: int,
) -> bytes:
    """Encode an integer ndarray suffix as contiguous big-endian uint64."""
    if token_ids.ndim != 1 or token_ids.dtype.kind not in ('i', 'u'):
        raise TypeError('token_ids must contain integers')
    suffix = token_ids[start:end]
    if suffix.dtype.itemsize > 8:
        raise ValueError('token IDs must fit in an unsigned 64-bit integer')
    if suffix.dtype.kind == 'i' and suffix.size and bool(np.any(suffix < 0)):
        raise ValueError('token IDs must fit in an unsigned 64-bit integer')
    return np.asarray(suffix, dtype='>u8', order='C').tobytes()


def build_prefix_block_hashes(
    token_ids: Sequence[int],
    block_size: int,
    *,
    extra_identity: bytes | bytearray | memoryview | str = b'',
    previous_hashes: Sequence[bytes] = (),
) -> tuple[bytes, ...]:
    """Build stable, prefix-chained hashes for complete token blocks.

    Each digest includes a versioned schema, the preceding digest, the block
    size, canonically encoded token IDs, and request attributes such as the
    active adapter from ``extra_identity``. A mutable partial tail is
    intentionally excluded. ``previous_hashes`` lets callers extend an
    append-only request without rehashing its existing prefix.
    """
    identity = _identity_bytes(extra_identity)
    if len(identity) >= 2**32:
        raise ValueError('extra_identity is too large')

    full_block_count = len(token_ids) // block_size
    block_hashes = []
    for block_hash in previous_hashes:
        if not isinstance(block_hash, (bytes, bytearray, memoryview)):
            raise TypeError('previous_hashes must contain bytes-like values')
        block_hash = bytes(block_hash)
        if len(block_hash) != MOONCAKE_BLOCK_HASH_BYTES:
            raise ValueError(
                f'previous hashes must contain {MOONCAKE_BLOCK_HASH_BYTES} bytes')
        block_hashes.append(block_hash)
    if len(block_hashes) > full_block_count:
        raise ValueError('previous_hashes exceed the complete token blocks')

    parent_hash = block_hashes[-1] if block_hashes else None

    first_new_block = len(block_hashes)
    if first_new_block == full_block_count:
        return tuple(block_hashes)

    first_new_token = first_new_block * block_size
    complete_token_end = full_block_count * block_size
    packed_numpy_tokens = None
    if (isinstance(token_ids, np.ndarray) and token_ids.ndim == 1
            and token_ids.dtype.kind in ('i', 'u')):
        packed_numpy_tokens = memoryview(
            _pack_numpy_token_suffix(
                token_ids,
                first_new_token,
                complete_token_end,
            ))

    block_struct = _token_block_struct(block_size)
    encoded_block_size = struct.pack('>I', block_size)
    encoded_identity = struct.pack('>I', len(identity)) + identity
    packed_block_bytes = block_struct.size
    for block_index in range(first_new_block, full_block_count):
        digest = hashlib.sha256()
        digest.update(_BLOCK_HASH_SCHEMA)
        if parent_hash is not None:
            digest.update(parent_hash)
        digest.update(encoded_block_size)
        if packed_numpy_tokens is not None:
            packed_offset = (block_index - first_new_block) * packed_block_bytes
            digest.update(
                packed_numpy_tokens[packed_offset:packed_offset + packed_block_bytes])
        else:
            start = block_index * block_size
            digest.update(
                _pack_python_token_block(
                    token_ids[start:start + block_size],
                    block_struct,
                ))
        digest.update(encoded_identity)
        parent_hash = digest.digest()
        block_hashes.append(parent_hash)
    return tuple(block_hashes)


@dataclass(frozen=True)
class MooncakeStoreKeyMetadata:
    """Stable namespace shared by Mooncake lookup and later transfers."""

    model_name: str
    cache_prefix: str
    tp_size: int
    block_size: int
    kv_head_replica_num: int = 1

    def __post_init__(self) -> None:
        validate_kv_head_replica_num(self.kv_head_replica_num, self.tp_size)

    @property
    def num_kv_head_shards(self) -> int:
        """Return the number of distinct KV-head namespaces."""
        return self.tp_size // self.kv_head_replica_num


def build_store_key(
    metadata: MooncakeStoreKeyMetadata,
    kv_head_rank: int,
    block_hash: bytes | bytearray | memoryview,
) -> str:
    """Build Mooncake key for one unique KV-head shard."""
    block_hash = bytes(block_hash)
    if len(block_hash) != MOONCAKE_BLOCK_HASH_BYTES:
        raise ValueError(f'block_hash must contain {MOONCAKE_BLOCK_HASH_BYTES} bytes')

    prefix = f'{metadata.cache_prefix}@' if metadata.cache_prefix else ''
    return (
        f'{prefix}{metadata.model_name}'
        f'@tp_rank:{kv_head_rank}'
        '@group:0'
        f'@{block_hash.hex()}'
    )


class BlobBlockHashes(Sequence[bytes]):
    """Lazy fixed-width hash view over one ZMQ payload frame."""

    def __init__(self, blob: memoryview, hash_len: int) -> None:
        if hash_len < 0:
            raise ValueError('hash_len must be non-negative')
        if hash_len == 0 and len(blob) != 0:
            raise ValueError('a non-empty hash payload requires hash_len greater than 0')
        if hash_len > 0 and len(blob) % hash_len != 0:
            raise ValueError('hash payload length must be divisible by hash_len')
        self._blob = blob
        self._hash_len = hash_len
        self._length = len(blob) // hash_len if hash_len else 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int | slice) -> bytes | list[bytes]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self._length))]
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        offset = index * self._hash_len
        return bytes(self._blob[offset:offset + self._hash_len])


def _parse_size(value: Any) -> int:
    """Parse a byte count or a size string using binary units."""
    if isinstance(value, bool):
        raise TypeError('Mooncake size must not be a boolean')
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise TypeError(f'unsupported Mooncake size type: {type(value).__name__}')

    match = re.fullmatch(r'\s*(\d+(?:\.\d*)?|\.\d+)\s*(gb|mb|kb|b)?\s*', value, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f'invalid Mooncake size: {value!r}')

    multipliers = {
        'gb': 1024**3,
        'mb': 1024**2,
        'kb': 1024,
        'b': 1,
    }
    number = float(match.group(1))
    unit = (match.group(2) or 'b').lower()
    return int(number * multipliers[unit])


@dataclass(frozen=True)
class MooncakeStoreConfig:
    """Validated configuration for an embedded Mooncake Store client."""

    metadata_server: str
    master_server_address: str
    protocol: str
    device_name: str
    mode: MooncakeMode = 'embedded'
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE
    enable_offload: bool = False

    def __post_init__(self) -> None:
        required_addresses = (
            ('metadata_server', self.metadata_server),
            ('master_server_address', self.master_server_address),
        )
        for field_name, value in required_addresses:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{field_name} must be a non-empty string')
        if not isinstance(self.protocol, str) or self.protocol not in ('rdma', 'tcp'):
            raise ValueError("protocol must be either 'rdma' or 'tcp'")
        if not isinstance(self.device_name, str):
            raise TypeError('device_name must be a string')
        if self.mode != 'embedded':
            raise ValueError('LMDeploy Mooncake Store currently supports only embedded mode')
        if self.enable_offload:
            raise ValueError('LMDeploy Mooncake Store does not support SSD offload yet')
        if isinstance(self.global_segment_size, bool) or not isinstance(self.global_segment_size, int):
            raise TypeError('global_segment_size must be an integer')
        if isinstance(self.local_buffer_size, bool) or not isinstance(self.local_buffer_size, int):
            raise TypeError('local_buffer_size must be an integer')
        if self.global_segment_size <= 0:
            raise ValueError('global_segment_size must be greater than 0')
        if self.local_buffer_size <= 0:
            raise ValueError('local_buffer_size must be greater than 0')

    @classmethod
    def from_file(cls, file_path: str | os.PathLike[str]) -> MooncakeStoreConfig:
        """Load and validate a Mooncake JSON configuration file."""
        with open(file_path, encoding='utf-8') as file:
            raw_config = json.load(file)
        if not isinstance(raw_config, Mapping):
            raise TypeError('Mooncake configuration must be a JSON object')

        enable_offload = raw_config.get('enable_offload', False)
        if not isinstance(enable_offload, bool):
            raise TypeError('enable_offload must be a boolean')
        enable_ssd_offload = raw_config.get('enable_ssd_offload', False)
        if not isinstance(enable_ssd_offload, bool):
            raise TypeError('enable_ssd_offload must be a boolean')
        ssd_offload_path = raw_config.get('ssd_offload_path', '')
        if not isinstance(ssd_offload_path, str):
            raise TypeError('ssd_offload_path must be a string')
        if enable_ssd_offload or ssd_offload_path:
            raise ValueError('LMDeploy Mooncake Store does not support SSD offload yet')
        return cls(
            metadata_server=raw_config.get('metadata_server', ''),
            master_server_address=raw_config.get('master_server_address', ''),
            protocol=raw_config.get('protocol', 'rdma'),
            device_name=raw_config.get('device_name', ''),
            mode=raw_config.get('mode', 'embedded'),
            global_segment_size=_parse_size(raw_config.get('global_segment_size', DEFAULT_GLOBAL_SEGMENT_SIZE)),
            local_buffer_size=_parse_size(raw_config.get('local_buffer_size', DEFAULT_LOCAL_BUFFER_SIZE)),
            enable_offload=enable_offload,
        )

    @classmethod
    def load_from_config(
        cls,
        config_path: str | os.PathLike[str] | None = None,
    ) -> MooncakeStoreConfig:
        """Load an explicit path, falling back to ``MOONCAKE_CONFIG_PATH``."""
        resolved_path = config_path or os.getenv(MOONCAKE_CONFIG_PATH_ENV)
        if not resolved_path:
            raise ValueError(
                "Mooncake config path is required: set kv_connector_extra_config['mooncake_config_path'] "
                f'or the {MOONCAKE_CONFIG_PATH_ENV} environment variable')
        return cls.from_file(resolved_path)


@dataclass(frozen=True)
class MooncakeStoreRegistration:
    """One contiguous GPU cache row registered with Mooncake Store."""

    name: str
    address: int
    size: int


@dataclass(frozen=True)
class MooncakeStoreLoadRequest:
    """One asynchronous load from Mooncake into allocated GPU blocks."""

    request_id: int
    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]
    remote_block_count: int = 0


@dataclass(frozen=True)
class MooncakeStoreSaveRequest:
    """One immutable full-block suffix saved after a model forward."""

    save_id: int
    request_id: int
    start_block: int
    block_ids: tuple[int, ...]
    logical_block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]


@dataclass(frozen=True)
class MooncakeStoreConnectorMetadata(KVConnectorMetadata):
    """Serializable Mooncake work issued by one scheduler step."""

    load_requests: tuple[MooncakeStoreLoadRequest, ...] = ()
    save_requests: tuple[MooncakeStoreSaveRequest, ...] = ()

    def get_save_block_leases(self) -> tuple[KVSaveBlockLease, ...]:
        return tuple(
            KVSaveBlockLease(
                operation_id=request.save_id,
                logical_block_ids=request.logical_block_ids,
            )
            for request in self.save_requests
        )
