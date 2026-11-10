# Copyright (c) OpenMMLab. All rights reserved.
"""Integer-only CPU R3 sidecars sharing the KV/state weight namespace."""

import ctypes
import struct

import numpy as np

from .data import build_store_key

_HEADER = struct.Struct('!8sIII')
_MAGIC = b'LMDR3v1\x00'


def encode_routes(routes: np.ndarray) -> bytes:
    """HistoryRouterExperts stores uint16 [token, layer, topk] IDs."""
    if (routes.dtype != np.uint16 or routes.ndim != 3 or not 0 < routes.shape[0] <= 2**20
            or not 0 < routes.shape[1] <= 1024 or not 0 < routes.shape[2] <= 128):
        raise ValueError('R3 sidecar requires nonempty uint16 [token, layer, topk] history')
    return _HEADER.pack(_MAGIC, *routes.shape) + routes.astype('<u2', copy=False).tobytes()


def decode_routes(payload: bytes | memoryview) -> np.ndarray:
    """Validate the whole wire record before interpreting its dimensions."""
    if len(payload) < _HEADER.size:
        raise ValueError('truncated R3 header')
    magic, rows, layers, topk = _HEADER.unpack_from(payload)
    if (magic != _MAGIC or not 0 < rows <= 2**20 or not 0 < layers <= 1024
            or not 0 < topk <= 128 or len(payload) != _HEADER.size + rows * layers * topk * 2):
        raise ValueError('invalid R3 schema, dimensions or byte count')
    return np.frombuffer(payload, dtype='<u2', offset=_HEADER.size).reshape(rows, layers, topk).copy()


def route_key(metadata, block_hash: bytes) -> str:
    # Routes collected by the engine are complete, not TP-sharded expert IDs.
    return build_store_key(metadata, 0, block_hash, group_id=2) + '@r3-u16-v1'


def put_route_blocks(store, metadata, block_hashes, routes, replicate_config=None):
    """Batch registered CPU payloads; never register or copy GPU routes
    again."""
    block_size = metadata.block_size
    if len(routes) != len(block_hashes) * block_size or not block_hashes:
        raise ValueError('R3 save rows must cover exactly its complete block hashes')
    parts = [encode_routes(routes[i * block_size:(i + 1) * block_size]) for i in range(len(block_hashes))]
    blob = b''.join(parts)
    buffer = ctypes.create_string_buffer(blob, len(blob))
    address = ctypes.addressof(buffer)
    if store.register_buffer(address, len(blob)) != 0:
        raise RuntimeError('R3 CPU buffer registration failed')
    try:
        size = len(parts[0])
        keys = [route_key(metadata, digest) for digest in block_hashes]
        pointers = [address + i * size for i in range(len(keys))]
        args = (keys, pointers, [size] * len(keys))
        result = (store.batch_put_from(*args, replicate_config)
                  if replicate_config is not None else store.batch_put_from(*args))
        if len(result) != len(keys) or any(code != 0 for code in result):
            raise RuntimeError(f'R3 sidecar batch put failed: {result}')
    finally:
        if store.unregister_buffer(address) != 0:
            raise RuntimeError('R3 CPU buffer unregister failed')
    return len(blob)


def get_route_blocks(store, metadata, block_hashes):
    """A racing eviction or malformed block fails closed, never partial R3."""
    keys = [route_key(metadata, digest) for digest in block_hashes]
    if not keys:
        raise ValueError('R3 load requires complete blocks')
    first = store.get(keys[0])
    sample = decode_routes(first)
    if len(sample) != metadata.block_size:
        raise ValueError('R3 sidecar row count does not match KV block size')
    size = len(first)
    buffer = ctypes.create_string_buffer(size * len(keys))
    address = ctypes.addressof(buffer)
    if store.register_buffer(address, len(buffer)) != 0:
        raise RuntimeError('R3 CPU receive buffer registration failed')
    try:
        result = store.batch_get_into(keys, [address + i * size for i in range(len(keys))], [size] * len(keys))
        if len(result) != len(keys) or any(code != size for code in result):
            raise RuntimeError(f'R3 sidecar batch get failed: {result}')
        view = memoryview(buffer).cast('B')
        parts = [decode_routes(view[i * size:(i + 1) * size]) for i in range(len(keys))]
        if any(part.shape != sample.shape for part in parts):
            raise ValueError('R3 sidecar block shapes disagree')
        return np.concatenate(parts, axis=0)
    finally:
        if store.unregister_buffer(address) != 0:
            raise RuntimeError('R3 CPU receive buffer unregister failed')
