# Copyright (c) OpenMMLab. All rights reserved.
import builtins
import os
import threading
import time
import uuid
from collections import deque

import pytest
import zmq

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.kv_connector.mooncake.store.data import BlobBlockHashes
from lmdeploy.pytorch.kv_connector.mooncake.store.lookup import LookupKeyClient, LookupKeyServer
from lmdeploy.pytorch.kv_connector.mooncake.store.protocol import LOOKUP_MSG, RESP_ERR


def _make_cache_config(endpoint: str, timeout_ms: int = 5000) -> CacheConfig:
    return CacheConfig(
        max_batches=1,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=1,
        kv_transfer_config=KVTransferConfig(
            kv_connector='MooncakeStoreConnector',
            kv_role='kv_both',
            kv_connector_extra_config={
                'lookup_rpc_path': endpoint,
                'lookup_rpc_timeout_ms': timeout_ms,
            },
        ),
    )


class _FakeStoreWorker:

    def __init__(self, results=()):
        self.results = deque(results)
        self.lookup_calls = []
        self.lookup_started = threading.Event()
        self.lookup_gate = None

    def lookup(self, token_len, block_hashes):
        self.lookup_calls.append((token_len, [bytes(block_hash) for block_hash in block_hashes]))
        gate = self.lookup_gate
        self.lookup_started.set()
        if gate is not None:
            assert gate.wait(timeout=2), 'lookup test gate was not released'
        result = self.results.popleft()
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def lookup_pair_factory():
    pairs = []

    def create(store_worker):
        endpoint = f'ipc:///tmp/lmd-lookup-{uuid.uuid4().hex}.sock'
        cache_config = _make_cache_config(endpoint)
        server = LookupKeyServer(store_worker, cache_config)
        client = LookupKeyClient(cache_config)
        pairs.append((client, server))
        return client, server, endpoint

    yield create

    for client, server in reversed(pairs):
        client.close()
        server.close()


def _poll_lookup(client, req_id, token_len, block_hashes, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.lookup(
            req_id,
            token_len,
            block_hashes,
            non_block=True,
        )
        if result is not None:
            return result
        time.sleep(0.005)
    pytest.fail(f'lookup for request {req_id} did not finish within {timeout}s')


def test_blob_block_hashes_is_a_lazy_sequence_view():
    hashes = [bytes([index]) * 16 for index in range(5)]
    view = BlobBlockHashes(memoryview(b''.join(hashes)), 16)

    assert len(view) == 5
    assert [bytes(block_hash) for block_hash in view] == hashes
    assert bytes(view[-1]) == hashes[-1]
    assert [bytes(block_hash) for block_hash in view[1:3]] == hashes[1:3]
    with pytest.raises(IndexError):
        _ = view[5]


def test_blob_block_hashes_accepts_an_empty_payload():
    view = BlobBlockHashes(memoryview(b''), 0)

    assert len(view) == 0
    assert list(view) == []


@pytest.mark.parametrize(
    ('blob', 'hash_len'),
    [(memoryview(b'x'), 0), (memoryview(b'abc'), 2), (memoryview(b''), -1)],
)
def test_blob_block_hashes_rejects_an_invalid_layout(blob, hash_len):
    with pytest.raises(ValueError):
        BlobBlockHashes(blob, hash_len)


@pytest.mark.parametrize(('store_result', 'expected'), [(37, 37), (0, 0)])
def test_sync_lookup_hit_and_miss_over_real_zmq(lookup_pair_factory, store_result, expected):
    store_worker = _FakeStoreWorker([store_result])
    client, _, _ = lookup_pair_factory(store_worker)
    hashes = [b'a' * 16, b'b' * 16]

    assert client.lookup(1, 128, hashes, non_block=False) == expected
    assert store_worker.lookup_calls == [(128, hashes)]


def test_async_lookup_is_pending_and_submitted_only_once(lookup_pair_factory):
    gate = threading.Event()
    store_worker = _FakeStoreWorker([23])
    store_worker.lookup_gate = gate
    client, _, _ = lookup_pair_factory(store_worker)
    hashes = [b'a' * 16]

    try:
        assert client.lookup(2, 64, hashes) is None
        assert store_worker.lookup_started.wait(timeout=2)
        assert client.is_pending(2)
        assert client.lookup(2, 64, hashes) is None
        assert store_worker.lookup_calls == [(64, hashes)]

        gate.set()
        assert _poll_lookup(client, 2, 64, hashes) == 23
        assert not client.is_pending(2)
    finally:
        gate.set()


def test_discard_drops_an_inflight_result_before_reusing_request_id(lookup_pair_factory):
    first_gate = threading.Event()
    second_gate = threading.Event()
    store_worker = _FakeStoreWorker([7, 29])
    store_worker.lookup_gate = first_gate
    client, _, _ = lookup_pair_factory(store_worker)
    hashes = [b'c' * 16]

    try:
        assert client.lookup(3, 64, hashes, non_block=True) is None
        assert store_worker.lookup_started.wait(timeout=2)
        client.discard(3)
        client.discard(999)

        store_worker.lookup_gate = second_gate
        first_gate.set()
        assert client.lookup(3, 64, hashes, non_block=True) is None
        second_gate.set()
        assert _poll_lookup(client, 3, 64, hashes) == 29
        assert store_worker.lookup_calls == [(64, hashes), (64, hashes)]
    finally:
        first_gate.set()
        second_gate.set()


def test_lookup_exception_is_a_miss_and_server_stays_available(lookup_pair_factory):
    store_worker = _FakeStoreWorker([RuntimeError('lookup failed'), 11])
    client, _, _ = lookup_pair_factory(store_worker)

    assert client.lookup(4, 64, [b'd' * 16], non_block=False) == 0
    assert client.lookup(5, 64, [b'e' * 16], non_block=False) == 11
    assert len(store_worker.lookup_calls) == 2


def test_timeout_is_a_miss_and_next_lookup_reconnects():
    endpoint = f'ipc:///tmp/lmd-lookup-{uuid.uuid4().hex}.sock'
    cache_config = _make_cache_config(endpoint, timeout_ms=200)
    first_server = LookupKeyServer(_FakeStoreWorker(), cache_config)
    client = LookupKeyClient(cache_config)
    first_server.close()

    try:
        assert client.lookup(40, 64, [b't' * 16], non_block=False) == 0

        second_worker = _FakeStoreWorker([31])
        second_server = LookupKeyServer(second_worker, cache_config)
        try:
            assert client.lookup(41, 64, [b'u' * 16], non_block=False) == 31
        finally:
            second_server.close()
    finally:
        client.close()


def test_duplicate_endpoint_is_rejected_without_disrupting_owner():
    endpoint = f'ipc:///tmp/lmd-lookup-{uuid.uuid4().hex}.sock'
    cache_config = _make_cache_config(endpoint)
    first_worker = _FakeStoreWorker([27])
    first_server = LookupKeyServer(first_worker, cache_config)

    try:
        with pytest.raises(RuntimeError, match='failed to bind'):
            LookupKeyServer(_FakeStoreWorker(), cache_config)

        client = LookupKeyClient(cache_config)
        try:
            assert client.lookup(42, 64, [b'v' * 16], non_block=False) == 27
        finally:
            client.close()
    finally:
        first_server.close()


def test_unknown_tag_is_rejected_without_disrupting_lookup(lookup_pair_factory):
    store_worker = _FakeStoreWorker([13])
    client, _, endpoint = lookup_pair_factory(store_worker)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 2000)
    try:
        socket.connect(endpoint)
        socket.send_multipart([b'unknown'])
        assert socket.recv() == RESP_ERR
    finally:
        socket.close(linger=0)
        context.term()

    assert client.lookup(6, 64, [b'f' * 16], non_block=False) == 13


def test_malformed_lookup_fails_closed_and_server_stays_available(lookup_pair_factory):
    store_worker = _FakeStoreWorker([17])
    client, _, endpoint = lookup_pair_factory(store_worker)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 2000)

    try:
        socket.connect(endpoint)
        socket.send_multipart([LOOKUP_MSG, (64).to_bytes(4, 'big')])
        assert socket.recv() == (0).to_bytes(4, 'big')
    finally:
        socket.close(linger=0)
        context.term()

    assert client.lookup(7, 64, [b'g' * 16], non_block=False) == 17


def test_close_is_idempotent_stops_server_and_removes_ipc_file(lookup_pair_factory):
    store_worker = _FakeStoreWorker()
    client, server, endpoint = lookup_pair_factory(store_worker)
    ipc_path = endpoint.removeprefix('ipc://')

    client.close()
    client.close()
    server.close()
    server.close()

    assert not server.thread.is_alive()
    assert not os.path.exists(ipc_path)
    assert not os.path.exists(f'{ipc_path}.lock')
    assert client.futures == {}
    with pytest.raises(RuntimeError, match='closed'):
        client.lookup(9, 64, [])


def test_close_drains_an_inflight_lookup_without_leaking_state(lookup_pair_factory):
    gate = threading.Event()
    store_worker = _FakeStoreWorker([31])
    store_worker.lookup_gate = gate
    client, _, _ = lookup_pair_factory(store_worker)

    assert client.lookup(10, 64, [b'i' * 16], non_block=True) is None
    assert store_worker.lookup_started.wait(timeout=2)

    close_thread = threading.Thread(target=client.close)
    close_thread.start()
    gate.set()
    close_thread.join(timeout=2)

    assert not close_thread.is_alive()
    assert client.futures == {}
    assert client.context is None
    assert client.socket is None


def test_lookup_channel_does_not_import_mooncake(monkeypatch, lookup_pair_factory):
    original_import = builtins.__import__

    def reject_mooncake(name, *args, **kwargs):
        if name == 'mooncake' or name.startswith('mooncake.'):
            raise AssertionError('LookupKey must not import Mooncake')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', reject_mooncake)
    store_worker = _FakeStoreWorker([19])
    client, _, _ = lookup_pair_factory(store_worker)

    assert client.lookup(8, 64, [b'h' * 16], non_block=False) == 19
