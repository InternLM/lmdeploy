# Copyright (c) OpenMMLab. All rights reserved.
import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.kv_connector.mooncake.store.data import MooncakeStoreConfig
from lmdeploy.pytorch.kv_connector.mooncake.store.transfer_backend import (
    ASCEND_FABRIC_MEM_ENV,
    AscendTransferBackend,
    CudaTransferBackend,
)


class FakeStore:

    def __init__(self, allocation_results=()):
        self.setup_calls = []
        self.mount_calls = []
        self.close_calls = 0
        self.allocation_results = iter(allocation_results)

    def setup(self, **kwargs):
        self.setup_calls.append(kwargs)
        return 0

    def allocate_and_mount_segment(self, size, *, protocol):
        self.mount_calls.append((size, protocol))
        return next(self.allocation_results, {'ret': 0, 'allocated_size': size})

    def close(self):
        self.close_calls += 1
        return 0


def make_config(protocol='ascend', global_segment_size=8 * 1024**3):
    return MooncakeStoreConfig(
        metadata_server='P2PHANDSHAKE',
        master_server_address='127.0.0.1:50051',
        protocol=protocol,
        device_name='',
        global_segment_size=global_segment_size,
    )


@pytest.fixture
def ascend_env(monkeypatch):
    monkeypatch.setenv(ASCEND_FABRIC_MEM_ENV, '1')
    monkeypatch.setitem(sys.modules, 'torch_npu', ModuleType('torch_npu'))
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(current_device=lambda: 3, device=lambda _: nullcontext()),
                        raising=False)
    return monkeypatch


def test_ascend_backend_requires_fabric_mem_env(ascend_env):
    ascend_env.delenv(ASCEND_FABRIC_MEM_ENV)

    with pytest.raises(RuntimeError, match=ASCEND_FABRIC_MEM_ENV):
        AscendTransferBackend()


@pytest.mark.parametrize('protocol', ['rdma', 'tcp'])
def test_fabric_rejects_non_ascend_protocol(ascend_env, protocol):
    store = FakeStore()

    with pytest.raises(ValueError, match='protocol="ascend"'):
        AscendTransferBackend().setup_store(store, make_config(protocol=protocol), '10.0.0.8')

    assert store.setup_calls == []


def test_fabric_setup_forwards_ascend_protocol(ascend_env):
    store = FakeStore()
    config = make_config()

    assert AscendTransferBackend().setup_store(store, config, '10.0.0.8') == 0

    assert store.setup_calls == [{
        'local_hostname': '10.0.0.8',
        'metadata_server': 'P2PHANDSHAKE',
        'global_segment_size': 8 * 1024**3,
        'local_buffer_size': 0,
        'protocol': 'ascend',
        'rdma_devices': '',
        'master_server_addr': '127.0.0.1:50051',
    }]
    assert store.mount_calls == []


@pytest.mark.parametrize(('total_gib', 'segment_gib'), [(16, [8, 8]), (17, [8, 8, 1]), (24, [8, 8, 8])])
def test_fabric_preserves_capacity_using_multiple_segments(ascend_env, total_gib, segment_gib):
    store = FakeStore()
    config = make_config(global_segment_size=total_gib * 1024**3)

    assert AscendTransferBackend().setup_store(store, config, '10.0.0.8') == 0

    assert store.setup_calls[0]['global_segment_size'] == 0
    assert store.mount_calls == [(gib * 1024**3, 'ascend') for gib in segment_gib]
    assert store.close_calls == 0


def test_fabric_rejects_missing_split_api_before_setup(ascend_env):
    store = FakeStore()
    store.allocate_and_mount_segment = None
    config = make_config(global_segment_size=16 * 1024**3)

    with pytest.raises(RuntimeError, match='allocate_and_mount_segment'):
        AscendTransferBackend().setup_store(store, config, '10.0.0.8')

    assert store.setup_calls == []


@pytest.mark.parametrize('result', [{'ret': -1, 'allocated_size': 0}, {'ret': 0, 'allocated_size': 4 * 1024**3}])
def test_fabric_rolls_back_after_incomplete_mount(ascend_env, result):
    store = FakeStore(allocation_results=[{'ret': 0, 'allocated_size': 8 * 1024**3}, result])
    config = make_config(global_segment_size=24 * 1024**3)

    with pytest.raises(RuntimeError, match='Failed to mount'):
        AscendTransferBackend().setup_store(store, config, '10.0.0.8')

    assert len(store.mount_calls) == 2
    assert store.close_calls == 1


def test_fabric_rejects_non_granular_segment_size(ascend_env):
    with pytest.raises(ValueError, match='multiple of 1GB'):
        AscendTransferBackend().setup_store(FakeStore(), make_config(global_segment_size=1024**3 + 1), '10.0.0.8')


def test_cuda_backend_close_store_closes_the_store():
    store = FakeStore()

    assert CudaTransferBackend().close_store(store) == 0
    assert store.close_calls == 1


def test_fabric_closes_store_on_the_setup_device(ascend_env):
    backend = AscendTransferBackend()
    close_devices = []

    @contextmanager
    def device(index):
        close_devices.append(index)
        yield

    ascend_env.setattr(torch.npu, 'device', device)
    store = FakeStore()
    assert backend.setup_store(store, make_config(), '10.0.0.8') == 0
    ascend_env.setattr(torch.npu, 'current_device', lambda: 1)

    assert backend.close_store(store) == 0

    assert close_devices == [3]
    assert store.close_calls == 1
