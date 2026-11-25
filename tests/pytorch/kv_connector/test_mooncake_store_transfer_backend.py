# Copyright (c) OpenMMLab. All rights reserved.
import gc
import sys
import weakref
from types import ModuleType

import pytest

from lmdeploy.pytorch.kv_connector.mooncake.store.data import MooncakeStoreConfig
from lmdeploy.pytorch.kv_connector.mooncake.store.transfer_backend import (
    ASCEND_FABRIC_MEM_ENV,
    AscendTransferBackend,
    CudaTransferBackend,
)


class FakeStore:

    def __init__(self):
        self.setup_calls = []
        self.close_calls = 0

    def setup(self, **kwargs):
        self.setup_calls.append(kwargs)
        return 0

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
    # AscendTransferBackend.__init__ only needs torch_npu to be importable.
    monkeypatch.setitem(sys.modules, 'torch_npu', ModuleType('torch_npu'))
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


def test_fabric_rejects_non_granular_segment_size(ascend_env):
    with pytest.raises(ValueError, match='multiple of 1GB'):
        AscendTransferBackend().setup_store(FakeStore(), make_config(global_segment_size=1024**3 + 1), '10.0.0.8')


def test_cuda_backend_close_store_closes_the_store():
    store = FakeStore()

    assert CudaTransferBackend().close_store(store) == 0
    assert store.close_calls == 1


def test_ascend_backend_close_store_parks_the_store(ascend_env):
    backend = AscendTransferBackend()
    store = FakeStore()
    store_ref = weakref.ref(store)

    assert backend.close_store(store) is None
    del store
    gc.collect()

    assert store_ref() is not None, 'store must survive garbage collection as long as the backend lives'
    assert store_ref().close_calls == 0
