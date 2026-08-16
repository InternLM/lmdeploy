# Copyright (c) OpenMMLab. All rights reserved.
import json
import pickle
from unittest.mock import MagicMock

import pytest

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.kv_connector import KVConnectorMetadata, KVConnectorRole
from lmdeploy.pytorch.kv_connector.mooncake.store import worker as worker_module
from lmdeploy.pytorch.kv_connector.mooncake.store.connector import MooncakeStoreConnector
from lmdeploy.pytorch.kv_connector.mooncake.store.data import MooncakeStoreConnectorMetadata
from lmdeploy.pytorch.kv_connector.mooncake.store.scheduler import MooncakeStoreScheduler
from lmdeploy.pytorch.kv_connector.mooncake.store.worker import MooncakeStoreWorker


class _FakeStore:

    def setup(self, *args):
        return 0

    def register_buffer(self, address, size):
        return 0

    def close(self):
        return None


@pytest.fixture
def cache_config(tmp_path, monkeypatch):
    config_path = tmp_path / 'mooncake.json'
    config_path.write_text(
        json.dumps({
            'metadata_server': 'P2PHANDSHAKE',
            'master_server_address': '127.0.0.1:50051',
            'protocol': 'tcp',
        }),
        encoding='utf-8',
    )
    monkeypatch.setattr(worker_module, '_get_local_hostname', lambda: '127.0.0.1')
    monkeypatch.setattr(worker_module, '_load_mooncake_store_factory', lambda: _FakeStore)
    return CacheConfig(
        max_batches=1,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=1,
        kv_transfer_config=KVTransferConfig(
            kv_connector='MooncakeStoreConnector',
            kv_role='kv_both',
            kv_connector_extra_config={'mooncake_config_path': str(config_path)},
        ),
    )


def test_connector_constructs_only_scheduler_delegate(cache_config):
    connector = MooncakeStoreConnector(KVConnectorRole.SCHEDULER, cache_config)

    assert isinstance(connector.connector_scheduler, MooncakeStoreScheduler)
    assert connector.connector_scheduler._cache_config is cache_config
    assert connector.connector_worker is None
    assert connector.kv_role == 'kv_both'


def test_connector_constructs_only_worker_delegate(cache_config):
    connector = MooncakeStoreConnector(
        KVConnectorRole.WORKER,
        cache_config,
        global_rank=7,
        tp_rank=3,
        tp_size=8,
    )

    assert connector.connector_scheduler is None
    assert isinstance(connector.connector_worker, MooncakeStoreWorker)
    assert connector.connector_worker._cache_config is cache_config
    assert connector.connector_worker.global_rank == 7
    assert connector.connector_worker.tp_rank == 3
    assert connector.connector_worker.tp_size == 8
    assert connector.kv_role == 'kv_both'


def test_connector_requires_enabled_kv_transfer_config():
    cache_config = CacheConfig(
        max_batches=1,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=1,
    )

    with pytest.raises(ValueError, match='enabled kv_transfer_config'):
        MooncakeStoreConnector(KVConnectorRole.SCHEDULER, cache_config)


def test_connector_rejects_a_different_connector_configuration():
    cache_config = CacheConfig(
        max_batches=1,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=1,
        kv_transfer_config=KVTransferConfig(
            kv_connector='OtherConnector',
            kv_role='kv_both',
        ),
    )

    with pytest.raises(ValueError, match="kv_connector='OtherConnector'"):
        MooncakeStoreConnector(KVConnectorRole.SCHEDULER, cache_config)


def test_scheduler_empty_implementations_are_fail_closed(cache_config):
    connector = MooncakeStoreConnector(KVConnectorRole.SCHEDULER, cache_config)
    request = object()

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.update_state_after_alloc(request, [1, 2], 0) is None
    assert isinstance(connector.build_connector_meta(object()), MooncakeStoreConnectorMetadata)
    assert connector.on_new_request(request) is None
    assert connector.update_connector_output(object()) is None
    assert connector.request_finished(request, [1, 2]) == (False, None)
    assert connector.shutdown() is None


def test_worker_unimplemented_hooks_are_safe_noops(cache_config):
    connector = MooncakeStoreConnector(KVConnectorRole.WORKER, cache_config)
    metadata = MooncakeStoreConnectorMetadata()
    connector.bind_connector_metadata(metadata)

    assert connector.handle_preemptions(metadata) is None
    assert connector.get_finished({1}) == (None, None)
    assert connector.get_block_ids_with_load_errors() == set()
    assert connector.shutdown() is None


def test_scheduler_methods_delegate_arguments_and_results(cache_config):
    connector = MooncakeStoreConnector(KVConnectorRole.SCHEDULER, cache_config)
    scheduler = connector.connector_scheduler
    assert scheduler is not None

    request = object()
    scheduler_output = object()
    metadata = MooncakeStoreConnectorMetadata()
    scheduler.get_num_new_matched_tokens = MagicMock(return_value=(17, True))
    scheduler.update_state_after_alloc = MagicMock(return_value=None)
    scheduler.build_connector_meta = MagicMock(return_value=metadata)
    scheduler.on_new_request = MagicMock(return_value=None)
    scheduler.update_connector_output = MagicMock(return_value=None)
    scheduler.request_finished = MagicMock(return_value=(True, {'stored': True}))
    scheduler.shutdown = MagicMock(return_value=None)

    assert connector.get_num_new_matched_tokens(request, 3) == (17, True)
    scheduler.get_num_new_matched_tokens.assert_called_once_with(request, 3)

    assert connector.update_state_after_alloc(request, [4, 5], 14) is None
    scheduler.update_state_after_alloc.assert_called_once_with(request, [4, 5], 14)

    assert connector.build_connector_meta(scheduler_output) is metadata
    scheduler.build_connector_meta.assert_called_once_with(scheduler_output)

    assert connector.on_new_request(request) is None
    scheduler.on_new_request.assert_called_once_with(request)

    connector_output = object()
    assert connector.update_connector_output(connector_output) is None
    scheduler.update_connector_output.assert_called_once_with(connector_output)

    assert connector.request_finished(request, [4, 5]) == (True, {'stored': True})
    scheduler.request_finished.assert_called_once_with(request, [4, 5])

    assert connector.shutdown() is None
    scheduler.shutdown.assert_called_once_with()


def test_worker_methods_delegate_arguments_and_results(cache_config):
    connector = MooncakeStoreConnector(KVConnectorRole.WORKER, cache_config)
    worker = connector.connector_worker
    assert worker is not None

    kv_caches = {'layer.0': object()}
    metadata = MooncakeStoreConnectorMetadata()
    worker.register_kv_caches = MagicMock(return_value=None)
    worker.handle_preemptions = MagicMock(return_value=None)
    worker.get_finished = MagicMock(return_value=({1}, {2}))
    worker.get_block_ids_with_load_errors = MagicMock(return_value={7, 8})
    worker.shutdown = MagicMock(return_value=None)

    assert connector.register_kv_caches(kv_caches) is None
    worker.register_kv_caches.assert_called_once_with(kv_caches)

    assert connector.handle_preemptions(metadata) is None
    worker.handle_preemptions.assert_called_once_with(metadata)

    connector.bind_connector_metadata(metadata)
    assert connector.get_finished({1, 3}) == ({1}, {2})
    worker.get_finished.assert_called_once_with({1, 3}, metadata)

    assert connector.get_block_ids_with_load_errors() == {7, 8}
    worker.get_block_ids_with_load_errors.assert_called_once_with()

    assert connector.shutdown() is None
    worker.shutdown.assert_called_once_with()


def test_get_finished_requires_bound_metadata(cache_config):
    connector = MooncakeStoreConnector(KVConnectorRole.WORKER, cache_config)

    with pytest.raises(AssertionError):
        connector.get_finished(set())


def test_get_finished_rejects_other_connector_metadata(cache_config):

    class OtherConnectorMetadata(KVConnectorMetadata):
        pass

    connector = MooncakeStoreConnector(KVConnectorRole.WORKER, cache_config)
    connector.bind_connector_metadata(OtherConnectorMetadata())

    with pytest.raises(TypeError):
        connector.get_finished(set())

    with pytest.raises(TypeError):
        connector.handle_preemptions(OtherConnectorMetadata())


def test_metadata_is_fresh_and_pickle_serializable(cache_config):
    connector = MooncakeStoreConnector(KVConnectorRole.SCHEDULER, cache_config)

    first = connector.build_connector_meta(object())
    second = connector.build_connector_meta(object())
    restored = pickle.loads(pickle.dumps(first))

    assert isinstance(first, MooncakeStoreConnectorMetadata)
    assert isinstance(second, MooncakeStoreConnectorMetadata)
    assert first is not second
    assert isinstance(restored, MooncakeStoreConnectorMetadata)
    assert restored is not first


def test_wrong_side_method_calls_fail_fast(cache_config):
    scheduler_connector = MooncakeStoreConnector(KVConnectorRole.SCHEDULER, cache_config)
    worker_connector = MooncakeStoreConnector(KVConnectorRole.WORKER, cache_config)

    with pytest.raises(RuntimeError):
        scheduler_connector.register_kv_caches({})
    with pytest.raises(RuntimeError):
        scheduler_connector.handle_preemptions(MooncakeStoreConnectorMetadata())
    with pytest.raises(RuntimeError):
        worker_connector.get_num_new_matched_tokens(object(), 0)
