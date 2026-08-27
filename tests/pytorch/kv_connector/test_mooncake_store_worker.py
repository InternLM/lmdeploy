# Copyright (c) OpenMMLab. All rights reserved.
import json
import sys

import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.kv_connector import KVConnectorOutput
from lmdeploy.pytorch.kv_connector.mooncake.store import worker as worker_module
from lmdeploy.pytorch.kv_connector.mooncake.store import worker_threads as worker_threads_module
from lmdeploy.pytorch.kv_connector.mooncake.store.data import (
    DEFAULT_GLOBAL_SEGMENT_SIZE,
    DEFAULT_LOCAL_BUFFER_SIZE,
    MooncakeStoreConfig,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreKeyMetadata,
    MooncakeStoreLoadRequest,
    MooncakeStoreSaveRequest,
    build_prefix_block_hashes,
    build_store_key,
)
from lmdeploy.pytorch.kv_connector.mooncake.store.worker import MooncakeStoreWorker


class FakeStore:

    def __init__(
        self,
        *,
        setup_ret=0,
        register_failure_at=None,
        register_error_at=None,
        close_error=None,
        close_ret=0,
        lookup_results=None,
        lookup_error=None,
        get_results=None,
        get_error=None,
        put_results=None,
        put_error=None,
        put_callback=None,
    ):
        self.setup_ret = setup_ret
        self.register_failure_at = register_failure_at
        self.register_error_at = register_error_at
        self.close_error = close_error
        self.close_ret = close_ret
        self.lookup_results = lookup_results
        self.lookup_error = lookup_error
        self.get_results = get_results
        self.get_error = get_error
        self.put_results = put_results
        self.put_error = put_error
        self.put_callback = put_callback
        self.setup_calls = []
        self.register_calls = []
        self.lookup_calls = []
        self.get_calls = []
        self.put_calls = []
        self.close_calls = 0

    def setup(self, *args):
        self.setup_calls.append(args)
        return self.setup_ret

    def register_buffer(self, address, size):
        self.register_calls.append((address, size))
        if len(self.register_calls) == self.register_error_at:
            raise RuntimeError('register failed')
        if len(self.register_calls) == self.register_failure_at:
            return -1
        return 0

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error
        return self.close_ret

    def batch_is_exist(self, keys):
        self.lookup_calls.append(list(keys))
        if self.lookup_error is not None:
            raise self.lookup_error
        if self.lookup_results is None:
            return [0] * len(keys)
        return self.lookup_results

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        self.get_calls.append((list(keys), addresses, sizes))
        if self.get_error is not None:
            raise self.get_error
        if self.get_results is None:
            return [0] * len(keys)
        return self.get_results

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, replicate_config):
        self.put_calls.append((list(keys), addresses, sizes, replicate_config))
        if self.put_callback is not None:
            self.put_callback()
        if self.put_error is not None:
            raise self.put_error
        if self.put_results is None:
            return [0] * len(keys)
        return self.put_results


class FakeTensor:

    def __init__(
        self,
        address,
        size=4096,
        *,
        is_cuda=True,
        contiguous=True,
        dtype='torch.uint8',
        device='cuda:0',
        storage_address=None,
        storage_size=None,
    ):
        self._address = address
        self._size = size
        self.is_cuda = is_cuda
        self._contiguous = contiguous
        self.dtype = dtype
        self.device = device
        self.shape = (size, )
        self._storage_address = address if storage_address is None else storage_address
        self._storage_size = size if storage_size is None else storage_size

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return self._size

    def element_size(self):
        return 1

    def data_ptr(self):
        return self._address

    def stride(self):
        return (1, )

    def untyped_storage(self):
        return FakeStorage(self._storage_address, self._storage_size)


class FakeStorage:

    def __init__(self, address, size):
        self._address = address
        self._size = size

    def data_ptr(self):
        return self._address

    def nbytes(self):
        return self._size


class RecordingLogger:

    def __init__(self):
        self.messages = []

    def _record(self, level, message, *args, **kwargs):
        del kwargs
        self.messages.append((level, message % args))

    def debug(self, message, *args, **kwargs):
        self._record('debug', message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._record('error', message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._record('warning', message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self._record('exception', message, *args, **kwargs)


class FakeCudaEvent:

    def __init__(self):
        self.record_calls = 0
        self.synchronize_calls = 0

    def record(self):
        self.record_calls += 1

    def synchronize(self):
        self.synchronize_calls += 1


@pytest.fixture(autouse=True)
def patch_worker_runtime(monkeypatch):
    original_is_tensor = torch.is_tensor
    recording_logger = RecordingLogger()
    monkeypatch.setattr(worker_module, '_get_local_hostname', lambda: '10.0.0.8')
    monkeypatch.setattr(
        worker_module.torch,
        'is_tensor',
        lambda value: isinstance(value, FakeTensor) or original_is_tensor(value),
    )
    monkeypatch.setattr(worker_module, 'logger', recording_logger)
    monkeypatch.setattr(worker_threads_module, 'logger', recording_logger)
    return recording_logger


def write_store_config(tmp_path, name='mooncake.json', **overrides):
    config = {
        'metadata_server': 'P2PHANDSHAKE',
        'master_server_address': '127.0.0.1:50051',
        'protocol': 'rdma',
        'device_name': 'mlx5_0,mlx5_1',
        'mode': 'embedded',
        'global_segment_size': '8GB',
        'local_buffer_size': '512MB',
        'enable_offload': False,
    }
    config.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(config), encoding='utf-8')
    return path


def make_cache_config(
    config_path=None,
    *,
    block_size=64,
    num_gpu_blocks=1,
    role='kv_both',
):
    extra_config = {}
    if config_path is not None:
        extra_config['mooncake_config_path'] = str(config_path)
    return CacheConfig(
        max_batches=1,
        block_size=block_size,
        num_cpu_blocks=0,
        num_gpu_blocks=num_gpu_blocks,
        kv_transfer_config=KVTransferConfig(
            kv_connector='MooncakeStoreConnector',
            kv_role=role,
            kv_connector_extra_config=extra_config,
        ),
    )


def make_worker(tmp_path, store=None, *, role='kv_both', **worker_kwargs):
    path = write_store_config(tmp_path)
    store = store or FakeStore()
    worker = MooncakeStoreWorker(
        make_cache_config(path, role=role),
        store_factory=lambda: store,
        **worker_kwargs,
    )
    return worker, store


def test_store_config_parses_sizes_and_defaults(tmp_path):
    path = write_store_config(
        tmp_path,
        global_segment_size='1.5GB',
        local_buffer_size='256 kb',
    )
    config = MooncakeStoreConfig.from_file(path)

    assert config.global_segment_size == int(1.5 * 1024**3)
    assert config.local_buffer_size == 256 * 1024

    default_path = tmp_path / 'defaults.json'
    default_path.write_text(
        json.dumps({
            'metadata_server': 'P2PHANDSHAKE',
            'master_server_address': '127.0.0.1:50051',
        }),
        encoding='utf-8',
    )
    defaults = MooncakeStoreConfig.from_file(default_path)
    assert defaults.global_segment_size == DEFAULT_GLOBAL_SEGMENT_SIZE
    assert defaults.local_buffer_size == DEFAULT_LOCAL_BUFFER_SIZE
    assert defaults.mode == 'embedded'
    assert defaults.enable_offload is False


@pytest.mark.parametrize(
    ('overrides', 'match'),
    [
        ({'mode': 'standalone-store', 'global_segment_size': 0}, 'only embedded'),
        ({'enable_offload': True}, 'does not support SSD'),
        ({'enable_ssd_offload': True}, 'does not support SSD'),
        ({'ssd_offload_path': '/mnt/ssd'}, 'does not support SSD'),
        ({'global_segment_size': 0}, 'global_segment_size'),
        ({'local_buffer_size': 0}, 'local_buffer_size'),
        ({'enable_offload': 'false'}, 'must be a boolean'),
        ({'enable_ssd_offload': 'false'}, 'must be a boolean'),
        ({'ssd_offload_path': []}, 'must be a string'),
        ({'metadata_server': ''}, 'metadata_server'),
        ({'master_server_address': ''}, 'master_server_address'),
        ({'protocol': 'nvmeof'}, 'protocol'),
        ({'device_name': []}, 'device_name'),
    ],
)
def test_store_config_rejects_unsupported_or_invalid_values(tmp_path, overrides, match):
    path = write_store_config(tmp_path, **overrides)

    with pytest.raises((TypeError, ValueError), match=match):
        MooncakeStoreConfig.from_file(path)


@pytest.mark.parametrize(
    ('contents', 'error_type', 'match'),
    [
        ('not-json', json.JSONDecodeError, None),
        ('[]', TypeError, 'JSON object'),
        ('{"metadata_server":"P2PHANDSHAKE","master_server_address":"master","local_buffer_size":1.5}',
         TypeError, 'size type'),
        ('{"metadata_server":"P2PHANDSHAKE","master_server_address":"master",'
         '"global_segment_size":"12XB"}', ValueError, 'invalid Mooncake size'),
    ],
)
def test_store_config_rejects_malformed_json(contents, error_type, match, tmp_path):
    path = tmp_path / 'invalid.json'
    path.write_text(contents, encoding='utf-8')

    with pytest.raises(error_type, match=match):
        MooncakeStoreConfig.from_file(path)


def test_explicit_config_path_precedes_environment(tmp_path, monkeypatch):
    explicit_path = write_store_config(tmp_path, name='explicit.json', metadata_server='explicit')
    env_path = write_store_config(tmp_path, name='env.json', metadata_server='environment')
    monkeypatch.setenv('MOONCAKE_CONFIG_PATH', str(env_path))
    store = FakeStore()

    worker = MooncakeStoreWorker(make_cache_config(explicit_path), store_factory=lambda: store)

    assert worker.store_config.metadata_server == 'explicit'
    assert store.setup_calls[0][1] == 'explicit'
    worker.shutdown()


def test_environment_config_path_is_fallback(tmp_path, monkeypatch):
    env_path = write_store_config(tmp_path, metadata_server='environment')
    monkeypatch.setenv('MOONCAKE_CONFIG_PATH', str(env_path))
    store = FakeStore()

    worker = MooncakeStoreWorker(make_cache_config(), store_factory=lambda: store)

    assert worker.store_config.metadata_server == 'environment'
    worker.shutdown()


def test_missing_config_path_fails_before_store_creation(monkeypatch):
    monkeypatch.delenv('MOONCAKE_CONFIG_PATH', raising=False)
    created = False

    def create_store():
        nonlocal created
        created = True
        return FakeStore()

    with pytest.raises(ValueError, match='Mooncake config path is required'):
        MooncakeStoreWorker(make_cache_config(), store_factory=create_store)
    assert created is False


def test_mooncake_import_is_lazy_and_has_actionable_error(tmp_path, monkeypatch):
    path = write_store_config(tmp_path)
    monkeypatch.setitem(sys.modules, 'mooncake', None)

    with pytest.raises(ImportError, match='mooncake-transfer-engine'):
        MooncakeStoreWorker(make_cache_config(path))


def test_store_create_setup_and_close_are_logged_with_ranks(tmp_path, patch_worker_runtime):
    path = write_store_config(tmp_path)
    store = FakeStore()

    worker = MooncakeStoreWorker(
        make_cache_config(path),
        global_rank=9,
        tp_rank=3,
        tp_size=8,
        store_factory=lambda: store,
    )

    assert store.setup_calls == [(
        '10.0.0.8',
        'P2PHANDSHAKE',
        8 * 1024**3,
        512 * 1024**2,
        'rdma',
        'mlx5_0,mlx5_1',
        '127.0.0.1:50051',
    )]
    worker.shutdown()
    assert {level for level, _ in patch_worker_runtime.messages} == {'debug'}
    messages = [message for _, message in patch_worker_runtime.messages]
    for operation in ('create', 'setup', 'close'):
        assert any(f'interaction before: operation={operation}' in message for message in messages)
        assert any(f'interaction after: operation={operation}' in message for message in messages)
    assert all('global_rank=9 tp_rank=3 tp_size=8' in message for message in messages)


@pytest.mark.parametrize('setup_result', [-1, None])
def test_setup_failure_propagates_without_cleanup(tmp_path, setup_result):
    path = write_store_config(tmp_path)
    store = FakeStore(setup_ret=setup_result)

    with pytest.raises(RuntimeError, match='setup failed'):
        MooncakeStoreWorker(make_cache_config(path), store_factory=lambda: store)

    assert store.close_calls == 0


def test_registers_99_logical_glm_cache_rows(tmp_path, patch_worker_runtime):
    worker, store = make_worker(tmp_path, global_rank=7, tp_rank=2, tp_size=8)
    rows = {
        **{
            f'mla.layer.{index}': FakeTensor(
                0x100000 + index * 0x1000,
                storage_address=0x100000,
                storage_size=78 * 0x1000,
            )
            for index in range(78)
        },
        **{
            f'dsa.layer.{index}': FakeTensor(
                0x200000 + index * 0x1000,
                storage_address=0x200000,
                storage_size=21 * 0x1000,
            )
            for index in range(21)
        },
    }
    worker.register_kv_caches(rows)

    assert len(store.register_calls) == 99
    assert store.register_calls[0] == (0x100000, 4096)
    assert store.register_calls[-1] == (0x200000 + 20 * 0x1000, 4096)
    messages = [message for _, message in patch_worker_runtime.messages]
    before = [message for message in messages if 'interaction before: operation=register_buffer' in message]
    after = [message for message in messages if 'interaction after: operation=register_buffer' in message]
    assert len(before) == 99
    assert len(after) == 99
    assert 'global_rank=7 tp_rank=2 tp_size=8 index=99/99' in before[-1]
    assert any('registration complete: global_rank=7 tp_rank=2 tp_size=8 '
               'backing_storages=2 registered_regions=99' in message
               for message in messages)
    worker.shutdown()


def test_sequence_values_are_flattened_to_rows(tmp_path):
    worker, store = make_worker(tmp_path)
    rows = [FakeTensor(0x1000), FakeTensor(0x2000)]

    worker.register_kv_caches({'main': rows})

    assert store.register_calls == [(0x1000, 4096), (0x2000, 4096)]
    worker.shutdown()


def test_lookup_server_starts_after_registration_only_on_local_tp_rank_zero(tmp_path):
    rank_zero_worker, _ = make_worker(tmp_path, global_rank=2, tp_rank=0, tp_size=2)
    rank_one_worker, _ = make_worker(tmp_path, global_rank=3, tp_rank=1, tp_size=2)

    assert rank_zero_worker.lookup_server is None
    assert rank_one_worker.lookup_server is None

    rank_zero_worker.register_kv_caches({'row': FakeTensor(0x1000)})
    rank_one_worker.register_kv_caches({'row': FakeTensor(0x2000)})

    assert rank_zero_worker.lookup_server is not None
    assert rank_zero_worker.lookup_server.thread.is_alive()
    assert rank_one_worker.lookup_server is None

    rank_zero_worker.shutdown()
    rank_one_worker.shutdown()


def test_producer_worker_does_not_start_lookup_server(tmp_path):
    worker, _ = make_worker(tmp_path, role='kv_producer')

    worker.register_kv_caches({'row': FakeTensor(0x1000)})

    assert worker.lookup_server is None
    assert worker.kv_recv_thread is None
    assert worker.kv_send_thread is not None
    worker.shutdown()


@pytest.mark.parametrize(
    ('replica_num', 'unique_ranks'),
    [(1, 8), (4, 2), (8, 1)],
)
def test_lookup_expands_unique_kv_namespaces_block_major(
    tmp_path,
    replica_num,
    unique_ranks,
):
    store = FakeStore(lookup_results=[1] * (3 * unique_ranks))
    worker, _ = make_worker(
        tmp_path,
        store=store,
        tp_size=8,
        kv_head_replica_num=replica_num,
    )
    block_hashes = build_prefix_block_hashes(range(192), 64)

    assert worker.lookup(192, block_hashes) == 192
    assert store.lookup_calls == [[
        build_store_key(worker.key_metadata, rank, block_hashes[block_index])
        for block_index in range(3)
        for rank in range(unique_ranks)
    ]]
    worker.shutdown()


def test_lookup_requires_all_namespaces_and_a_contiguous_prefix(tmp_path):
    store = FakeStore(lookup_results=[1, 1, 1, 0, 1, 1])
    worker, _ = make_worker(
        tmp_path,
        store=store,
        tp_size=8,
        kv_head_replica_num=4,
    )

    assert worker.lookup(192, build_prefix_block_hashes(range(192), 64)) == 64
    worker.shutdown()


@pytest.mark.parametrize(
    'store',
    [
        FakeStore(lookup_results=[1]),
        FakeStore(lookup_error=RuntimeError('lookup failed')),
    ],
)
def test_lookup_external_errors_fail_closed(tmp_path, store):
    worker, _ = make_worker(
        tmp_path,
        store=store,
        tp_size=8,
        kv_head_replica_num=4,
    )

    assert worker.lookup(128, build_prefix_block_hashes(range(128), 64)) == 0
    worker.shutdown()


def test_store_key_uses_lmdeploy_namespace_without_unsupported_geometry():
    metadata = MooncakeStoreKeyMetadata(
        model_name='test-model',
        cache_prefix='tenant-a',
        tp_size=8,
        block_size=4,
        kv_head_replica_num=4,
    )
    block_hash = build_prefix_block_hashes(range(4), 4)[0]

    assert build_store_key(metadata, 1, block_hash) == (
        f'tenant-a@test-model@tp_rank:1@group:0@{block_hash.hex()}')


def test_lookup_server_start_failure_propagates_without_cleanup(tmp_path, monkeypatch):
    worker, store = make_worker(tmp_path)

    def fail_to_start(*args, **kwargs):
        raise RuntimeError('lookup bind failed')

    monkeypatch.setattr(worker_module, 'LookupKeyServer', fail_to_start)
    with pytest.raises(RuntimeError, match='lookup bind failed'):
        worker.register_kv_caches({'row': FakeTensor(0x1000)})

    assert store.register_calls == [(0x1000, 4096)]
    assert store.close_calls == 0
    assert worker.store is store
    assert worker.lookup_server is None


def test_empty_registration_mapping_fails_fast(tmp_path):
    worker, store = make_worker(tmp_path)

    with pytest.raises(ValueError, match='No KV cache rows'):
        worker.register_kv_caches({})

    assert store.register_calls == []
    worker.shutdown()


def test_register_failure_propagates_without_cleanup(tmp_path):
    store = FakeStore(register_failure_at=3)
    worker, _ = make_worker(tmp_path, store=store)
    rows = {f'row.{index}': FakeTensor(0x1000 + index * 0x1000) for index in range(5)}

    with pytest.raises(RuntimeError, match="'row.2'.*return code -1"):
        worker.register_kv_caches(rows)

    assert len(store.register_calls) == 3
    assert store.close_calls == 0
    assert worker.store is store


def test_register_exception_propagates_without_cleanup(tmp_path):
    store = FakeStore(register_error_at=2)
    worker, _ = make_worker(tmp_path, store=store)
    rows = {f'row.{index}': FakeTensor(0x1000 + index * 0x1000) for index in range(3)}

    with pytest.raises(RuntimeError, match='register failed'):
        worker.register_kv_caches(rows)

    assert len(store.register_calls) == 2
    assert store.close_calls == 0
    assert worker.store is store


def test_registration_rejects_non_cuda_noncontiguous_and_empty_rows(tmp_path):
    worker, store = make_worker(tmp_path)

    with pytest.raises(ValueError, match='CUDA tensor'):
        worker.register_kv_caches({'cpu': torch.empty(4)})
    with pytest.raises(ValueError, match='contiguous'):
        worker.register_kv_caches({'strided': FakeTensor(0x1000, contiguous=False)})

    assert store.register_calls == []
    assert worker.store is store
    worker.shutdown()


def test_async_load_writes_allocated_blocks_and_reports_partial_failure(tmp_path):
    path = write_store_config(tmp_path)
    store = FakeStore(get_results=[0, -1])
    worker = MooncakeStoreWorker(
        make_cache_config(path, num_gpu_blocks=4),
        global_rank=1,
        tp_rank=5,
        tp_size=8,
        kv_head_replica_num=4,
        store_factory=lambda: store,
    )
    worker.register_kv_caches({
        'row.0': FakeTensor(0x1000, size=400),
        'row.1': FakeTensor(0x2000, size=800),
    })
    block_hashes = build_prefix_block_hashes(range(128), 64)
    request = MooncakeStoreLoadRequest(
        request_id=19,
        block_ids=(3, 1),
        block_hashes=block_hashes,
    )

    worker.start_load_kv(MooncakeStoreConnectorMetadata(load_requests=(request, )))
    worker.kv_recv_thread.request_queue.join()

    assert worker.get_finished() == KVConnectorOutput(
        finished_receiving={19},
        invalid_block_ids={1},
    )
    assert worker.get_finished() == KVConnectorOutput()
    assert store.get_calls == [(
        [
            build_store_key(worker.key_metadata, 1, block_hash)
            for block_hash in block_hashes
        ],
        [
            [0x1000 + 3 * 100, 0x2000 + 3 * 200],
            [0x1000 + 1 * 100, 0x2000 + 1 * 200],
        ],
        [[100, 200], [100, 200]],
    )]
    worker.shutdown()


def test_async_save_waits_for_forward_and_writes_only_owned_missing_blocks(
    tmp_path,
    monkeypatch,
):
    path = write_store_config(tmp_path)
    event = FakeCudaEvent()
    replicate_config = object()

    def assert_forward_is_ready():
        assert event.synchronize_calls == 1

    store = FakeStore(
        lookup_results=[0, 1],
        put_results=[0],
        put_callback=assert_forward_is_ready,
    )
    worker = MooncakeStoreWorker(
        make_cache_config(path, num_gpu_blocks=4),
        global_rank=1,
        tp_rank=3,
        tp_size=4,
        kv_head_replica_num=2,
        store_factory=lambda: store,
        replicate_config=replicate_config,
    )
    worker.register_kv_caches({
        'row.0': FakeTensor(0x1000, size=400),
        'row.1': FakeTensor(0x2000, size=800),
    })
    monkeypatch.setattr(worker_module.torch.cuda, 'Event', lambda: event)
    block_hashes = build_prefix_block_hashes(range(320), 64)
    request = MooncakeStoreSaveRequest(
        save_id=23,
        request_id=19,
        start_block=1,
        block_ids=(3, 2, 1, 0),
        logical_block_ids=(13, 12, 11, 10),
        block_hashes=block_hashes[1:5],
    )

    worker.start_save_kv(MooncakeStoreConnectorMetadata(save_requests=(request, )))
    worker.kv_send_thread.request_queue.join()

    # tp_rank=3 owns KV-head shard 1 and replica phase 1, so absolute
    # blocks 1 and 3 are queried. The second key already exists.
    expected_keys = [
        build_store_key(worker.key_metadata, 1, block_hashes[index])
        for index in (1, 3)
    ]
    assert store.lookup_calls == [expected_keys]
    assert store.put_calls == [(
        [expected_keys[0]],
        [[0x1000 + 3 * 100, 0x2000 + 3 * 200]],
        [[100, 200]],
        replicate_config,
    )]
    assert event.record_calls == 1
    assert event.synchronize_calls == 1
    assert worker.get_finished() == KVConnectorOutput(completed_save_ids={23})
    assert worker.get_finished() == KVConnectorOutput()
    worker.shutdown()


@pytest.mark.parametrize('failure_stage', ['lookup', 'put', 'partial_put'])
def test_async_save_failure_is_a_terminal_completion(
    tmp_path,
    monkeypatch,
    failure_stage,
):
    path = write_store_config(tmp_path)
    event = FakeCudaEvent()
    store = FakeStore(
        lookup_results=[0],
        lookup_error=(RuntimeError('lookup failed')
                      if failure_stage == 'lookup' else None),
        put_results=([-1] if failure_stage == 'partial_put' else None),
        put_error=(RuntimeError('put failed')
                   if failure_stage == 'put' else None),
    )
    worker = MooncakeStoreWorker(
        make_cache_config(path, num_gpu_blocks=2),
        global_rank=1,
        store_factory=lambda: store,
        replicate_config=object(),
    )
    worker.register_kv_caches({'row': FakeTensor(0x1000, size=200)})
    monkeypatch.setattr(worker_module.torch.cuda, 'Event', lambda: event)
    request = MooncakeStoreSaveRequest(
        save_id=29,
        request_id=7,
        start_block=0,
        block_ids=(1, ),
        logical_block_ids=(9, ),
        block_hashes=build_prefix_block_hashes(range(64), 64),
    )

    worker.start_save_kv(MooncakeStoreConnectorMetadata(save_requests=(request, )))
    worker.kv_send_thread.request_queue.join()

    assert event.record_calls == 1
    assert event.synchronize_calls == 1
    assert len(store.put_calls) == (0 if failure_stage == 'lookup' else 1)
    assert worker.get_finished() == KVConnectorOutput(completed_save_ids={29})
    worker.shutdown()


def test_shutdown_is_idempotent_and_propagates_close_error(tmp_path):
    store = FakeStore(close_error=RuntimeError('close failed'))
    worker, _ = make_worker(tmp_path, store=store)

    with pytest.raises(RuntimeError, match='close failed'):
        worker.shutdown()
    worker.shutdown()

    assert store.close_calls == 1
    assert worker.store is None
