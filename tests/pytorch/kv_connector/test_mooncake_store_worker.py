# Copyright (c) OpenMMLab. All rights reserved.
import json
import pickle
import sys
import threading
from collections import deque

import numpy as np
import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.kv_connector.base import KVConnectorOutput
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
    ):
        self.setup_ret = setup_ret
        self.register_failure_at = register_failure_at
        self.register_error_at = register_error_at
        self.close_error = close_error
        self.close_ret = close_ret
        self.setup_calls = []
        self.register_calls = []
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

    def info(self, message, *args, **kwargs):
        self._record('info', message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._record('error', message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._record('warning', message, *args, **kwargs)

    def exception(self, message, *args, **kwargs):
        self._record('error', message, *args, **kwargs)


class AsyncFakeStore(FakeStore):

    def __init__(self, exists_results=(), put_results=(), get_results=()):
        super().__init__()
        self.exists_results = deque(exists_results)
        self.put_results = deque(put_results)
        self.get_results = deque(get_results)
        self.exists_calls = []
        self.put_calls = []
        self.get_calls = []
        self.operations = []
        self.exists_started = threading.Event()
        self.get_started = threading.Event()
        self.exists_gate = None
        self.get_gate = None
        self.close_event = threading.Event()

    def batch_is_exist(self, keys):
        self.operations.append('query')
        self.exists_calls.append(list(keys))
        self.exists_started.set()
        if self.exists_gate is not None:
            assert self.exists_gate.wait(timeout=2), 'batch_is_exist gate was not released'
        result = self.exists_results.popleft()
        if isinstance(result, BaseException):
            raise result
        return result

    def batch_put_from_multi_buffers(self, keys, addresses, sizes, replicate_config):
        self.operations.append('put')
        self.put_calls.append((list(keys), addresses, sizes, replicate_config))
        result = self.put_results.popleft()
        if isinstance(result, BaseException):
            raise result
        return result

    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        self.operations.append('get')
        self.get_calls.append((list(keys), addresses, sizes))
        self.get_started.set()
        if self.get_gate is not None:
            assert self.get_gate.wait(timeout=2), 'batch_get gate was not released'
        result = self.get_results.popleft()
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self):
        result = super().close()
        self.close_event.set()
        return result


class FakeReadyEvent:

    def __init__(self, operations):
        self.operations = operations
        self.calls = 0

    def synchronize(self):
        self.operations.append('event')
        self.calls += 1


@pytest.fixture(autouse=True)
def patch_worker_runtime(monkeypatch):
    original_is_tensor = worker_module._is_tensor
    recording_logger = RecordingLogger()
    monkeypatch.setattr(worker_module, '_get_local_hostname', lambda: '10.0.0.8')
    monkeypatch.setattr(
        worker_module,
        '_is_tensor',
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
    extra_config=None,
    kv_role='kv_both',
    window_size=-1,
    states_shapes=None,
):
    extra_config = dict(extra_config or {})
    if config_path is not None:
        extra_config['mooncake_config_path'] = str(config_path)
    return CacheConfig(
        max_batches=1,
        block_size=block_size,
        num_cpu_blocks=0,
        num_gpu_blocks=num_gpu_blocks,
        window_size=window_size,
        states_shapes=states_shapes or [],
        kv_transfer_config=KVTransferConfig(
            kv_connector='MooncakeStoreConnector',
            kv_role=kv_role,
            kv_connector_extra_config=extra_config,
        ),
    )


def make_worker(tmp_path, store=None, **worker_kwargs):
    path = write_store_config(tmp_path)
    store = store or FakeStore()
    worker = MooncakeStoreWorker(
        make_cache_config(path),
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


@pytest.mark.parametrize(
    ('cache_overrides', 'match'),
    [
        ({'window_size': 128}, 'sliding-window'),
        ({'states_shapes': [(4, 8)]}, 'linear-attention'),
    ],
)
def test_unsupported_cache_families_fail_before_store_creation(
    tmp_path,
    cache_overrides,
    match,
):
    path = write_store_config(tmp_path)
    created = False

    def create_store():
        nonlocal created
        created = True
        return FakeStore()

    with pytest.raises(ValueError, match=match):
        MooncakeStoreWorker(
            make_cache_config(path, **cache_overrides),
            store_factory=create_store,
        )
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
    messages = [message for _, message in patch_worker_runtime.messages]
    for operation in ('create', 'setup', 'close'):
        assert any(f'interaction before: operation={operation}' in message for message in messages)
        assert any(f'interaction after: operation={operation}' in message for message in messages)
    assert all('global_rank=9 tp_rank=3 tp_size=8' in message for message in messages)


def test_store_setup_accepts_explicit_local_hostname(tmp_path):
    path = write_store_config(tmp_path)
    store = FakeStore()
    cache_config = make_cache_config(
        path,
        extra_config={'local_hostname': '10.10.0.7'},
    )

    worker = MooncakeStoreWorker(
        cache_config,
        store_factory=lambda: store,
    )

    assert store.setup_calls[0][0] == '10.10.0.7'
    worker.shutdown()


@pytest.mark.parametrize('setup_result', [-1, None])
def test_setup_failure_closes_partial_store(tmp_path, setup_result):
    path = write_store_config(tmp_path)
    store = FakeStore(setup_ret=setup_result)

    with pytest.raises(RuntimeError, match='setup failed'):
        MooncakeStoreWorker(make_cache_config(path), store_factory=lambda: store)

    assert store.close_calls == 1


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
    assert [(region.name, region.address, region.size) for region in worker._registered_regions] == [
        ('main[0]', 0x1000, 4096),
        ('main[1]', 0x2000, 4096),
    ]
    worker.shutdown()


def test_lookup_server_starts_after_registration_only_on_global_rank_zero(tmp_path):
    rank_zero_worker, _ = make_worker(tmp_path, global_rank=0, tp_rank=0, tp_size=2)
    rank_one_worker, _ = make_worker(tmp_path, global_rank=1, tp_rank=1, tp_size=2)

    assert rank_zero_worker.lookup_server is None
    assert rank_one_worker.lookup_server is None

    rank_zero_worker.register_kv_caches({'row': FakeTensor(0x1000)})
    rank_one_worker.register_kv_caches({'row': FakeTensor(0x2000)})

    assert rank_zero_worker.lookup_server is not None
    assert rank_zero_worker.lookup_server.thread.is_alive()
    assert rank_one_worker.lookup_server is None

    rank_zero_worker.shutdown()
    rank_one_worker.shutdown()


def test_lookup_server_start_failure_rolls_back_registration(tmp_path, monkeypatch):
    worker, store = make_worker(tmp_path)

    def fail_to_start(*args, **kwargs):
        raise RuntimeError('lookup bind failed')

    monkeypatch.setattr(worker_module, 'LookupKeyServer', fail_to_start)
    with pytest.raises(RuntimeError, match='lookup bind failed'):
        worker.register_kv_caches({'row': FakeTensor(0x1000)})

    assert store.register_calls == [(0x1000, 4096)]
    assert store.close_calls == 1
    assert worker.store is None
    assert worker.lookup_server is None
    assert worker._registered_regions is None


def test_equivalent_registration_is_noop_but_different_mapping_fails(tmp_path):
    worker, store = make_worker(tmp_path)
    rows = {
        'first': FakeTensor(0x1000),
        'second': FakeTensor(0x2000),
    }
    worker.register_kv_caches(rows)

    worker.register_kv_caches(rows)
    assert len(store.register_calls) == 2

    worker.register_kv_caches(dict(reversed(list(rows.items()))))
    assert len(store.register_calls) == 2
    assert [region.name for region in worker._registered_regions] == ['first', 'second']

    with pytest.raises(RuntimeError, match='different mapping'):
        worker.register_kv_caches({'first': FakeTensor(0x3000)})
    assert len(store.register_calls) == 2
    worker.shutdown()
    assert worker._registered_regions is None


def test_empty_registration_mapping_fails_fast(tmp_path):
    worker, store = make_worker(tmp_path)

    with pytest.raises(ValueError, match='No KV cache rows'):
        worker.register_kv_caches({})

    assert store.register_calls == []
    worker.shutdown()


def test_register_failure_stops_and_closes_store(tmp_path):
    store = FakeStore(register_failure_at=3)
    worker, _ = make_worker(tmp_path, store=store)
    rows = {f'row.{index}': FakeTensor(0x1000 + index * 0x1000) for index in range(5)}

    with pytest.raises(RuntimeError, match="'row.2'.*return code -1"):
        worker.register_kv_caches(rows)

    assert len(store.register_calls) == 3
    assert store.close_calls == 1
    assert worker.store is None
    assert worker._registered_regions is None


def test_register_exception_stops_and_closes_store(tmp_path):
    store = FakeStore(register_error_at=2)
    worker, _ = make_worker(tmp_path, store=store)
    rows = {f'row.{index}': FakeTensor(0x1000 + index * 0x1000) for index in range(3)}

    with pytest.raises(RuntimeError, match="raised for 'row.1'"):
        worker.register_kv_caches(rows)

    assert len(store.register_calls) == 2
    assert store.close_calls == 1
    assert worker.store is None
    assert worker._registered_regions is None


def test_registration_rejects_non_cuda_noncontiguous_and_empty_rows(tmp_path):
    worker, store = make_worker(tmp_path)

    with pytest.raises(ValueError, match='CUDA tensor'):
        worker.register_kv_caches({'cpu': torch.empty(4)})
    with pytest.raises(ValueError, match='contiguous'):
        worker.register_kv_caches({'strided': FakeTensor(0x1000, contiguous=False)})
    with pytest.raises(ValueError, match='size must be greater than 0'):
        worker.register_kv_caches({'empty': FakeTensor(0x1000, size=0)})

    assert store.register_calls == []
    assert worker.store is store
    worker.shutdown()


def test_shutdown_is_idempotent_and_swallows_close_error(tmp_path, patch_worker_runtime):
    store = FakeStore(close_error=RuntimeError('close failed'))
    worker, _ = make_worker(tmp_path, store=store)

    worker.shutdown()
    worker.shutdown()

    assert store.close_calls == 1
    assert worker.store is None
    assert any('operation=close' in message and 'status=error' in message
               for _, message in patch_worker_runtime.messages)


def _make_async_worker(
    tmp_path,
    store,
    *,
    num_gpu_blocks=4,
    tp_rank=1,
    tp_size=2,
    kv_head_replica_num=1,
    kv_role='kv_both',
    rows=None,
):
    path = write_store_config(tmp_path)
    replicate_config = object()
    worker = MooncakeStoreWorker(
        make_cache_config(
            path,
            num_gpu_blocks=num_gpu_blocks,
            extra_config={
                'model_name': 'test-model',
                'cache_prefix': 'tenant/a',
            },
            kv_role=kv_role,
        ),
        global_rank=1,
        tp_rank=tp_rank,
        tp_size=tp_size,
        kv_head_replica_num=kv_head_replica_num,
        store_factory=lambda: store,
        replicate_config=replicate_config,
    )
    worker.register_kv_caches(rows or {
        'row.0': FakeTensor(0x1000, size=400),
        'row.1': FakeTensor(0x2000, size=800),
    })
    return worker, replicate_config


def _save_request(save_id, *, block_ids=(3, 1, 2), token_len=192):
    block_hashes = build_prefix_block_hashes(np.arange(token_len, dtype=np.int64), 64)
    return MooncakeStoreSaveRequest(
        req_id=9,
        save_id=save_id,
        generation=2,
        token_len=token_len,
        block_ids=block_ids,
        block_hashes=block_hashes,
    )


def _load_request(
    load_id,
    *,
    block_ids=(1, 2),
    local_token_len=64,
    remote_token_len=192,
):
    block_hashes = build_prefix_block_hashes(
        np.arange(remote_token_len, dtype=np.int64),
        64,
    )
    local_blocks = local_token_len // 64
    remote_blocks = remote_token_len // 64
    return MooncakeStoreLoadRequest(
        req_id=19,
        load_id=load_id,
        generation=3,
        local_token_len=local_token_len,
        remote_token_len=remote_token_len,
        block_ids=block_ids,
        block_hashes=block_hashes[local_blocks:remote_blocks],
    )


def test_submit_transfers_is_separate_from_sticky_completion_poll(tmp_path):
    store = AsyncFakeStore(exists_results=([1], ))
    worker, _ = _make_async_worker(tmp_path, store)
    request = _save_request(30, block_ids=(0, ), token_len=64)
    metadata = MooncakeStoreConnectorMetadata(save_requests=(request, ))
    save_ready_event = FakeReadyEvent(store.operations)
    original_poll_finished = worker.poll_finished

    assert not worker.has_pending_step_transfers(MooncakeStoreConnectorMetadata())
    assert worker.has_pending_step_transfers(metadata)

    def fail_poll(*args, **kwargs):
        raise AssertionError('submit_transfers must not poll completions')

    try:
        worker.poll_finished = fail_poll
        assert worker.submit_transfers(
            metadata,
            save_ready_event=save_ready_event,
        ) is None
        worker.kv_send_thread.request_queue.join()

        worker.poll_finished = original_poll_finished
        assert worker.poll_finished() == KVConnectorOutput(completed_save_ids={30})
    finally:
        worker.poll_finished = original_poll_finished
        worker.shutdown()


def test_prefix_hashes_are_stable_chained_and_accept_numpy_tokens():
    tokens = np.arange(10, dtype=np.int64)

    hashes = build_prefix_block_hashes(tokens, 4, extra_identity='adapter:7')
    same_hashes = build_prefix_block_hashes(tokens.tolist(), 4, extra_identity=b'adapter:7')
    object_hashes = build_prefix_block_hashes(tokens.astype(object), 4, extra_identity='adapter:7')
    first_hash = build_prefix_block_hashes(tokens[:4], 4, extra_identity='adapter:7')
    incremental_hashes = build_prefix_block_hashes(
        tokens,
        4,
        extra_identity='adapter:7',
        previous_hashes=first_hash,
    )
    changed_first = tokens.copy()
    changed_first[0] = 99
    changed_second = tokens.copy()
    changed_second[4] = 99

    assert hashes == same_hashes
    assert hashes == object_hashes
    assert hashes == incremental_hashes
    assert [block_hash.hex() for block_hash in hashes] == [
        '297a562fafde70395287c3a5f82fb0b765ce17447d0f612d1a137f790633363b',
        '09c45d88b9833036ada7bcbf1bbac37314da48ceeb153fd6108af67b642a225d',
    ]
    assert len(hashes) == 2
    assert all(len(block_hash) == 32 for block_hash in hashes)
    assert all(left != right for left, right in zip(
        hashes,
        build_prefix_block_hashes(changed_first, 4, extra_identity='adapter:7'),
        strict=True,
    ))
    second_hashes = build_prefix_block_hashes(changed_second, 4, extra_identity='adapter:7')
    assert hashes[0] == second_hashes[0]
    assert hashes[1] != second_hashes[1]
    assert hashes != build_prefix_block_hashes(tokens, 4, extra_identity='adapter:8')


def test_prefix_hashes_preserve_token_validation_and_ignore_partial_tail():
    assert build_prefix_block_hashes([1.5], 4) == ()
    first_hash = build_prefix_block_hashes(np.arange(4, dtype=np.int64), 4)
    assert build_prefix_block_hashes(
        [0, 1, 2, 3, 4.5],
        4,
        previous_hashes=first_hash,
    ) == first_hash
    completed_hashes = build_prefix_block_hashes(
        np.arange(8, dtype=np.int64),
        4,
        previous_hashes=first_hash,
    )
    assert completed_hashes == build_prefix_block_hashes(
        np.arange(8, dtype=np.int64),
        4,
    )
    with pytest.raises(TypeError, match='must contain integers'):
        build_prefix_block_hashes([0, 1, 2, 3.5], 4)
    with pytest.raises(TypeError, match='must contain integers'):
        build_prefix_block_hashes([0, 1, 2, True], 4)
    with pytest.raises(TypeError, match='must contain integers'):
        build_prefix_block_hashes(np.arange(4, dtype=np.float64), 4)
    with pytest.raises(TypeError, match='must contain integers'):
        build_prefix_block_hashes(np.ones(4, dtype=np.bool_), 4)
    with pytest.raises(ValueError, match='unsigned 64-bit'):
        build_prefix_block_hashes([0, 1, 2, -1], 4)
    with pytest.raises(ValueError, match='unsigned 64-bit'):
        build_prefix_block_hashes(np.array([0, 1, 2, -1], dtype=np.int64), 4)
    with pytest.raises(ValueError, match='previous_hashes exceed'):
        build_prefix_block_hashes(
            np.arange(4, dtype=np.int64),
            4,
            previous_hashes=(b'x' * 32, ) * 2,
        )


def test_store_key_matches_vllm_format_for_unique_kv_head_shard():
    metadata = MooncakeStoreKeyMetadata(
        model_name='model/name',
        cache_prefix='tenant one',
        tp_size=8,
        block_size=64,
        kv_head_replica_num=4,
    )
    block_hash = b'\xab' * 32

    rank_zero = build_store_key(metadata, 0, block_hash)
    rank_one = build_store_key(metadata, 1, block_hash)

    assert metadata.num_kv_head_shards == 2
    assert rank_zero == (
        f'tenant one@model/name@tp_rank:0@group:0@{block_hash.hex()}')
    assert rank_one == (
        f'tenant one@model/name@tp_rank:1@group:0@{block_hash.hex()}')
    assert 'tp_size' not in rank_zero
    assert 'kv_head_replica_num' not in rank_zero
    assert rank_zero != rank_one
    with pytest.raises(ValueError, match='kv_head_rank'):
        build_store_key(metadata, 2, block_hash)
    with pytest.raises(ValueError, match='32 bytes'):
        build_store_key(metadata, 0, b'short')


def test_store_key_metadata_rejects_non_divisible_kv_head_replica_num():
    with pytest.raises(ValueError, match='must be divisible'):
        MooncakeStoreKeyMetadata(
            model_name='model',
            cache_prefix='',
            tp_size=8,
            block_size=64,
            kv_head_replica_num=3,
        )


def test_save_metadata_is_pickle_serializable():
    request = _save_request(31, block_ids=(0, 1, 2))
    metadata = MooncakeStoreConnectorMetadata(
        save_requests=(request, ),
        preempted_save_ids=(30, ),
    )

    assert pickle.loads(pickle.dumps(metadata)) == metadata


def test_load_metadata_is_suffix_only_and_pickle_serializable():
    request = _load_request(
        32,
        block_ids=(3, 0),
        local_token_len=64,
        remote_token_len=192,
    )
    metadata = MooncakeStoreConnectorMetadata(load_requests=(request, ))

    restored = pickle.loads(pickle.dumps(metadata))
    assert restored == metadata
    assert len(restored.load_requests[0].block_ids) == 2
    assert len(restored.load_requests[0].block_hashes) == 2

    with pytest.raises(ValueError, match='same length'):
        MooncakeStoreLoadRequest(
            req_id=1,
            load_id=1,
            generation=0,
            local_token_len=64,
            remote_token_len=192,
            block_ids=(0, 1),
            block_hashes=(b'x' * 32, ),
        )


def test_receiver_enqueue_is_async_and_loads_suffix_into_allocated_blocks(
    tmp_path,
    patch_worker_runtime,
):
    store = AsyncFakeStore(get_results=([0], ))
    store.get_gate = threading.Event()
    worker, _ = _make_async_worker(tmp_path, store)
    request = _load_request(
        60,
        block_ids=(3, ),
        local_token_len=64,
        remote_token_len=128,
    )

    try:
        assert worker.submit_loads(
            MooncakeStoreConnectorMetadata(load_requests=(request, )),
        ) is None
        assert store.get_started.wait(timeout=2)
        assert worker.poll_finished() == KVConnectorOutput()

        store.get_gate.set()
        worker.kv_recv_thread.request_queue.join()
        assert worker.poll_finished() == KVConnectorOutput(
            completed_load_ids={60},
        )
        assert store.get_calls == [(
            [build_store_key(worker._key_metadata, 1, request.block_hashes[0])],
            [[0x1000 + 3 * 100, 0x2000 + 3 * 200]],
            [[100, 200]],
        )]
        messages = [message for _, message in patch_worker_runtime.messages]
        assert any(
            'interaction before: operation=load_batch_get_into_multi_buffers' in message
            for message in messages
        )
        assert any(
            'interaction after: operation=load_batch_get_into_multi_buffers' in message
            for message in messages
        )
    finally:
        store.get_gate.set()
        worker.shutdown()


@pytest.mark.parametrize(
    ('tp_rank', 'kv_head_replica_num', 'key_rank'),
    [
        (0, 8, 0),
        (1, 8, 0),
        (4, 4, 1),
        (5, 4, 1),
    ],
)
def test_receiver_loads_full_suffix_on_every_kv_replica(
    tmp_path,
    tp_rank,
    kv_head_replica_num,
    key_rank,
):
    store = AsyncFakeStore(get_results=([0, 0], ))
    worker, _ = _make_async_worker(
        tmp_path,
        store,
        num_gpu_blocks=16,
        tp_rank=tp_rank,
        tp_size=8,
        kv_head_replica_num=kv_head_replica_num,
    )
    request = _load_request(70 + tp_rank, block_ids=(7, 0))

    worker.submit_loads(
        MooncakeStoreConnectorMetadata(load_requests=(request, )),
    )
    worker.kv_recv_thread.request_queue.join()

    keys, addresses, sizes = store.get_calls[0]
    assert keys == [
        build_store_key(worker._key_metadata, key_rank, block_hash)
        for block_hash in request.block_hashes
    ]
    # Saving is replica-striped, but every rank must restore the complete
    # remote suffix into its own allocated GPU cache copy.
    assert len(keys) == len(addresses) == len(sizes) == 2
    assert addresses == [
        [0x1000 + 7 * 25, 0x2000 + 7 * 50],
        [0x1000, 0x2000],
    ]
    assert sizes == [[25, 50], [25, 50]]
    assert worker.poll_finished() == KVConnectorOutput(
        completed_load_ids={70 + tp_rank},
    )
    worker.shutdown()


def test_receiver_one_key_scatters_all_99_registered_rows(tmp_path):
    store = AsyncFakeStore(get_results=([0], ))
    rows = {
        f'row.{index}': FakeTensor(0x100000 + index * 0x1000, size=400)
        for index in range(99)
    }
    worker, _ = _make_async_worker(tmp_path, store, rows=rows)
    request = _load_request(
        80,
        block_ids=(2, ),
        local_token_len=64,
        remote_token_len=128,
    )

    worker.submit_loads(
        MooncakeStoreConnectorMetadata(load_requests=(request, )),
    )
    worker.kv_recv_thread.request_queue.join()

    _, addresses, sizes = store.get_calls[0]
    assert len(addresses) == len(sizes) == 1
    assert len(addresses[0]) == len(sizes[0]) == 99
    assert addresses[0][0] == 0x100000 + 2 * 100
    assert addresses[0][-1] == 0x100000 + 98 * 0x1000 + 2 * 100
    assert sizes[0] == [100] * 99
    worker.shutdown()


def test_failed_load_is_terminal_sticky_and_has_independent_receive_ack(tmp_path):
    store = AsyncFakeStore(get_results=([0, -5], ))
    worker, _ = _make_async_worker(tmp_path, store)
    request = _load_request(90, block_ids=(1, 2))

    worker.submit_loads(
        MooncakeStoreConnectorMetadata(load_requests=(request, )),
    )
    worker.kv_recv_thread.request_queue.join()

    expected = KVConnectorOutput(
        completed_load_ids={90},
        failed_load_ids={90},
    )
    assert worker.poll_finished() == expected
    assert worker.poll_finished(acknowledged_sending={90}) == expected
    assert worker.get_block_ids_with_load_errors() == {2}
    assert worker.get_block_ids_with_load_errors() == set()
    assert worker.poll_finished(acknowledged_recving={90}) == KVConnectorOutput()
    worker.shutdown()


def test_ack_watermarks_bound_long_running_state_and_reject_old_metadata(
    tmp_path,
    monkeypatch,
):
    store = AsyncFakeStore()
    worker, _ = _make_async_worker(tmp_path, store)

    # Exercise many independent waves in batches.  Once the executor sends
    # the final ACK, only two integer watermarks remain; there is no
    # per-operation acknowledged tombstone collection to grow forever.
    num_waves = 4096
    ack_batch_size = 64
    for start in range(0, num_waves, ack_batch_size):
        operation_ids = set(range(start, start + ack_batch_size))
        for operation_id in operation_ids:
            worker._mark_save_finished(operation_id)
            worker._mark_load_finished(operation_id, set())
        output = worker.poll_finished()
        assert operation_ids <= output.completed_save_ids
        assert operation_ids <= output.completed_load_ids
        assert worker.poll_finished(
            acknowledged_sending=operation_ids,
            acknowledged_recving=operation_ids,
        ) == KVConnectorOutput()

    assert worker._acknowledged_save_watermark == num_waves - 1
    assert worker._acknowledged_load_watermark == num_waves - 1
    assert worker._completed_save_ids == set()
    assert worker._completed_load_ids == set()
    assert worker._failed_load_ids == set()

    send_calls = []
    recv_calls = []
    assert worker.kv_send_thread is not None
    assert worker.kv_recv_thread is not None
    monkeypatch.setattr(
        worker.kv_send_thread,
        'add_request',
        lambda request, ready_event: send_calls.append((request, ready_event)),
    )
    monkeypatch.setattr(
        worker.kv_recv_thread,
        'add_request',
        recv_calls.append,
    )

    old_id = num_waves // 2
    worker.submit_transfers(
        MooncakeStoreConnectorMetadata(
            save_requests=(
                _save_request(old_id, block_ids=(0, ), token_len=64),
            ),
            load_requests=(
                _load_request(
                    old_id,
                    block_ids=(0, ),
                    local_token_len=64,
                    remote_token_len=128,
                ),
            ),
        ),
    )
    assert send_calls == []
    assert recv_calls == []

    new_id = num_waves
    new_save = _save_request(new_id, block_ids=(0, ), token_len=64)
    new_load = _load_request(
        new_id,
        block_ids=(0, ),
        local_token_len=64,
        remote_token_len=128,
    )
    worker.submit_transfers(
        MooncakeStoreConnectorMetadata(
            save_requests=(new_save, ),
            load_requests=(new_load, ),
        ),
    )
    assert send_calls == [(new_save, None)]
    assert recv_calls == [new_load]
    assert worker._inflight_save_ids == {new_id}
    assert worker._inflight_load_ids == {new_id}

    worker._mark_save_finished(new_id)
    worker._mark_load_finished(new_id, set())
    assert worker.poll_finished() == KVConnectorOutput(
        completed_save_ids={new_id},
        completed_load_ids={new_id},
    )
    assert worker.poll_finished(
        acknowledged_sending={new_id},
        acknowledged_recving={new_id},
    ) == KVConnectorOutput()
    worker.shutdown()


def test_ack_watermark_does_not_hide_an_older_inflight_completion(tmp_path):
    store = AsyncFakeStore()
    worker, _ = _make_async_worker(tmp_path, store)
    older_id = 7
    newer_id = 8
    worker._inflight_save_ids.add(older_id)
    worker._inflight_load_ids.add(older_id)

    # Out-of-order terminal notifications are legal (for example an
    # unsubmitted preempted save can finish before an earlier Store RPC).
    worker.poll_finished(
        acknowledged_sending={newer_id},
        acknowledged_recving={newer_id},
    )
    worker._mark_save_finished(older_id)
    worker._mark_load_finished(older_id, set())

    assert worker.poll_finished() == KVConnectorOutput(
        completed_save_ids={older_id},
        completed_load_ids={older_id},
    )
    worker.shutdown()


def test_get_exception_completes_failure_and_receiver_processes_next_job(
    tmp_path,
    patch_worker_runtime,
):
    store = AsyncFakeStore(get_results=(RuntimeError('get failed'), [0]))
    worker, _ = _make_async_worker(tmp_path, store)
    failed_request = _load_request(
        100,
        block_ids=(0, ),
        local_token_len=64,
        remote_token_len=128,
    )
    good_request = _load_request(
        101,
        block_ids=(1, ),
        local_token_len=64,
        remote_token_len=128,
    )

    worker.submit_loads(
        MooncakeStoreConnectorMetadata(
            load_requests=(failed_request, good_request),
        ),
    )
    worker.kv_recv_thread.request_queue.join()

    assert worker.kv_recv_thread.is_alive()
    assert worker.poll_finished() == KVConnectorOutput(
        completed_load_ids={100, 101},
        failed_load_ids={100},
    )
    assert worker.get_block_ids_with_load_errors() == {0}
    assert any(
        'operation=load_batch_get_into_multi_buffers' in message
        and 'status=error' in message
        and 'get failed' in message
        for level, message in patch_worker_runtime.messages
        if level == 'error'
    )
    worker.shutdown()


def test_producer_only_worker_does_not_start_receiver(tmp_path):
    store = AsyncFakeStore()
    worker, _ = _make_async_worker(
        tmp_path,
        store,
        kv_role='kv_producer',
    )

    assert worker.kv_recv_thread is None
    assert worker.kv_send_thread is not None
    worker.shutdown()


def test_sender_queries_then_waits_and_puts_only_missing_scatter(tmp_path, patch_worker_runtime):
    store = AsyncFakeStore(exists_results=([1, 0, 1], ), put_results=([0], ))
    worker, replicate_config = _make_async_worker(tmp_path, store)
    ready_event = FakeReadyEvent(store.operations)
    request = _save_request(41)

    worker.get_finished(
        set(),
        MooncakeStoreConnectorMetadata(save_requests=(request, )),
        ready_event=ready_event,
    )
    worker.kv_send_thread.request_queue.join()

    assert store.operations == ['query', 'event', 'put']
    assert ready_event.calls == 1
    assert len(store.put_calls) == 1
    keys, addresses, sizes, passed_config = store.put_calls[0]
    assert keys == [store.exists_calls[0][1]]
    assert addresses == [[0x1000 + 100, 0x2000 + 200]]
    assert sizes == [[100, 200]]
    assert passed_config is replicate_config
    assert worker.poll_finished() == KVConnectorOutput(completed_save_ids={41})
    assert worker.poll_finished() == KVConnectorOutput(completed_save_ids={41})
    assert worker.poll_finished({41}) == KVConnectorOutput()

    messages = [message for _, message in patch_worker_runtime.messages]
    for operation in ('save_batch_is_exist', 'save_batch_put_from_multi_buffers'):
        assert any(f'interaction before: operation={operation}' in message for message in messages)
        assert any(f'interaction after: operation={operation}' in message for message in messages)
    worker.shutdown()


@pytest.mark.parametrize(
    ('tp_rank', 'kv_head_replica_num', 'owned_ordinals', 'key_rank'),
    [
        (0, 8, (0, 8), 0),
        (1, 8, (1, ), 0),
        (4, 4, (0, 4, 8), 1),
        (5, 4, (1, 5), 1),
    ],
)
def test_sender_stripes_replicated_kv_by_absolute_logical_ordinal(
    tmp_path,
    tp_rank,
    kv_head_replica_num,
    owned_ordinals,
    key_rank,
):
    store = AsyncFakeStore(
        exists_results=([0] * len(owned_ordinals), ),
        put_results=([0] * len(owned_ordinals), ),
    )
    worker, _ = _make_async_worker(
        tmp_path,
        store,
        num_gpu_blocks=16,
        tp_rank=tp_rank,
        tp_size=8,
        kv_head_replica_num=kv_head_replica_num,
    )
    # Deliberately choose physical IDs whose modulo does not match the logical
    # ordinal. Assignment must remain stable across allocator reuse.
    physical_block_ids = (7, 0, 8, 1, 9, 2, 10, 3, 11)
    request = _save_request(
        140 + tp_rank,
        block_ids=physical_block_ids,
        token_len=9 * 64,
    )
    ready_event = FakeReadyEvent(store.operations)

    worker.get_finished(
        set(),
        MooncakeStoreConnectorMetadata(save_requests=(request, )),
        ready_event=ready_event,
    )
    worker.kv_send_thread.request_queue.join()

    expected_keys = [
        build_store_key(worker._key_metadata, key_rank, request.block_hashes[index])
        for index in owned_ordinals
    ]
    assert store.exists_calls == [expected_keys]
    assert store.put_calls[0][0] == expected_keys
    assert all(f'@tp_rank:{key_rank}' in key for key in expected_keys)
    assert store.put_calls[0][1] == [
        [
            0x1000 + physical_block_ids[index] * 25,
            0x2000 + physical_block_ids[index] * 50,
        ]
        for index in owned_ordinals
    ]
    assert store.operations == ['query', 'event', 'put']
    assert ready_event.calls == 1
    assert worker.poll_finished() == KVConnectorOutput(
        completed_save_ids={140 + tp_rank},
    )
    worker.shutdown()


def test_sender_empty_replica_wave_waits_ready_before_completion(tmp_path):
    store = AsyncFakeStore()
    worker, _ = _make_async_worker(
        tmp_path,
        store,
        tp_rank=7,
        tp_size=8,
        kv_head_replica_num=8,
    )
    ready_event = FakeReadyEvent(store.operations)

    worker.get_finished(
        set(),
        MooncakeStoreConnectorMetadata(
            save_requests=(_save_request(159, block_ids=(3, ), token_len=64), ),
        ),
        ready_event=ready_event,
    )
    worker.kv_send_thread.request_queue.join()

    assert store.operations == ['event']
    assert store.exists_calls == []
    assert store.put_calls == []
    assert ready_event.calls == 1
    assert worker.poll_finished() == KVConnectorOutput(completed_save_ids={159})
    worker.shutdown()


def test_all_existing_still_waits_for_forward_before_completion(tmp_path):
    store = AsyncFakeStore(exists_results=([1, 1, 1], ))
    worker, _ = _make_async_worker(tmp_path, store)
    ready_event = FakeReadyEvent(store.operations)

    worker.get_finished(
        set(),
        MooncakeStoreConnectorMetadata(save_requests=(_save_request(42), )),
        ready_event=ready_event,
    )
    worker.kv_send_thread.request_queue.join()

    assert store.operations == ['query', 'event']
    assert ready_event.calls == 1
    assert store.put_calls == []
    assert worker.poll_finished() == KVConnectorOutput(completed_save_ids={42})
    worker.shutdown()


def test_sender_enqueue_does_not_block_on_store_query(tmp_path):
    store = AsyncFakeStore(exists_results=([1], ))
    store.exists_gate = threading.Event()
    worker, _ = _make_async_worker(tmp_path, store)
    ready_event = FakeReadyEvent(store.operations)
    request = _save_request(43, block_ids=(0, ), token_len=64)

    try:
        assert worker.get_finished(
            set(),
            MooncakeStoreConnectorMetadata(save_requests=(request, )),
            ready_event=ready_event,
        ) == KVConnectorOutput()
        assert store.exists_started.wait(timeout=2)
        assert ready_event.calls == 0
        assert worker.poll_finished() == KVConnectorOutput()

        store.exists_gate.set()
        worker.kv_send_thread.request_queue.join()
        assert worker.poll_finished() == KVConnectorOutput(completed_save_ids={43})
    finally:
        store.exists_gate.set()
        worker.shutdown()


def test_query_error_completes_save_and_sender_processes_next_job(tmp_path):
    store = AsyncFakeStore(exists_results=([-1], [1]))
    worker, _ = _make_async_worker(tmp_path, store)
    first_event = FakeReadyEvent(store.operations)
    second_event = FakeReadyEvent(store.operations)

    worker.get_finished(
        set(),
        MooncakeStoreConnectorMetadata(save_requests=(_save_request(44, block_ids=(0, ), token_len=64), )),
        ready_event=first_event,
    )
    worker.get_finished(
        set(),
        MooncakeStoreConnectorMetadata(save_requests=(_save_request(45, block_ids=(1, ), token_len=64), )),
        ready_event=second_event,
    )
    worker.kv_send_thread.request_queue.join()

    assert first_event.calls == second_event.calls == 1
    assert store.put_calls == []
    assert worker.kv_send_thread.is_alive()
    assert worker.poll_finished() == KVConnectorOutput(completed_save_ids={44, 45})
    worker.shutdown()


def test_put_exception_completes_save_and_sender_processes_next_job(
    tmp_path,
    patch_worker_runtime,
):
    store = AsyncFakeStore(
        exists_results=([0], [1]),
        put_results=(RuntimeError('put failed'), ),
    )
    worker, _ = _make_async_worker(tmp_path, store)

    worker.get_finished(
        set(),
        MooncakeStoreConnectorMetadata(save_requests=(_save_request(47, block_ids=(0, ), token_len=64), )),
    )
    worker.get_finished(
        set(),
        MooncakeStoreConnectorMetadata(save_requests=(_save_request(48, block_ids=(1, ), token_len=64), )),
    )
    worker.kv_send_thread.request_queue.join()

    assert worker.kv_send_thread.is_alive()
    assert worker.poll_finished() == KVConnectorOutput(completed_save_ids={47, 48})
    assert any(
        'operation=save_batch_put_from_multi_buffers' in message
        and 'status=error' in message
        and 'put failed' in message
        for level, message in patch_worker_runtime.messages
        if level == 'error'
    )
    worker.shutdown()


def test_one_glm_key_scatters_all_99_registered_rows(tmp_path):
    store = AsyncFakeStore(exists_results=([0], ), put_results=([3], ))
    rows = {
        f'row.{index}': FakeTensor(0x100000 + index * 0x1000, size=400)
        for index in range(99)
    }
    worker, _ = _make_async_worker(tmp_path, store, rows=rows)

    worker.get_finished(
        set(),
        MooncakeStoreConnectorMetadata(
            save_requests=(_save_request(46, block_ids=(2, ), token_len=64), ),
        ),
    )
    worker.kv_send_thread.request_queue.join()

    _, addresses, sizes, _ = store.put_calls[0]
    assert len(addresses) == len(sizes) == 1
    assert len(addresses[0]) == len(sizes[0]) == 99
    assert addresses[0][0] == 0x100000 + 2 * 100
    assert addresses[0][-1] == 0x100000 + 98 * 0x1000 + 2 * 100
    assert sizes[0] == [100] * 99
    worker.shutdown()


def test_lookup_requires_every_unique_rank_without_kv_replication(tmp_path):
    store = AsyncFakeStore(exists_results=([1, 1, 1, 0], ))
    worker, _ = _make_async_worker(tmp_path, store, tp_rank=0, tp_size=2)
    hashes = build_prefix_block_hashes(np.arange(128, dtype=np.int64), 64)

    assert worker.lookup(128, hashes) == 64
    keys = store.exists_calls[0]
    assert len(keys) == 4
    assert '@tp_rank:0' in keys[0]
    assert '@tp_rank:1' in keys[1]
    assert '@tp_rank:0' in keys[2]
    assert '@tp_rank:1' in keys[3]
    worker.shutdown()


@pytest.mark.parametrize(
    ('kv_head_replica_num', 'unique_ranks', 'exists_results'),
    [
        (8, (0, ), [1, 0]),
        (4, (0, 1), [1, 1, 1, 0]),
    ],
)
def test_lookup_requires_every_unique_kv_shard_for_replicated_heads(
    tmp_path,
    kv_head_replica_num,
    unique_ranks,
    exists_results,
):
    store = AsyncFakeStore(exists_results=(exists_results, ))
    worker, _ = _make_async_worker(
        tmp_path,
        store,
        tp_rank=0,
        tp_size=8,
        kv_head_replica_num=kv_head_replica_num,
    )
    hashes = build_prefix_block_hashes(np.arange(128, dtype=np.int64), 64)

    assert worker.lookup(128, hashes) == 64
    assert store.exists_calls[0] == [
        build_store_key(worker._key_metadata, rank, hashes[block_index])
        for block_index in range(2)
        for rank in unique_ranks
    ]
    worker.shutdown()


def test_preempted_unsubmitted_save_is_sticky_completed(tmp_path):
    store = AsyncFakeStore()
    worker, _ = _make_async_worker(tmp_path, store)

    worker.handle_preemptions(MooncakeStoreConnectorMetadata(preempted_save_ids=(51, )))

    assert worker.poll_finished() == KVConnectorOutput(completed_save_ids={51})
    assert worker.poll_finished({51}) == KVConnectorOutput()
    worker.shutdown()


def test_preemption_does_not_cancel_an_inflight_save(tmp_path):
    store = AsyncFakeStore(exists_results=([1], ))
    store.exists_gate = threading.Event()
    worker, _ = _make_async_worker(tmp_path, store)
    ready_event = FakeReadyEvent(store.operations)

    try:
        worker.get_finished(
            set(),
            MooncakeStoreConnectorMetadata(
                save_requests=(_save_request(52, block_ids=(0, ), token_len=64), ),
            ),
            ready_event=ready_event,
        )
        assert store.exists_started.wait(timeout=2)

        worker.handle_preemptions(MooncakeStoreConnectorMetadata(preempted_save_ids=(52, )))
        assert worker.poll_finished() == KVConnectorOutput()

        store.exists_gate.set()
        worker.kv_send_thread.request_queue.join()
        assert ready_event.calls == 1
        assert worker.poll_finished() == KVConnectorOutput(completed_save_ids={52})
    finally:
        store.exists_gate.set()
        worker.shutdown()


def test_invalid_physical_block_fails_before_store_interaction_but_completes(tmp_path):
    store = AsyncFakeStore()
    worker, _ = _make_async_worker(tmp_path, store, num_gpu_blocks=4)
    ready_event = FakeReadyEvent(store.operations)

    worker.get_finished(
        set(),
        MooncakeStoreConnectorMetadata(
            save_requests=(_save_request(53, block_ids=(4, ), token_len=64), ),
        ),
        ready_event=ready_event,
    )
    worker.kv_send_thread.request_queue.join()

    assert store.exists_calls == []
    assert store.put_calls == []
    assert ready_event.calls == 1
    assert worker.poll_finished() == KVConnectorOutput(completed_save_ids={53})
    worker.shutdown()


def test_shutdown_drains_sender_before_closing_store(tmp_path):
    store = AsyncFakeStore(exists_results=([1], ))
    store.exists_gate = threading.Event()
    worker, _ = _make_async_worker(tmp_path, store)
    worker.get_finished(
        set(),
        MooncakeStoreConnectorMetadata(
            save_requests=(_save_request(54, block_ids=(0, ), token_len=64), ),
        ),
    )
    assert store.exists_started.wait(timeout=2)

    shutdown_thread = threading.Thread(target=worker.shutdown)
    shutdown_thread.start()
    try:
        assert not store.close_event.wait(timeout=0.1)
        assert shutdown_thread.is_alive()
    finally:
        store.exists_gate.set()
    shutdown_thread.join(timeout=2)

    assert not shutdown_thread.is_alive()
    assert store.close_event.is_set()
    assert store.close_calls == 1


def test_shutdown_drains_receiver_before_closing_store(tmp_path):
    store = AsyncFakeStore(get_results=([0], ))
    store.get_gate = threading.Event()
    worker, _ = _make_async_worker(tmp_path, store)
    request = _load_request(
        110,
        block_ids=(0, ),
        local_token_len=64,
        remote_token_len=128,
    )
    worker.submit_loads(
        MooncakeStoreConnectorMetadata(load_requests=(request, )),
    )
    assert store.get_started.wait(timeout=2)

    shutdown_thread = threading.Thread(target=worker.shutdown)
    shutdown_thread.start()
    try:
        assert not store.close_event.wait(timeout=0.1)
        assert shutdown_thread.is_alive()
    finally:
        store.get_gate.set()
    shutdown_thread.join(timeout=2)

    assert not shutdown_thread.is_alive()
    assert store.close_event.is_set()
    assert store.close_calls == 1
