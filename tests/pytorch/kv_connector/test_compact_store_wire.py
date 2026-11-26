# Copyright (c) OpenMMLab. All rights reserved.
"""Compact wire spans and mode isolation; BytesStore is not a real Mooncake
server."""
import pickle
from dataclasses import replace

import pytest
import torch
from test_mooncake_store_worker import (  # noqa: F401
    BytesStore,
    FakeCudaEvent,
    FakeStore,
    make_cache_config,
    patch_worker_runtime,
    write_store_config,
)

from lmdeploy.pytorch.config import SSMStateCapacity, StateCacheSpec
from lmdeploy.pytorch.engine.cache_engine.state import SSMStatePlan
from lmdeploy.pytorch.kv_connector import KVCachePool
from lmdeploy.pytorch.kv_connector.mooncake.store import worker as worker_module
from lmdeploy.pytorch.kv_connector.mooncake.store.data import (
    MooncakeStoreCompactStateSave,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreKeyMetadata,
    MooncakeStoreLoadRequest,
    MooncakeStoreSaveRequest,
    MooncakeStoreStateSave,
    build_compact_state_schema,
    build_prefix_block_hashes,
    build_store_key,
)
from lmdeploy.pytorch.kv_connector.mooncake.store.worker import MooncakeStoreWorker


def _plan(units_budget=2, dtype=torch.float32, layers=(1, 3), shape=(3,)):
    return SSMStatePlan([
        StateCacheSpec('recurrent', shape, dtype, layer_ids=list(layers)),
        StateCacheSpec('conv_history', (2, 3), torch.bfloat16, layer_ids=list(layers)),
        StateCacheSpec('kpool_tail', (8,), torch.float32, prefix_checkpoint=False, store_cacheable=False),
    ], SSMStateCapacity(3, 1, units_budget, 2))


def _schema(plan, **kwargs):
    options = dict(spec_identity='target-ar', tp_size=1)
    options.update(kwargs)
    return build_compact_state_schema(plan.unit_specs, **options)


@pytest.mark.parametrize('change', ['dtype', 'shape', 'layers', 'order', 'spec', 'tp'])
def test_compact_schema_covers_recovery_semantics_and_excludes_capacity(change):
    plan = _plan()
    expected = _schema(plan)
    assert _schema(_plan(units_budget=9)) == expected
    kwargs = {}
    if change == 'dtype':
        plan = _plan(dtype=torch.float16)
    elif change == 'shape':
        plan = _plan(shape=(1, 3))
    elif change == 'layers':
        plan = _plan(layers=(3, 1))
    elif change == 'order':
        plan.unit_specs = plan.unit_specs[::-1]
    else:
        kwargs[dict(spec='spec_identity', tp='tp_size')[change]] = (
            2 if change == 'tp' else 'different')
    assert _schema(plan, **kwargs) != expected


def test_versioned_compact_keys_isolate_both_groups_from_legacy():
    legacy = MooncakeStoreKeyMetadata('model', 'test-isolated-prefix', 1, 64)
    compact = replace(legacy, state_schema=_schema(_plan()))
    h = bytes(32)
    assert build_store_key(legacy, 0, h) != build_store_key(compact, 0, h)
    assert build_store_key(legacy, 0, h, group_id=1) != build_store_key(compact, 0, h, group_id=1)
    assert pickle.loads(pickle.dumps(compact)) == compact
    with pytest.raises(ValueError, match='schema fingerprint'):
        replace(compact, state_schema='arbitrary')


@pytest.mark.parametrize('identity', ['', None, []])
def test_schema_requires_valid_execution_contract(identity):
    with pytest.raises(ValueError, match='execution contract'):
        _schema(_plan(), spec_identity=identity)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compact_allocation_exports_only_recovery_fields_without_padding_or_auxiliary(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires an allowed GPU')
    plan = _plan()
    units, auxiliary = plan.allocate(device)
    pool = units.pools[0]
    assert pool.entry_axis == 0 and pool.tensor.shape[0] == plan.capacity.num_units
    assert pool.transfer_ranges == ((0, 24), (256, 24))
    assert pool.tensor.shape[1] == 512
    assert sum(size for _, size in pool.transfer_ranges) == 48
    assert auxiliary.pools[0].tensor.untyped_storage().data_ptr() != pool.tensor.untyped_storage().data_ptr()
    assert [spec.name for spec in plan.unit_specs] == ['recurrent', 'conv_history']


class _HostTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


class _CompactBytesStore(BytesStore):
    def batch_get_into_multi_buffers(self, keys, addresses, sizes):
        results = super().batch_get_into_multi_buffers(keys, addresses, sizes)
        return [sum(parts) for parts in sizes] if self.get_results is None else results


def _worker(tmp_path, monkeypatch, *, device='cpu'):
    plan = _plan()
    units, auxiliary = plan.allocate(device)
    config = replace(make_cache_config(write_store_config(tmp_path), num_gpu_blocks=4),
                     compact_state_capacity=plan.capacity, states_shapes=[((3,), torch.float32)],
                     prefill_save_alignment=64)
    monkeypatch.setattr(MooncakeStoreWorker, '_start_lookup_server', lambda self: None)
    store = _CompactBytesStore()
    store.existing_keys = store.values
    worker = MooncakeStoreWorker(config, store_factory=lambda: store, replicate_config=object())
    kv = torch.arange(256, dtype=torch.uint8, device=device).reshape(4, 64)
    pool = units.pools[0]
    wrap = (lambda x: x.as_subclass(_HostTensor)) if device == 'cpu' else (lambda x: x)
    state_pool = KVCachePool(wrap(pool.tensor), pool.entry_axis, pool.transfer_ranges)
    return worker, store, plan, units, auxiliary, wrap(kv), state_pool


def test_direct_unit_roundtrip_has_no_snapshot_and_uses_c_not_e(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_module.torch.cuda, 'Event', FakeCudaEvent)
    worker, store, plan, units, auxiliary, kv, pool = _worker(tmp_path, monkeypatch)
    worker.register_kv_caches({'kv': kv}, state_cache_pools=(pool,), state_cache_schema=_schema(plan))
    payload = units.pools[0].tensor
    payload.fill_(91)  # poison all padding, including source and destination.
    for spec, tensor in zip(plan.unit_specs, units.tensor_views):
        row = tensor.select(1 if spec.layer_rows else 0, 2)
        row.copy_(torch.arange(row.numel()).reshape(row.shape) + 1)
    expected = b''.join(bytes(payload[2, offset:offset + size].tolist()) for offset, size in pool.transfer_ranges)
    aux_before = auxiliary.pools[0].tensor.clone()
    hashes = build_prefix_block_hashes(range(136), 64)
    state = MooncakeStoreCompactStateSave(2, 64, 136, hashes[0])
    request = MooncakeStoreSaveRequest(11, 12, 0, (0, 1), (0, 1), hashes, state=state)
    metadata = MooncakeStoreConnectorMetadata(save_requests=(request,))
    assert pickle.loads(pickle.dumps(metadata)) == metadata
    assert metadata.get_state_save_copies() == ()
    worker.start_save_kv(metadata)
    worker.kv_send_thread.request_queue.join()
    assert worker.get_finished().completed_save_ids == {11}
    key = build_store_key(worker.key_metadata, 0, hashes[0], group_id=1)
    assert b''.join(store.values[key]) == expected and len(expected) == 48
    assert build_store_key(worker.key_metadata, 0, hashes[1], group_id=1) not in store.values
    assert worker.lookup(128, hashes) == 64  # complete KV does not invent state at e.
    state_position = store.put_calls[0][0].index(key)
    assert store.put_calls[0][1][state_position] == [payload.data_ptr() + 2 * 512,
                                                   payload.data_ptr() + 2 * 512 + 256]
    assert store.put_calls[0][2][state_position] == [24, 24]
    # Subsequent runtime writes use another private unit; the saved input stays unchanged.
    payload[1].fill_(77)
    load = MooncakeStoreLoadRequest(13, (3,), (hashes[0],), 1, state_unit_id=4)
    worker.start_load_kv(MooncakeStoreConnectorMetadata(load_requests=(load,)))
    worker.kv_recv_thread.request_queue.join()
    assert worker.get_finished().finished_receiving == {13}
    meaningful = torch.zeros(512, dtype=torch.bool)
    for offset, size in pool.transfer_ranges:
        meaningful[offset:offset + size] = True
    assert torch.equal(payload[4, meaningful], payload[2, meaningful])
    assert (payload[4, ~meaningful] == 91).all()  # no padding transfer/restore copy
    assert torch.equal(auxiliary.pools[0].tensor, aux_before)
    legacy = replace(worker.key_metadata, state_schema=None)
    legacy_key = build_store_key(legacy, 0, hashes[0], group_id=1)
    assert legacy_key not in store.values
    store.values[legacy_key] = store.values.pop(key)
    assert worker.lookup(128, hashes) == 0  # never dual-read the legacy namespace
    worker.shutdown()


@pytest.mark.parametrize('bad', ['missing_schema', 'legacy_schema', 'slice', 'auxiliary', 'whole_entry', 'overlap'])
def test_registration_rejects_wrong_pool_or_schema_before_registering(tmp_path, monkeypatch, bad):
    worker, store, plan, units, auxiliary, kv, pool = _worker(tmp_path, monkeypatch)
    schema = _schema(plan)
    if bad == 'missing_schema':
        schema = None
    elif bad == 'legacy_schema':
        worker._cache_config.compact_state_capacity = None
    elif bad == 'slice':
        pool = replace(pool, tensor=pool.tensor[:, :256])
    elif bad == 'auxiliary':
        pool = replace(pool, tensor=auxiliary.pools[0].tensor.as_subclass(_HostTensor))
    elif bad == 'whole_entry':
        pool = replace(pool, transfer_ranges=None)
    else:
        pool = replace(pool, transfer_ranges=((0, 30), (20, 8)))
    with pytest.raises(ValueError, match='Compact Store'):
        worker.register_kv_caches({'kv': kv}, state_cache_pools=(pool,), state_cache_schema=schema)
    assert not store.register_calls
    worker.shutdown()


@pytest.mark.parametrize('direction', ['save', 'load'])
def test_worker_rejects_mixed_legacy_metadata_before_enqueue(tmp_path, monkeypatch, direction):
    worker, store, plan, units, auxiliary, kv, pool = _worker(tmp_path, monkeypatch)
    worker.register_kv_caches({'kv': kv}, state_cache_pools=(pool,), state_cache_schema=_schema(plan))
    h = bytes(32)
    if direction == 'save':
        metadata = MooncakeStoreConnectorMetadata(save_requests=(
            MooncakeStoreSaveRequest(1, 1, 0, (0,), (0,), (h,), MooncakeStoreStateSave(1, 2, 64)),))
        call = worker.start_save_kv
    else:
        metadata = MooncakeStoreConnectorMetadata(load_requests=(MooncakeStoreLoadRequest(1, (0,), (h,), 1, 2),))
        call = worker.start_load_kv
    with pytest.raises(ValueError, match='schema mode'):
        call(metadata)
    assert not worker._inflight_save_ids and not worker._inflight_loads
    assert not store.get_calls and not store.put_calls
    worker.shutdown()


@pytest.mark.parametrize('bad', ['zero', 'bad_boundary', 'bad_hash'])
def test_invalid_direct_save_record_rejected(bad):
    values = dict(unit_id=1, boundary_tokens=64, forward_end=72, boundary_hash=bytes(32))
    values[dict(zero='unit_id', bad_boundary='boundary_tokens', bad_hash='boundary_hash')[bad]] = (
        0 if bad == 'zero' else 80 if bad == 'bad_boundary' else b'short')
    with pytest.raises(ValueError):
        MooncakeStoreCompactStateSave(**values)


def test_load_cannot_alias_legacy_slot_and_compact_unit():
    with pytest.raises(ValueError, match='legacy state slot'):
        MooncakeStoreLoadRequest(1, (0,), (bytes(32),), 1, state_slot=1, state_unit_id=2)


@pytest.mark.parametrize('bad', ['checkpoint_hash', 'checkpoint_alignment', 'unfinished_kv'])
def test_sender_rejects_inconsistent_c_e_or_prefix_identity(tmp_path, monkeypatch, bad):
    worker, store, plan, units, auxiliary, kv, pool = _worker(tmp_path, monkeypatch)
    worker.register_kv_caches({'kv': kv}, state_cache_pools=(pool,), state_cache_schema=_schema(plan))
    hashes = build_prefix_block_hashes(range(136), 64)
    state = MooncakeStoreCompactStateSave(1, 64, 136, hashes[0])
    if bad == 'checkpoint_hash':
        state = replace(state, boundary_hash=hashes[1])
    elif bad == 'checkpoint_alignment':
        state = replace(state, boundary_tokens=65)
    else:
        state = replace(state, forward_end=100)
    request = MooncakeStoreSaveRequest(1, 2, 0, (0, 1), (0, 1), hashes, state)
    with pytest.raises(ValueError, match='Compact'):
        worker.kv_send_thread._owned_entries(request)
    assert not store.put_calls
    worker.shutdown()


@pytest.mark.parametrize('results', [[63, 48], [64, 47], [64, 0], [64, 49], [64, -1]])
def test_compact_load_rejects_short_oversize_or_failed_object(tmp_path, monkeypatch, results):
    monkeypatch.setattr(worker_module.torch.cuda, 'Event', FakeCudaEvent)
    worker, store, plan, units, auxiliary, kv, pool = _worker(tmp_path, monkeypatch)
    worker.register_kv_caches({'kv': kv}, state_cache_pools=(pool,), state_cache_schema=_schema(plan))
    h = bytes(32)
    store.values[build_store_key(worker.key_metadata, 0, h)] = [bytes(64)]
    store.values[build_store_key(worker.key_metadata, 0, h, group_id=1)] = [bytes(24), bytes(24)]
    store.get_results = results
    worker.start_load_kv(MooncakeStoreConnectorMetadata(load_requests=(
        MooncakeStoreLoadRequest(9, (0,), (h,), 1, state_unit_id=2),)))
    worker.kv_recv_thread.request_queue.join()
    output = worker.get_finished()
    assert output.finished_receiving == output.failed_receiving == {9}
    worker.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires an allowed GPU')
@pytest.mark.parametrize('load_result', ['ready', 'cancel', 'short'])
def test_real_cuda_event_and_direct_pool_addresses_survive_delayed_store_reader(tmp_path, monkeypatch, load_result):
    """GPU-address oracle with host byte storage, NOT real network/zero-copy
    I/O."""
    import ctypes
    import threading

    from lmdeploy.pytorch.paging.compact_state import SSMStateManager

    runtime = ctypes.CDLL(f'libcudart.so.{torch.version.cuda.split(".")[0]}')
    runtime.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    runtime.cudaMemcpy.restype = ctypes.c_int
    entered, allow = threading.Event(), threading.Event()
    load_entered, load_allow = threading.Event(), threading.Event()

    class GPUAddressStore(_CompactBytesStore):
        def batch_put_from_multi_buffers(self, keys, addresses, sizes, config):
            torch.cuda.set_device(0)
            entered.set()  # worker already waited for its actual producer event
            assert allow.wait(10)
            self.put_calls.append((keys, addresses, sizes, config))
            for key, pointers, lengths in zip(keys, addresses, sizes):
                values = []
                for pointer, length in zip(pointers, lengths):
                    buffer = ctypes.create_string_buffer(length)
                    assert runtime.cudaMemcpy(buffer, pointer, length, 2) == 0  # device to host oracle
                    values.append(buffer.raw)
                self.values[key] = values
            return [0] * len(keys)

        def batch_get_into_multi_buffers(self, keys, addresses, sizes):
            torch.cuda.set_device(0)
            load_entered.set()
            assert load_allow.wait(10)
            self.get_calls.append((keys, addresses, sizes))
            for key, pointers, lengths in zip(keys, addresses, sizes):
                for pointer, length, value in zip(pointers, lengths, self.values[key]):
                    assert len(value) == length
                    buffer = ctypes.create_string_buffer(value)
                    assert runtime.cudaMemcpy(pointer, buffer, length, 1) == 0  # host to device oracle
            return [sum(parts) - int(load_result == 'short') for parts in sizes]

    worker, _, plan, units, auxiliary, kv, pool = _worker(tmp_path, monkeypatch, device='cuda')
    store = GPUAddressStore()
    store.existing_keys = store.values
    worker.store = store
    worker.register_kv_caches({'kv': kv}, state_cache_pools=(pool,), state_cache_schema=_schema(plan))
    owner = SSMStateManager(plan.capacity)
    bank = owner.allocate_bank()
    reference = owner.reserve_checkpoint(bank, 64)
    producer = owner.submit_decode((bank,), checkpoints=(reference,))
    lease = owner.retain(reference, 'store')
    io = owner.submit_save(lease, stream=1, wait_for=(producer,))
    payload = units.pools[0].tensor
    payload.fill_(93)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    h = build_prefix_block_hashes(range(64), 64)[0]
    state = MooncakeStoreCompactStateSave(reference.unit_id, 64, 72, h)
    request = MooncakeStoreSaveRequest(17, 18, 0, (0,), (0,), (h,), state)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
        for offset, length in pool.transfer_ranges:
            payload[reference.unit_id, offset:offset + length].fill_(37)
        worker.start_save_kv(MooncakeStoreConnectorMetadata(save_requests=(request,)))
    assert entered.wait(10)
    # Completion of the producer, request cancellation and logical lease release
    # still cannot recycle a unit held by an actual in-flight Store reader.
    payload[bank.unit_ids[0]].fill_(77)
    owner.release_bank(bank)
    owner.release_reference(reference)
    owner.complete(producer)
    owner.release_reference(lease)
    assert reference.unit_id in owner._units and not io.finished
    allow.set()
    worker.kv_send_thread.request_queue.join()
    assert worker.get_finished().completed_save_ids == {17}
    key = build_store_key(worker.key_metadata, 0, h, group_id=1)
    assert b''.join(store.values[key]) == bytes([37]) * 48
    owner.complete(io)
    assert not owner._units and not owner._banks and not owner._accesses
    target = owner.allocate_bank()
    target_id = target.unit_ids[0]
    payload[target_id].fill_(93)
    load_access = owner.submit_load(target, 64)
    worker.start_load_kv(MooncakeStoreConnectorMetadata(load_requests=(
        MooncakeStoreLoadRequest(19, (3,), (h,), 1, state_unit_id=target_id),)))
    assert load_entered.wait(10)
    with pytest.raises(ValueError, match='Store I/O'):
        owner.submit_decode((target,))
    if load_result == 'cancel':
        owner.release_bank(target)
    assert target_id in owner._units and target.pending_load is load_access
    load_allow.set()
    worker.kv_recv_thread.request_queue.join()
    result = worker.get_finished()
    assert result.finished_receiving == {19}
    assert result.failed_receiving == ({19} if load_result == 'short' else set())
    mask = torch.zeros(512, dtype=torch.bool, device='cuda')
    for offset, length in pool.transfer_ranges:
        mask[offset:offset + length] = True
    assert (payload[target_id, mask] == 37).all() and (payload[target_id, ~mask] == 93).all()
    owner.complete(load_access, success=not result.failed_receiving)
    assert target.loaded_input == ((target_id, 64) if load_result == 'ready' else None)
    assert target.failed == (load_result == 'short')
    owner.release_bank(target)
    assert not owner._units and not owner._banks and not owner._accesses
    worker.shutdown()


@pytest.mark.parametrize('missing', ['none', 'state_rank', 'kv_shard'])
def test_compact_lookup_requires_all_state_ranks_and_complete_kv(tmp_path, missing):
    from lmdeploy.pytorch.kv_connector.mooncake.store.data import MooncakeStoreRegistration
    from lmdeploy.pytorch.kv_connector.mooncake.store.worker_threads import KVCacheStoreSendingThread

    plan = _plan()
    metadata = MooncakeStoreKeyMetadata('model', 'isolated', 8, 64, 4, _schema(plan, tp_size=8))
    hashes = build_prefix_block_hashes(range(128), 64)
    request = MooncakeStoreSaveRequest(1, 2, 0, (0, 1), (0, 1), hashes,
                                      MooncakeStoreCompactStateSave(2, 64, 136, hashes[0]))
    keys = []
    for rank in range(8):
        sender = KVCacheStoreSendingThread(store=FakeStore(),
            registrations=(MooncakeStoreRegistration('kv', 0x1000, 128),), row_block_sizes=(64,),
            num_gpu_blocks=2, key_metadata=metadata, global_rank=rank, tp_rank=rank, tp_size=8,
            completion_callback=lambda _: None, replicate_config=object())
        keys.extend(entry.key for entry in sender._owned_entries(request))
    states = [build_store_key(metadata, rank, hashes[0], group_id=1) for rank in range(8)]
    assert all(keys.count(key) == 1 for key in states)
    assert len(keys) == 12  # four KV keys (two blocks x two shards) plus eight states
    existing = set(keys)
    if missing == 'state_rank':
        existing.remove(states[-1])
    elif missing == 'kv_shard':
        existing.remove(build_store_key(metadata, 1, hashes[0]))
    config = replace(make_cache_config(write_store_config(tmp_path)), states_shapes=[((3,), torch.float32)],
                     compact_state_capacity=plan.capacity, prefill_save_alignment=64)
    worker = MooncakeStoreWorker(config, tp_size=8, kv_head_replica_num=4,
                                store_factory=lambda: FakeStore(existing_keys=existing))
    worker.key_metadata = metadata  # lookup-only unit test; no buffer registration / I/O
    assert worker.lookup(128, hashes) == (64 if missing == 'none' else 0)
    worker.shutdown()


def test_adjacent_meaningful_fields_coalesce_without_changing_payload_order():
    plan = SSMStatePlan([StateCacheSpec('recurrent', (64,), torch.float32),
                             StateCacheSpec('conv', (128,), torch.float16)], SSMStateCapacity(2, 1, 1, 1))
    units, _ = plan.allocate('meta')
    assert units.pools[0].transfer_ranges == ((0, 512),)
