# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.engine.inputs_maker import _ForwardInputsResult, _ForwardInputsTask
from lmdeploy.pytorch.kv_connector.base import KVConnectorOutput
from lmdeploy.pytorch.kv_connector.mooncake.store import scheduler as scheduler_module
from lmdeploy.pytorch.kv_connector.mooncake.store.data import (
    MOONCAKE_BLOCK_HASH_BYTES,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreLoadRequest,
    MooncakeStoreSaveRequest,
    build_prefix_block_hashes,
)
from lmdeploy.pytorch.kv_connector.mooncake.store.scheduler import MooncakeStoreScheduler
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.paging.scheduler import Scheduler, SchedulerOutput
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


class _FakeLookupClient:

    def __init__(self, _cache_config):
        self.discarded = []
        self.futures = {}
        self.closed = False

    def discard(self, req_id):
        self.discarded.append(req_id)
        self.futures.pop(req_id, None)

    def close(self):
        self.closed = True


@pytest.fixture
def mooncake_scheduler(monkeypatch):
    monkeypatch.setattr(scheduler_module, 'LookupKeyClient', _FakeLookupClient)
    cache_config = CacheConfig(
        max_batches=2,
        block_size=4,
        num_cpu_blocks=0,
        num_gpu_blocks=8,
        kv_transfer_config=KVTransferConfig(
            kv_connector='MooncakeStoreConnector',
            kv_role='kv_both',
        ),
    )
    return MooncakeStoreScheduler(cache_config)


def _connector_output(request, token_len, block_ids, generation=0, preempted=()):
    return SchedulerOutput(
        running=[request],
        swap_in_map={},
        swap_out_map={},
        copy_map={},
        connector_token_lens=(token_len, ),
        connector_block_ids=(block_ids, ),
        connector_logical_block_ids=(block_ids, ),
        connector_generations=(generation, ),
        preempted_save_ids=preempted,
    )


def test_mooncake_scheduler_builds_unique_incremental_prefill_waves(mooncake_scheduler):
    tokens = np.arange(8, dtype=np.int64)
    request = SimpleNamespace(
        seq_id=11,
        adapter_name='adapter-a',
        all_ids=tokens,
        get_prefix_cache_extra_identity=lambda _start, _end: (),
    )

    first = mooncake_scheduler.build_connector_meta(_connector_output(request, 4, (7, )))
    duplicate = mooncake_scheduler.build_connector_meta(_connector_output(request, 4, (7, )))
    second = mooncake_scheduler.build_connector_meta(_connector_output(request, 8, (7, 9)))
    resumed = mooncake_scheduler.build_connector_meta(
        _connector_output(request, 4, (2, ), generation=1, preempted=(0, 1)))

    assert len(first.save_requests) == 1
    assert duplicate.save_requests == ()
    assert first.save_requests[0].save_id == 0
    assert first.save_requests[0].token_len == 4
    assert second.save_requests[0].save_id == 1
    assert second.save_requests[0].token_len == 8
    assert second.save_requests[0].block_hashes[0] == first.save_requests[0].block_hashes[0]
    assert resumed.save_requests[0].save_id == 2
    assert resumed.save_requests[0].generation == 1
    assert resumed.preempted_save_ids == (0, 1)
    assert first.save_requests[0].block_hashes == build_prefix_block_hashes(
        tokens[:4].tolist(),
        4,
        extra_identity=repr(('adapter-a', ())),
    )


def test_mooncake_scheduler_reuses_hashes_across_chunks_and_generations(
    mooncake_scheduler,
    monkeypatch,
):
    original_build_hashes = scheduler_module.build_prefix_block_hashes
    hash_calls = []

    def record_build_hashes(token_ids, block_size, **kwargs):
        hash_calls.append((
            len(token_ids),
            len(kwargs.get('previous_hashes', ())),
            kwargs['extra_identity'],
        ))
        return original_build_hashes(token_ids, block_size, **kwargs)

    monkeypatch.setattr(
        scheduler_module,
        'build_prefix_block_hashes',
        record_build_hashes,
    )
    request = SimpleNamespace(
        seq_id=13,
        adapter_name='adapter-a',
        all_ids=np.arange(12, dtype=np.int64),
        get_prefix_cache_extra_identity=lambda _start, _end: (),
    )

    first = mooncake_scheduler.build_connector_meta(
        _connector_output(request, 4, (7, )))
    second = mooncake_scheduler.build_connector_meta(
        _connector_output(request, 8, (7, 9)))
    resumed_short = mooncake_scheduler.build_connector_meta(
        _connector_output(request, 4, (2, ), generation=1))
    resumed = mooncake_scheduler.build_connector_meta(
        _connector_output(request, 12, (2, 4, 6), generation=1))

    assert [(token_len, previous_count)
            for token_len, previous_count, _identity in hash_calls] == [
                (4, 0),
                (8, 1),
                (12, 2),
            ]
    assert second.save_requests[0].block_hashes[:1] == first.save_requests[0].block_hashes
    assert resumed_short.save_requests[0].block_hashes == first.save_requests[0].block_hashes
    assert resumed.save_requests[0].block_hashes[:2] == second.save_requests[0].block_hashes

    request.adapter_name = 'adapter-b'
    changed_identity = mooncake_scheduler.build_connector_meta(
        _connector_output(request, 12, (3, 5, 7), generation=2))

    assert hash_calls[-1][1:] == (0, repr(('adapter-b', ())))
    assert all(
        old_hash != new_hash
        for old_hash, new_hash in zip(
            resumed.save_requests[0].block_hashes,
            changed_identity.save_requests[0].block_hashes,
            strict=True,
        ))


def test_mooncake_scheduler_rollback_retries_and_finished_lifecycle(mooncake_scheduler):
    request = SimpleNamespace(
        seq_id=17,
        adapter_name=None,
        all_ids=np.arange(4, dtype=np.int64),
        get_prefix_cache_extra_identity=lambda _start, _end: (),
    )
    first = mooncake_scheduler.build_connector_meta(_connector_output(request, 4, (3, )))
    first_id = first.save_requests[0].save_id
    cached_hashes = mooncake_scheduler._request_hash_trackers[17].block_hashes

    mooncake_scheduler.update_connector_output({'rolled_back_save_ids': {first_id}})
    assert mooncake_scheduler._request_hash_trackers[17].block_hashes is cached_hashes
    retry = mooncake_scheduler.build_connector_meta(_connector_output(request, 4, (3, )))
    retry_id = retry.save_requests[0].save_id
    assert retry.save_requests[0].block_hashes == cached_hashes
    mooncake_scheduler.request_finished(request, (3, ))

    assert mooncake_scheduler.client.discarded == [17]
    assert mooncake_scheduler.has_pending_kv_connector_work()
    assert mooncake_scheduler._request_hash_trackers == {}
    assert mooncake_scheduler._finished_requests == {17}

    mooncake_scheduler.update_connector_output({retry_id})

    assert not mooncake_scheduler.has_pending_kv_connector_work()
    assert mooncake_scheduler._request_trackers == {}
    assert mooncake_scheduler._request_hash_trackers == {}
    assert mooncake_scheduler._finished_requests == set()


def test_mooncake_scheduler_async_lookup_builds_suffix_load_and_blocks_failed_retry(
    mooncake_scheduler,
):
    request = SimpleNamespace(
        seq_id=23,
        adapter_name=None,
        all_ids=np.arange(13, dtype=np.int64),
        get_prefix_cache_max_match_step=lambda: 12,
        get_prefix_cache_extra_identity=lambda _start, _end: (),
    )
    lookup_results = iter((None, 12))
    lookup_calls = []

    def lookup(req_id, token_len, block_hashes, non_block):
        lookup_calls.append((req_id, token_len, tuple(block_hashes), non_block))
        return next(lookup_results)

    mooncake_scheduler.client.lookup = lookup

    assert mooncake_scheduler.get_num_new_matched_tokens(request, 4) == (None, False)
    assert mooncake_scheduler.get_num_new_matched_tokens(request, 4) == (8, True)
    load_request = mooncake_scheduler.update_state_after_alloc(
        request,
        (31, 32),
        8,
        generation=3,
    )

    assert load_request.local_token_len == 4
    assert load_request.remote_token_len == 12
    assert load_request.block_ids == (31, 32)
    assert len(load_request.block_hashes) == 2
    assert load_request.block_hashes == lookup_calls[-1][2][1:3]

    empty_output = SchedulerOutput([], {}, {}, {})
    first_meta = mooncake_scheduler.build_connector_meta(empty_output)
    retry_meta = mooncake_scheduler.build_connector_meta(empty_output)
    assert first_meta.load_requests == (load_request, )
    assert retry_meta.load_requests == ()

    mooncake_scheduler.update_connector_output({
        'rolled_back_load_ids': {load_request.load_id},
    })
    retry_meta = mooncake_scheduler.build_connector_meta(empty_output)
    assert retry_meta.load_requests == (load_request, )
    mooncake_scheduler.mark_connector_meta_dispatched(retry_meta)
    assert mooncake_scheduler.build_connector_meta(empty_output).load_requests == ()
    mooncake_scheduler.update_connector_output({
        'completed_load_ids': {load_request.load_id},
        'failed_load_ids': {load_request.load_id},
    })

    # A failed immutable signature must fall back to compute instead of
    # entering an infinite lookup/load retry loop.
    assert mooncake_scheduler.get_num_new_matched_tokens(request, 4) == (0, False)
    assert len(lookup_calls) == 2


def test_mooncake_scheduler_clamps_short_hit_before_multimodal_span_boundary(
    mooncake_scheduler,
):
    paging = _paging_scheduler(
        None,
        enable_prefix_caching=True,
        block_size=4,
    )
    multimodals = {
        'image': [
            MultiModalData(
                data=torch.ones((2, 2)),
                start=6,
                end=11,
                content_hash='image-a',
            ),
        ],
    }
    request = paging.add_session(30).add_sequence(
        np.arange(17, dtype=np.int64),
        multimodals=multimodals,
    )
    lookup_calls = []

    def lookup(req_id, token_len, block_hashes, non_block):
        lookup_calls.append((req_id, token_len, len(block_hashes), non_block))
        # Block boundary 8 is inside the multimodal span [6, 11).  The safe
        # resume point is its start rounded down to block boundary 4.
        return 8

    mooncake_scheduler.client.lookup = lookup

    assert mooncake_scheduler.get_num_new_matched_tokens(request, 0) == (4, True)
    load_request = mooncake_scheduler.update_state_after_alloc(
        request,
        (37, ),
        4,
    )

    assert lookup_calls == [(request.seq_id, 16, 4, True)]
    assert load_request.local_token_len == 0
    assert load_request.remote_token_len == 4
    assert len(load_request.block_hashes) == 1


def test_mooncake_scheduler_cancelled_partial_dispatch_retries_same_load_id(
    mooncake_scheduler,
):
    request = SimpleNamespace(
        seq_id=29,
        adapter_name=None,
        all_ids=np.arange(9, dtype=np.int64),
        get_prefix_cache_max_match_step=lambda: 8,
        get_prefix_cache_extra_identity=lambda _start, _end: (),
    )
    mooncake_scheduler.client.lookup = (
        lambda _req_id, _token_len, _hashes, non_block: 8)
    assert mooncake_scheduler.get_num_new_matched_tokens(request, 0) == (8, True)
    load_request = mooncake_scheduler.update_state_after_alloc(
        request,
        (41, 42),
        8,
        generation=1,
    )
    empty_output = SchedulerOutput([], {}, {}, {})
    mooncake_scheduler.build_connector_meta(empty_output)

    # request_finished races a collective where only a subset of ranks may
    # have accepted metadata. Rollback must keep a retryable fence wave.
    mooncake_scheduler.request_finished(request, (41, 42))
    mooncake_scheduler.update_connector_output({
        'rolled_back_load_ids': {load_request.load_id},
    })
    retry = mooncake_scheduler.build_connector_meta(empty_output)

    assert retry.load_requests == (load_request, )
    mooncake_scheduler.mark_connector_meta_dispatched(retry)
    mooncake_scheduler.update_connector_output({
        'completed_load_ids': {load_request.load_id},
    })
    assert not mooncake_scheduler.has_pending_kv_transfer_work()


def test_mooncake_scheduler_cancel_lookup_keeps_save_state(mooncake_scheduler):
    request_id = 31
    mooncake_scheduler.client.futures[request_id] = object()
    mooncake_scheduler._lookup_plans[request_id] = object()
    save_tracker = object()
    mooncake_scheduler._request_hash_trackers[request_id] = save_tracker

    mooncake_scheduler.cancel_lookup(request_id)

    assert request_id not in mooncake_scheduler.client.futures
    assert request_id not in mooncake_scheduler._lookup_plans
    assert mooncake_scheduler._request_hash_trackers[request_id] is save_tracker


class _PinConnector:

    def __init__(self):
        self.next_save_id = 0
        self.outputs = []
        self.updates = []
        self.finished = []

    def on_new_request(self, request):
        return None

    def cancel_lookup(self, request_id):
        return None

    def has_pending_kv_lookup_work(self):
        return False

    def build_connector_meta(self, scheduler_output):
        self.outputs.append(scheduler_output)
        save_requests = ()
        if scheduler_output.connector_token_lens:
            token_len = scheduler_output.connector_token_lens[0]
            num_blocks = token_len // 4
            save_requests = (
                MooncakeStoreSaveRequest(
                    req_id=scheduler_output.running[0].seq_id,
                    save_id=self.next_save_id,
                    generation=scheduler_output.connector_generations[0],
                    token_len=token_len,
                    block_ids=scheduler_output.connector_block_ids[0],
                    block_hashes=(b'x' * MOONCAKE_BLOCK_HASH_BYTES, ) * num_blocks,
                ),
            )
            self.next_save_id += 1
        return MooncakeStoreConnectorMetadata(
            save_requests=save_requests,
            preempted_save_ids=scheduler_output.preempted_save_ids,
        )

    def update_connector_output(self, output):
        self.updates.append(output)

    def request_finished(self, request, block_ids):
        self.finished.append((request, block_ids))
        return False, None

    def shutdown(self):
        return None


class _LoadConnector:

    def __init__(self, remote_token_len=8):
        self.remote_token_len = remote_token_len
        self.next_load_id = 0
        self.ready = {}
        self.dispatching = set()
        self.dispatched = set()
        self.cancelled = set()
        self.needs_fence = set()
        self.failed_signature = False
        self.updates = []
        self.new_requests = []
        self.lookup_calls = []

    def on_new_request(self, request):
        self.new_requests.append(request.seq_id)

    def cancel_lookup(self, request_id):
        return None

    def has_pending_kv_lookup_work(self):
        return False

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.lookup_calls.append((request.seq_id, num_computed_tokens))
        if self.failed_signature:
            return 0, False
        return max(0, self.remote_token_len - num_computed_tokens), True

    def update_state_after_alloc(
        self,
        request,
        block_ids,
        num_external_tokens,
        generation=0,
    ):
        load_id = self.next_load_id
        self.next_load_id += 1
        load_request = MooncakeStoreLoadRequest(
            req_id=request.seq_id,
            load_id=load_id,
            generation=generation,
            local_token_len=self.remote_token_len - num_external_tokens,
            remote_token_len=self.remote_token_len,
            block_ids=tuple(block_ids),
            block_hashes=(b'l' * MOONCAKE_BLOCK_HASH_BYTES, ) * len(block_ids),
        )
        self.ready[load_id] = load_request
        return load_request

    def build_connector_meta(self, _scheduler_output):
        load_requests = tuple(
            request
            for load_id, request in sorted(self.ready.items())
            if (load_id not in self.dispatching
                and load_id not in self.dispatched
                and (load_id not in self.cancelled
                     or load_id in self.needs_fence))
        )
        self.dispatching.update(request.load_id for request in load_requests)
        return MooncakeStoreConnectorMetadata(load_requests=load_requests)

    def mark_connector_meta_dispatched(self, metadata):
        load_ids = {request.load_id for request in metadata.load_requests}
        self.dispatching.difference_update(load_ids)
        self.needs_fence.difference_update(load_ids)
        self.dispatched.update(load_ids)

    def update_connector_output(self, output):
        self.updates.append(output)
        if isinstance(output, dict):
            completed = output.get('completed_load_ids', ())
            failed = output.get('failed_load_ids', ())
            cancelled = output.get('cancelled_load_ids', ())
            rolled_back = output.get('rolled_back_load_ids', ())
        else:
            completed = getattr(output, 'completed_load_ids', ()) or ()
            failed = getattr(output, 'failed_load_ids', ()) or ()
            cancelled = ()
            rolled_back = ()
        if failed:
            self.failed_signature = True
        for load_id in rolled_back:
            self.dispatching.discard(int(load_id))
            if int(load_id) in self.cancelled:
                self.needs_fence.add(int(load_id))
        for load_id in cancelled:
            load_id = int(load_id)
            self.cancelled.add(load_id)
            if load_id not in self.dispatching and load_id not in self.dispatched:
                self.ready.pop(load_id, None)
        for load_id in completed:
            self.ready.pop(int(load_id), None)

    def request_finished(self, _request, _block_ids):
        return False, None

    def shutdown(self):
        return None


def _paging_scheduler(
    connector,
    *,
    num_gpu_blocks=4,
    enable_prefix_caching=False,
    block_size=4,
    max_batches=1,
    max_prefill_token_num=8192,
    max_session_len=None,
):
    if max_session_len is None:
        max_session_len = block_size * 4
    cache_config = CacheConfig(
        max_batches=max_batches,
        block_size=block_size,
        num_cpu_blocks=0,
        num_gpu_blocks=num_gpu_blocks,
        enable_prefix_caching=enable_prefix_caching,
        max_prefill_token_num=max_prefill_token_num,
    )
    scheduler_config = SchedulerConfig(
        max_batches=max_batches,
        max_session_len=max_session_len,
        max_request_output_len=8,
        eviction_type='recompute',
    )
    return Scheduler(
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        seq_meta=SequenceMeta(block_size, strategy=ARSequenceStrategy()),
        kv_connector=connector,
    )


def test_paging_scheduler_remote_load_blocks_forward_until_all_tp_completion():
    connector = _LoadConnector(remote_token_len=8)
    scheduler = _paging_scheduler(connector)
    seq = scheduler.add_session(7).add_sequence(np.arange(9, dtype=np.int64))

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert seq.num_history_ids == 0
    # Two load destinations plus one reserved tail-forward block.
    assert len(seq.logical_blocks) == 3
    pending = next(iter(scheduler._pending_kv_loads.values()))
    assert len(pending.logical_block_ids) == 2
    assert np.array_equal(
        scheduler.block_manager.allocator.get_ref_count(pending.logical_block_ids),
        np.array([2, 2]),
    )

    metadata = scheduler.build_kv_connector_progress_metadata()
    assert metadata.load_requests[0].block_ids == tuple(
        scheduler.block_manager.resolve_gpu_block_offsets(pending.logical_block_ids))
    scheduler.rollback_kv_connector_metadata(metadata)
    retry_metadata = scheduler.build_kv_connector_progress_metadata()
    assert retry_metadata.load_requests[0].load_id == metadata.load_requests[0].load_id
    metadata = retry_metadata
    scheduler.mark_kv_connector_metadata_dispatched(metadata)
    assert scheduler.build_kv_connector_progress_metadata() is None

    scheduler.update_connector_output({
        'completed_load_ids': {pending.load_id},
        'failed_load_ids': set(),
    })

    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 8
    assert len(seq.logical_blocks) == 3
    assert seq.seq_id in scheduler._remote_prefill_reservations
    assert not scheduler.has_pending_kv_transfer_work()
    output = scheduler.schedule(is_prefill=True)
    assert output.running == [seq]
    assert seq.status == MessageStatus.READY
    assert seq.seq_id not in scheduler._remote_prefill_reservations


def test_paging_scheduler_failed_remote_load_truncates_and_falls_back_once():
    connector = _LoadConnector(remote_token_len=8)
    scheduler = _paging_scheduler(connector)
    seq = scheduler.add_session(8).add_sequence(np.arange(9, dtype=np.int64))
    scheduler.schedule(is_prefill=True)
    pending = next(iter(scheduler._pending_kv_loads.values()))
    metadata = scheduler.build_kv_connector_progress_metadata()
    scheduler.mark_kv_connector_metadata_dispatched(metadata)

    scheduler.update_connector_output({
        'completed_load_ids': {pending.load_id},
        'failed_load_ids': {pending.load_id},
    })

    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 0
    assert len(seq.logical_blocks) == 0
    assert scheduler._remote_prefill_reservations == {}
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4
    output = scheduler.schedule(is_prefill=True)
    assert output.running == [seq]


def test_completed_remote_load_is_admitted_before_new_waiter_can_evict_it():
    connector = _LoadConnector(remote_token_len=8)
    scheduler = _paging_scheduler(
        connector,
        num_gpu_blocks=5,
        max_batches=1,
    )
    loaded = scheduler.add_session(30).add_sequence(
        np.arange(17, dtype=np.int64))

    assert scheduler.schedule(is_prefill=True).running == []
    pending = next(iter(scheduler._pending_kv_loads.values()))
    assert pending.prefill_reservation_blocks == 2
    scheduler.mark_kv_connector_metadata_dispatched(
        scheduler.build_kv_connector_progress_metadata())
    scheduler.update_connector_output({
        'completed_load_ids': {pending.load_id},
        'failed_load_ids': set(),
    })
    assert loaded.status == MessageStatus.WAITING
    assert loaded.seq_id in scheduler._remote_prefill_reservations
    lookup_count = len(connector.lookup_calls)

    # Make a smaller new request look older. Without the completed-load lane,
    # it is selected first and frees ``loaded`` to make room for another load.
    newcomer = scheduler.add_session(31).add_sequence(
        np.arange(9, dtype=np.int64))
    newcomer.arrive_time = loaded.arrive_time - 1.0
    preempted_req_ids = []
    original_preempt = scheduler.mark_kv_connector_preempted

    def _record_preempt(seq):
        preempted_req_ids.append(int(seq.seq_id))
        return original_preempt(seq)

    scheduler.mark_kv_connector_preempted = _record_preempt

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [loaded]
    assert loaded.status == MessageStatus.READY
    assert newcomer.status == MessageStatus.WAITING
    assert loaded.num_history_ids == 8
    assert loaded.seq_id not in scheduler._remote_prefill_reservations
    # First admission consumes the established result without another lookup.
    assert len(connector.lookup_calls) == lookup_count
    assert connector.next_load_id == 1
    assert loaded.seq_id not in preempted_req_ids


def test_remote_loads_started_in_one_schedule_consume_batch_slots():
    connector = _LoadConnector(remote_token_len=8)
    scheduler = _paging_scheduler(
        connector,
        num_gpu_blocks=20,
        max_batches=2,
    )
    seqs = [
        scheduler.add_session(40 + index).add_sequence(
            np.arange(9, dtype=np.int64))
        for index in range(4)
    ]

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert connector.next_load_id == 2
    assert scheduler.num_remote_loading() == 2
    assert scheduler.num_waiting() == 2
    assert [seq.status for seq in seqs].count(
        MessageStatus.WAITING_FOR_REMOTE_KVS) == 2


def test_remote_load_admission_reserves_its_own_and_prior_prefill_suffixes():
    connector = _LoadConnector(remote_token_len=8)
    scheduler = _paging_scheduler(
        connector,
        num_gpu_blocks=6,
        max_batches=4,
    )
    first = scheduler.add_session(50).add_sequence(
        np.arange(17, dtype=np.int64))
    second = scheduler.add_session(51).add_sequence(
        np.arange(9, dtype=np.int64))

    assert scheduler.schedule(is_prefill=True).running == []

    # The first load occupies three rows and reserves two for its 9-token
    # suffix. The three actually-free rows could fit the second load by itself,
    # but cannot fit it while honoring the first request's reservation.
    assert first.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert second.status == MessageStatus.WAITING
    assert connector.next_load_id == 1
    pending = next(iter(scheduler._pending_kv_loads.values()))
    assert pending.prefill_reservation_blocks == 2
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 3


def test_full_isl_reservation_throttles_two_remote_multichunk_prefills():
    connector = _LoadConnector(remote_token_len=16)
    scheduler = _paging_scheduler(
        connector,
        num_gpu_blocks=12,
        max_batches=2,
        max_prefill_token_num=8,
        max_session_len=64,
    )
    first = scheduler.add_session(70).add_sequence(
        np.arange(33, dtype=np.int64))
    second = scheduler.add_session(71).add_sequence(
        np.arange(33, dtype=np.int64))

    assert scheduler.schedule(is_prefill=True).running == []

    # A load owns five rows (four remote blocks plus the safety tail), but
    # needs nine rows to finish all input chunks.  Only the first load may
    # enter; admitting two based on next-chunk headroom wedges after chunk 1.
    assert first.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert second.status == MessageStatus.WAITING
    assert connector.next_load_id == 1
    pending = next(iter(scheduler._pending_kv_loads.values()))
    assert pending.prefill_target_blocks == 9
    assert pending.prefill_reservation_blocks == 4
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 7

    scheduler.mark_kv_connector_metadata_dispatched(
        scheduler.build_kv_connector_progress_metadata())
    scheduler.update_connector_output({
        'completed_load_ids': {pending.load_id},
        'failed_load_ids': set(),
    })
    assert scheduler.schedule(is_prefill=True).running == [first]
    assert first.seq_id not in scheduler._remote_prefill_reservations
    assert scheduler._remote_prefill_reserved_blocks() == 3

    # The full-ISL target remains live and shrinks after each successful
    # chunk. It is not consumed merely because the loaded request ran once.
    first.state.activate()
    first.set_step(24)
    assert scheduler.reserve_long_context_chunk(first, 8)
    assert scheduler._remote_prefill_reserved_blocks() == 1
    first.set_step(32)
    assert scheduler.reserve_long_context_chunk(
        first, 1, prealloc_size=1, is_last_chunk=True)
    assert first.seq_id in scheduler._prefill_reservation_targets
    assert scheduler._remote_prefill_reserved_blocks() == 0

    # Final allocation is not final completion. Only the observed model
    # output boundary releases the reservation and lets request 2 load.
    first.set_step(33)
    assert scheduler.schedule_running(
        [first], num_required_tokens=0, prealloc_size=0) == [True]
    assert first.seq_id not in scheduler._prefill_reservation_targets
    scheduler.end_session(first.session_id)

    assert scheduler.schedule(is_prefill=True).running == []
    assert second.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert connector.next_load_id == 2


def test_new_multiturn_prefill_refreshes_stale_full_isl_reservation():
    connector = _LoadConnector(remote_token_len=0)
    scheduler = _paging_scheduler(
        connector,
        num_gpu_blocks=12,
        max_batches=2,
        max_prefill_token_num=8,
        max_session_len=64,
    )
    first = scheduler.add_session(76).add_sequence(
        np.arange(17, dtype=np.int64))

    # Complete a local multi-chunk turn without another scheduler output in
    # between its final model output and the next user turn.  The old absolute
    # target is intentionally still present at this point.
    assert scheduler.schedule(is_prefill=True).running == [first]
    first.state.activate()
    first.set_step(8)
    assert scheduler.reserve_long_context_chunk(first, 8)
    first.set_step(16)
    assert scheduler.reserve_long_context_chunk(
        first, 1, prealloc_size=1, is_last_chunk=True)
    first.update_token_ids(
        np.array([900], dtype=np.int64), mode=UpdateTokenMode.PREFILL)
    first.state.finish()
    assert scheduler._prefill_reservation_targets[first.seq_id][1] == 5

    # A second request can begin a remote load while the completed turn's
    # zero-sized stale reservation remains in the map.
    connector.remote_token_len = 8
    second = scheduler.add_session(77).add_sequence(
        np.arange(17, dtype=np.int64))
    assert scheduler.schedule(is_prefill=True).running == []
    pending = next(iter(scheduler._pending_kv_loads.values()))
    assert pending.req_id == second.seq_id
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4

    # Append a new long turn to the same sequence immediately.  Admission
    # must recompute the current full-ISL target instead of treating the old
    # map membership as proof that the longer turn is already reserved.
    first.update_token_ids(
        np.arange(16, dtype=np.int64), mode=UpdateTokenMode.INPUTS)
    first.state.activate()
    assert scheduler.schedule(is_prefill=True).running == []
    assert first.status == MessageStatus.WAITING
    assert first.num_blocks == 5
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4
    assert pending.prefill_reservation_blocks == 2

    # The pending load can still complete and make forward progress; neither
    # request wedges after consuming only its first chunk.
    scheduler.mark_kv_connector_metadata_dispatched(
        scheduler.build_kv_connector_progress_metadata())
    scheduler.update_connector_output({
        'completed_load_ids': {pending.load_id},
        'failed_load_ids': set(),
    })
    assert scheduler.schedule(is_prefill=True).running == [second]
    assert second.num_blocks == 4
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 3
    assert scheduler._remote_prefill_reserved_blocks() == 1


def test_active_local_long_prefill_reservation_blocks_remote_load():
    connector = _LoadConnector(remote_token_len=0)
    scheduler = _paging_scheduler(
        connector,
        num_gpu_blocks=6,
        max_batches=2,
        max_prefill_token_num=8,
        max_session_len=32,
    )
    local = scheduler.add_session(72).add_sequence(
        np.arange(17, dtype=np.int64))

    assert scheduler.schedule(is_prefill=True).running == [local]
    assert local.num_blocks == 2
    assert scheduler._remote_prefill_reserved_blocks() == 3
    local.state.activate()
    local.set_step(8)

    connector.remote_token_len = 8
    remote = scheduler.add_session(73).add_sequence(
        np.arange(17, dtype=np.int64))
    assert scheduler.schedule(is_prefill=True).running == []

    # Four rows are physically free and could hold the remote destination,
    # but three belong to the active local request's later input chunks.
    assert remote.status == MessageStatus.WAITING
    assert connector.next_load_id == 0
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4


def test_running_growth_cannot_spend_pending_remote_prefill_reservation():
    connector = _LoadConnector(remote_token_len=0)
    scheduler = _paging_scheduler(
        connector,
        num_gpu_blocks=10,
        max_batches=2,
        max_prefill_token_num=8,
        max_session_len=64,
    )
    running = scheduler.add_session(74).add_sequence(
        np.arange(4, dtype=np.int64))
    assert scheduler.schedule(is_prefill=True).running == [running]
    running.state.activate()

    connector.remote_token_len = 16
    loading = scheduler.add_session(75).add_sequence(
        np.arange(33, dtype=np.int64))
    assert scheduler.schedule(is_prefill=True).running == []
    pending = next(iter(scheduler._pending_kv_loads.values()))
    assert pending.req_id == loading.seq_id
    assert pending.prefill_reservation_blocks == 4
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4

    # Growing the existing request by one row would consume the pending
    # request's four-row continuation guarantee, so it is preempted instead.
    assert scheduler.schedule_running(
        [running], num_required_tokens=1, prealloc_size=1) == [False]
    assert running.status == MessageStatus.WAITING
    assert loading.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert pending.prefill_reservation_blocks == 4


@pytest.mark.parametrize('num_gpu_blocks', [3, 4])
def test_remote_load_requires_own_first_prefill_headroom(num_gpu_blocks):
    connector = _LoadConnector(remote_token_len=8)
    scheduler = _paging_scheduler(
        connector,
        num_gpu_blocks=num_gpu_blocks,
        max_batches=1,
    )
    seq = scheduler.add_session(55).add_sequence(
        np.arange(17, dtype=np.int64))

    output = scheduler.schedule(is_prefill=True)

    # Loading the 8-token prefix plus its one-token safety tail needs three
    # rows, but the first real prefill needs five in total. Do not start a
    # transfer that can never progress after completion.
    assert output.running == []
    assert seq.status == MessageStatus.WAITING
    assert len(seq.logical_blocks) == 0
    assert connector.next_load_id == 0
    assert scheduler._pending_kv_loads == {}
    assert scheduler._remote_prefill_reservations == {}
    assert scheduler.block_manager.get_num_free_gpu_blocks() == num_gpu_blocks


def test_completed_remote_load_zero_reservation_is_still_cleared_on_stop():
    connector = _LoadConnector(remote_token_len=8)
    scheduler = _paging_scheduler(connector)
    seq = scheduler.add_session(60).add_sequence(
        np.arange(9, dtype=np.int64))
    scheduler.schedule(is_prefill=True)
    pending = next(iter(scheduler._pending_kv_loads.values()))
    scheduler.mark_kv_connector_metadata_dispatched(
        scheduler.build_kv_connector_progress_metadata())
    scheduler.update_connector_output({
        'completed_load_ids': {pending.load_id},
        'failed_load_ids': set(),
    })

    assert scheduler._remote_prefill_reservations == {seq.seq_id: 0}
    scheduler.stop_session(seq.session_id)
    assert scheduler._remote_prefill_reservations == {}


def test_paging_scheduler_cancelled_inflight_load_keeps_tombstone_pin():
    connector = _LoadConnector(remote_token_len=8)
    scheduler = _paging_scheduler(connector)
    seq = scheduler.add_session(9).add_sequence(np.arange(9, dtype=np.int64))
    scheduler.schedule(is_prefill=True)
    pending = next(iter(scheduler._pending_kv_loads.values()))
    pinned_ids = pending.logical_block_ids.copy()
    metadata = scheduler.build_kv_connector_progress_metadata()

    # Cancel exactly in the build -> awaited RPC -> commit window.  The
    # request reference is detached, but the dispatching lease must keep the
    # destination alive because the worker may already have received metadata.
    seq.state.stop()

    assert seq.status == MessageStatus.STOPPED
    assert len(seq.logical_blocks) == 0
    assert np.array_equal(
        scheduler.block_manager.allocator.get_ref_count(pinned_ids),
        np.array([1, 1]),
    )
    # Treat an RPC exception as partial delivery: rank 0 may already be
    # writing while rank 1 failed. Keep the pin and retry the same idempotent
    # load ID to establish an all-rank completion fence.
    scheduler.rollback_kv_connector_metadata(metadata)
    assert np.array_equal(
        scheduler.block_manager.allocator.get_ref_count(pinned_ids),
        np.array([1, 1]),
    )
    retry_metadata = scheduler.build_kv_connector_progress_metadata()
    assert retry_metadata.load_requests[0].load_id == pending.load_id
    scheduler.mark_kv_connector_metadata_dispatched(retry_metadata)
    scheduler.update_connector_output({
        'completed_load_ids': {pending.load_id},
        'failed_load_ids': set(),
    })
    assert np.array_equal(
        scheduler.block_manager.allocator.get_ref_count(pinned_ids),
        np.array([0, 0]),
    )


def test_paging_scheduler_remote_load_reserves_tail_forward_block():
    connector = _LoadConnector(remote_token_len=8)
    scheduler = _paging_scheduler(connector, num_gpu_blocks=2)
    seq = scheduler.add_session(10).add_sequence(np.arange(9, dtype=np.int64))

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING
    assert len(seq.logical_blocks) == 0
    assert scheduler._pending_kv_loads == {}


def test_paging_scheduler_skips_remote_load_for_routed_expert_outputs():
    connector = _LoadConnector(remote_token_len=8)
    scheduler = _paging_scheduler(connector)
    seq = scheduler.add_session(11).add_sequence(np.arange(9, dtype=np.int64))
    seq.sampling_param.return_routed_experts = True

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [seq]
    assert connector.lookup_calls == []
    assert scheduler._pending_kv_loads == {}


def test_paging_scheduler_rolls_back_tentative_l1_while_lookup_is_pending():

    class _PendingOnceLoadConnector(_LoadConnector):

        def __init__(self):
            super().__init__(remote_token_len=8)
            self.pending_once = True

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            self.lookup_calls.append((request.seq_id, num_computed_tokens))
            if self.pending_once:
                self.pending_once = False
                return None, False
            return self.remote_token_len - num_computed_tokens, True

    scheduler = _paging_scheduler(
        None,
        num_gpu_blocks=5,
        enable_prefix_caching=True,
    )
    cached = scheduler.add_session(12).add_sequence(
        np.array([1, 1, 1, 1, 9], dtype=np.int64))
    scheduler.schedule(is_prefill=True)
    cached.state.stop()

    connector = _PendingOnceLoadConnector()
    scheduler.kv_connector = connector
    seq = scheduler.add_session(13).add_sequence(
        np.array([1, 1, 1, 1, 2, 2, 2, 2, 3], dtype=np.int64))
    scheduler.block_trie.stats.reset()

    first = scheduler.schedule(is_prefill=True)

    assert first.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 0
    assert len(seq.logical_blocks) == 0
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0

    second = scheduler.schedule(is_prefill=True)

    assert second.running == []
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert connector.lookup_calls == [(seq.seq_id, 4), (seq.seq_id, 4)]


@pytest.mark.parametrize('failed', [False, True])
def test_paging_scheduler_unaligned_multiturn_load_preserves_partial_l1(failed):

    class _PendingOnceLoadConnector(_LoadConnector):

        def __init__(self):
            super().__init__(remote_token_len=128)
            self.pending_once = True

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            self.lookup_calls.append((request.seq_id, num_computed_tokens))
            if self.pending_once:
                self.pending_once = False
                return None, False
            return self.remote_token_len - num_computed_tokens, True

    scheduler = _paging_scheduler(
        None,
        num_gpu_blocks=10,
        enable_prefix_caching=True,
        block_size=64,
    )

    # Publish both complete blocks, including the block [64, 128) which must
    # not be appended behind the preserved request's old partial block.
    cached = scheduler.add_session(20).add_sequence(
        np.arange(129, dtype=np.int64))
    assert scheduler.schedule(is_prefill=True).running == [cached]
    cached.state.stop()

    seq = scheduler.add_session(21).add_sequence(
        np.arange(70, dtype=np.int64),
        preserve_cache=True,
    )
    assert scheduler.schedule(is_prefill=True).running == [seq]
    assert seq.prefix_cache.trie_cursor.prefix_len == 64
    seq.state.stop()
    seq.set_step(70)
    seq.update_token_ids(np.arange(70, 129, dtype=np.int64))
    seq.state.activate()
    baseline_ids = seq.logical_blocks.get_real_blocks().copy()
    partial_id = int(baseline_ids[1])
    baseline_refs = scheduler.block_manager.allocator.get_ref_count(
        baseline_ids).copy()

    connector = _PendingOnceLoadConnector()
    scheduler.kv_connector = connector
    first = scheduler.schedule(is_prefill=True)

    assert first.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 70
    assert np.array_equal(seq.logical_blocks.get_real_blocks(), baseline_ids)
    assert np.array_equal(
        scheduler.block_manager.allocator.get_ref_count(baseline_ids),
        baseline_refs,
    )

    second = scheduler.schedule(is_prefill=True)
    assert second.running == []
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert connector.lookup_calls == [(seq.seq_id, 64), (seq.seq_id, 64)]
    pending = next(iter(scheduler._pending_kv_loads.values()))
    assert pending.local_token_len == 64
    assert pending.remote_token_len == 128
    assert pending.fallback_step == 70
    assert tuple(pending.fallback_logical_block_ids) == (partial_id, )
    assert int(seq.logical_blocks[1]) != partial_id

    metadata = scheduler.build_kv_connector_progress_metadata()
    scheduler.mark_kv_connector_metadata_dispatched(metadata)
    scheduler.update_connector_output({
        'completed_load_ids': {pending.load_id},
        'failed_load_ids': {pending.load_id} if failed else set(),
    })

    assert seq.status == MessageStatus.WAITING
    if failed:
        assert seq.num_history_ids == 70
        assert int(seq.logical_blocks[1]) == partial_id
        assert scheduler.block_manager.allocator.get_ref_count(
            np.array([partial_id]))[0] > 0
    else:
        assert seq.num_history_ids == 128
        assert int(seq.logical_blocks[1]) != partial_id
        assert scheduler.block_manager.allocator.get_ref_count(
            np.array([partial_id]))[0] == 0


def test_paging_scheduler_aligned_step_keeps_private_preallocated_tail():

    class _PendingLookupConnector(_LoadConnector):

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            self.lookup_calls.append((request.seq_id, num_computed_tokens))
            return None, False

    scheduler = _paging_scheduler(
        None,
        num_gpu_blocks=8,
        enable_prefix_caching=True,
        block_size=64,
    )
    cached = scheduler.add_session(23).add_sequence(
        np.arange(129, dtype=np.int64))
    assert scheduler.schedule(is_prefill=True).running == [cached]
    cached.state.stop()

    seq = scheduler.add_session(24).add_sequence(
        np.arange(65, dtype=np.int64),
        preserve_cache=True,
    )
    assert scheduler.schedule(is_prefill=True).running == [seq]
    seq.state.stop()
    seq.set_step(64)
    seq.update_token_ids(np.arange(65, 129, dtype=np.int64))
    seq.state.activate()
    baseline_ids = seq.logical_blocks.get_real_blocks().copy()
    assert len(baseline_ids) == 2

    connector = _PendingLookupConnector(remote_token_len=128)
    scheduler.kv_connector = connector
    assert scheduler.schedule(is_prefill=True).running == []

    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 64
    assert np.array_equal(seq.logical_blocks.get_real_blocks(), baseline_ids)
    assert connector.lookup_calls == [(seq.seq_id, 64)]


def test_paging_scheduler_stop_discards_pending_lookup_and_resume_requeries():

    class _PendingLookupConnector(_LoadConnector):

        def __init__(self):
            super().__init__()
            self.lookup_pending = False
            self.lookup_payloads = []
            self.cancelled_lookups = []

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            self.lookup_payloads.append(tuple(int(token) for token in request.all_ids))
            self.lookup_pending = True
            return None, False

        def cancel_lookup(self, request_id):
            self.cancelled_lookups.append(int(request_id))
            self.lookup_pending = False

        def has_pending_kv_lookup_work(self):
            return self.lookup_pending

    connector = _PendingLookupConnector()
    scheduler = _paging_scheduler(connector)
    session = scheduler.add_session(22)
    seq = session.add_sequence(np.arange(9, dtype=np.int64))

    assert scheduler.schedule(is_prefill=True).running == []
    assert scheduler.has_pending_kv_lookup_work()
    scheduler.stop_session(session.session_id)

    assert seq.status == MessageStatus.STOPPED
    assert connector.cancelled_lookups == [seq.seq_id]
    assert not scheduler.has_pending_kv_lookup_work()

    seq.update_token_ids(np.array([99], dtype=np.int64))
    seq.state.activate()
    assert scheduler.schedule(is_prefill=True).running == []
    assert scheduler.has_pending_kv_lookup_work()
    assert len(connector.lookup_payloads) == 2
    assert connector.lookup_payloads[0] != connector.lookup_payloads[1]


def test_paging_scheduler_pins_until_all_tp_completion_and_rolls_back_dispatch():
    connector = _PinConnector()
    scheduler = _paging_scheduler(connector)
    seq = scheduler.add_session(1).add_sequence(np.arange(8, dtype=np.int64))
    scheduler.block_manager.allocate(seq)
    logical_ids = seq.logical_blocks.get_real_blocks().copy()
    allocator = scheduler.block_manager.allocator

    metadata = scheduler.build_kv_connector_metadata([seq], (8, ))

    assert np.array_equal(allocator.get_ref_count(logical_ids), np.array([2, 2]))
    assert scheduler.has_pending_kv_connector_work()

    scheduler.block_manager.free(seq)
    assert np.array_equal(allocator.get_ref_count(logical_ids), np.array([1, 1]))
    scheduler.update_connector_output({metadata.save_requests[0].save_id})
    assert np.array_equal(allocator.get_ref_count(logical_ids), np.array([0, 0]))
    assert not scheduler.has_pending_kv_connector_work()

    retry_seq = scheduler.sessions[1].add_sequence(np.arange(4, dtype=np.int64))
    scheduler.block_manager.allocate(retry_seq)
    retry_logical_ids = retry_seq.logical_blocks.get_real_blocks().copy()
    retry_metadata = scheduler.build_kv_connector_metadata([retry_seq], (4, ))
    scheduler.rollback_kv_connector_metadata(retry_metadata)

    assert np.array_equal(allocator.get_ref_count(retry_logical_ids), np.array([1]))
    assert connector.updates[-1] == {
        'rolled_back_save_ids': {retry_metadata.save_requests[0].save_id}
    }
    scheduler.block_manager.free(retry_seq)


def test_paging_scheduler_eviction_kv_roundtrip(mooncake_scheduler):
    """Saved KV survives eviction and restores the same request generation."""
    stored_hashes = set()

    def lookup(_req_id, token_len, block_hashes, non_block):
        assert non_block
        matched = 0
        for block_hash in block_hashes:
            if block_hash not in stored_hashes:
                break
            matched += mooncake_scheduler._cache_config.block_size
        return min(matched, token_len)

    mooncake_scheduler.client.lookup = lookup
    scheduler = _paging_scheduler(
        mooncake_scheduler,
        num_gpu_blocks=3,
        max_session_len=16,
    )
    seq = scheduler.add_session(4).add_sequence(np.arange(9, dtype=np.int64))

    first = scheduler.schedule(is_prefill=True)
    assert first.running == [seq]
    save_metadata = scheduler.build_kv_connector_metadata([seq], (8, ))
    save_request = save_metadata.save_requests[0]
    assert save_request.generation == 0
    stored_hashes.update(save_request.block_hashes)
    scheduler.update_connector_output(
        KVConnectorOutput(completed_save_ids={save_request.save_id}))
    assert not scheduler.has_pending_kv_transfer_work()

    seq.state.evict()
    pressure = scheduler.add_session(5).add_sequence(
        np.arange(9, dtype=np.int64))
    assert scheduler.eviction_helper.evict_for_seq(pressure, [seq], 0)
    pressure.session.remove_sequence(pressure)
    assert seq.status == MessageStatus.WAITING
    assert len(seq.logical_blocks) == 0
    assert scheduler._kv_seq_generations[seq.seq_id] == 1

    loading = scheduler.schedule(is_prefill=True)
    assert loading.running == []
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    load_metadata = scheduler.build_kv_connector_progress_metadata()
    load_request = load_metadata.load_requests[0]
    assert load_request.generation == 1
    assert load_request.remote_token_len == 8
    assert load_request.block_hashes == save_request.block_hashes
    scheduler.mark_kv_connector_metadata_dispatched(load_metadata)
    scheduler.update_connector_output(
        KVConnectorOutput(completed_load_ids={load_request.load_id}))

    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 8
    assert not scheduler.has_pending_kv_transfer_work()
    resumed = scheduler.schedule(is_prefill=True)
    assert resumed.running == [seq]
    assert seq.status == MessageStatus.READY


def test_paging_scheduler_notifies_connector_before_end_session_free():
    connector = _PinConnector()
    scheduler = _paging_scheduler(connector)
    seq = scheduler.add_session(3).add_sequence(np.arange(4, dtype=np.int64))
    scheduler.block_manager.allocate(seq)
    expected_block_ids = tuple(
        int(block_id)
        for block_id in scheduler.block_manager.get_block_table(seq)
    )

    scheduler.end_session(3)

    assert connector.finished == [(seq, expected_block_ids)]
    assert len(seq.logical_blocks) == 0


def test_paging_scheduler_preemption_isolates_generation_without_cancelling_pinned_job():
    connector = _PinConnector()
    scheduler = _paging_scheduler(connector)
    seq = scheduler.add_session(5).add_sequence(np.arange(4, dtype=np.int64))
    scheduler.block_manager.allocate(seq)
    logical_ids = seq.logical_blocks.get_real_blocks().copy()

    old_metadata = scheduler.build_kv_connector_metadata([seq], (4, ))
    scheduler.mark_kv_connector_preempted(seq)
    resumed_metadata = scheduler.build_kv_connector_metadata([seq], (4, ))

    assert old_metadata.save_requests[0].generation == 0
    assert resumed_metadata.save_requests[0].generation == 1
    assert old_metadata.preempted_save_ids == ()
    assert resumed_metadata.preempted_save_ids == ()
    assert np.array_equal(
        scheduler.block_manager.allocator.get_ref_count(logical_ids),
        np.array([3]),
    )

    scheduler.update_connector_output({
        old_metadata.save_requests[0].save_id,
        resumed_metadata.save_requests[0].save_id,
    })
    assert np.array_equal(
        scheduler.block_manager.allocator.get_ref_count(logical_ids),
        np.array([1]),
    )
    scheduler.block_manager.free(seq)


def test_decode_eviction_advances_connector_generation_once_per_tick():
    """The decode and deferred-eviction layers represent one preemption."""
    connector = _PinConnector()
    scheduler = _paging_scheduler(
        connector,
        num_gpu_blocks=1,
        max_session_len=8,
    )
    seq = scheduler.add_session(6).add_sequence(np.arange(4, dtype=np.int64))
    scheduler.block_manager.allocate(seq)
    scheduler.build_kv_connector_metadata([seq], (4, ))
    seq.state.activate()
    scheduler.activate_seqs([seq])

    valid_mask = scheduler.schedule_running(
        [seq],
        num_required_tokens=1,
        prealloc_size=1,
    )
    assert valid_mask == [False]
    assert seq.status == MessageStatus.WAITING

    # InputsMaker later revisits the same invalid sequence through
    # deactivate_evict_seqs().  It must not create another generation.
    scheduler.deactivate_seqs([seq])
    scheduler.evict_seqs([seq])
    assert scheduler._kv_seq_generations[seq.seq_id] == 1

    resumed = scheduler.build_kv_connector_metadata([seq], (4, ))
    assert resumed.save_requests[0].generation == 1

    # A later forward tick starts a genuinely new preemption generation.
    scheduler.tick()
    scheduler.mark_kv_connector_preempted(seq)
    assert scheduler._kv_seq_generations[seq.seq_id] == 2


@pytest.mark.parametrize('is_chunk', [False, True])
def test_forward_payload_appends_prefill_connector_metadata(is_chunk):
    calls = []
    metadata = object()

    class _Scheduler:

        kv_connector = object()

        def build_kv_connector_metadata(self, running, token_lens):
            calls.append((running, token_lens))
            return metadata

    seq = SimpleNamespace(
        return_logits=False,
        return_routed_experts=False,
        return_ce_loss=False,
    )
    inputs = SimpleNamespace(
        is_decoding=False,
        is_chunk=is_chunk,
        history_lengths=torch.tensor([4]),
        seq_length=torch.tensor([6]),
    )
    maker = SimpleNamespace(
        scheduler=_Scheduler(),
        spec_decoding=False,
        sampling_strategy=SimpleNamespace(make_sampling_inputs=lambda _running: None),
        model_agent_strategy=SimpleNamespace(make_stopping_criteria=lambda _running: None),
    )
    task = _ForwardInputsTask.__new__(_ForwardInputsTask)
    task.maker = maker
    task.scheduler = maker.scheduler
    task.result = _ForwardInputsResult(running=[seq], inputs=inputs)

    payload = task._build_payload()

    assert calls == [([seq], (10, ))]
    assert list(payload)[-1] == 'kv_connector_metadata'
    assert payload['kv_connector_metadata'] is metadata
