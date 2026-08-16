# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.engine.inputs_maker import _ForwardInputsResult, _ForwardInputsTask
from lmdeploy.pytorch.kv_connector.mooncake.store import scheduler as scheduler_module
from lmdeploy.pytorch.kv_connector.mooncake.store.data import (
    MOONCAKE_BLOCK_HASH_BYTES,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreSaveRequest,
    build_prefix_block_hashes,
)
from lmdeploy.pytorch.kv_connector.mooncake.store.scheduler import MooncakeStoreScheduler
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging.scheduler import Scheduler, SchedulerOutput
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


class _FakeLookupClient:

    def __init__(self, _cache_config):
        self.discarded = []
        self.closed = False

    def discard(self, req_id):
        self.discarded.append(req_id)

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


class _PinConnector:

    def __init__(self):
        self.next_save_id = 0
        self.outputs = []
        self.updates = []
        self.finished = []

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


def _paging_scheduler(connector):
    cache_config = CacheConfig(
        max_batches=1,
        block_size=4,
        num_cpu_blocks=0,
        num_gpu_blocks=4,
    )
    scheduler_config = SchedulerConfig(
        max_batches=1,
        max_session_len=32,
        max_request_output_len=8,
        eviction_type='recompute',
    )
    return Scheduler(
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        seq_meta=SequenceMeta(4, strategy=ARSequenceStrategy()),
        kv_connector=connector,
    )


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


@pytest.mark.parametrize('is_chunk', [False, True])
def test_forward_payload_appends_prefill_connector_metadata(is_chunk):
    calls = []
    metadata = object()

    class _Scheduler:

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
