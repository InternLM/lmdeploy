# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.kv_connector import (
    KVConnectorMetadata,
    KVConnectorOutput,
    KVConnectorResult,
    KVLoadResult,
    KVSaveBlockLease,
)
from lmdeploy.pytorch.kv_connector.mooncake.store.scheduler import MooncakeStoreScheduler
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta
from lmdeploy.pytorch.paging.scheduler import Scheduler


class _AsyncLookupConnector:

    def __init__(self, results, failed_ids=()):
        self.results = iter(results)
        self.failed_ids = set(failed_ids)
        self.pending_ids = set()
        self.lookup_calls = []
        self.cancelled = []
        self.finished = []
        self.allocations = []

    def on_new_request(self, request):
        pass

    def is_lookup_pending(self, request_id):
        return request_id in self.pending_ids

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self.lookup_calls.append((request.seq_id, num_computed_tokens))
        result = next(self.results)
        if result[0] is None:
            self.pending_ids.add(request.seq_id)
        else:
            self.pending_ids.discard(request.seq_id)
        return result

    def cancel_lookup(self, request_id):
        self.pending_ids.discard(request_id)
        self.cancelled.append(request_id)

    def update_state_after_alloc(self, request, block_ids, num_external_tokens):
        self.allocations.append((request.seq_id, tuple(block_ids), num_external_tokens))

    def build_connector_meta(self, scheduler_output):
        return None

    def update_connector_output(self, connector_output):
        return KVConnectorResult(
            load_results=tuple(
                KVLoadResult(
                    request_id=request_id,
                    success=request_id not in self.failed_ids,
                )
                for request_id in (connector_output.finished_receiving or set())
            )
        )

    def request_finished(self, request):
        self.finished.append(request.seq_id)

    def finish_transfers_after_worker_drain(self):
        pass

    def shutdown(self):
        pass


def _make_async_lookup_scheduler(
    connector,
    *,
    enable_prefix_caching=True,
    max_batches=1,
    num_gpu_blocks=16,
    max_prefill_token_num=8192,
):
    from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
    block_size = 4
    return Scheduler(
        scheduler_config=SchedulerConfig(
            max_batches=max_batches,
            max_session_len=64,
            max_request_output_len=16,
            eviction_type='recompute',
        ),
        cache_config=CacheConfig(
            max_batches=max_batches,
            block_size=block_size,
            num_cpu_blocks=0,
            num_gpu_blocks=num_gpu_blocks,
            max_prefill_token_num=max_prefill_token_num,
            enable_prefix_caching=enable_prefix_caching,
            kv_transfer_config=KVTransferConfig(
                kv_connector='MooncakeStoreConnector',
                kv_role='kv_both',
            ),
        ),
        seq_meta=SequenceMeta(block_size, strategy=ARSequenceStrategy()),
        kv_connector=connector,
    )


def test_async_lookup_pending_rolls_back_a_new_request_once():
    connector = _AsyncLookupConnector([(None, False), (0, False)])
    scheduler = _make_async_lookup_scheduler(connector)
    seq = scheduler.add_session(70).add_sequence(torch.arange(9))

    first = scheduler.schedule(is_prefill=True)
    assert first.running == []
    assert scheduler.last_schedule_had_pending_lookup
    assert seq.num_history_ids == 0
    assert seq.num_blocks == 0
    assert seq.prefix_cache.trie_cursor is None
    assert seq.prefix_cache.match_start_step == -1
    assert scheduler.block_trie.stats.num_query_tokens == 0

    second = scheduler.schedule(is_prefill=True)
    assert second.running == []
    assert connector.lookup_calls == [(seq.seq_id, 0)]
    assert scheduler.block_trie.stats.num_query_tokens == 0

    connector.pending_ids.clear()
    third = scheduler.schedule(is_prefill=True)
    assert third.running == [seq]
    assert connector.lookup_calls == [(seq.seq_id, 0), (seq.seq_id, 0)]


def test_async_lookup_rebases_remote_hit_after_local_trie_grows(monkeypatch):
    connector = MooncakeStoreScheduler(
        CacheConfig(
            max_batches=1,
            block_size=4,
            num_cpu_blocks=0,
            num_gpu_blocks=16,
            enable_prefix_caching=True,
            kv_transfer_config=KVTransferConfig(
                kv_connector='MooncakeStoreConnector',
                kv_role='kv_both',
            ),
        ))
    assert connector.client is not None
    # The same asynchronous lookup first reports pending, then returns its
    # absolute remote prefix boundary from the original snapshot.
    monkeypatch.setattr(connector.client, 'lookup', Mock(side_effect=(None, 16)))
    observed_results = []
    get_matched_tokens = connector.get_num_new_matched_tokens

    def _record_result(request, num_computed_tokens):
        result = get_matched_tokens(request, num_computed_tokens)
        observed_results.append((num_computed_tokens, result))
        return result

    monkeypatch.setattr(connector, 'get_num_new_matched_tokens', _record_result)
    scheduler = _make_async_lookup_scheduler(connector)
    tokens = torch.arange(17)

    cached_to_8 = scheduler.add_session(80).add_sequence(tokens[:9])
    scheduler.block_manager.allocate(cached_to_8)
    scheduler.block_trie.allocate(cached_to_8)
    cached_to_8.state.stop()

    seq = scheduler.add_session(81).add_sequence(tokens)
    first = scheduler.schedule(is_prefill=True)

    assert first.running == []
    assert seq.num_history_ids == 0
    assert observed_results == [(8, (None, False))]

    # While the remote lookup is pending, another sequence publishes the next
    # complete local block. The retried match must now advance from 8 to 12.
    cached_to_12 = scheduler.add_session(82).add_sequence(tokens[:13])
    scheduler.block_manager.allocate(cached_to_12)
    scheduler.block_trie.allocate(cached_to_12)
    cached_to_12.state.stop()

    second = scheduler.schedule(is_prefill=True)

    assert second.running == []
    assert seq.num_history_ids == 12
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert observed_results == [
        (8, (None, False)),
        (12, (4, True)),
    ]

    metadata = connector.build_connector_meta(second)
    assert metadata is not None
    assert len(metadata.load_requests) == 1
    load_request = metadata.load_requests[0]
    block_table = scheduler.block_manager.get_block_table(seq)
    assert load_request.block_ids == (int(block_table[3]), )
    assert load_request.remote_block_count == 4
    scheduler.shutdown()


def test_async_lookup_pending_request_does_not_block_later_waiter():
    connector = _AsyncLookupConnector([(None, False), (0, False)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
    )
    pending = scheduler.add_session(74).add_sequence(torch.arange(9))
    schedulable = scheduler.add_session(75).add_sequence(torch.arange(9, 18))

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [schedulable]
    assert pending.status == MessageStatus.WAITING
    assert schedulable.status == MessageStatus.READY
    assert pending.seq_id in connector.pending_ids
    assert connector.lookup_calls == [
        (pending.seq_id, 0),
        (schedulable.seq_id, 0),
    ]
    assert scheduler.last_schedule_had_pending_lookup


def test_async_lookup_precisely_restores_a_multiturn_local_prefix():
    connector = _AsyncLookupConnector([(None, False), (4, True)])
    scheduler = _make_async_lookup_scheduler(connector)
    tokens = torch.arange(13)
    seq = scheduler.add_session(71).add_sequence(tokens)

    seq.kv_token_limit = 4
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    seq.set_step(4)
    seq.kv_token_limit = 5
    seq.cached_tokens = 3
    seq.model_meta = {'state': 'keep'}
    seq.prefix_cache.recompute_overlap.fresh_block_range = range(0, 1)
    seq.prefix_cache.recompute_overlap.trie_block_map[0] = seq.logical_blocks[0]
    baseline_blocks = seq.logical_blocks.get_real_blocks().copy()
    baseline_cursor = seq.prefix_cache.trie_cursor

    cached = scheduler.add_session(72).add_sequence(tokens[:9])
    scheduler.block_manager.allocate(cached)
    scheduler.block_trie.allocate(cached)
    cached.state.stop()
    matched_block = cached.logical_blocks[1]
    matched_ref_count = scheduler.block_manager.allocator.get_ref_count(
        cached.logical_blocks.get_real_blocks()[1:2]).copy()
    scheduler.block_trie.stats.reset()

    first = scheduler.schedule(is_prefill=True)
    assert first.running == []
    assert scheduler.last_schedule_had_pending_lookup
    assert seq.num_history_ids == 4
    assert seq.num_blocks == 1
    assert torch.equal(torch.from_numpy(seq.logical_blocks.get_real_blocks()),
                       torch.from_numpy(baseline_blocks))
    assert seq.prefix_cache.trie_cursor is baseline_cursor
    assert seq.prefix_cache.match_start_step == -1
    assert seq.prefix_cache.recompute_overlap.fresh_block_range == range(0, 1)
    assert seq.prefix_cache.recompute_overlap.trie_block_map == {0: baseline_blocks[0]}
    assert seq.cached_tokens == 3
    assert seq.kv_token_limit == 5
    assert seq.model_meta == {'state': 'keep'}
    assert scheduler.block_manager.allocator.get_ref_count(
        cached.logical_blocks.get_real_blocks()[1:2]).tolist() == matched_ref_count.tolist()
    assert scheduler.block_trie.stats.num_query_tokens == 0

    second = scheduler.schedule(is_prefill=True)
    assert second.running == []
    assert connector.lookup_calls == [(seq.seq_id, 8)]
    assert seq.num_history_ids == 4
    assert scheduler.block_trie.stats.num_query_tokens == 0

    connector.pending_ids.clear()
    seq.kv_token_limit = None
    seq.prefix_cache.recompute_overlap.clear_tracking()
    third = scheduler.schedule(is_prefill=True)
    assert third.running == []
    assert connector.lookup_calls == [(seq.seq_id, 8), (seq.seq_id, 8)]
    assert seq.num_history_ids == 8
    assert matched_block in seq.logical_blocks.get_real_blocks()
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert connector.allocations[0][2] == 4

    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert seq.num_history_ids == 12
    assert seq.status == MessageStatus.WAITING

    fourth = scheduler.schedule(is_prefill=True)
    assert fourth.running == [seq]


def test_async_lookup_pending_preserves_private_partial_prefix():
    connector = _AsyncLookupConnector([(None, False)])
    scheduler = _make_async_lookup_scheduler(connector)
    tokens = torch.arange(13)
    seq = scheduler.add_session(72).add_sequence(tokens)

    seq.kv_token_limit = 5
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    seq.set_step(5)
    seq.kv_token_limit = 7
    seq.cached_tokens = 3
    seq.model_meta = {'state': 'keep'}
    baseline_blocks = seq.logical_blocks.get_real_blocks().copy()
    baseline_cursor = seq.prefix_cache.trie_cursor
    scheduler.block_trie.stats.reset()

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert scheduler.last_schedule_had_pending_lookup
    assert connector.lookup_calls == [(seq.seq_id, 5)]
    assert seq.num_history_ids == 5
    assert torch.equal(torch.from_numpy(seq.logical_blocks.get_real_blocks()),
                       torch.from_numpy(baseline_blocks))
    assert seq.prefix_cache.trie_cursor is baseline_cursor
    assert seq.prefix_cache.match_start_step == -1
    assert seq.cached_tokens == 3
    assert seq.kv_token_limit == 7
    assert seq.model_meta == {'state': 'keep'}
    assert scheduler.block_trie.stats.num_query_tokens == 0


def test_external_cached_tokens_survive_remote_ready_admission():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(connector)
    seq = scheduler.add_session(73).add_sequence(torch.arange(13))

    started = scheduler.schedule(is_prefill=True)
    assert started.running == []

    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert seq.num_history_ids == 8
    assert seq.cached_tokens == 8
    assert seq.prefix_cache.match_start_step == 0

    admitted = scheduler.schedule(is_prefill=True)

    assert admitted.running == [seq]
    assert seq.cached_tokens == 8
    assert seq.prefix_cache.match_start_step == 0


def test_remote_ready_long_prefill_respects_short_only_turn():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        max_prefill_token_num=4,
    )
    seq = scheduler.add_session(74).add_sequence(torch.arange(17))

    started = scheduler.schedule(is_prefill=True)
    assert started.running == []
    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert seq.num_history_ids == 8
    assert scheduler.kv_load_coordinator.is_remote_ready(seq)

    short_turn = scheduler.schedule(is_prefill=True, allow_long_prefill=False)

    assert short_turn.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 8
    assert scheduler.kv_load_coordinator.is_remote_ready(seq)

    long_turn = scheduler.schedule(is_prefill=True)
    assert long_turn.running == [seq]
    assert seq.status == MessageStatus.READY


def test_external_cached_tokens_survive_prefill_budget_rejection():
    connector = _AsyncLookupConnector([(8, True), (8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        max_batches=2,
        max_prefill_token_num=8,
    )
    admitted_seq = scheduler.add_session(74).add_sequence(torch.arange(13))
    waiting_seq = scheduler.add_session(75).add_sequence(torch.arange(20, 33))

    started = scheduler.schedule(is_prefill=True)
    assert started.running == []
    scheduler.update_connector_output(
        KVConnectorOutput(
            finished_receiving={admitted_seq.seq_id, waiting_seq.seq_id}))

    admitted = scheduler.schedule(is_prefill=True)

    assert admitted.running == [admitted_seq]
    assert waiting_seq.status == MessageStatus.WAITING
    assert waiting_seq.num_history_ids == 8
    assert waiting_seq.num_blocks == 2
    assert waiting_seq.cached_tokens == 8
    assert waiting_seq.prefix_cache.match_start_step == 0
    assert scheduler.kv_load_coordinator.is_remote_ready(waiting_seq)


def test_async_load_keeps_a_private_partial_block_at_the_suffix_start():
    connector = _AsyncLookupConnector([(7, True)])
    scheduler = _make_async_lookup_scheduler(connector)
    tokens = torch.arange(13)
    cached = scheduler.add_session(76).add_sequence(tokens)
    scheduler.block_manager.allocate(cached)
    scheduler.block_trie.allocate(cached)
    cached.state.stop()

    seq = scheduler.add_session(77).add_sequence(tokens)
    seq.kv_token_limit = 5
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    seq.set_step(5)
    seq.kv_token_limit = None
    private_block = int(scheduler.block_manager.get_block_table(seq)[1])

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert seq.num_blocks == 3
    assert connector.lookup_calls == [(seq.seq_id, 5)]
    load_blocks = connector.allocations[0][1]
    assert load_blocks == tuple(
        int(block_id)
        for block_id in scheduler.block_manager.get_block_table(seq)[1:3]
    )
    assert load_blocks[0] == private_block


def test_async_load_requires_capacity_for_the_complete_prefill():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=3,
    )
    seq = scheduler.add_session(76).add_sequence(torch.arange(13))

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_blocks == 0
    assert connector.allocations == []


def test_async_load_capacity_failure_restores_tentative_local_prefix(monkeypatch):
    connector = _AsyncLookupConnector([(4, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        num_gpu_blocks=3,
    )
    tokens = torch.arange(13)
    cached = scheduler.add_session(76).add_sequence(tokens[:5])
    scheduler.block_manager.allocate(cached)
    scheduler.block_trie.allocate(cached)
    cached.state.stop()
    cached_block = cached.logical_blocks.get_real_blocks()[:1]
    ref_count = scheduler.block_manager.allocator.get_ref_count(cached_block).copy()
    scheduler.block_trie.stats.reset()

    seq = scheduler.add_session(77).add_sequence(tokens)
    evict_for_seq = Mock(return_value=False)
    monkeypatch.setattr(scheduler.eviction_helper, 'evict_for_seq', evict_for_seq)

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 0
    assert seq.num_blocks == 0
    assert seq.kv_token_limit is None
    assert seq.cached_tokens == 0
    assert seq.prefix_cache.trie_cursor is None
    assert seq.prefix_cache.match_start_step == -1
    assert connector.lookup_calls == [(seq.seq_id, 4)]
    assert connector.allocations == []
    assert evict_for_seq.call_count == 1
    assert scheduler.block_manager.allocator.get_ref_count(cached_block).tolist() == ref_count.tolist()
    assert scheduler.block_trie.stats.num_query_tokens == 0
    assert scheduler.block_trie.stats.num_hit_tokens == 0


def test_async_load_binding_failure_releases_allocated_destinations():
    connector = _AsyncLookupConnector([(8, True)])
    connector.update_state_after_alloc = Mock(
        side_effect=RuntimeError('binding failed'))
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=4,
    )
    seq = scheduler.add_session(77).add_sequence(torch.arange(13))

    with pytest.raises(RuntimeError, match='binding failed'):
        scheduler.schedule(is_prefill=True)

    assert seq.status == MessageStatus.WAITING
    assert seq.num_blocks == 0
    assert seq.kv_token_limit is None
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4
    assert not scheduler.has_remote_loading()


def test_async_load_does_not_consume_model_batch_slot():
    connector = _AsyncLookupConnector([(8, True), (0, False)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=5,
    )
    loading = scheduler.add_session(77).add_sequence(torch.arange(13))
    later = scheduler.add_session(78).add_sequence(torch.arange(8))

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [later]
    assert loading.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert loading.num_blocks == 2
    assert later.status == MessageStatus.READY
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 2
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 1

    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={loading.seq_id}))
    assert loading.num_history_ids == 8
    assert loading.cached_tokens == 8


def test_multiple_async_loads_start_in_one_prefill_turn():
    connector = _AsyncLookupConnector([(8, True), (8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        max_batches=1,
        num_gpu_blocks=16,
    )
    first = scheduler.add_session(91).add_sequence(torch.arange(13))
    second = scheduler.add_session(92).add_sequence(torch.arange(20, 33))

    output = scheduler.schedule(is_prefill=True)

    assert output.running == []
    assert first.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert second.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert [allocation[0] for allocation in connector.allocations] == [
        first.seq_id,
        second.seq_id,
    ]


def test_soft_reservation_blocks_new_load_until_capacity_is_released():
    connector = _AsyncLookupConnector([
        (4, True),
        (12, True),
        (12, True),
    ])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=6,
    )
    first = scheduler.add_session(86).add_sequence(torch.arange(13))

    scheduler.schedule(is_prefill=True)
    assert first.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert first.num_blocks == 1
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 3

    second = scheduler.add_session(87).add_sequence(torch.arange(17))
    blocked = scheduler.schedule(is_prefill=True)
    assert blocked.running == []
    assert second.status == MessageStatus.WAITING
    assert second.num_blocks == 0
    assert [allocation[0] for allocation in connector.allocations] == [first.seq_id]

    scheduler.end_session(86)
    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={first.seq_id}))
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 0

    retried = scheduler.schedule(is_prefill=True)
    assert retried.running == []
    assert second.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert second.num_blocks == 3
    assert [allocation[0] for allocation in connector.allocations] == [
        first.seq_id,
        second.seq_id,
    ]


def test_failed_async_load_preserves_local_prefix_and_releases_remote_tail():
    connector = _AsyncLookupConnector([(4, True)])
    scheduler = _make_async_lookup_scheduler(connector)
    tokens = torch.arange(13)
    cached = scheduler.add_session(79).add_sequence(tokens[:9])
    scheduler.block_manager.allocate(cached)
    scheduler.block_trie.allocate(cached)
    cached.state.stop()
    seq = scheduler.add_session(80).add_sequence(tokens)
    scheduler.schedule(is_prefill=True)
    connector.failed_ids.add(seq.seq_id)

    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))

    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == 8
    assert seq.num_blocks == 2
    assert seq.cached_tokens == 8
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 0


def test_async_load_soft_reservation_shrinks_across_chunks():
    connector = _AsyncLookupConnector([(4, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        max_prefill_token_num=4,
    )
    seq = scheduler.add_session(81).add_sequence(torch.arange(13))

    scheduler.schedule(is_prefill=True)
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 3
    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))

    assert scheduler.schedule(is_prefill=True).running == [seq]
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 2
    seq.set_step(8)
    scheduler.release_completed_prefill_reservations([seq])
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 2

    assert scheduler.reserve_long_context_chunk(seq, chunk_size=4)
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 1
    seq.set_step(12)
    assert scheduler.reserve_long_context_chunk(seq, chunk_size=1, is_last_chunk=True)
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 0


def test_stop_session_waits_for_active_async_load_before_stopping():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
    )
    seq = scheduler.add_session(82).add_sequence(torch.arange(13))
    scheduler.schedule(is_prefill=True)

    scheduler.stop_session(82)
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert seq.num_blocks == 2

    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert seq.status == MessageStatus.STOPPED
    assert seq.num_blocks == 0
    assert 82 in scheduler.sessions
    assert connector.finished == []


def test_end_session_overrides_stop_deferred_during_active_async_load():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
    )
    seq = scheduler.add_session(82).add_sequence(torch.arange(13))
    scheduler.schedule(is_prefill=True)

    scheduler.stop_session(82)
    scheduler.end_session(82)
    assert 82 in scheduler.sessions
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert seq.num_blocks == 2
    assert connector.finished == []

    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert 82 not in scheduler.sessions
    assert connector.finished == [seq.seq_id]


def test_worker_drain_finishes_an_ended_session_with_a_dropped_load_output():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
    )
    seq = scheduler.add_session(85).add_sequence(torch.arange(13))
    scheduler.schedule(is_prefill=True)
    scheduler.end_session(85)

    scheduler.finish_deferred_kv_transfers_after_worker_drain()

    assert 85 not in scheduler.sessions
    assert connector.finished == [seq.seq_id]
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 0


def test_completed_async_load_is_admitted_before_an_older_waiter():
    connector = _AsyncLookupConnector([(8, True)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=4,
    )
    loaded = scheduler.add_session(83).add_sequence(torch.arange(13))
    scheduler.schedule(is_prefill=True)
    scheduler.update_connector_output(
        KVConnectorOutput(finished_receiving={loaded.seq_id}))
    newcomer = scheduler.add_session(84).add_sequence(torch.arange(4))
    newcomer.arrive_time = loaded.arrive_time - 1

    output = scheduler.schedule(is_prefill=True)

    assert output.running == [loaded]
    assert newcomer.status == MessageStatus.WAITING


def test_stop_and_end_session_cancel_lookup_before_request_cleanup():
    connector = _AsyncLookupConnector([(None, False)])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
    )
    seq = scheduler.add_session(73).add_sequence(torch.arange(9))
    scheduler.schedule(is_prefill=True)

    scheduler.stop_session(73)
    assert connector.cancelled == [seq.seq_id]

    scheduler.end_session(73)
    assert connector.finished == [seq.seq_id]


def test_async_save_lease_keeps_exact_blocks_alive_until_all_tp_complete():

    class _SaveMetadata(KVConnectorMetadata):

        def __init__(self, logical_block_ids):
            self.logical_block_ids = tuple(logical_block_ids)

        def get_save_block_leases(self):
            return (KVSaveBlockLease(7, self.logical_block_ids), )

    class _SaveConnector(_AsyncLookupConnector):

        def __init__(self):
            super().__init__([])
            self.metadata = None

        def build_connector_meta(self, scheduler_output):
            metadata, self.metadata = self.metadata, None
            return metadata

        def update_connector_output(self, connector_output):
            return KVConnectorResult(
                completed_save_ids=frozenset(
                    connector_output.completed_save_ids or ()),
            )

    connector = _SaveConnector()
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=4,
    )
    seq = scheduler.add_session(88).add_sequence(torch.arange(8))
    scheduler.block_manager.allocate(seq)
    logical_blocks = seq.logical_blocks.get_real_blocks().copy()
    allocator = scheduler.block_manager.allocator
    connector.metadata = _SaveMetadata(logical_blocks)

    metadata = scheduler.build_connector_meta(
        [seq],
        connector_token_lens=(8, ),
    )

    assert metadata is not None
    assert allocator.get_ref_count(logical_blocks).tolist() == [2, 2]
    assert scheduler.has_unfinished()

    # Sequence ownership may disappear before remote I/O completes. The save
    # lease remains as the only reference and prevents physical reuse.
    scheduler.block_manager.free(seq)
    assert allocator.get_ref_count(logical_blocks).tolist() == [1, 1]
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 2

    scheduler.update_connector_output(
        KVConnectorOutput(completed_save_ids={7}))
    assert allocator.get_ref_count(logical_blocks).tolist() == [0, 0]
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4
    assert not scheduler.kv_save_coordinator.has_pending()


def test_worker_drain_releases_save_leases_when_outputs_are_discarded():

    class _SaveMetadata(KVConnectorMetadata):

        def __init__(self, logical_block_ids):
            self.logical_block_ids = tuple(logical_block_ids)

        def get_save_block_leases(self):
            return (KVSaveBlockLease(9, self.logical_block_ids), )

    connector = _AsyncLookupConnector([])
    scheduler = _make_async_lookup_scheduler(
        connector,
        enable_prefix_caching=False,
        num_gpu_blocks=2,
    )
    seq = scheduler.add_session(89).add_sequence(torch.arange(8))
    scheduler.block_manager.allocate(seq)
    logical_blocks = seq.logical_blocks.get_real_blocks().copy()
    metadata = _SaveMetadata(logical_blocks)
    connector.build_connector_meta = lambda scheduler_output: metadata

    scheduler.build_connector_meta([seq], connector_token_lens=(8, ))
    scheduler.block_manager.free(seq)
    scheduler.finish_deferred_kv_transfers_after_worker_drain()

    assert scheduler.block_manager.get_num_free_gpu_blocks() == 2
    assert not scheduler.kv_save_coordinator.has_pending()
