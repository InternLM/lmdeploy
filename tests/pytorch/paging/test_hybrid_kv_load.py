# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.engine.inputs_maker import InputsMakerAsync
from lmdeploy.pytorch.kv_connector import KVConnectorOutput, KVConnectorStepInput
from lmdeploy.pytorch.kv_connector.base import KVConnectorOutputAggregator
from lmdeploy.pytorch.kv_connector.mooncake.store.scheduler import MooncakeStoreScheduler
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.paging.kv_load_coordinator import KVLoadAdmission
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


@pytest.fixture
def make_scheduler():
    schedulers = []

    def create(*, apc=True, role='kv_both', state_budget=1, remote_hit=16, num_gpu_blocks=16,
               max_batches=1, max_prefill_token_num=4):
        save_slots = 2 if role in ('kv_producer', 'kv_both') else 0
        config = CacheConfig(
            max_batches=max_batches, block_size=4, num_cpu_blocks=0, num_gpu_blocks=num_gpu_blocks,
            max_prefill_token_num=max_prefill_token_num, enable_prefix_caching=apc,
            states_shapes=[((2,), torch.float32)],
            num_state_caches=1 + max_batches + state_budget + save_slots,
            prefix_cache_state_budget=state_budget, mooncake_state_save_slots=2,
            mooncake_prefill_save_alignment=8,
            kv_transfer_config=KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role=role))
        connector = MooncakeStoreScheduler(config)
        if connector.client is not None:
            connector.client.lookup = Mock(return_value=remote_hit)
        scheduler = Scheduler(
            SchedulerConfig(max_batches=max_batches, max_session_len=64, max_request_output_len=16), config,
            SequenceMeta(4, strategy=ARSequenceStrategy()), kv_connector=connector)
        schedulers.append(scheduler)
        return scheduler, connector

    yield create
    for scheduler in schedulers:
        scheduler.shutdown()


def _publish_local_checkpoint(scheduler, step):
    seq = scheduler.add_session(0).add_sequence(torch.arange(step))
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    assert scheduler.block_trie.state_checkpoints.reserve_save(seq) > 0
    assert scheduler.block_trie.state_checkpoints.publish_save(seq)
    checkpoint = seq.prefix_cache.trie_cursor.state_checkpoint
    scheduler.end_session(0)
    return checkpoint


def test_mooncake_producer_reports_zero_external_hit_rate_without_lookup(make_scheduler):
    scheduler, connector = make_scheduler(apc=False, role='kv_producer')
    assert scheduler.schedule_metrics.external_prefix_cache_hit_rate == 0
    seq = scheduler.add_session(1).add_sequence(torch.arange(21))
    assert scheduler.schedule(is_prefill=True).running == [seq]
    assert connector.client is None
    assert scheduler.schedule_metrics.external_prefix_cache_queries == 21
    assert scheduler.schedule_metrics.external_prefix_cache_hits == 0
    assert scheduler.schedule_metrics.external_prefix_cache_hit_rate == 0


@pytest.mark.parametrize('local_step', [4, 6, 10])
@pytest.mark.parametrize('remote_hit', [0, 16])
def test_hybrid_load_preserves_exact_local_hit_or_restores_exact_remote_boundary(
        make_scheduler, local_step, remote_hit):
    scheduler, connector = make_scheduler(remote_hit=remote_hit)
    checkpoint = _publish_local_checkpoint(scheduler, local_step)
    seq = scheduler.add_session(1).add_sequence(torch.arange(21))
    output = scheduler.schedule(is_prefill=True)
    if remote_hit == 0:
        assert output.running == [seq]
        assert seq.num_history_ids == local_step
        assert seq.prefix_cache.restore.slot == checkpoint.slot
        assert connector.build_connector_meta(output) is None
        assert scheduler.schedule_metrics.prefix_cache_hit_rate == pytest.approx(local_step / 21)
        assert scheduler.schedule_metrics.external_prefix_cache_queries == 21 - local_step
        assert scheduler.schedule_metrics.external_prefix_cache_hits == 0
        return

    assert output.running == []
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert seq.num_history_ids == local_step
    load = connector.build_connector_meta(output).load_requests[0]
    assert load.state_slot == seq.logical_state > 0
    assert load.remote_block_count == 4
    assert len(load.block_ids) == (16 - local_step // 4 * 4) // 4
    if checkpoint.frozen_block_id >= 0:
        frozen_block = scheduler.block_manager.allocator.get_physical_blocks([checkpoint.frozen_block_id])[0]
        assert frozen_block not in load.block_ids
    assert checkpoint.pin_count == 0
    assert not seq.prefix_cache.restore.is_selected
    assert scheduler.state_manager.get_num_free_runtime() == 0
    # Remote loading replaces the partial local block, as in vLLM's adopted
    # prefix accounting. Count the lookup now, before any worker completes.
    local_tokens = local_step // 4 * 4
    metrics = scheduler.schedule_metrics
    assert scheduler.block_trie.stats.num_query_tokens == 21
    assert scheduler.block_trie.stats.num_hit_tokens == local_tokens
    assert metrics.external_prefix_cache_queries == 21 - local_tokens
    assert metrics.external_prefix_cache_hits == 16 - local_tokens
    assert metrics.external_prefix_cache_hit_rate == pytest.approx((16 - local_tokens) / (21 - local_tokens))

    scheduler.update_connector_output(KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert seq.num_history_ids == seq.cached_tokens == 16
    assert scheduler.schedule_metrics.prefix_cache_hit_rate == pytest.approx(local_tokens / 21)
    assert seq.logical_state == load.state_slot
    assert seq.prefix_cache.recompute_overlap.fresh_block_range is None
    assert not seq.prefix_cache.restore.is_selected
    # There is no spare runtime capacity. Remote-ready admission must reuse D.
    assert scheduler.schedule(is_prefill=True).running == [seq]
    assert seq.num_history_ids == 16
    assert scheduler.block_trie.stats.num_query_tokens == 21
    assert scheduler.block_trie.stats.num_hit_tokens == local_tokens
    assert scheduler.schedule_metrics.external_prefix_cache_queries == metrics.external_prefix_cache_queries
    assert scheduler.schedule_metrics.external_prefix_cache_hits == metrics.external_prefix_cache_hits
    connector.client.lookup.assert_called_once()
    # Build the next forward's real copy plans: the earlier local checkpoint
    # must not overwrite the state that was just loaded at H.
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(is_ssm=True, enable_prefix_caching=True)
    maker.state_checkpoints = scheduler.state_checkpoints
    maker.scheduler = scheduler
    cache_inputs = maker._prepare_prefill_cache_inputs([seq], save_steps=(20,))
    assert cache_inputs is None or cache_inputs.state_restore_plan is None
    assert cache_inputs is None or cache_inputs.kv_restore_plan is None


@pytest.mark.parametrize('apc', [False, True])
@pytest.mark.parametrize('role', ['kv_consumer', 'kv_both'])
def test_hybrid_load_needs_no_checkpoint_or_save_slot(make_scheduler, apc, role):
    scheduler, connector = make_scheduler(apc=apc, role=role, state_budget=0)
    if apc:
        # The only runtime slot is borrowed by an unpinned checkpoint. Load
        # admission may evict it instead of retaining a fallback snapshot.
        _publish_local_checkpoint(scheduler, 6)
    seq = scheduler.add_session(1).add_sequence(torch.arange(21))
    output = scheduler.schedule(is_prefill=True)
    load = connector.build_connector_meta(output).load_requests[0]
    assert load.state_slot == 1
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 0
    scheduler.update_connector_output(KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert scheduler.schedule(is_prefill=True).running == [seq]
    assert seq.num_history_ids == seq.cached_tokens == 16
    local_tokens = 4 if apc else 0
    assert scheduler.block_trie.stats.num_query_tokens == (21 if apc else 0)
    assert scheduler.block_trie.stats.num_hit_tokens == local_tokens
    assert scheduler.schedule_metrics.external_prefix_cache_queries == 21 - local_tokens
    assert scheduler.schedule_metrics.external_prefix_cache_hits == 16 - local_tokens


@pytest.mark.parametrize('fail_binding', [False, True])
def test_hybrid_load_reuses_existing_runtime_and_private_partial_block(make_scheduler, fail_binding):
    scheduler, connector = make_scheduler(state_budget=0)
    seq = scheduler.add_session(1).add_sequence(torch.arange(21))
    seq.kv_token_limit = 6
    scheduler.block_manager.allocate(seq)
    scheduler.state_manager.allocate(seq)
    seq.set_step(6)
    original_slot = seq.logical_state
    original_blocks = tuple(scheduler.block_manager.get_block_table(seq))
    if fail_binding:
        connector.update_state_after_alloc = Mock(side_effect=ValueError('invalid metadata'))
        with pytest.raises(ValueError, match='invalid metadata'):
            scheduler.schedule(is_prefill=True)
        assert tuple(scheduler.block_manager.get_block_table(seq)) == original_blocks
        assert seq.kv_token_limit == seq.num_history_ids == 6
    else:
        load = connector.build_connector_meta(scheduler.schedule(is_prefill=True)).load_requests[0]
        assert load.state_slot == original_slot
        assert load.block_ids[0] == original_blocks[1]
        assert len(load.block_ids) == 3
        scheduler.update_connector_output(KVConnectorOutput(finished_receiving={seq.seq_id}))
        assert scheduler.schedule(is_prefill=True).running == [seq]
        assert seq.num_history_ids == 16
        assert seq.cached_tokens == 10
        assert scheduler.block_trie.stats.num_query_tokens == 0
        assert scheduler.block_trie.stats.num_hit_tokens == 0
        assert scheduler.schedule_metrics.external_prefix_cache_queries == 17
        assert scheduler.schedule_metrics.external_prefix_cache_hits == 12
    assert seq.logical_state == original_slot
    assert scheduler.state_manager.get_num_runtime_states() == 1


@pytest.mark.parametrize('retain_checkpoint', [False, True])
def test_failed_hybrid_load_waits_for_all_ranks_then_rematches(make_scheduler, retain_checkpoint):
    scheduler, connector = make_scheduler()
    checkpoint = _publish_local_checkpoint(scheduler, 6)
    seq = scheduler.add_session(1).add_sequence(torch.arange(21))
    load = connector.build_connector_meta(scheduler.schedule(is_prefill=True)).load_requests[0]
    original_blocks = seq.logical_blocks.get_real_blocks().copy()
    aggregator = KVConnectorOutputAggregator(world_size=2)
    first_rank = KVConnectorOutput(finished_receiving={seq.seq_id}, failed_receiving={seq.seq_id})
    scheduler.update_connector_output(aggregator.aggregate([first_rank, None]))
    assert seq.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert seq.logical_state == load.state_slot
    assert list(seq.logical_blocks.get_real_blocks()) == list(original_blocks)
    assert scheduler.schedule(is_prefill=True).running == []
    assert scheduler.state_manager.get_num_runtime_states() == 1
    if not retain_checkpoint:
        scheduler.block_trie.state_checkpoints.release_checkpoint(seq.prefix_cache.trie_cursor)
    assert checkpoint.pin_count == 0

    # A later successful rank must not erase the earlier failure.
    scheduler.update_connector_output(aggregator.aggregate([
        None, KVConnectorOutput(finished_receiving={seq.seq_id})]))
    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == seq.cached_tokens == seq.num_blocks == 0
    assert seq.logical_state == -1
    assert scheduler.state_manager.get_num_runtime_states() == 0
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 0
    assert connector._next_save_block[seq.seq_id] == 0

    output = scheduler.schedule(is_prefill=True)
    assert output.running == [seq]
    assert seq.num_history_ids == (6 if retain_checkpoint else 0)
    assert seq.prefix_cache.restore.is_selected == retain_checkpoint
    assert connector.build_connector_meta(output) is None
    connector.client.lookup.assert_called_once()
    # Failed I/O does not undo an admitted lookup hit. The rematch is a new
    # admitted query, after hybrid rollback released the entire request state.
    assert scheduler.block_trie.stats.num_query_tokens == 42
    assert scheduler.block_trie.stats.num_hit_tokens == 4 + (6 if retain_checkpoint else 0)
    assert scheduler.schedule_metrics.external_prefix_cache_queries == 17 + (15 if retain_checkpoint else 21)
    assert scheduler.schedule_metrics.external_prefix_cache_hits == 12


@pytest.mark.parametrize('apc', [False, True])
def test_concurrent_hybrid_loads_isolate_mixed_rank_results_at_runtime_capacity(make_scheduler, apc):
    scheduler, connector = make_scheduler(
        apc=apc, role='kv_consumer', state_budget=0, max_batches=2, max_prefill_token_num=32)
    failed, successful = [scheduler.add_session(i + 1).add_sequence(torch.arange(21) + 100 * i)
                          for i in range(2)]
    output = scheduler.schedule(is_prefill=True)
    assert output.running == []
    loads = {load.request_id: load for load in connector.build_connector_meta(output).load_requests}
    failed_load, successful_load = loads[failed.seq_id], loads[successful.seq_id]
    assert failed_load.state_slot != successful_load.state_slot
    assert set(failed_load.block_ids).isdisjoint(successful_load.block_ids)
    assert scheduler.state_manager.get_num_runtime_states() == 2
    assert scheduler.state_manager.get_num_free_runtime() == 0
    successful_blocks = tuple(scheduler.block_manager.get_block_table(successful))

    aggregator = KVConnectorOutputAggregator(world_size=2)
    scheduler.update_connector_output(aggregator.aggregate([
        KVConnectorOutput(finished_receiving={failed.seq_id}, failed_receiving={failed.seq_id}), None]))
    assert failed.status == successful.status == MessageStatus.WAITING_FOR_REMOTE_KVS
    assert failed.logical_state == failed_load.state_slot
    assert successful.logical_state == successful_load.state_slot
    assert scheduler.state_manager.get_num_runtime_states() == 2

    scheduler.update_connector_output(aggregator.aggregate([
        KVConnectorOutput(finished_receiving={successful.seq_id}),
        KVConnectorOutput(finished_receiving={failed.seq_id, successful.seq_id})]))
    assert failed.logical_state == -1
    assert failed.num_history_ids == failed.cached_tokens == failed.num_blocks == 0
    assert successful.logical_state == successful_load.state_slot
    assert successful.num_history_ids == successful.cached_tokens == 16
    assert tuple(scheduler.block_manager.get_block_table(successful)) == successful_blocks
    assert scheduler.state_manager.get_num_runtime_states() == 1

    output = scheduler.schedule(is_prefill=True)
    assert successful in output.running
    assert failed in output.running
    assert failed.logical_state != successful.logical_state
    assert successful.logical_state == successful_load.state_slot
    assert successful.num_history_ids == 16
    assert tuple(scheduler.block_manager.get_block_table(successful))[:4] == successful_blocks
    assert connector.client.lookup.call_count == 2
    assert scheduler.block_trie.stats.num_query_tokens == (63 if apc else 0)
    assert scheduler.block_trie.stats.num_hit_tokens == 0
    assert scheduler.schedule_metrics.external_prefix_cache_queries == 63
    assert scheduler.schedule_metrics.external_prefix_cache_hits == 32


@pytest.mark.parametrize('apc', [False, True])
def test_hybrid_hit_rate_counts_admitted_lookups_without_recounting_load_completion(make_scheduler, apc):
    scheduler, connector = make_scheduler(apc=apc, remote_hit=0)
    missed = scheduler.add_session(1).add_sequence(torch.arange(21))
    assert scheduler.schedule(is_prefill=True).running == [missed]
    scheduler.end_session(1)
    assert scheduler.block_trie.stats.num_query_tokens == (21 if apc else 0)
    assert scheduler.schedule_metrics.external_prefix_cache_queries == 21

    seq = scheduler.add_session(2).add_sequence(torch.arange(21) + 100)
    connector.client.lookup.side_effect = [None, 16]
    # A pending lookup does not count; accepted hits count before load completion.
    assert scheduler.schedule(is_prefill=True).running == []
    assert scheduler.schedule_metrics.external_prefix_cache_queries == 21
    assert scheduler.schedule(is_prefill=True).running == []
    connector.build_connector_meta(KVConnectorStepInput())
    assert scheduler.block_trie.stats.num_query_tokens == (42 if apc else 0)
    assert scheduler.block_trie.stats.num_hit_tokens == 0
    assert scheduler.schedule_metrics.external_prefix_cache_queries == 42
    assert scheduler.schedule_metrics.external_prefix_cache_hits == 16

    completed = KVConnectorOutput(finished_receiving={seq.seq_id})
    scheduler.update_connector_output(completed)
    scheduler.update_connector_output(completed)
    assert scheduler.schedule(is_prefill=True).running == [seq]
    assert seq.cached_tokens == 16
    assert scheduler.block_trie.stats.num_query_tokens == (42 if apc else 0)
    assert scheduler.block_trie.stats.num_hit_tokens == 0
    assert scheduler.schedule_metrics.prefix_cache_hit_rate == 0
    assert scheduler.schedule_metrics.external_prefix_cache_queries == 42
    assert scheduler.schedule_metrics.external_prefix_cache_hits == 16
    assert scheduler.schedule_metrics.external_prefix_cache_hit_rate == pytest.approx(16 / 42)


@pytest.mark.parametrize('apc', [False, True])
@pytest.mark.parametrize('remote_hit', [0, 16])
def test_hybrid_remote_recompute_does_not_count_hits_again(make_scheduler, apc, remote_hit):
    scheduler, connector = make_scheduler(apc=apc, remote_hit=0, num_gpu_blocks=6, state_budget=0)
    seq = scheduler.add_session(1).add_sequence(torch.arange(21))
    assert scheduler.schedule(is_prefill=True).running == [seq]
    seq.set_step(4)
    seq.state.evict()
    # Exercise real eviction with local caching disabled as well: the flag
    # must be set by paging, not supplied by the test.
    pressure = scheduler.add_session(2).add_sequence(torch.arange(24) + 100)
    assert scheduler.eviction_helper.try_make_capacity_for(pressure, [seq], 0)
    assert seq.prefix_cache.suppress_match_stats
    scheduler.end_session(2)

    connector.client.lookup.return_value = remote_hit
    output = scheduler.schedule(is_prefill=True)
    if remote_hit:
        connector.build_connector_meta(output)
        scheduler.update_connector_output(KVConnectorOutput(finished_receiving={seq.seq_id}))
        assert seq.cached_tokens == 0
        assert scheduler.schedule(is_prefill=True).running == [seq]
    else:
        assert output.running == [seq]
    assert seq.cached_tokens == 0
    assert not seq.prefix_cache.suppress_match_stats
    assert scheduler.block_trie.stats.num_query_tokens == (21 if apc else 0)
    assert scheduler.block_trie.stats.num_hit_tokens == 0
    assert scheduler.schedule_metrics.external_prefix_cache_queries == 21
    assert scheduler.schedule_metrics.external_prefix_cache_hits == 0


def test_hybrid_load_waits_for_pinned_local_restore_before_reusing_checkpoint_slot(make_scheduler):
    scheduler, connector = make_scheduler(
        role='kv_consumer', state_budget=0, max_batches=2, max_prefill_token_num=32)
    checkpoint = _publish_local_checkpoint(scheduler, 6)
    checkpoint_slot = checkpoint.slot
    local = scheduler.add_session(1).add_sequence(torch.arange(9))
    remote = scheduler.add_session(2).add_sequence(torch.arange(21) + 100)
    connector.client.lookup.side_effect = lambda request_id, *args, **kwargs: 0 if request_id == local.seq_id else 16

    output = scheduler.schedule(is_prefill=True)
    assert output.running == [local]
    assert local.num_history_ids == 6
    local_slot = local.logical_state
    assert local_slot != checkpoint_slot
    assert checkpoint.pin_count == 1
    assert scheduler.state_manager.get_num_free_runtime() == 0
    assert remote.status == MessageStatus.WAITING
    assert remote.logical_state == -1
    assert connector.build_connector_meta(output) is None
    copy_plan = scheduler.state_checkpoints.prepare_restore_batch([local])
    assert copy_plan.state_pairs == ((checkpoint_slot, local_slot),)
    assert copy_plan.kv_block_pairs == ((checkpoint.frozen_block_id, local.logical_blocks[1]),)

    # Dispatch releases the source pin. It does not prove GPU copy completion;
    # the worker's ready event fences reuse of this slot by the subsequent GET.
    scheduler.state_checkpoints.finish_forward_dispatch([local], has_save_plan=False)
    assert checkpoint.pin_count == 0
    output = scheduler.schedule(is_prefill=True)
    load = connector.build_connector_meta(output).load_requests[0]
    assert load.request_id == remote.seq_id
    assert load.state_slot == checkpoint_slot
    assert local.logical_state == local_slot
    assert scheduler.state_manager.get_num_runtime_states() == 2
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 0

    scheduler.update_connector_output(KVConnectorOutput(finished_receiving={remote.seq_id}))
    assert remote.num_history_ids == 16
    assert remote.logical_state == checkpoint_slot
    assert local.num_history_ids == 6
    assert local.logical_state == local_slot


@pytest.mark.parametrize('invalid_boundary', ['upper_bound', 'multimodal'])
def test_hybrid_coordinator_rejects_invalid_exact_boundary_without_allocating(make_scheduler, invalid_boundary):
    scheduler, connector = make_scheduler()
    multimodals = None
    if invalid_boundary == 'multimodal':
        multimodals = {'image': [MultiModalData(torch.tensor([1]), start=12, end=18)]}
    seq = scheduler.add_session(1).add_sequence(torch.arange(21), multimodals=multimodals)
    seq.kv_token_limit = 6
    scheduler.block_manager.allocate(seq)
    scheduler.state_manager.allocate(seq)
    seq.set_step(6)
    original_slot = seq.logical_state
    original_blocks = tuple(scheduler.block_manager.get_block_table(seq))
    free_blocks = scheduler.block_manager.get_num_free_gpu_blocks()
    remote_step = 24 if invalid_boundary == 'upper_bound' else 16
    # Bypass connector validation to exercise the coordinator's own guard.
    # Clipping H to 20 or 12 would mismatch the checkpoint state returned by GET.
    connector.get_num_new_matched_tokens = Mock(return_value=(remote_step - 6, True))
    connector.update_state_after_alloc = Mock()
    eviction_helper = Mock()
    admission = scheduler.kv_load_coordinator.try_load(
        seq, prealloc_size=0, evictable_seqs=[], eviction_helper=eviction_helper)
    assert admission is KVLoadAdmission.NO_LOAD
    eviction_helper.try_make_capacity_for.assert_not_called()
    connector.update_state_after_alloc.assert_not_called()
    assert seq.status == MessageStatus.WAITING
    assert seq.kv_token_limit == seq.num_history_ids == 6
    assert seq.logical_state == original_slot
    assert tuple(scheduler.block_manager.get_block_table(seq)) == original_blocks
    assert scheduler.block_manager.get_num_free_gpu_blocks() == free_blocks
    assert scheduler.state_manager.get_num_runtime_states() == 1
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 0
    assert connector.build_connector_meta(KVConnectorStepInput()) is None


@pytest.mark.parametrize('cleanup', ['stop', 'end', 'drain'])
def test_cancelled_hybrid_load_retains_destinations_until_terminal(make_scheduler, cleanup):
    scheduler, connector = make_scheduler(state_budget=0)
    seq = scheduler.add_session(1).add_sequence(torch.arange(21))
    load = connector.build_connector_meta(scheduler.schedule(is_prefill=True)).load_requests[0]
    if cleanup == 'stop':
        scheduler.stop_session(1)
    else:
        scheduler.end_session(1)
    assert seq.logical_state == load.state_slot
    assert seq.num_blocks == 4
    if cleanup == 'drain':
        scheduler.finish_kv_transfers_after_worker_drain()
    else:
        scheduler.update_connector_output(KVConnectorOutput(finished_receiving={seq.seq_id}))
    assert seq.logical_state == -1
    assert seq.num_blocks == seq.num_history_ids == 0
    assert scheduler.state_manager.get_num_runtime_states() == 0
    assert scheduler.kv_load_coordinator.soft_reserved_blocks() == 0
    assert (1 in scheduler.sessions) == (cleanup == 'stop')
    assert scheduler.block_trie.stats.num_query_tokens == 21
    assert scheduler.block_trie.stats.num_hit_tokens == 0
    assert scheduler.schedule_metrics.external_prefix_cache_queries == 21
    assert scheduler.schedule_metrics.external_prefix_cache_hits == 16


@pytest.mark.parametrize('rejection', ['pending_lookup', 'capacity', 'metadata_binding'])
def test_rejected_hybrid_admission_releases_tentative_local_restore(make_scheduler, rejection):
    scheduler, connector = make_scheduler(num_gpu_blocks=5 if rejection == 'capacity' else 16)
    checkpoint = _publish_local_checkpoint(scheduler, 6)
    seq = scheduler.add_session(1).add_sequence(torch.arange(25))
    if rejection == 'pending_lookup':
        connector.client.lookup.return_value = None
    elif rejection == 'metadata_binding':
        connector.update_state_after_alloc = Mock(side_effect=ValueError('invalid metadata'))
    if rejection == 'metadata_binding':
        with pytest.raises(ValueError, match='invalid metadata'):
            scheduler.schedule(is_prefill=True)
    else:
        assert scheduler.schedule(is_prefill=True).running == []
    assert seq.status == MessageStatus.WAITING
    assert seq.num_history_ids == seq.num_blocks == seq.cached_tokens == 0
    assert seq.logical_state == -1
    assert checkpoint.pin_count == 0
    assert not seq.prefix_cache.restore.is_selected
    assert scheduler.state_manager.get_num_runtime_states() == 0
    assert connector.build_connector_meta(KVConnectorStepInput()) is None
    assert scheduler.schedule_metrics.external_prefix_cache_queries == 0
    assert scheduler.schedule_metrics.external_prefix_cache_hits == 0
