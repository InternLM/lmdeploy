# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
from lmdeploy.pytorch.engine.engine_loop import EngineLoop, EngineLoopConfig
from lmdeploy.pytorch.engine.model_agent.agent import BatchedOutputs
from lmdeploy.pytorch.kv_connector import KVConnectorOutput
from lmdeploy.pytorch.kv_connector.base import KVConnectorOutputAggregator
from lmdeploy.pytorch.kv_connector.mooncake.store.scheduler import MooncakeStoreScheduler
from lmdeploy.pytorch.long_context import get_decode_save_step
from lmdeploy.pytorch.messages import SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy


def _scheduler(local=True, store=False, *, speculative=True, alignment=128, interval=64,
               save_first=True, last_only=True):
    cache = CacheConfig(max_batches=2, block_size=64, num_cpu_blocks=0, num_gpu_blocks=32,
                        num_state_caches=16, states_shapes=[((2,), torch.float32)],
                        enable_prefix_caching=local, prefix_cache_state_budget=8,
                        prefix_cache_token_lookahead=int(speculative), prefix_cache_block_aligned=True,
                        prefill_save_alignment=alignment, prefill_save_last_only=last_only,
                        prefix_cache_decode_save_first=save_first, prefix_cache_decode_state_interval=interval,
                        mooncake_state_save_slots=2,
                        kv_transfer_config=KVTransferConfig(kv_connector='MooncakeStoreConnector',
                                                           kv_role='kv_producer') if store else None)
    strategy = ARSpecSequenceStrategy() if speculative else ARSequenceStrategy()
    return Scheduler(SchedulerConfig(max_batches=2, max_session_len=4096), cache,
                     seq_meta=SequenceMeta(64, strategy=strategy),
                     kv_connector=MooncakeStoreScheduler(cache) if store else None)


def _store_save(scheduler, seq, step):
    scheduler.block_manager.allocate(seq)
    metadata = scheduler.build_connector_meta(
        [seq], connector_token_lens=(step,), connector_state_ids=(int(seq.logical_state),),
        is_decode_checkpoint=True)
    return metadata.save_requests[0]


@pytest.mark.parametrize(('save_first', 'interval', 'first', 'second'), [
    (False, 0, None, None),
    (True, 0, 256, None),
    (True, 64, 256, 320),
    (False, 64, 192, 256),
])
def test_decode_policy_modes(save_first, interval, first, second):
    scheduler = _scheduler(save_first=save_first, interval=interval)
    seq = scheduler.add_session(0).add_sequence(list(range(190)))
    assert get_decode_save_step(seq, 190, 258) == first
    assert seq.prefix_cache.last_decode_save_step == -1
    if first is not None:
        assert get_decode_save_step(seq, first - 2, first - 1) is None
        assert get_decode_save_step(seq, first - 2, first + 2) == first
        seq.prefix_cache.last_decode_save_step = first
    start = first if first is not None else 256
    assert get_decode_save_step(seq, start + 1, start + 64) == second


def test_decode_retry_boundaries_remain_discoverable_by_store():
    scheduler = _scheduler(alignment=256, interval=1024)
    seq = scheduler.add_session(0).add_sequence(list(range(190)))
    assert get_decode_save_step(seq, 254, 258) == 256
    assert seq.prefix_cache.last_decode_save_step == -1
    # The first copy failed and its state has since been overwritten.
    assert get_decode_save_step(seq, 318, 322) is None
    first_retry = get_decode_save_step(seq, 510, 514)
    assert first_retry == 512
    seq.prefix_cache.last_decode_save_step = first_retry
    assert get_decode_save_step(seq, 1534, 1538) == 1536
    # A later failed copy must not retry at 1600: Store probes multiples of 256.
    assert get_decode_save_step(seq, 1598, 1602) is None
    retry = get_decode_save_step(seq, 1790, 1794)
    assert retry == 1792
    assert retry % scheduler.cache_config.state_lookup_alignment == 0
    seq.prefix_cache.last_decode_save_step = retry
    assert get_decode_save_step(seq, 2814, 2818) == 2816


@pytest.mark.parametrize('speculative', [False, True])
def test_decode_save_position_resets_only_for_new_input_turn(speculative):
    scheduler = _scheduler(speculative=speculative, alignment=256, interval=1024)
    seq = scheduler.add_session(0).add_sequence(list(range(256)))
    # An aligned prompt can supply the first checkpoint after acceptance.
    assert get_decode_save_step(seq, 256, 256) == 256
    seq.update_token_ids([7], draft_token_ids=[8, 9], mode=UpdateTokenMode.PREFILL)
    seq.prefix_cache.last_decode_save_step = 256
    seq.update_token_ids([8], draft_token_ids=[], stop_pos=0, mode=UpdateTokenMode.DECODE)
    seq.set_step(128)  # Recompute within this turn preserves the successful save.
    assert seq.input_end_pos == seq.prefix_cache.last_decode_save_step == 256
    seq.update_token_ids([11] * 100, mode=UpdateTokenMode.INPUTS)
    assert seq.input_end_pos == 358
    assert seq.prefix_cache.last_decode_save_step == -1
    assert get_decode_save_step(seq, 510, 514) == 512


@pytest.mark.parametrize(('local', 'store'), [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize('last_only', [False, True])
def test_prefetched_mtp_checkpoints_use_accepted_tokens_and_frozen_slots(local, store, last_only):
    scheduler = _scheduler(local, store, last_only=last_only)
    seq = scheduler.add_session(0).add_sequence(list(range(254)))
    scheduler.block_manager.allocate(seq, prealloc_size=8)
    scheduler.state_manager.allocate(seq)
    seq.update_token_ids([50], draft_token_ids=[51, 52, 53], mode=UpdateTokenMode.PREFILL)
    coordinator = scheduler.decode_checkpoints
    # Both dispatches see the same stale CPU progress. The worker decides
    # independently, using each forward's actual history and accepted output.
    first_id, first_plan = coordinator.prepare([seq], None, SimpleNamespace(max_q_seqlen=4))
    second_id, second_plan = coordinator.prepare([seq], None, SimpleNamespace(max_q_seqlen=4))
    assert first_plan[1].tolist() == second_plan[1].tolist() == [256] * (int(local) + int(store))
    assert set(first_plan[3].tolist()).isdisjoint(second_plan[3].tolist())
    for batch_id, plan, copied, tokens in (
        (first_id, first_plan, False, [51, -1, -1, -1]),
        (second_id, second_plan, True, [52, 53, -1, -1]),
    ):
        # Worker acceptance is an input to scheduler publication. The actual
        # device predicate is covered by the CUDA operator tests.
        mask = torch.full((plan.size(1),), copied, dtype=torch.bool)
        seq.update_token_ids(tokens, draft_token_ids=[54, 55, 56], mode=UpdateTokenMode.DECODE)
        coordinator.finish(batch_id, mask)
    assert seq.num_history_ids == 257
    if local:
        assert seq.prefix_cache.decode_checkpoint_node.state_checkpoint.step == 256
    if store:
        # Accepted token identity is attached now; the frozen slot is sent on
        # the next normal/polling step without recopying the live runtime state.
        metadata = scheduler.build_connector_meta([])
        saved = metadata.save_requests[0]
        assert saved.state.boundary_tokens == 256
        assert saved.state.source_slot == saved.state.snapshot_slot
        assert metadata.get_state_save_copies() == ()
        assert scheduler.cache_config.state_lookup_alignment == 64
        assert seq.prefix_cache.last_decode_save_step == (256 if local else -1)
        scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={saved.save_id}))
        assert not scheduler.kv_save_coordinator.has_pending()
    assert seq.prefix_cache.last_decode_save_step == 256

    for changed in (False, True):
        tokens = seq.history_cache[:seq.num_valid_ids].tolist()
        tokens[256] += int(changed)
        repeat = scheduler.add_session(1 + int(changed)).add_sequence(tokens + [99])
        if local:
            scheduler.block_trie.match(repeat)
            assert repeat.num_history_ids == (0 if changed else 256)
        if store:
            hashes = scheduler.kv_connector._get_request_block_hashes(repeat, repeat.seq_id, 256, 64)
            assert (hashes[-1] == saved.block_hashes[-1]) is not changed


def test_mtp_decode_saves_first_64_boundary_then_every_1024_tokens():
    scheduler = _scheduler(store=True, alignment=64, interval=1024)
    seq = scheduler.add_session(0).add_sequence(list(range(190)))
    scheduler.state_manager.allocate(seq)
    seq.update_token_ids([50], draft_token_ids=[51, 52, 53], mode=UpdateTokenMode.PREFILL)
    for step in (192, 1216):
        # Fast-forward accepted progress to the next verification window.
        seq.update_token_ids([51] * (step - 2 - seq.num_history_ids),
                             draft_token_ids=[51, 52, 53], mode=UpdateTokenMode.DECODE)
        scheduler.block_manager.allocate(seq, prealloc_size=8)
        batch_id, plan = scheduler.decode_checkpoints.prepare([seq], None, SimpleNamespace(max_q_seqlen=4))
        assert plan[1].tolist() == [step, step]
        mask = torch.ones(plan.size(1), dtype=torch.bool)
        seq.update_token_ids([51, 52], draft_token_ids=[53, 54, 55], mode=UpdateTokenMode.DECODE)
        scheduler.decode_checkpoints.finish(batch_id, mask)
        assert seq.prefix_cache.last_decode_save_step == step
        save = scheduler.build_connector_meta([]).save_requests[0]
        scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={save.save_id}))
        assert get_decode_save_step(seq, step + 1, step + 1023) is None
        assert get_decode_save_step(seq, step + 1024, step + 1024) == step + 1024
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 1
    assert scheduler.kv_connector._state_slots.get_num_free() == 2


@pytest.mark.parametrize('interval', [0, 1024])
def test_store_decode_save_waits_for_all_ranks_and_retries_failure(interval):
    scheduler = _scheduler(local=False, store=True, alignment=256, interval=interval)
    store = scheduler.kv_connector
    seq = scheduler.add_session(0).add_sequence(list(range(256)))
    scheduler.state_manager.allocate(seq)
    seq.update_token_ids([50], draft_token_ids=[], mode=UpdateTokenMode.PREFILL)
    first = _store_save(scheduler, seq, 256)
    assert seq.prefix_cache.last_decode_save_step == -1
    assert get_decode_save_step(seq, 510, 514) is None  # No extra copies while Store is pending.

    ranks = KVConnectorOutputAggregator(2)
    scheduler.update_connector_output(ranks.aggregate([
        KVConnectorOutput(completed_save_ids={first.save_id}, failed_save_ids={first.save_id}), None]))
    assert store._state_slots.get_num_free() == 1
    assert scheduler.kv_save_coordinator.has_pending()
    scheduler.update_connector_output(ranks.aggregate([
        None, KVConnectorOutput(completed_save_ids={first.save_id})]))
    assert store._state_slots.get_num_free() == 2
    assert not scheduler.kv_save_coordinator.has_pending()
    assert not seq.prefix_cache.decode_store_save_pending
    assert seq.prefix_cache.last_decode_save_step == -1
    assert get_decode_save_step(seq, 510, 514) == 512

    seq.update_token_ids([51] * 256, draft_token_ids=[], mode=UpdateTokenMode.DECODE)
    retry = _store_save(scheduler, seq, 512)
    assert retry.start_block == 0  # Repair the failed FA prefix as well as saving the new state.
    scheduler.update_connector_output(ranks.aggregate([
        KVConnectorOutput(completed_save_ids={retry.save_id}),
        KVConnectorOutput(completed_save_ids={retry.save_id})]))
    assert seq.prefix_cache.last_decode_save_step == 512
    assert get_decode_save_step(seq, 1534, 1538) == (1536 if interval else None)
    scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={retry.save_id}))
    assert store._state_slots.get_num_free() == 2
    assert not scheduler.kv_save_coordinator.has_pending()


def test_decode_store_save_covers_unconfirmed_prefill_waves():
    scheduler = _scheduler(local=False, store=True, alignment=256, interval=1024)
    seq = scheduler.add_session(0).add_sequence(list(range(512)))
    scheduler.state_manager.allocate(seq)
    scheduler.block_manager.allocate(seq)
    prefill = scheduler.build_connector_meta(
        [seq], connector_token_lens=(256,), connector_state_ids=(int(seq.logical_state),)).save_requests[0]
    seq.update_token_ids([50], draft_token_ids=[], mode=UpdateTokenMode.PREFILL)
    decode = _store_save(scheduler, seq, 512)
    assert decode.start_block == prefill.start_block == 0
    assert len(decode.block_ids) == 8
    scheduler.update_connector_output(KVConnectorOutput(
        completed_save_ids={prefill.save_id}, failed_save_ids={prefill.save_id}))
    assert seq.prefix_cache.last_decode_save_step == -1
    scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={decode.save_id}))
    assert seq.prefix_cache.last_decode_save_step == 512
    assert get_decode_save_step(seq, 1534, 1538) == 1536
    assert not scheduler.kv_save_coordinator.has_pending()


@pytest.mark.parametrize('local', [False, True])
def test_aligned_prefill_publication_counts_as_the_first_checkpoint(local):
    scheduler = _scheduler(local=local, store=True, speculative=False, alignment=256,
                           interval=1024, last_only=False)
    seq = scheduler.add_session(0).add_sequence(list(range(256)))
    scheduler.state_manager.allocate(seq)
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    scheduler.state_checkpoints.reserve_prefill_save_batch([seq], (256,))
    save = scheduler.build_connector_meta(
        [seq], connector_token_lens=(256,), connector_state_ids=(int(seq.logical_state),)).save_requests[0]
    scheduler.state_checkpoints.publish_saves([seq])
    assert seq.prefix_cache.last_decode_save_step == (256 if local else -1)
    assert get_decode_save_step(seq, 256, 256) is None
    scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={save.save_id}))
    assert seq.prefix_cache.last_decode_save_step == 256
    assert get_decode_save_step(seq, 1278, 1282) == 1280


def test_late_store_completion_preserves_new_turn_and_newer_local_checkpoint():
    scheduler = _scheduler(store=True)
    seq = scheduler.add_session(0).add_sequence(list(range(256)))
    scheduler.state_manager.allocate(seq)
    seq.update_token_ids([50], draft_token_ids=[], mode=UpdateTokenMode.PREFILL)
    first = _store_save(scheduler, seq, 256)
    # Local saves can continue while Store is pending.
    seq.prefix_cache.last_decode_save_step = 256
    assert get_decode_save_step(seq, 318, 322) == 320
    seq.update_token_ids([99], mode=UpdateTokenMode.INPUTS)
    seq.set_step(seq.num_history_ids)
    seq.update_token_ids([50], draft_token_ids=[], mode=UpdateTokenMode.PREFILL)
    seq.update_token_ids([51] * 254, draft_token_ids=[], mode=UpdateTokenMode.DECODE)
    second = _store_save(scheduler, seq, 512)
    scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={first.save_id}))
    assert seq.prefix_cache.decode_store_save_pending
    assert seq.prefix_cache.last_decode_save_step == -1
    seq.prefix_cache.last_decode_save_step = 640  # A later local publication won the race.
    scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={second.save_id}))
    assert not seq.prefix_cache.decode_store_save_pending
    assert seq.prefix_cache.last_decode_save_step == 640


@pytest.mark.parametrize('drain', [False, True])
def test_ended_decode_store_save_releases_leases_without_recreating_progress(drain):
    scheduler = _scheduler(local=False, store=True)
    store = scheduler.kv_connector
    seq = scheduler.add_session(0).add_sequence(list(range(256)))
    scheduler.state_manager.allocate(seq)
    seq.update_token_ids([50], draft_token_ids=[], mode=UpdateTokenMode.PREFILL)
    save = _store_save(scheduler, seq, 256)
    scheduler.end_session(0)
    assert scheduler.kv_save_coordinator.has_pending()
    assert store._state_slots.get_num_free() == 1
    if drain:
        scheduler.finish_kv_transfers_after_worker_drain()
    else:
        scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={save.save_id}))
    assert store._state_slots.get_num_free() == 2
    assert not scheduler.kv_save_coordinator.has_pending()
    assert seq.seq_id not in store._next_save_block
    assert seq.prefix_cache.last_decode_save_step == -1
    assert not seq.prefix_cache.decode_store_save_pending


@pytest.mark.parametrize('speculative', [False, True])
def test_first_checkpoint_can_use_final_prefill_but_not_an_intermediate_chunk(speculative):
    scheduler = _scheduler(speculative=speculative)
    seq = scheduler.add_session(0).add_sequence(list(range(256)))
    scheduler.block_manager.allocate(seq, prealloc_size=4)
    scheduler.state_manager.allocate(seq)
    coordinator = scheduler.decode_checkpoints
    inputs = SimpleNamespace(is_dummy=False, is_chunk=True, is_last_chunk=False)
    assert coordinator.prepare([seq], inputs, None) is None
    inputs.is_last_chunk = True
    batch_id, plan = coordinator.prepare([seq], inputs, None)
    mask = torch.ones(plan.size(1), dtype=torch.bool)
    seq.update_token_ids([50], draft_token_ids=[51, 52, 53], mode=UpdateTokenMode.PREFILL)
    coordinator.finish(batch_id, mask)
    assert seq.prefix_cache.last_decode_save_step == 256
    inputs.is_dummy = True
    assert coordinator.prepare([seq], inputs, None) is None


def test_first_checkpoint_retries_after_state_capacity_recovers():
    scheduler = _scheduler(alignment=64, interval=1024)
    states = scheduler.state_manager
    seq = scheduler.add_session(0).add_sequence(list(range(256)))
    states.allocate(seq)
    scheduler.block_manager.allocate(seq)
    held = [states.allocate_checkpoint_state() for _ in range(states.get_num_free_checkpoint())]
    assert scheduler.decode_checkpoints.prepare(
        [seq], SimpleNamespace(is_dummy=False, is_chunk=False), None) is None
    assert seq.prefix_cache.last_decode_save_step == -1
    states.free_checkpoint_state(held.pop())
    seq.update_token_ids([50], draft_token_ids=[51, 52, 53], mode=UpdateTokenMode.PREFILL)
    seq.update_token_ids([51] * 62, draft_token_ids=[52, 53, 54], mode=UpdateTokenMode.DECODE)
    scheduler.block_manager.allocate(seq, prealloc_size=4)
    batch_id, plan = scheduler.decode_checkpoints.prepare([seq], None, SimpleNamespace(max_q_seqlen=4))
    assert plan[1].tolist() == [320]
    seq.update_token_ids([52, 53], draft_token_ids=[54, 55, 56], mode=UpdateTokenMode.DECODE)
    scheduler.decode_checkpoints.finish(batch_id, torch.tensor([True]))
    assert seq.prefix_cache.last_decode_save_step == 320
    assert get_decode_save_step(seq, 1343, 1345) == 1344
    for slot in held:
        states.free_checkpoint_state(slot)
    assert states.get_num_allocated_checkpoint_states() == 1


def test_queued_checkpoint_does_not_advance_a_new_input_turn():
    scheduler = _scheduler(store=True)
    seq = scheduler.add_session(0).add_sequence(list(range(256)))
    scheduler.state_manager.allocate(seq)
    scheduler.block_manager.allocate(seq)
    batch_id, plan = scheduler.decode_checkpoints.prepare(
        [seq], SimpleNamespace(is_dummy=False, is_chunk=False), None)
    seq.update_token_ids([50], draft_token_ids=[], mode=UpdateTokenMode.PREFILL)
    seq.update_token_ids([99], mode=UpdateTokenMode.INPUTS)
    scheduler.decode_checkpoints.finish(batch_id, torch.ones(plan.size(1), dtype=torch.bool))
    assert seq.prefix_cache.decode_checkpoint_node is None
    assert seq.prefix_cache.last_decode_save_step == -1
    assert get_decode_save_step(seq, 382, 386) == 384
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 0
    assert scheduler.kv_connector._state_slots.get_num_free() == 2


@pytest.mark.parametrize('stop_pos', [0, 2])
def test_terminal_checkpoint_respects_the_truncated_accepted_prefix(stop_pos):
    scheduler = _scheduler()
    seq = scheduler.add_session(0).add_sequence(list(range(254)))
    scheduler.block_manager.allocate(seq, prealloc_size=8)
    scheduler.state_manager.allocate(seq)
    seq.activate()
    scheduler.activate_seqs([seq])
    seq.update_token_ids([50], draft_token_ids=[51, 52, 53], mode=UpdateTokenMode.PREFILL)
    batch_id, plan = scheduler.decode_checkpoints.prepare([seq], None, SimpleNamespace(max_q_seqlen=4))
    tokens = [51, 52, 53, 54]
    # CPU publication independently checks the stop-truncated sequence even
    # if a worker reports that its candidate was copied.
    mask = torch.ones(plan.size(1), dtype=torch.bool)
    seq.update_token_ids(tokens, draft_token_ids=[], stop_pos=stop_pos, mode=UpdateTokenMode.DECODE)
    seq.finish()
    scheduler.decode_checkpoints.finish(batch_id, mask)
    assert seq.prefix_cache.last_decode_save_step == (256 if stop_pos == 2 else -1)


@pytest.mark.parametrize('copied', [None, False, True])
def test_engine_always_prefetches_before_checkpoint_confirmation(copied):
    events = []
    response = {'finished': True}
    mask = None if copied is None else torch.tensor([copied])
    model_output = BatchedOutputs(next_token_ids=torch.tensor([2]), stopped=torch.tensor([True]),
                                  decode_checkpoint_mask=mask)

    async def get_output():
        events.append('output')
        return model_output

    async def prefetch():
        events.append('prefetch')
        return None, None

    def finish(*args):
        events.append('acceptance')
        return response

    def confirm(batch_id, actual):
        assert batch_id == 7 and actual is mask
        events.append('confirm')

    loop = EngineLoop.__new__(EngineLoop)
    loop.config = EngineLoopConfig(role=EngineRole.Hybrid)
    loop._sleep_requested = False
    loop.inputs_maker = SimpleNamespace(update_running_seqs=lambda *args: None, prefetch_next_inputs=prefetch)
    loop.state_checkpoints = SimpleNamespace(
        finish_forward_dispatch=lambda *args, **kwargs: None,
        unpin_saves=lambda *args: events.append('unpin'))
    loop.executor = SimpleNamespace(get_output_async=get_output)
    loop.scheduler = SimpleNamespace(decode_checkpoints=SimpleNamespace(finish=confirm))
    loop.resp_queue = SimpleNamespace(put_nowait=lambda output: events.append('response'))
    loop._finish_forward_output = finish
    cache_inputs = None if copied is None else CacheCheckpointInputs(decode_save_id=7)
    asyncio.run(loop._main_loop_get_outputs([object()], dict(inputs=None, delta=object(), cache_inputs=cache_inputs)))
    assert events == (['prefetch', 'output', 'unpin', 'acceptance']
                      + ([] if copied is None else ['confirm']) + ['response'])


@pytest.mark.parametrize('drain', [False, True])
def test_provisional_decode_copy_owns_slots_across_session_end(drain):
    scheduler = _scheduler(store=True)
    seq = scheduler.add_session(0).add_sequence(list(range(256)))
    scheduler.state_manager.allocate(seq)
    scheduler.block_manager.allocate(seq)
    source_slot = seq.logical_state
    batch_id, plan = scheduler.decode_checkpoints.prepare(
        [seq], SimpleNamespace(is_dummy=False, is_chunk=False), None)
    scheduler.end_session(0)
    assert seq.logical_state == -1
    assert scheduler.state_manager.get_num_runtime_states() == 1
    assert scheduler.state_checkpoints.evict(1) == 0
    other_slot = scheduler.state_manager.allocate_state()
    assert other_slot != source_slot
    scheduler.state_manager.free_state(other_slot)
    if drain:
        scheduler.finish_kv_transfers_after_worker_drain()
    else:
        scheduler.decode_checkpoints.finish(batch_id, torch.ones(plan.size(1), dtype=torch.bool))
    assert scheduler.state_manager.get_num_runtime_states() == 0
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 0
    assert scheduler.kv_connector._state_slots.get_num_free() == 2
    assert seq.prefix_cache.last_decode_save_step == -1


def test_decode_publication_failure_releases_provisional_resources(monkeypatch):
    scheduler = _scheduler(store=True)
    seq = scheduler.add_session(0).add_sequence(list(range(256)))
    scheduler.state_manager.allocate(seq)
    scheduler.block_manager.allocate(seq)
    batch_id, plan = scheduler.decode_checkpoints.prepare(
        [seq], SimpleNamespace(is_dummy=False, is_chunk=False), None)
    seq.update_token_ids([50], draft_token_ids=[], mode=UpdateTokenMode.PREFILL)
    index = scheduler.state_checkpoints._index
    add = index.add

    def fail_after_index(node):
        add(node)
        raise RuntimeError('injected index failure')

    monkeypatch.setattr(index, 'add', fail_after_index)
    with pytest.raises(RuntimeError, match='injected index failure'):
        scheduler.decode_checkpoints.finish(batch_id, torch.ones(plan.size(1), dtype=torch.bool))
    assert seq.prefix_cache.trie_cursor.state_checkpoint is None
    assert not index._buckets
    assert seq.prefix_cache.last_decode_save_step == -1
    assert scheduler.state_manager.get_num_allocated_checkpoint_states() == 0
    assert scheduler.kv_connector._state_slots.get_num_free() == 2
    scheduler.end_session(0)
    assert scheduler.state_manager.get_num_runtime_states() == 0


def test_decode_publication_preserves_blocks_used_by_prefetched_forward():
    scheduler = _scheduler(store=True)
    sequences = [scheduler.add_session(i).add_sequence(list(range(254))) for i in range(2)]
    for seq in sequences:
        scheduler.state_manager.allocate(seq)
        scheduler.block_manager.allocate(seq, prealloc_size=8)
        seq.update_token_ids([50], draft_token_ids=[51, 52, 53], mode=UpdateTokenMode.PREFILL)
    batch_id, plan = scheduler.decode_checkpoints.prepare(sequences, None, SimpleNamespace(max_q_seqlen=4))
    # Both next forwards have already received these private block tables.
    prefetched_blocks = [seq.logical_blocks.get_real_blocks().copy() for seq in sequences]
    for seq in sequences:
        seq.update_token_ids([51, 52], draft_token_ids=[53, 54, 55], mode=UpdateTokenMode.DECODE)
    scheduler.decode_checkpoints.finish(batch_id, torch.ones(plan.size(1), dtype=torch.bool))
    assert sequences[0].prefix_cache.trie_cursor is sequences[1].prefix_cache.trie_cursor
    metadata = scheduler.build_connector_meta([])
    for seq, blocks, save in zip(sequences, prefetched_blocks, metadata.save_requests):
        assert seq.logical_blocks.get_real_blocks().tolist() == blocks.tolist()
        assert list(save.logical_block_ids) == blocks[:4].tolist()
    scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={save.save_id
                                                                          for save in metadata.save_requests}))
    assert not scheduler.kv_save_coordinator.has_pending()
    # Sequence ownership keeps the second forward's private prefix allocated
    # after both temporary copy pins and Store leases have been released.
    refs = scheduler.block_manager.allocator.get_ref_count(prefetched_blocks[1])
    assert (refs > 0).all()
