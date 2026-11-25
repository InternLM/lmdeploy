# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig, SSMStateCapacity
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest, PDControlRequest, PDHandoff
from lmdeploy.pytorch.disagg.messages import PDTransferResult
from lmdeploy.pytorch.messages import SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.paging.pd_coordinator import prompt_digest
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


def make_export(cached_tokens=0):
    cache = CacheConfig(max_batches=2, block_size=4, num_gpu_blocks=4, num_cpu_blocks=0)
    scheduler = Scheduler(SchedulerConfig(max_batches=2, max_session_len=32), cache,
                          SequenceMeta(4, strategy=ARSequenceStrategy()))
    seq = scheduler.add_session(1).add_sequence(np.arange(8), preserve_cache=True)
    seq.pd_prompt_tokens = np.arange(8)
    seq.cached_tokens = cached_tokens
    scheduler.block_manager.allocate(seq)
    seq.sampling_param.random_seed = 42
    seq.update_token_ids(np.array([8]), mode=UpdateTokenMode.PREFILL)
    handoff = scheduler.pd_coordinator.publish(seq, 'schema', [])
    return scheduler, seq, handoff


@pytest.mark.parametrize('cached_tokens', [0, 4])
def test_handoff_preserves_prefill_cache_hits_separately_from_imported_history(cached_tokens):
    _, _, handoff = make_export(cached_tokens)
    assert handoff.prefill_cached_tokens == cached_tokens
    cache = CacheConfig(max_batches=1, block_size=4, num_gpu_blocks=4, num_cpu_blocks=0)
    scheduler = Scheduler(SchedulerConfig(max_batches=1, max_session_len=32), cache,
                          SequenceMeta(4, strategy=ARSequenceStrategy()))
    migration = MigrationRequest(protocol=MigrationProtocol.RDMA, remote_engine_id='prefill',
                                 remote_session_id=1, handoff=handoff)
    seq = scheduler.add_session(2).add_sequence(np.arange(8), migration_request=migration)
    assert scheduler.schedule_migration() == [seq]
    result = PDTransferResult(transfer_id=handoff.transfer_id, status='completed')
    assert scheduler.pd_coordinator.finish_receive(seq, result) == 'ready'
    assert seq.num_history_ids == 8
    assert seq.remote_prefill_cached_tokens == cached_tokens
    assert seq.cached_tokens == 0
    seq.update_token_ids(np.arange(2))
    assert seq.remote_prefill_cached_tokens is None


def test_granted_export_survives_real_session_end_and_rejects_wrong_receiver():
    scheduler, seq, handoff = make_export()
    owner = scheduler.pd_coordinator
    request = dict(transfer_id=handoff.transfer_id, engine_epoch=handoff.engine_epoch, receiver_epoch='decode')
    owner.control(PDControlRequest(action='claim', **request))
    free = scheduler.block_manager.get_num_free_gpu_blocks()
    scheduler.end_session(seq.session_id)
    assert scheduler.get_session(1) is not None
    assert scheduler.block_manager.get_num_free_gpu_blocks() == free
    for reset in (scheduler.shutdown, scheduler.reset_cache_after_worker_drain):
        with pytest.raises(RuntimeError, match='PD cache is still owned'):
            reset()
    assert owner.expire_exports(owner.exports[handoff.transfer_id].renewed_at + 299) == []
    with pytest.raises(ValueError, match='granted receiver'):
        owner.control(PDControlRequest(action='release', **(request | dict(receiver_epoch='other'))))
    assert owner.control(PDControlRequest(action='release', **request)) is seq
    scheduler.end_session(1)
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4
    assert owner.control(PDControlRequest(action='release', **request)) is None
    scheduler.reset_cache_after_worker_drain()
    assert owner.epoch != handoff.engine_epoch


def test_revoke_before_grant_prevents_late_reader():
    scheduler, seq, handoff = make_export()
    assert scheduler.pd_coordinator.expire_exports(float('inf')) == [seq]
    scheduler.end_session(seq.session_id)
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4
    with pytest.raises(ValueError, match='no longer available'):
        scheduler.pd_coordinator.control(PDControlRequest(
            action='claim', transfer_id=handoff.transfer_id, engine_epoch=handoff.engine_epoch,
            receiver_epoch='late'))


@pytest.mark.parametrize('status', ['failed_drained', 'completion_unknown'])
def test_receive_reuses_prefix_and_cancellation_retains_private_destination(status):
    cache = CacheConfig(max_batches=2, block_size=4, num_gpu_blocks=8, num_cpu_blocks=0,
                        enable_prefix_caching=True)
    scheduler = Scheduler(SchedulerConfig(max_batches=2, max_session_len=32), cache,
                          SequenceMeta(4, strategy=ARSequenceStrategy()))
    original = scheduler.add_session(1).add_sequence(np.arange(10))
    scheduler.block_manager.allocate(original)
    scheduler.block_trie.allocate(original, token_limit=8)
    prefix = scheduler.block_manager.get_block_table(original)[:2].copy()
    scheduler.end_session(1)
    migration = MigrationRequest(protocol=MigrationProtocol.RDMA, remote_engine_id='prefill', remote_session_id=2)
    seq = scheduler.add_session(2).add_sequence(np.arange(10), migration_request=migration)
    migration.handoff = PDHandoff(transfer_id='read', engine_epoch='prefill-epoch', cache_schema='model',
                                   prompt_digest=prompt_digest(seq, seq.valid_ids), prompt_token_count=10,
                                   valid_token_count=10, first_token_id=10, random_seed=42, random_offset=11,
                                   block_size=4, block_ids=[0, 1, 2])
    assert scheduler.schedule_migration() == [seq]
    owner = scheduler.pd_coordinator
    receive = owner.receives[seq.seq_id]
    assert seq.num_history_ids == 8
    assert np.array_equal(scheduler.block_manager.get_block_table(seq)[:2], prefix)
    assert len(receive.request.block_pairs) == 1
    free = scheduler.block_manager.get_num_free_gpu_blocks()
    scheduler.end_session(2)
    assert scheduler.block_manager.get_num_free_gpu_blocks() == free
    action = owner.finish_receive(seq, PDTransferResult(transfer_id='read', status=status))
    if status == 'completion_unknown':
        scheduler.end_session(2)
        assert action == 'quarantined' and owner.snapshot()['quarantined'] == 1
        assert scheduler.block_manager.get_num_free_gpu_blocks() == free
        assert not owner.expire_receive(seq, 'read', receive.failed_at + 299)
        assert owner.expire_receive(seq, 'read', receive.failed_at + 300)
        scheduler.end_session(2)
        assert scheduler.block_manager.get_num_free_gpu_blocks() == free + 1
        assert owner.finish_receive(seq, PDTransferResult(transfer_id='read', status='completed')) == 'stale'
    else:
        assert action == 'end'
        scheduler.end_session(2)
        assert scheduler.block_manager.get_num_free_gpu_blocks() == free + 1


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('expire', [False, True])
def test_receive_state_lifetime_through_direct_decode_or_expiry(compact, expire, monkeypatch):
    cache = CacheConfig(max_batches=1, block_size=4, num_gpu_blocks=8, num_cpu_blocks=0,
                        states_shapes=[((2,), torch.float32)], num_state_caches=2,
                        enable_prefix_caching=True,
                        compact_state_capacity=SSMStateCapacity(2, 4, 0) if compact else None)
    scheduler = Scheduler(SchedulerConfig(max_batches=1, max_session_len=32), cache,
                          SequenceMeta(4, strategy=ARSequenceStrategy()))
    migration = MigrationRequest(protocol=MigrationProtocol.RDMA, remote_engine_id='prefill', remote_session_id=2)
    seq = scheduler.add_session(2).add_sequence(np.arange(10), migration_request=migration)
    migration.handoff = PDHandoff(transfer_id='read', engine_epoch='source', cache_schema='model',
                                   prompt_digest=prompt_digest(seq, seq.valid_ids), prompt_token_count=10,
                                   valid_token_count=10, first_token_id=10, random_seed=42, random_offset=11,
                                   block_size=4, block_ids=[0, 1, 2], state_ids=[17],
                                   auxiliary_id=4, legacy_state_id=None if compact else 4)
    assert scheduler.schedule_migration() == [seq]
    seq.state.activate()
    state_id = seq.logical_state
    if compact:
        bank = scheduler.state_manager.bank_for(seq)
        assert bank.pending_load is not None and bank.loaded_input is None
    result = PDTransferResult(transfer_id='read', status='completed')
    if expire:
        owner = scheduler.pd_coordinator
        assert owner.finish_receive(seq, result.model_copy(update={'status': 'completion_unknown'})) == 'quarantined'
        assert owner.expire_receive(seq, 'read', owner.receives[seq.seq_id].failed_at + 300)
        scheduler.end_session(2)
        replacement = scheduler.add_session(3).add_sequence(np.arange(10))
        scheduler.block_manager.allocate(replacement)
        scheduler.state_manager.allocate(replacement)
        assert replacement.logical_state == state_id
        assert owner.finish_receive(seq, result) == 'stale'
        assert replacement.logical_state == state_id
        return
    assert scheduler.pd_coordinator.finish_receive(seq, result) == 'ready'
    scheduler.block_trie.allocate(seq, token_limit=10)
    seq.state.deactivate()
    scheduler.resume_completed_migrations()

    def unexpected_match(seq):
        pytest.fail('PD replay must retain its imported state without another prefix match.')

    monkeypatch.setattr(scheduler.block_trie, 'match', unexpected_match)
    assert scheduler.schedule_pd_decode() == [seq]
    assert seq.logical_state == state_id
    if compact:
        assert bank.loaded_input == (bank.unit_ids[0], 10)
    assert seq.num_history_ids == 10 and seq.token_ids.tolist() == [10]
    assert seq.num_new_tokens == 1


def test_grant_renewal_abort_and_expiry_preserve_other_owners(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('lmdeploy.pytorch.paging.pd_coordinator.monotonic', lambda: clock[0])
    scheduler, seq, handoff = make_export()
    owner = scheduler.pd_coordinator
    control = dict(transfer_id=handoff.transfer_id, engine_epoch=owner.epoch,
                   receiver_epoch='decode', receiver_engine_id='decode-endpoint')
    owner.control(PDControlRequest(action='claim', **control))
    clock[0] = 390
    owner.control(PDControlRequest(action='claim', **control))
    assert owner.expire_exports(400) == []

    # A concurrent Store save owns its own reference to the same allocation.
    blocks = seq.logical_blocks.get_real_blocks().copy()
    scheduler.block_manager.pin_logical_blocks(blocks)
    scheduler.end_session(seq.session_id)
    peer, notification = owner.notifications.pop()
    assert peer == 'decode-endpoint' and notification.action == 'abort'
    assert notification.reason == 'Prefill request was cancelled.'
    clock[0] = 500
    owner.control(PDControlRequest(action='revoke', **control))
    assert owner.expire_exports(689) == []
    assert owner.expire_exports(690) == [seq]
    scheduler.end_session(seq.session_id)
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 2
    scheduler.block_manager.release_logical_blocks(blocks)
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4
    assert owner.control(PDControlRequest(action='release', **control)) is None
    assert owner.snapshot()['forced_releases'] == 1
