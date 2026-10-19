# Copyright (c) OpenMMLab. All rights reserved.
"""Actual phase-aware scheduler admission and compact lifetime contracts."""
import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, CompactStateCapacity, SchedulerConfig
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta
from lmdeploy.pytorch.paging.compact_state import CompactStateManager
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


def _scheduler(width=4, requests=2, **kwargs):
    config = CacheConfig(max_batches=requests, block_size=16, num_cpu_blocks=0, num_gpu_blocks=32,
                          states_shapes=[((2,), torch.float32)], num_state_caches=requests + 1,
                          **kwargs)
    capacity = CompactStateCapacity(requests + 1, width)
    return Scheduler(SchedulerConfig(max_batches=requests, max_session_len=256), config,
                     seq_meta=SequenceMeta(16, strategy=ARSequenceStrategy()), compact_capacity=capacity)


@pytest.mark.parametrize('width', [1, 4, 6])
def test_real_scheduler_admits_disjoint_banks_and_retains_cancelled_accesses(width):
    scheduler = _scheduler(width)
    owner = scheduler.state_manager
    assert isinstance(owner, CompactStateManager)
    sequences = [scheduler.add_session(i).add_sequence([1] * 17) for i in range(3)]
    output = scheduler.schedule(is_prefill=True)
    assert output.running == sequences[:2]
    banks = [owner.bank_for(seq) for seq in output.running]
    assert [seq.logical_state for seq in output.running] == [bank.auxiliary_id for bank in banks]
    assert all(len(bank.unit_ids) == 1 for bank in banks)
    assert not set(banks[0].unit_ids).intersection(banks[1].unit_ids)
    assert owner.get_num_free_runtime() == 0
    access = owner.submit_prefill(tuple(banks))
    scheduler.end_session(0)
    assert sequences[0].logical_state == -1 and banks[0].closing
    # A request slot and its whole bank stay held even though the sequence is gone.
    assert owner.get_num_free_runtime() == 0
    assert scheduler.schedule(is_prefill=True).running == []
    assert sequences[2].logical_state == -1 and sequences[2].num_blocks == 0
    owner.acknowledge(access.access_id)
    assert owner.get_num_free_runtime() == 1
    assert scheduler.schedule(is_prefill=True).running == [sequences[2]]
    replacement = owner.bank_for(sequences[2])
    assert not set(replacement.unit_ids).intersection(banks[1].unit_ids)
    scheduler.end_session(1)
    scheduler.end_session(2)
    assert owner.get_num_free_runtime() == 2
    assert not owner._banks and not owner._units and not owner._accesses


def test_stopped_sequence_keeps_bank_until_evicted_not_until_batch_reindex():
    scheduler = _scheduler(requests=1)
    owner = scheduler.state_manager
    first = scheduler.add_session(0).add_sequence([1] * 17)
    scheduler.schedule(is_prefill=True)
    bank = owner.bank_for(first)
    first.state.stop()
    assert owner.bank_for(first) is bank
    owner.allocate(first)  # Idempotent admission never allocates another bank.
    assert owner.get_num_free_runtime() == 0
    scheduler.end_session(0)
    assert owner.get_num_free_runtime() == 1
    owner.free(first)  # Sequence lifecycle cleanup is idempotent.
    with pytest.raises(ValueError, match='does not own'):
        owner.bank_for(first)


@pytest.mark.parametrize('case', ['connector', 'geometry', 'stateless'])
def test_unintegrated_scheduler_modes_fail_before_allocating_pools(case):
    config = CacheConfig(max_batches=1, block_size=16, num_cpu_blocks=0, num_gpu_blocks=32,
                          num_state_caches=2, states_shapes=[] if case == 'stateless' else [((2,), torch.float32)],
                          enable_prefix_caching=False)
    capacity = CompactStateCapacity(3 if case == 'geometry' else 2, 4)
    exception = NotImplementedError if case == 'connector' else ValueError
    with pytest.raises(exception, match='Compact scheduler'):
        Scheduler(SchedulerConfig(max_batches=1, max_session_len=256), config,
                  kv_connector=object() if case == 'connector' else None, compact_capacity=capacity)


@pytest.mark.parametrize('width', [1, 4, 6])
def test_no_apc_sleep_drain_restores_all_request_capacity(width):
    scheduler = _scheduler(width=width)
    owner = scheduler.state_manager
    sequences = [scheduler.add_session(i).add_sequence([1] * 17) for i in range(2)]
    assert scheduler.schedule(is_prefill=True).running == sequences
    access = owner.submit_prefill(tuple(owner.bank_for(seq) for seq in sequences))
    for i in range(2):
        scheduler.end_session(i)
    assert owner.get_num_free_runtime() == 0
    scheduler.finish_kv_transfers_after_worker_drain()
    assert access.finished and owner.get_num_free_runtime() == 2
    assert not owner._units and not owner._banks and not owner._accesses


@pytest.mark.parametrize('role', ['kv_producer', 'kv_consumer', 'kv_both'])
@pytest.mark.parametrize('apc', [False, True])
def test_constructor_binds_only_existing_owner_before_admission(role, apc):
    from lmdeploy.messages import KVTransferConfig
    from lmdeploy.pytorch.kv_connector import KVConnectorRole, build_kv_connector

    budget, credits = (2 if apc else 0), (1 if role != 'kv_consumer' else 0)
    capacity = CompactStateCapacity(3, 4, budget, credits)
    config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=32,
        states_shapes=[((2,), torch.float32)], num_state_caches=3 + budget + credits,
        prefix_cache_state_budget=budget, enable_prefix_caching=apc, compact_state_capacity=capacity,
        mooncake_state_save_slots=1, prefill_save_alignment=64,
        kv_transfer_config=KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role=role))
    connector = build_kv_connector(KVConnectorRole.SCHEDULER, config)
    scheduler = Scheduler(SchedulerConfig(max_batches=1, max_session_len=256), config,
        kv_connector=connector, compact_capacity=capacity)
    owner, store = scheduler.state_manager, connector.connector_scheduler
    assert isinstance(owner, CompactStateManager)
    assert store._compact_owner is owner and store._state_slots is None
    assert scheduler.sequence_lifecycle._connector is connector
    assert scheduler.kv_load_coordinator.connector is connector
    assert scheduler.kv_load_coordinator.lookup_enabled == (role != 'kv_producer')
    assert scheduler.seq_meta.compact_prefill_checkpoint_stride == (64 if apc or credits else 0)
    with pytest.raises(ValueError, match='exactly once'):
        connector.bind_compact_state_manager(owner)
    scheduler.shutdown()


@pytest.mark.parametrize('width', [1, 4, 6])
def test_decode_promotion_runs_even_without_kv_growth(width, monkeypatch):
    scheduler = _scheduler(width)
    owner = scheduler.state_manager
    seqs = [scheduler.add_session(i).add_sequence([1] * 17) for i in range(2)]
    assert scheduler.schedule(is_prefill=True).running == seqs
    banks = tuple(owner.bank_for(seq) for seq in seqs)
    prefill = owner.submit_prefill(banks)
    original = prefill.bank_ids
    scheduler.activate_seqs(seqs)
    monkeypatch.setattr(scheduler.block_manager, 'num_required_blocks', lambda *args: 0)
    assert scheduler.schedule_running(seqs) == [True, True]
    assert all(bank.phase == 'decode' and len(bank.unit_ids) == width + 1 for bank in banks)
    assert prefill.bank_ids == original and all(len(ids) == 1 for ids in original)
    assert all(bank.unit_ids[:1] == ids for bank, ids in zip(banks, original))
    for i in range(2):
        scheduler.end_session(i)
    owner.complete(prefill)
    assert not owner._units and not owner._banks


def test_decode_expansion_invariant_failure_never_preempts_prefill(monkeypatch):
    scheduler = _scheduler(requests=1)
    owner = scheduler.state_manager
    seq = scheduler.add_session(0).add_sequence([1] * 17)
    scheduler.schedule(is_prefill=True)
    scheduler.activate_seqs([seq])
    bank = owner.bank_for(seq)
    original, blocks = bank.unit_ids, seq.logical_blocks.get_real_blocks().copy()
    monkeypatch.setattr(owner, 'promote_to_decode', lambda bank: False)
    with pytest.raises(RuntimeError, match='reserved capacity'):
        scheduler.schedule_running([seq])
    assert owner.bank_for(seq) is bank and bank.unit_ids is original
    assert (seq.logical_blocks.get_real_blocks() == blocks).all()
    assert seq.status == MessageStatus.RUNNING
    scheduler.end_session(0)
