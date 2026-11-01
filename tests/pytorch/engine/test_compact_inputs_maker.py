# Copyright (c) OpenMMLab. All rights reserved.
"""Real scheduler/InputMaker bank snapshots before worker-side ID binding."""
import asyncio
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.compact_state import SSMStateCursor
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig, SSMStateCapacity
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
from lmdeploy.pytorch.engine.inputs_maker import InputsMakerAsync, InputsMakerConfig
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.model_inputs import get_model_inputs_next_decoding
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
from lmdeploy.pytorch.strategies.ar.step_inputs import _reindex_model_inputs
from lmdeploy.pytorch.strategies.ar_spec.step_inputs import _reindex_model_inputs_arspec
from tests.pytorch.compact_state_reference import checkpoint_ready


def _maker(width=4, budget=256, requests=2, *, prefix=False, block_size=16, checkpoint_budget=0, compact=True,
           store_role=None, store_credits=1):
    from lmdeploy.messages import KVTransferConfig
    from lmdeploy.pytorch.kv_connector import KVConnectorRole, build_kv_connector

    credits = store_credits if store_role in ('kv_producer', 'kv_both') else 0
    transfer = None if store_role is None else KVTransferConfig(
        kv_connector='MooncakeStoreConnector', kv_role=store_role)
    cache = CacheConfig(max_batches=requests, block_size=block_size, num_cpu_blocks=0, num_gpu_blocks=128,
                        states_shapes=[((2,), torch.float32)],
                        num_state_caches=requests + 1 + checkpoint_budget + credits,
                        kv_transfer_config=transfer, mooncake_state_save_slots=store_credits,
                        prefix_cache_state_budget=checkpoint_budget, enable_prefix_caching=prefix,
                        prefill_save_alignment=block_size, max_prefill_token_num=budget)
    capacity = SSMStateCapacity(requests + 1, width, checkpoint_budget, credits) if compact else None
    cache.compact_state_capacity = capacity
    connector = build_kv_connector(KVConnectorRole.SCHEDULER, cache)
    scheduler = Scheduler(SchedulerConfig(max_batches=requests, max_session_len=4096), cache,
                          seq_meta=SequenceMeta(block_size, strategy=ARSequenceStrategy()),
                          kv_connector=connector)
    config = InputsMakerConfig(requests, budget, EngineRole.Hybrid, block_size, block_size, -1, prefix, 0,
                               is_ssm=True, spec_decoding=width > 1)
    return InputsMakerAsync(
        executor=SimpleNamespace(device_type='cuda'), scheduler=scheduler,
        state_checkpoints=scheduler.state_checkpoints,
        adapter_manager=SimpleNamespace(num_adapters=lambda: 1),
        engine_strategy=SimpleNamespace(get_prealloc_size=lambda _: 0, get_num_decode_tokens=lambda: width,
                                        get_num_required_tokens=lambda: width),
        sampling_strategy=SimpleNamespace(make_sampling_inputs=lambda _: None),
        model_agent_strategy=SimpleNamespace(make_stopping_criteria=lambda _: None,
                                             make_extra_inputs=lambda *args: None),
        config=config, compact_state_manager=scheduler.state_manager if compact else None)


@pytest.mark.parametrize('width', [1, 4, 6])
def test_real_prefill_and_decode_have_distinct_owner_tickets_and_held_snapshots(width):
    maker = _maker(width)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    seqs = [scheduler.add_session(i).add_sequence([1] * n) for i, n in enumerate((72, 65))]
    prefill = maker._make_forward_inputs(True)
    inputs = prefill['inputs']
    assert prefill['running'] == seqs
    assert inputs.ssm_state_meta.partial_bank
    assert inputs.ssm_state_meta.state_ids.tolist() == [
        list(owner.bank_for(seq).unit_ids) + [-1] * (owner.capacity.bank_width - 1) for seq in seqs
    ]
    assert all(len(owner.bank_for(seq).unit_ids) == 1 for seq in seqs)
    assert inputs.ssm_state_meta.input_state_ids.tolist() == [-1, -1]
    assert inputs.ssm_state_meta.output_state_ids.tolist() == [[-1], [-1]]
    assert inputs.state_offsets.tolist() == [1, 2]
    carrier = prefill['cache_inputs']
    assert inputs.ssm_prefill_meta.cu_seqlens.tolist() == [0, 72, 137]
    assert carrier.state_restore_plan is carrier.state_save_plan is inputs.ssm_source_state_ids is None
    scheduler.activate_seqs(seqs)
    maker.update_running_seqs(seqs, inputs)
    # Host prefetch intentionally precedes the previous output/acknowledgment.
    decode = maker._make_forward_inputs(False)
    assert decode['inputs'] is None
    ticket = decode['cache_inputs'].ssm_state_access_id
    assert ticket != carrier.ssm_state_access_id
    assert decode['delta'].ssm_state_ids.tolist() == [list(owner.bank_for(seq).unit_ids) for seq in seqs]
    assert all(len(owner.bank_for(seq).unit_ids) == owner.capacity.bank_width for seq in seqs)
    assert decode['delta'].ssm_state_ids[:, 0].tolist() == inputs.ssm_state_meta.state_ids[:, 0].tolist()
    assert len(owner._accesses) == 2
    original = inputs.ssm_state_meta.state_ids.clone()
    scheduler.end_session(0)
    scheduler.end_session(1)
    owner.acknowledge(carrier.ssm_state_access_id)
    assert owner.get_num_free_runtime() == 0
    owner.acknowledge(ticket)
    assert owner.get_num_free_runtime() == 2
    torch.testing.assert_close(inputs.ssm_state_meta.state_ids, original, rtol=0, atol=0)
    assert not owner._accesses


def test_real_long_context_each_chunk_gets_exact_layout_and_new_ticket():
    maker = _maker(1, budget=64, requests=1)
    seq = maker.scheduler.add_session(0).add_sequence([1] * 137)
    tickets = []
    for length, history in ((64, 0), (64, 64), (9, 128)):
        payload = maker._make_forward_inputs(True)
        inputs, carrier = payload['inputs'], payload['cache_inputs']
        assert inputs.seq_length.tolist() == [length]
        assert inputs.history_lengths.tolist() == [history]
        assert inputs.is_chunk and inputs.is_last_chunk == (history == 128)
        assert inputs.ssm_prefill_meta.cu_seqlens.tolist() == [0, length]
        tickets.append(carrier.ssm_state_access_id)
        maker.scheduler.activate_seqs([seq])
        maker.update_running_seqs([seq], inputs)
        maker.compact_state_manager.acknowledge(tickets[-1])
    assert len(set(tickets)) == 3
    assert not maker.has_pending_long_context_chunk()


@pytest.mark.parametrize('failure', ['sampling', 'layout', 'carrier'])
def test_unsubmitted_preparation_failures_leave_no_access_holds(monkeypatch, failure):
    maker = _maker()
    maker.scheduler.add_session(0).add_sequence([1] * 72)
    owner = maker.compact_state_manager

    def fail(*args, **kwargs):
        raise RuntimeError('injected preparation failure')

    if failure == 'sampling':
        monkeypatch.setattr(maker.sampling_strategy, 'make_sampling_inputs', fail)
    elif failure == 'layout':
        monkeypatch.setattr('lmdeploy.pytorch.engine.inputs_maker.SSMPrefillMetadata.from_lengths', fail)
    else:
        # This particular replacement is after submit_decode, but before the
        # payload can be observed by any executor/worker.
        monkeypatch.setattr('lmdeploy.pytorch.engine.inputs_maker.replace', fail)
    with pytest.raises(RuntimeError, match='injected'):
        maker._make_forward_inputs(True)
    assert not owner._accesses
    assert all(not bank.accesses for bank in owner._banks.values())
    maker.scheduler.end_session(0)
    assert owner.get_num_free_runtime() == 2


def test_dispatch_failure_does_not_claim_gpu_quiescence():
    maker = _maker()
    maker.scheduler.add_session(0).add_sequence([1] * 72)

    async def fail(_):
        raise RuntimeError('worker delivery may already have occurred')

    maker.executor.forward_async = fail
    with pytest.raises(RuntimeError, match='delivery'):
        asyncio.run(maker.send_next_inputs())
    owner = maker.compact_state_manager
    assert len(owner._accesses) == 1
    maker.scheduler.end_session(0)
    assert owner.get_num_free_runtime() == 1
    # Test transport never submitted GPU work; production must drain workers.
    owner.acknowledge(next(iter(owner._accesses)))
    assert owner.get_num_free_runtime() == 2


def test_unsupported_copy_packet_rejects_before_holds():
    maker = _maker()
    maker.scheduler.add_session(0).add_sequence([1] * 72)
    maker._prepare_prefill_cache_inputs = lambda _: CacheCheckpointInputs(state_save_plan=((1,), (2,)))
    with pytest.raises(NotImplementedError, match='checkpoint/connector'):
        maker._make_forward_inputs(True)
    assert not maker.compact_state_manager._accesses


@pytest.mark.parametrize('width', [1, 4, 6])
@pytest.mark.parametrize('replace_bank', [False, True])
def test_real_host_packets_bind_then_decode_after_worker_batch_shrink(width, replace_bank):
    maker = _maker(width)
    owner = maker.compact_state_manager
    seqs = [maker.scheduler.add_session(i).add_sequence([1] * n) for i, n in enumerate((72, 65))]
    payload = maker._make_forward_inputs(True)
    inputs = payload['inputs'].to_device('cuda')
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.ssm_state_cursor = SSMStateCursor(owner.capacity.request_slots)
    agent._prev_chunk_output = None
    inputs = agent._prepare_inputs_prefill(inputs, None)
    torch.testing.assert_close(inputs.ssm_state_meta.input_state_ids, torch.zeros(2, device='cuda', dtype=torch.long))
    decoding = get_model_inputs_next_decoding(inputs, torch.zeros((1, 2 * width), device='cuda', dtype=torch.long),
                                              width, None, ssm_state_cursor=agent.ssm_state_cursor)
    maker.scheduler.activate_seqs(seqs)
    maker.update_running_seqs(seqs, payload['inputs'])
    maker.scheduler.end_session(0)
    frozen = None
    if replace_bank:
        torch.cuda.synchronize()
        owner.acknowledge(payload['cache_inputs'].ssm_state_access_id)
        bank = owner.bank_for(seqs[1])
        # Promotion adds non-current destinations. Freeze one before the
        # first full decode packet; the prefill current source stays private.
        assert owner.promote_to_decode(bank)
        frozen = checkpoint_ready(owner, bank, boundary=64)
    packet = maker._make_forward_inputs(False)
    delta = packet['delta'].to_device('cuda')
    delta.record_stream(torch.cuda.current_stream())
    assert delta.indice_cpu.tolist() == [1]
    steps = SimpleNamespace(model_inputs=decoding, extra_inputs=None, stopping_criteria=None, sampling_delta=None)

    def reindex(update):
        steps.model_inputs = (_reindex_model_inputs(steps.model_inputs, update) if width == 1 else
                              _reindex_model_inputs_arspec(steps.model_inputs, update, width - 1))

    steps.reindex = reindex
    from lmdeploy.pytorch.strategies.base.step_inputs import StepInputs
    steps.ssm_state_cursor = agent.ssm_state_cursor
    steps.apply_delta = lambda delta: StepInputs.apply_delta(steps, delta)
    agent.step_inputs = steps
    previous = decoding.ssm_state_meta
    actual, *_ = agent._get_inputs_from_delta(delta, SimpleNamespace(update_delta=lambda _: None))
    assert actual.state_offsets.tolist() == [seqs[1].logical_state]
    torch.testing.assert_close(actual.ssm_state_meta.input_state_ids, previous.input_state_ids[1:], rtol=0, atol=0)
    torch.testing.assert_close(actual.ssm_state_meta.state_ids, delta.ssm_state_ids, rtol=0, atol=0)
    assert not torch.isin(actual.ssm_state_meta.output_state_ids[:, :-1], actual.ssm_state_meta.input_state_ids).any()
    if not (replace_bank and width == 1):
        torch.testing.assert_close(actual.ssm_state_meta.output_state_ids[:, -1], actual.ssm_state_meta.input_state_ids)
    assert actual.ssm_state_meta is not previous
    if frozen is not None:
        assert frozen.unit_id not in previous.state_ids[1].tolist()
        assert frozen.unit_id not in actual.ssm_state_meta.state_ids[0].tolist()
        assert frozen.unit_id not in actual.ssm_state_meta.output_state_ids[0].tolist()
    # Numerical operator tests are separate; this test covers actual CPU packet
    # construction, H2D, worker reindex/binding and owner snapshot lifetimes.
    torch.cuda.synchronize()
    owner.acknowledge(payload['cache_inputs'].ssm_state_access_id)
    owner.acknowledge(packet['cache_inputs'].ssm_state_access_id)
    maker.scheduler.end_session(1)
    if frozen is not None:
        owner.release_reference(frozen)
    assert owner.get_num_free_runtime() == 2


@pytest.mark.parametrize('case', ['dummy', 'no_work', 'auxiliary_mismatch', 'decode_shape'])
def test_binding_ignores_nonwork_and_rejects_bad_row_identity_without_holds(case):
    maker = _maker()
    seq = maker.scheduler.add_session(0).add_sequence([1] * 72)
    maker.scheduler.schedule(is_prefill=True)
    inputs = maker.create_model_inputs([seq], True)
    payload = dict(inputs=inputs, delta=None, running=[seq], cache_inputs=None, kv_connector_metadata=None)
    if case == 'dummy':
        inputs.is_dummy = True
    elif case == 'no_work':
        payload['inputs'] = None
    elif case == 'auxiliary_mismatch':
        inputs.state_offsets = torch.tensor([2])
    else:
        payload['inputs'] = None
        payload['delta'] = SimpleNamespace(block_offsets=torch.zeros(2, 1),
                            ssm_state_ids=None, ssm_checkpoint_state_ids=None)
    if case in ('dummy', 'no_work'):
        maker._bind_compact_state(payload)
        assert payload['cache_inputs'] is None and inputs.ssm_state_meta is None
    else:
        with pytest.raises(ValueError, match='auxiliary IDs|request count'):
            maker._bind_compact_state(payload)
    assert not maker.compact_state_manager._accesses


def _prefix_fixture(width=4):
    """Actual trie/InputsMaker owners with an explicitly bound test trie."""
    from lmdeploy.pytorch.paging.block_trie import BlockTrie
    from lmdeploy.pytorch.paging.compact_state import SSMStateManager

    maker = _maker(width)
    scheduler = Scheduler(SchedulerConfig(max_batches=4, max_session_len=4096),
                          CacheConfig(max_batches=4, block_size=64, num_cpu_blocks=0, num_gpu_blocks=512,
                                      enable_prefix_caching=True),
                          seq_meta=SequenceMeta(64, strategy=ARSequenceStrategy()))
    owner = SSMStateManager(SSMStateCapacity(5, width, 8))
    trie = BlockTrie(allocator=scheduler.block_manager.allocator, block_size=64, enabled=True,
                     checkpoint_state_manager=owner)
    maker.scheduler, maker.compact_state_manager, maker.state_checkpoints = scheduler, owner, trie.state_checkpoints
    maker.config.enable_prefix_caching = True
    maker.config.block_size = maker.config.kernel_block_size = 64
    maker.kernel_blocks_per_kv = 1

    def add(tokens, match=False):
        seq = scheduler.add_session(len(scheduler.sessions)).add_sequence(tokens)
        if match:
            trie.match(seq)
        owner.allocate(seq)
        scheduler.block_manager.allocate(seq)
        trie.allocate(seq)
        return seq

    return maker, trie, add


def _seed_prefix(maker, add, tokens, step):
    seq = add(tokens)
    lifecycle, owner = maker.state_checkpoints, maker.compact_state_manager
    slot = lifecycle.reserve_save(seq, step=step)
    node = seq.prefix_cache.pending_save.node
    reference = node.state_checkpoint.state_reference
    access = owner.submit_prefill((owner.bank_for(seq),), checkpoints=(reference,))
    lifecycle.publish_save(seq)
    owner.acknowledge(access.access_id)
    return seq, slot, node


def _prefix_packet(maker, seqs, steps):
    return dict(inputs=maker.create_model_inputs(seqs, True), delta=None, running=seqs,
                cache_inputs=maker._prepare_prefill_cache_inputs(seqs, save_steps=steps), kv_connector_metadata=None)


@pytest.mark.parametrize('width', [1, 4, 6])
def test_real_trie_to_inputmaker_checkpoint_source_and_internal_producer_packet(width):
    maker, trie, add = _prefix_fixture(width)
    tokens = list(range(2120))
    _, slot, node = _seed_prefix(maker, add, tokens, 2048)
    reader = add(tokens, match=True)
    payload = _prefix_packet(maker, [reader], (2112,))
    owner = maker.compact_state_manager
    assert not owner._accesses and owner.bank_for(reader).pending_checkpoint is None
    maker._bind_compact_state(payload)
    carrier, inputs = payload['cache_inputs'], payload['inputs']
    access = owner._accesses[carrier.ssm_state_access_id]
    reference = reader.prefix_cache.pending_save.node.state_checkpoint.state_reference
    assert carrier.ssm_checkpoint_steps is None
    assert inputs.ssm_source_state_ids.tolist() == [slot]
    assert inputs.ssm_prefill_meta.checkpoint_ids.tolist() == [reference.unit_id]
    assert inputs.ssm_prefill_meta.checkpoint_offsets.tolist() == [64]
    assert inputs.ssm_prefill_meta.cu_seqlens.tolist() == [0, 72]
    assert inputs.history_lengths.tolist() == [2048] and inputs.seq_length.tolist() == [72]
    assert reference.unit_id not in access.bank_ids[0]
    assert access.read_only_input_ids == (slot,) and reference.producing
    assert carrier.state_restore_plan is carrier.state_save_plan is None
    assert carrier.kv_restore_plan is carrier.kv_save_plan is None
    # The real worker consumes the exact one-forward override, without reading
    # acceptance/current IDs back to the host. No model forward is run here.
    gpu_inputs = inputs.to_device('cuda')
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.ssm_state_cursor = SSMStateCursor(owner.capacity.request_slots)
    agent._prev_chunk_output = None
    gpu_inputs = agent._prepare_inputs_prefill(gpu_inputs, None)
    assert gpu_inputs.ssm_state_meta.input_state_ids.tolist() == [slot]
    assert not torch.isin(gpu_inputs.ssm_state_meta.output_state_ids, gpu_inputs.ssm_prefill_meta.checkpoint_ids).any()
    assert node.state_checkpoint.pin_count == 1
    maker.state_checkpoints.publish_save(reader)
    assert not reader.prefix_cache.restore.is_selected  # Pin transferred to exact forward.
    assert node.state_checkpoint.pin_count == 1  # Dispatch is not consumption.
    torch.cuda.synchronize()
    owner.acknowledge(access.access_id)
    maker.state_checkpoints.complete_forward(access.access_id)
    assert owner.bank_for(reader).input_ref is None
    assert node.state_checkpoint.pin_count == 0


def test_partial_checkpoint_packet_copies_only_frozen_kv_tail():
    maker, trie, add = _prefix_fixture(1)
    tokens = list(range(160))
    _, slot, node = _seed_prefix(maker, add, tokens, 80)
    reader = add(tokens, match=True)
    packet = _prefix_packet(maker, [reader], (144,))
    maker._bind_compact_state(packet)
    carrier = packet['cache_inputs']
    assert packet['inputs'].ssm_source_state_ids.tolist() == [slot]
    assert packet['inputs'].ssm_prefill_meta.checkpoint_offsets.tolist() == [64]
    assert carrier.kv_restore_plan.shape == carrier.kv_save_plan.shape == (2, 1)
    assert carrier.state_restore_plan is carrier.state_save_plan is None
    # Only the 64-token KV block containing each partial boundary is copied.
    assert carrier.kv_restore_plan.tolist() == maker._make_kv_prefix_cache_copy_plan([
        (node.state_checkpoint.frozen_block_id, int(reader.logical_blocks[1]))]).tolist()


def test_unsupported_internal_checkpoint_offset_fails_before_reservation_or_pin():
    maker, trie, add = _prefix_fixture()
    tokens = list(range(2120))
    _, _, node = _seed_prefix(maker, add, tokens, 2048)
    reader = add(tokens, match=True)
    owner = maker.compact_state_manager
    free = owner.get_num_free()
    payload = _prefix_packet(maker, [reader], (2104,))
    with pytest.raises(ValueError, match='chunk boundary'):
        maker._bind_compact_state(payload)
    assert owner.get_num_free() == free and not owner._accesses
    assert owner.bank_for(reader).input_ref is owner.bank_for(reader).pending_checkpoint is None
    assert node.state_checkpoint.pin_count == 0


@pytest.mark.parametrize('failure', ['kv_plan', 'second_bind', 'submit', 'carrier_after_submit'])
@pytest.mark.parametrize('admission_pin', [False, True])
def test_checkpoint_packet_pre_dispatch_failure_rolls_back_exact_resources(monkeypatch, failure, admission_pin):
    maker, trie, add = _prefix_fixture()
    tokens = list(range(160))
    _, _, node = _seed_prefix(maker, add, tokens, 80)
    readers = [add(tokens, match=True) for _ in range(2)]
    owner, lifecycle = maker.compact_state_manager, maker.state_checkpoints
    if admission_pin:
        lifecycle.pin_restores(readers)
    free = owner.get_num_free()
    free_kv = maker.scheduler.block_manager.get_num_free_gpu_blocks()
    payload = _prefix_packet(maker, readers, (144, 144))

    def fail(*args, **kwargs):
        raise RuntimeError('injected checkpoint preparation failure')

    if failure == 'kv_plan':
        monkeypatch.setattr(maker, '_make_kv_prefix_cache_copy_plan', fail)
    elif failure == 'second_bind':
        bind = lifecycle.bind_restore

        def bind_one(seq):
            return fail() if seq is readers[1] else bind(seq)

        monkeypatch.setattr(lifecycle, 'bind_restore', bind_one)
    elif failure == 'submit':
        monkeypatch.setattr(owner, 'submit_prefill', fail)
    else:
        from dataclasses import replace

        def replace_carrier(obj, **kwargs):
            return fail() if 'ssm_state_access_id' in kwargs else replace(obj, **kwargs)

        monkeypatch.setattr('lmdeploy.pytorch.engine.inputs_maker.replace', replace_carrier)
    with pytest.raises(RuntimeError, match='injected checkpoint'):
        maker._bind_compact_state(payload)
    assert not owner._accesses and owner.get_num_free() == free
    assert maker.scheduler.block_manager.get_num_free_gpu_blocks() == free_kv
    assert not lifecycle._retired_tails or all(not cp.state_reference.producing for cp in lifecycle._retired_tails)
    assert node.state_checkpoint.pin_count == 2 * int(admission_pin)
    for reader in readers:
        assert owner.bank_for(reader).input_ref is owner.bank_for(reader).pending_checkpoint is None
        assert reader.prefix_cache.restore.pinned == admission_pin
        assert reader.prefix_cache.restore.is_selected and not reader.prefix_cache.pending_save.is_pending
    assert payload['inputs'].ssm_state_meta is None


def test_unbound_checkpoint_plan_cannot_be_sent_to_worker():
    with pytest.raises(ValueError, match='Unbound compact checkpoint'):
        CacheCheckpointInputs(ssm_checkpoint_steps=(64,)).to_device('cuda')


@pytest.mark.parametrize('failure', ['sampling', 'stopping'])
def test_checkpoint_plans_take_no_resources_before_final_payload_binding(monkeypatch, failure):
    from lmdeploy.pytorch.engine.inputs_maker import _ForwardInputsTask

    maker, _, add = _prefix_fixture(1)
    seq = add(list(range(72)))
    owner = maker.compact_state_manager
    task = _ForwardInputsTask(maker, True, False)
    task.result.running = [seq]
    task.result.inputs = maker.create_model_inputs([seq], True)
    task.result.cache_inputs = maker._prepare_prefill_cache_inputs([seq], (64,))
    free = owner.get_num_free()

    def fail(*args):
        raise RuntimeError('sampling/stopping failed')

    if failure == 'sampling':
        monkeypatch.setattr(maker.sampling_strategy, 'make_sampling_inputs', fail)
    else:
        monkeypatch.setattr(maker.model_agent_strategy, 'make_stopping_criteria', fail)
    with pytest.raises(RuntimeError, match='sampling/stopping failed'):
        task._build_payload()
    assert owner.get_num_free() == free and not owner._accesses
    assert owner.bank_for(seq).pending_checkpoint is None


@pytest.mark.parametrize('skip_save', [False, True])
def test_cold_and_skipped_checkpoint_packets_preserve_runtime_cold_binding(skip_save):
    maker, _, add = _prefix_fixture(1)
    seq = add(list(range(72)))
    payload = _prefix_packet(maker, [seq], (0 if skip_save else 72,))
    maker._bind_compact_state(payload)
    carrier = payload['cache_inputs']
    assert payload['inputs'].ssm_source_state_ids.tolist() == [-1]
    assert payload['inputs'].ssm_prefill_meta.is_init_token[0]
    if skip_save:
        assert payload['inputs'].ssm_prefill_meta.checkpoint_ids.tolist() == [-1]
        assert not seq.prefix_cache.pending_save.is_pending
    else:
        assert payload['inputs'].ssm_prefill_meta.checkpoint_offsets.tolist() == [72]
        assert seq.prefix_cache.pending_save.step == 72
    assert carrier.state_restore_plan is carrier.state_save_plan is None


def test_selected_source_boundary_must_match_actual_input_history_before_reservation():
    maker, _, add = _prefix_fixture()
    tokens = list(range(160))
    _, _, node = _seed_prefix(maker, add, tokens, 80)
    reader = add(tokens, match=True)
    packet = _prefix_packet(maker, [reader], (144,))
    packet['inputs'].history_lengths = packet['inputs'].history_lengths + 8
    owner = maker.compact_state_manager
    free = owner.get_num_free()
    with pytest.raises(ValueError, match='source boundary'):
        maker._bind_compact_state(packet)
    assert owner.get_num_free() == free and not owner._accesses
    assert owner.bank_for(reader).input_ref is owner.bank_for(reader).pending_checkpoint is None
    assert not node.state_checkpoint.pin_count


@pytest.mark.parametrize('published', [False, True])
def test_ordered_prefetch_preserves_first_producer_and_respects_live_reservation(published):
    maker, _, add = _prefix_fixture(1)
    seq = add(list(range(128)))
    first = _prefix_packet(maker, [seq], (64,))
    maker._bind_compact_state(first)
    owner = maker.compact_state_manager
    reference = owner.bank_for(seq).pending_checkpoint
    node = seq.prefix_cache.pending_save.node
    first_checkpoint = node.state_checkpoint
    if published:
        maker.state_checkpoints.publish_save(seq)
        assert not reference.active and owner.bank_for(seq).pending_checkpoint is None
    else:
        # A live, unpublished reservation is still exclusive. Production
        # publishes before prefetch; this branch tests the owner's guard.
        assert owner.bank_for(seq).pending_checkpoint is reference
        with pytest.raises(ValueError, match='One pending checkpoint'):
            owner.reserve_checkpoint(owner.bank_for(seq), 128)
        maker.state_checkpoints.publish_save(seq)
    # Isolate metadata prefetch: the exact next history/length are normally
    # supplied by scheduler chunk planning.
    second = _prefix_packet(maker, [seq], (128,))
    second['inputs'].history_lengths = torch.tensor([64])
    second['inputs'].seq_length = torch.tensor([64])
    maker._bind_compact_state(second)
    next_reference = owner.bank_for(seq).pending_checkpoint
    assert next_reference is not None and next_reference.unit_id != reference.unit_id
    assert second['inputs'].ssm_prefill_meta.checkpoint_ids.tolist() == [next_reference.unit_id]
    assert reference.producing and first_checkpoint.published
    assert first_checkpoint.state_reference.active
    assert len(owner._accesses) == 2
    owner.acknowledge(first['cache_inputs'].ssm_state_access_id)
    maker.state_checkpoints.complete_forward(first['cache_inputs'].ssm_state_access_id)
    assert owner.bank_for(seq).pending_checkpoint is next_reference
    assert first_checkpoint.state_reference.active and reference.unit_id in owner._units
    maker.state_checkpoints.publish_forward(second['cache_inputs'].ssm_state_access_id)
    owner.acknowledge(second['cache_inputs'].ssm_state_access_id)
    maker.state_checkpoints.complete_forward(second['cache_inputs'].ssm_state_access_id)
    for node in tuple(maker.state_checkpoints._index.unique_nodes()):
        maker.state_checkpoints.release_checkpoint(node)
    owner.free(seq)
    assert not owner._units and not owner._accesses and not maker.state_checkpoints._forwards


@pytest.mark.parametrize('action', ['cancel', 'new_selection', 'new_binding'])
def test_real_engine_loop_holds_exact_checkpoint_pin_through_delayed_cuda_completion(action):
    from lmdeploy.pytorch.engine.engine_loop import EngineLoop
    from lmdeploy.pytorch.engine.model_agent.agent import BatchedOutputs

    maker, _, add = _prefix_fixture(1)
    tokens = list(range(160))
    _, slot, source = _seed_prefix(maker, add, tokens, 80)
    reader = add(tokens, match=True)
    packet = _prefix_packet(maker, [reader], (144,))
    maker._bind_compact_state(packet)
    lifecycle, owner = maker.state_checkpoints, maker.compact_state_manager
    ticket = packet['cache_inputs'].ssm_state_access_id
    access = owner._accesses[ticket]
    bank = owner.bank_for(reader)
    destination = reader.prefix_cache.pending_save.node
    restore = reader.prefix_cache.restore
    assert not restore.is_selected and source.state_checkpoint.pin_count == 1
    with pytest.raises(RuntimeError, match='actual owner completion'):
        lifecycle.complete_forward(ticket)
    loop = EngineLoop.__new__(EngineLoop)
    loop.compact_state_manager = owner
    loop.state_checkpoints = lifecycle
    loop.inputs_maker = SimpleNamespace(update_running_seqs=lambda *_: None)
    delivered = []
    loop._finish_forward_output = lambda output, *_: delivered.append(output)
    later = []
    replacement = []
    value = torch.zeros(1, device='cuda')
    done = torch.cuda.Event()
    torch.cuda.synchronize()
    torch.cuda._sleep(150_000_000)
    value.fill_(9)
    done.record()

    async def prefetch():
        assert destination.state_checkpoint.published  # Published before prefetch, not after GPU completion.
        assert source.state_checkpoint.pin_count == 1 and not done.query()
        assert owner.promote_to_decode(bank)
        later.append(owner.submit_decode((bank,)))
        if action == 'cancel':
            assert not lifecycle.unpin_restore(reader)  # Request cancellation cannot release transferred pin.
            owner.free(reader)
        elif action == 'new_selection':
            restore.select(slot, source)
            lifecycle.pin_restore(reader)
        else:
            new_reference = destination.state_checkpoint.state_reference
            owner.bind_input(bank, new_reference)
            replacement.append(bank.input_ref)
        return None, None

    async def output():
        await asyncio.to_thread(done.synchronize)
        return BatchedOutputs(None, None, ssm_state_access_id=ticket)

    loop._prefetch_next_inputs = prefetch
    loop.executor = SimpleNamespace(get_output_async=output)
    assert asyncio.run(loop._main_loop_get_outputs([reader], packet)) == (None, None)
    assert value.item() == 9 and access.finished and delivered
    assert ticket not in lifecycle._forwards
    assert not later[0].finished  # Earlier completion cannot release prefetched access.
    lifecycle.complete_forward(ticket)  # Duplicate report is harmless.
    if action == 'new_selection':
        assert restore.is_selected and restore.pinned and source.state_checkpoint.pin_count == 1
        lifecycle.unpin_restore(reader)
    else:
        assert source.state_checkpoint.pin_count == 0
    if action == 'new_binding':
        assert bank.input_ref is replacement[0] and replacement[0].active
    elif action != 'cancel':
        assert bank.input_ref is None
    owner.acknowledge(later[0].access_id)
    lifecycle.release_checkpoint(source)
    lifecycle.release_checkpoint(destination)
    owner.free(reader)


@pytest.mark.parametrize('response', ['wrong', 'failure'])
def test_checkpoint_engine_failure_retains_transferred_pins_and_access(response):
    from lmdeploy.pytorch.engine.engine_loop import EngineLoop
    from lmdeploy.pytorch.engine.model_agent.agent import BatchedOutputs

    maker, _, add = _prefix_fixture(1)
    tokens = list(range(160))
    _, _, source = _seed_prefix(maker, add, tokens, 80)
    reader = add(tokens, match=True)
    packet = _prefix_packet(maker, [reader], (144,))
    maker._bind_compact_state(packet)
    lifecycle, owner = maker.state_checkpoints, maker.compact_state_manager
    ticket = packet['cache_inputs'].ssm_state_access_id
    loop = EngineLoop.__new__(EngineLoop)
    loop.compact_state_manager, loop.state_checkpoints = owner, lifecycle
    loop.inputs_maker = SimpleNamespace(update_running_seqs=lambda *_: None)

    async def prefetch():
        return None, None

    async def output():
        if response == 'failure':
            raise RuntimeError('worker failed')
        return BatchedOutputs(None, None, ssm_state_access_id=ticket + 1)

    loop._prefetch_next_inputs = prefetch
    loop.executor = SimpleNamespace(get_output_async=output)
    with pytest.raises(RuntimeError, match='match|worker failed'):
        asyncio.run(loop._main_loop_get_outputs([reader], packet))
    assert ticket in lifecycle._forwards and not owner._accesses[ticket].finished
    assert source.state_checkpoint.pin_count == 1 and not reader.prefix_cache.restore.is_selected
    # No production failure drain is claimed: the fixture has dispatched no
    # GPU work. A real coordinator must prove quiescence and withdraw outputs.
    owner.complete(owner._accesses[ticket], success=False)
    lifecycle.complete_forward(ticket)
    assert source.state_checkpoint.pin_count == 0


def test_same_anchor_save_cannot_replace_the_selected_source_before_binding():
    maker, _, add = _prefix_fixture(1)
    tokens = list(range(96))
    _, slot, node = _seed_prefix(maker, add, tokens, 80)
    reader = add(tokens, match=True)
    checkpoint = node.state_checkpoint
    packet = _prefix_packet(maker, [reader], (96,))
    maker._bind_compact_state(packet)
    assert packet['inputs'].ssm_source_state_ids.tolist() == [slot]
    assert packet['inputs'].ssm_prefill_meta.checkpoint_ids.tolist() == [-1]
    assert node.state_checkpoint is checkpoint and checkpoint.pin_count == 1
    owner = maker.compact_state_manager
    assert owner.bank_for(reader).input_ref.unit_id == slot
    ticket = packet['cache_inputs'].ssm_state_access_id
    owner.acknowledge(ticket)
    maker.state_checkpoints.complete_forward(ticket)
    assert checkpoint.pin_count == 0


def _seed_real_scheduler(maker, tokens, boundary):
    """Preseed exact Trie ownership only; no computed payload or model
    claim."""
    scheduler = maker.scheduler
    seq = scheduler.add_session(0).add_sequence(tokens)
    scheduler.state_manager.allocate(seq)
    scheduler.block_manager.allocate(seq)
    scheduler.block_trie.allocate(seq)
    lifecycle = scheduler.state_checkpoints
    slot = lifecycle.reserve_save(seq, step=boundary)
    assert slot > 0
    if maker.compact_state_manager is not None:
        owner = maker.compact_state_manager
        reference = seq.prefix_cache.pending_save.node.state_checkpoint.state_reference
        access = owner.submit_prefill((owner.bank_for(seq),), checkpoints=(reference,))
        owner.acknowledge(access.access_id)
    lifecycle.publish_save(seq)
    scheduler.end_session(0)
    return slot


@pytest.mark.parametrize('compact', [False, True])
def test_real_scheduler_and_inputmaker_emit_72_or_legacy_checkpoint_cut(compact):
    maker = _maker(1, budget=4096, prefix=True, block_size=64, checkpoint_budget=4, compact=compact)
    tokens = list(range(2120))
    slot = _seed_real_scheduler(maker, tokens[:2056], 2048)
    seq = maker.scheduler.add_session(1).add_sequence(tokens)
    packet = maker._make_forward_inputs(True)
    inputs, carrier = packet['inputs'], packet['cache_inputs']
    assert packet['running'] == [seq]
    assert inputs.history_lengths.tolist() == [2048]
    assert inputs.seq_length.tolist() == [72 if compact else 64]
    assert seq.kv_token_limit == (None if compact else 2112)
    assert inputs.is_chunk == (not compact)
    maker.scheduler.activate_seqs([seq])
    assert maker.has_pending_long_context_chunk() == (not compact)
    assert seq.prefix_cache.pending_save.step == 2112
    if compact:
        assert inputs.ssm_source_state_ids.tolist() == [slot]
        assert inputs.ssm_prefill_meta.checkpoint_offsets.tolist() == [64]
        assert carrier.state_restore_plan is carrier.state_save_plan is None
        ticket = carrier.ssm_state_access_id
        maker.state_checkpoints.publish_forward(ticket)
        maker.compact_state_manager.acknowledge(ticket)
        maker.state_checkpoints.complete_forward(ticket)
    else:
        assert carrier.state_restore_plan[0] == (slot,)
        assert carrier.state_save_plan is not None
    # This assertion distinguishes the real old checkpoint-cut scheduling from
    # the indexed path. It is not evidence of one full model forward yet.


def test_real_scheduler_unsupported_relative56_falls_back_before_submission():
    maker = _maker(1, budget=4096, prefix=True, block_size=64, checkpoint_budget=4)
    tokens = list(range(2120))
    _seed_real_scheduler(maker, tokens, 2056)
    seq = maker.scheduler.add_session(1).add_sequence(tokens)
    packet = maker._make_forward_inputs(True)
    assert packet['inputs'].history_lengths.tolist() == [2056]
    assert packet['inputs'].seq_length.tolist() == [56]
    assert packet['inputs'].is_chunk and not packet['inputs'].is_last_chunk
    assert seq.kv_token_limit == 2112
    assert packet['inputs'].ssm_prefill_meta.checkpoint_offsets.tolist() == [56]
    assert seq.prefix_cache.pending_save.step == 2112
    #56 is legal here only because it is the final output boundary of this
    # fallback forward; no unsupported internal checkpoint was submitted.


@pytest.mark.parametrize('history,end,budget,expected_e,expected_c', [
    (2048, 2120, 72, 2120, 2112),
    (2048, 2120, 64, 2112, 2112),
    (2056, 2120, 72, 2112, 2112),
    (2048, 2112, 128, 2112, 2112),
    (0, 137, 72, 72, 64),
    (72, 137, 72, 128, 128),
    (128, 137, 72, 137, 137),
    (0, 1024, 4096, 1024, 960),
])
def test_compact_shared_planner_separates_forward_and_checkpoint(history, end, budget, expected_e, expected_c):
    from lmdeploy.pytorch.engine.inputs_maker import LongContextChunker
    from lmdeploy.pytorch.long_context import get_long_context_chunk_limit, plan_long_context_chunk

    maker = _maker(1, budget=budget, prefix=True, block_size=64, checkpoint_budget=4)
    seq = maker.scheduler.add_session(0).add_sequence(list(range(end)))
    seq.set_step(history)
    planner = maker.scheduler._prefill_scheduler
    plan = plan_long_context_chunk(seq, get_long_context_chunk_limit(seq, budget))
    assert (plan.chunk_end, plan.checkpoint_step) == (expected_e, expected_c)
    assert planner._next_long_context_chunk_end(seq) == expected_e
    assert planner._prefill_admission_token_count(seq) == expected_e - history
    chunker = LongContextChunker(budget)
    assert chunker.is_long_context(seq) == (expected_e != end)
    chunker.set_seq(seq)
    assert chunker.next_chunk_size()[0] == expected_e - history
    assert chunker.is_last_chunk() == (expected_e == end)


@pytest.mark.parametrize('evict', [False, True])
def test_final_checkpoint_binding_survives_producer_then_releases_after_consumer(evict):
    maker, _, add = _prefix_fixture(1)
    seq = add(list(range(128)))
    first = _prefix_packet(maker, [seq], (128,))
    maker._bind_compact_state(first)
    owner, lifecycle = maker.compact_state_manager, maker.state_checkpoints
    bank = owner.bank_for(seq)
    ref = bank.input_ref
    node = seq.prefix_cache.pending_save.node
    assert ref is not None and ref.unit_id == node.state_checkpoint.slot
    assert ref.unit_id not in bank.unit_ids
    assert first['inputs'].ssm_state_meta.state_ids.tolist() == [list(bank.unit_ids)]
    first_id = first['cache_inputs'].ssm_state_access_id
    lifecycle.publish_forward(first_id)
    # Prepare another forward BEFORE producer completion, as EngineLoop does.
    second = _prefix_packet(maker, [seq], (0,))
    second['inputs'].history_lengths = torch.tensor([128])
    second['inputs'].seq_length = torch.tensor([8])
    maker._bind_compact_state(second)
    second_id = second['cache_inputs'].ssm_state_access_id
    assert owner._accesses[second_id].read_only_input_ids == (ref.unit_id,)
    assert ref.unit_id not in owner._accesses[second_id].bank_ids[0]
    owner.acknowledge(first_id)
    lifecycle.complete_forward(first_id)
    assert bank.input_ref is ref and ref.active
    if evict:
        lifecycle.release_checkpoint(node)
    # Consumer still owns the physical read even after trie withdrawal.
    assert ref.unit_id in owner._units
    owner.acknowledge(second_id)
    lifecycle.complete_forward(second_id)
    assert bank.input_ref is None and not ref.active
    if evict:
        assert ref.unit_id not in owner._units
    assert not lifecycle._forwards and not owner._accesses
    owner.free(seq)


def test_final_checkpoint_replaces_restored_source_without_early_pin_release():
    maker, _, add = _prefix_fixture(1)
    tokens = list(range(160))
    _, source_id, source = _seed_prefix(maker, add, tokens, 80)
    seq = add(tokens, match=True)
    packet = _prefix_packet(maker, [seq], (160,))
    maker._bind_compact_state(packet)
    owner, lifecycle = maker.compact_state_manager, maker.state_checkpoints
    bank = owner.bank_for(seq)
    final_ref = bank.input_ref
    assert final_ref.unit_id != source_id
    assert source.state_checkpoint.pin_count == 1
    ticket = packet['cache_inputs'].ssm_state_access_id
    assert owner._accesses[ticket].read_only_input_ids == (source_id,)
    lifecycle.publish_forward(ticket)
    owner.acknowledge(ticket)
    lifecycle.complete_forward(ticket)
    assert source.state_checkpoint.pin_count == 0
    assert bank.input_ref is final_ref and final_ref.active


def test_final_checkpoint_batch_retain_failure_rolls_back_before_pin_transfer(monkeypatch):
    maker, _, add = _prefix_fixture(1)
    seqs = [add(list(range(160))), add(list(range(160, 320)))]
    owner, lifecycle = maker.compact_state_manager, maker.state_checkpoints
    free = owner.get_num_free()
    packet = _prefix_packet(maker, seqs, (160, 160))
    original = owner.retain
    count = 0

    def fail_second(reference, kind):
        nonlocal count
        if kind == 'request':
            count += 1
            if count == 2:
                raise RuntimeError('injected final retain failure')
        return original(reference, kind)

    monkeypatch.setattr(owner, 'retain', fail_second)
    with pytest.raises(RuntimeError, match='injected final retain'):
        maker._bind_compact_state(packet)
    assert count == 2 and owner.get_num_free() == free
    assert not owner._accesses and not lifecycle._forwards
    for seq in seqs:
        bank = owner.bank_for(seq)
        assert bank.input_ref is bank.pending_checkpoint is None
        assert not seq.prefix_cache.pending_save.is_pending


@pytest.mark.parametrize('cancel', [False, True])
def test_engine_loop_final_checkpoint_prefetch_has_no_future_writer(cancel):
    from lmdeploy.pytorch.engine.engine_loop import EngineLoop
    from lmdeploy.pytorch.engine.model_agent.agent import BatchedOutputs

    maker, _, add = _prefix_fixture(1)
    seq = add(list(range(128)))
    first = _prefix_packet(maker, [seq], (128,))
    maker._bind_compact_state(first)
    owner, lifecycle = maker.compact_state_manager, maker.state_checkpoints
    bank = owner.bank_for(seq)
    ref = bank.input_ref
    node = seq.prefix_cache.pending_save.node
    runtime = SSMStateCursor(owner.capacity.request_slots)
    initial = first['inputs'].to_device('cuda')
    state = runtime.bind(initial.ssm_state_meta, initial.state_offsets, initial.history_lengths,
                         initial.ssm_source_state_ids, prefill=initial.ssm_prefill_meta)
    assert state.output_state_ids.tolist() == [[ref.unit_id]]
    original_bank = initial.ssm_state_meta.state_ids.clone()
    stream = torch.cuda.Stream()
    first_done, second_done = torch.cuda.Event(), torch.cuda.Event()
    pool = torch.zeros(owner.capacity.num_units, device='cuda')  # Test payload, not an implementation adapter.
    consumed = torch.ones(1, device='cuda', dtype=torch.int64)
    # Compile before the delayed region: host JIT time is not a lifecycle race.
    pool.index_fill_(0, state.output_state_ids.flatten(), 0)  # Warm CUDA's lazy module load too.
    runtime.advance(state, initial.state_offsets, consumed)
    runtime.current_ids.fill_(-1)
    torch.cuda.synchronize()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(150_000_000)
        pool.index_fill_(0, state.output_state_ids.flatten(), 17)
        runtime.advance(state, initial.state_offsets, consumed)
        first_done.record()
    second = []
    loop = EngineLoop.__new__(EngineLoop)
    loop.compact_state_manager, loop.state_checkpoints = owner, lifecycle
    loop.inputs_maker = SimpleNamespace(update_running_seqs=lambda *_: None)
    loop._finish_forward_output = lambda *_: None

    async def prefetch():
        assert node.state_checkpoint.published and not first_done.query()
        packet = _prefix_packet(maker, [seq], (0,))
        packet['inputs'].history_lengths = torch.tensor([128])
        packet['inputs'].seq_length = torch.tensor([8])
        maker._bind_compact_state(packet)
        second.append(packet)
        assert ref.unit_id not in owner._accesses[packet['cache_inputs'].ssm_state_access_id].bank_ids[0]
        with torch.cuda.stream(stream):
            inputs = packet['inputs'].to_device('cuda', non_blocking=True)
            following = runtime.bind(inputs.ssm_state_meta, inputs.state_offsets, inputs.history_lengths,
                                     inputs.ssm_source_state_ids, prefill=inputs.ssm_prefill_meta)
            pool.index_fill_(0, following.output_state_ids.flatten(), 29)
            second_done.record()
        if cancel:
            owner.free(seq)  # Submitted tables/units survive request cancellation.
        return None, None

    async def output():
        await asyncio.to_thread(first_done.synchronize)
        return BatchedOutputs(None, None, ssm_state_access_id=first['cache_inputs'].ssm_state_access_id)

    loop._prefetch_next_inputs = prefetch
    loop.executor = SimpleNamespace(get_output_async=output)
    asyncio.run(loop._main_loop_get_outputs([seq], first))
    assert second and ref.unit_id in owner._units
    if not cancel:
        assert bank.input_ref is ref
    lifecycle.release_checkpoint(node)
    assert ref.unit_id in owner._units  # Future consumer access keeps it live.
    second_done.synchronize()
    assert pool[ref.unit_id].item() == 17  # Later write could not touch the frozen final.
    torch.testing.assert_close(initial.ssm_state_meta.state_ids, original_bank, rtol=0, atol=0)
    ticket = second[0]['cache_inputs'].ssm_state_access_id
    owner.acknowledge(ticket)
    lifecycle.complete_forward(ticket)
    assert ref.unit_id not in owner._units
    assert not owner._accesses and not lifecycle._forwards
    if not cancel:
        owner.free(seq)


@pytest.mark.parametrize('pending', [False, True])
def test_decode_checkpoint_boundary_accounts_for_queued_output_before_cpu_append(pending):
    from lmdeploy.pytorch.messages import UpdateTokenMode
    maker, trie, add = _prefix_fixture(width=1)
    maker.config.prefix_cache_decode_state_interval = 64
    seq = add(list(range(127 if pending else 128)))
    owner, lifecycle = maker.compact_state_manager, maker.state_checkpoints
    bank = owner.bank_for(seq)
    assert owner.promote_to_decode(bank)
    earlier = owner.submit_decode((bank,)) if pending else None
    delta = SimpleNamespace(max_q_seqlen=1, block_offsets=torch.zeros((1, 2), dtype=torch.long),
                            ssm_state_ids=None, ssm_checkpoint_state_ids=None)
    carrier = maker._make_decode_cache_inputs([seq], delta)
    assert carrier.ssm_checkpoint_steps == (128,)
    packet = dict(inputs=None, delta=delta, running=[seq], cache_inputs=carrier, kv_connector_metadata=None)
    maker._bind_compact_state(packet)
    carrier = packet['cache_inputs']
    assert packet['inputs'] is None
    destination = delta.ssm_checkpoint_state_ids.item()
    assert destination not in bank.unit_ids
    assert bank.input_ref.unit_id == destination
    if pending:
        assert seq.num_valid_ids == 127  # No token identity was guessed at reserve time.
        owner.acknowledge(earlier.access_id)
        seq.update_token_ids([127], mode=UpdateTokenMode.DECODE)
    lifecycle.publish_forward(carrier.ssm_state_access_id)
    node = seq.prefix_cache.decode_checkpoint_node
    assert node.state_checkpoint.step == 128
    assert node.state_checkpoint.exact_match_data.token_ids.tolist() == list(range(128))
    owner.acknowledge(carrier.ssm_state_access_id)
    lifecycle.complete_forward(carrier.ssm_state_access_id)
    assert bank.input_ref.unit_id == destination
    following = owner.submit_decode((bank,))
    owner.acknowledge(following.access_id, consumed_inputs=(True,))
    assert bank.input_ref is None
    lifecycle.release_checkpoint(node)
    owner.free(seq)
    assert not owner._units and not owner._accesses


@pytest.mark.parametrize('kind', ['stale', 'wide', 'double'])
def test_bad_decode_checkpoint_packet_rejects_before_any_reservation(kind):
    maker, trie, add = _prefix_fixture(width=1)
    seq = add(list(range(128)))
    owner = maker.compact_state_manager
    delta = SimpleNamespace(max_q_seqlen=2 if kind == 'wide' else 1,
                            block_offsets=torch.zeros((1, 2), dtype=torch.long),
                            ssm_state_ids=None, ssm_checkpoint_state_ids=None)
    delta.ssm_checkpoint_state_ids = torch.tensor([99]) if kind == 'double' else None
    carrier = CacheCheckpointInputs(ssm_checkpoint_steps=(64 if kind == 'stale' else 128,))
    before = owner.get_num_free()
    with pytest.raises(ValueError):
        maker._bind_compact_state(dict(inputs=None, delta=delta, running=[seq], cache_inputs=carrier,
                                      kv_connector_metadata=None))
    assert not owner._accesses and owner.bank_for(seq).pending_checkpoint is None
    assert owner.get_num_free() == before


def test_decode_checkpoint_bind_failure_rolls_back_without_releasing_old_node(monkeypatch):
    maker, trie, add = _prefix_fixture(width=1)
    seq, _, old_node = _seed_prefix(maker, add, list(range(256)), 64)
    seq.prefix_cache.decode_checkpoint_node = old_node
    owner, lifecycle = maker.compact_state_manager, maker.state_checkpoints
    assert owner.promote_to_decode(owner.bank_for(seq))
    old, free_before = old_node.state_checkpoint, owner.get_num_free()
    delta = SimpleNamespace(max_q_seqlen=1, block_offsets=torch.zeros((1, 4), dtype=torch.long),
                            ssm_state_ids=None, ssm_checkpoint_state_ids=None)

    def fail(*args, **kwargs):
        raise RuntimeError('injected decode attachment failure')

    monkeypatch.setattr(lifecycle, 'attach_forward', fail)
    with pytest.raises(RuntimeError, match='injected decode'):
        maker._bind_compact_state(dict(inputs=None, delta=delta, running=[seq], kv_connector_metadata=None,
                                      cache_inputs=CacheCheckpointInputs(ssm_checkpoint_steps=(256,))))
    assert old_node.state_checkpoint is old and old.pin_count == 0 and old.published
    assert owner.get_num_free() == free_before
    assert not owner._accesses and not lifecycle._forwards
    assert owner.bank_for(seq).pending_checkpoint is None and delta.ssm_state_ids is None


@pytest.mark.parametrize('case', ['mtp', 'not_due', 'disabled', 'pending'])
def test_optional_decode_checkpoint_skip_has_no_lifecycle_side_effects(case):
    maker, trie, add = _prefix_fixture(width=4 if case == 'mtp' else 1)
    maker.config.prefix_cache_decode_state_interval = 0 if case == 'disabled' else 64
    seq = add(list(range(127 if case == 'not_due' else 128)))
    owner, lifecycle = maker.compact_state_manager, maker.state_checkpoints
    bank = owner.bank_for(seq)
    delta = SimpleNamespace(max_q_seqlen=4 if case == 'mtp' else 1,
                            block_offsets=torch.zeros((1, 2), dtype=torch.long),
                            ssm_state_ids=None, ssm_checkpoint_state_ids=None)
    if case == 'pending':
        reference = owner.reserve_checkpoint(bank, 64)
        assert owner.promote_to_decode(bank)
        previous = owner.submit_decode((bank,), checkpoints=(reference,))
        assert lifecycle.prepare_decode([seq], [128]) == ()
        assert bank.pending_checkpoint is reference
        owner.acknowledge(previous.access_id)
        owner.release_reference(reference)
    else:
        assert maker._make_decode_cache_inputs([seq], delta) is None
    assert not owner._accesses and bank.input_ref is None and not lifecycle._forwards


@pytest.mark.parametrize('step', [8, 120, 128])
@pytest.mark.parametrize('published', [False, True])
def test_compact_sleep_drain_retires_dropped_prefill_and_invalidates_entire_trie(step, published, monkeypatch):
    maker = _maker(width=1, prefix=True, block_size=64, checkpoint_budget=8)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    scheduler.seq_meta.compact_prefill_checkpoint_stride = 0
    scheduler.seq_meta.prefill_save_alignment = 0
    seq = scheduler.add_session(0).add_sequence(list(range(step)))
    assert scheduler.schedule(is_prefill=True).running == [seq]
    packet = _prefix_packet(maker, [seq], (step,))
    maker._bind_compact_state(packet)
    ticket = packet['cache_inputs'].ssm_state_access_id
    access = owner._accesses[ticket]
    node = seq.prefix_cache.pending_save.node
    checkpoint = node.state_checkpoint
    assert checkpoint is not None
    if published:
        scheduler.state_checkpoints.publish_forward(ticket)
    scheduler.end_session(0)
    clears = []
    clear = scheduler.block_trie.clear

    def tracked_clear():
        assert access.finished  # Physical completion precedes identity removal.
        clears.append(True)
        return clear()

    monkeypatch.setattr(scheduler.block_trie, 'clear', tracked_clear)
    scheduler.block_trie.stats.num_hit_tokens = 17
    assert not access.finished and owner._units
    scheduler.reset_cache_after_worker_drain()
    assert clears == [True] and scheduler.block_trie.stats.num_hit_tokens == 0
    assert access.finished and checkpoint.state_reference.phase == 'failed'
    assert not owner._units and not owner._banks and not owner._accesses
    assert not scheduler.state_checkpoints._forwards and not scheduler.state_checkpoints._retired_tails
    assert not scheduler.block_trie._roots and not scheduler.block_trie.leaves
    assert not tuple(scheduler.block_trie._checkpoint_index.unique_nodes())
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 128
    scheduler.reset_cache_after_worker_drain()  # Idempotent, without resetting ticket identities.
    resumed = scheduler.add_session(1).add_sequence(list(range(step + 1)))
    assert scheduler.schedule(is_prefill=True).running == [resumed]
    assert resumed.num_history_ids == 0 and not resumed.prefix_cache.restore.is_selected
    next_access = owner.submit_prefill((owner.bank_for(resumed),))
    owner.acknowledge(ticket)
    assert not next_access.finished
    scheduler.end_session(1)
    scheduler.reset_cache_after_worker_drain()


def test_compact_sleep_rejects_live_sessions_before_mutating_resources():
    maker = _maker(width=1, prefix=True, block_size=64, checkpoint_budget=8)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    seq = scheduler.add_session(0).add_sequence(list(range(128)))
    payload = maker._make_forward_inputs(True)
    access = owner._accesses[payload['cache_inputs'].ssm_state_access_id]
    before = owner.get_num_free()
    with pytest.raises(RuntimeError, match='sessions'):
        scheduler.reset_cache_after_worker_drain()
    assert not access.finished and owner.get_num_free() == before
    assert seq.logical_state > 0
    scheduler.end_session(0)
    scheduler.reset_cache_after_worker_drain()


def test_compact_sleep_unpins_selected_inflight_restore_before_cache_invalidation():
    maker = _maker(width=1, prefix=True, block_size=64, checkpoint_budget=8)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    scheduler.seq_meta.compact_prefill_checkpoint_stride = 0
    scheduler.seq_meta.prefill_save_alignment = 0
    first = scheduler.add_session(0).add_sequence(list(range(128)))
    packet = maker._make_forward_inputs(True)
    ticket = packet['cache_inputs'].ssm_state_access_id
    node = first.prefix_cache.pending_save.node
    scheduler.state_checkpoints.publish_forward(ticket)
    owner.acknowledge(ticket)
    scheduler.state_checkpoints.complete_forward(ticket)
    scheduler.end_session(0)
    checkpoint = node.state_checkpoint
    reader = scheduler.add_session(1).add_sequence(list(range(136)))
    restore_packet = maker._make_forward_inputs(True)
    assert restore_packet['inputs'].history_lengths.tolist() == [128]
    assert checkpoint.pin_count == 1 and not reader.prefix_cache.restore.is_selected
    scheduler.end_session(1)
    assert checkpoint.pin_count == 1
    scheduler.reset_cache_after_worker_drain()
    assert checkpoint.pin_count == 0 and node.state_checkpoint is None
    assert not owner._units and not scheduler.state_checkpoints._forwards
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 128


def test_compact_sleep_unpins_discarded_decode_replacement_old_node():
    maker = _maker(width=1, prefix=True, block_size=64, checkpoint_budget=8)
    scheduler, owner, lifecycle = maker.scheduler, maker.compact_state_manager, maker.state_checkpoints
    seq = scheduler.add_session(0).add_sequence(list(range(256)))
    scheduler.schedule(is_prefill=True)
    for step in (64, 128):
        reservations = lifecycle.prepare_decode([seq], [step])
        reference = reservations[0].reference
        access = owner.submit_prefill((owner.bank_for(seq),), checkpoints=(reference,))
        lifecycle.attach_forward(access, [seq], final_checkpoints=(reference,), decode_reservations=reservations)
        if step == 64:
            lifecycle.publish_forward(access.access_id)
            owner.acknowledge(access.access_id)
            lifecycle.complete_forward(access.access_id)
            old = seq.prefix_cache.decode_checkpoint_node.state_checkpoint
    assert old.pin_count == 1
    scheduler.end_session(0)
    assert not access.finished and old.pin_count == 1
    scheduler.reset_cache_after_worker_drain()
    assert old.pin_count == 0 and access.finished
    assert not owner._units and not lifecycle._forwards
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 128


@pytest.mark.parametrize('failure', [False, True])
def test_engine_sleep_keeps_compact_holds_until_successful_worker_drain(failure):
    from lmdeploy.pytorch.engine.engine import Engine
    from lmdeploy.pytorch.engine.request import RequestManager

    maker = _maker(width=1, prefix=True, block_size=64, checkpoint_budget=8)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    scheduler.add_session(0).add_sequence(list(range(128)))
    packet = maker._make_forward_inputs(True)
    ticket = packet['cache_inputs'].ssm_state_access_id
    access = owner._accesses[ticket]
    engine = Engine.__new__(Engine)
    engine.req_manager, engine.scheduler = RequestManager(), scheduler
    engine.end_session = scheduler.end_session
    events = []

    async def scheduling_drain():
        events.append('schedule-paused')

    async def worker_drain(level):
        assert not scheduler.sessions and not access.finished and owner._units
        events.append('worker-enter')
        await asyncio.sleep(0)
        assert not access.finished and owner._units
        if failure:
            raise RuntimeError('injected worker drain failure')
        events.append('worker-complete')

    engine._engine_loop = SimpleNamespace(drain_for_sleep=scheduling_drain,
                                          reset_runtime_state=lambda: events.append('reset-runtime'))
    engine.executor = SimpleNamespace(sleep=worker_drain)
    if failure:
        with pytest.raises(RuntimeError, match='injected worker'):
            asyncio.run(engine.sleep())
        assert not access.finished and owner._units and events == ['schedule-paused', 'worker-enter']
        scheduler.reset_cache_after_worker_drain()  # Test fixture submitted no actual GPU work.
    else:
        asyncio.run(engine.sleep())
        assert access.finished and not owner._units
        assert events == ['schedule-paused', 'worker-enter', 'worker-complete', 'reset-runtime']


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two allowed GPUs')
@pytest.mark.parametrize('slow_rank', [0, 1])
def test_compact_sleep_reset_waits_for_both_real_device_events(slow_rank):
    from lmdeploy.pytorch.engine.engine import Engine
    from lmdeploy.pytorch.engine.request import RequestManager

    maker = _maker(width=1, prefix=True, block_size=64, checkpoint_budget=8)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    scheduler.add_session(0).add_sequence(list(range(128)))
    packet = maker._make_forward_inputs(True)
    access = owner._accesses[packet['cache_inputs'].ssm_state_access_id]
    buffers, events, streams = [], [], []
    for rank in range(2):
        with torch.cuda.device(rank):
            buffers.append(torch.zeros(1, device=f'cuda:{rank}'))
            streams.append(torch.cuda.Stream())
            events.append(torch.cuda.Event())
    engine = Engine.__new__(Engine)
    engine.req_manager, engine.scheduler = RequestManager(), scheduler
    engine.end_session = scheduler.end_session

    async def schedule_paused():
        pass

    engine._engine_loop = SimpleNamespace(drain_for_sleep=schedule_paused, reset_runtime_state=lambda: None)

    async def scenario():
        fast_finished = asyncio.Event()

        async def worker(rank):
            if rank == slow_rank:
                await fast_finished.wait()
                # The fast worker's real event is not all-rank completion.
                assert not access.finished and owner._units
            with torch.cuda.device(rank), torch.cuda.stream(streams[rank]):
                torch.cuda._sleep(10_000_000)
                buffers[rank].fill_(rank + 1)
                events[rank].record()
            while not events[rank].query():
                assert not access.finished and owner._units
                await asyncio.sleep(.001)
            assert buffers[rank].item() == rank + 1
            if rank != slow_rank:
                fast_finished.set()

        async def workers(level):
            await asyncio.gather(worker(0), worker(1))
            assert all(event.query() for event in events)
            assert not access.finished

        engine.executor = SimpleNamespace(sleep=workers)
        await engine.sleep()
        assert access.finished and not owner._units and not scheduler.state_checkpoints._forwards
    asyncio.run(scenario())


def test_selected_failed_checkpoint_rolls_back_to_uncached_scheduler_admission(monkeypatch):
    maker = _maker(width=1, prefix=True, block_size=64, checkpoint_budget=8)
    scheduler, owner, lifecycle = maker.scheduler, maker.compact_state_manager, maker.state_checkpoints
    scheduler.seq_meta.compact_prefill_checkpoint_stride = 0
    scheduler.seq_meta.prefill_save_alignment = 0
    scheduler.add_session(0).add_sequence(list(range(128)))
    packet = maker._make_forward_inputs(True)
    ticket = packet['cache_inputs'].ssm_state_access_id
    lifecycle.publish_forward(ticket)
    scheduler.end_session(0)
    reader = scheduler.add_session(1).add_sequence(list(range(136)))
    pin = lifecycle.pin_restore
    failed = []

    def fail_between_match_and_pin(seq):
        if seq is reader and seq.prefix_cache.restore.is_selected:
            assert seq.num_history_ids == 128
            failed.append(lifecycle.fail_forward(ticket))
            owner.complete(owner._accesses[ticket], success=False)
            lifecycle.complete_forward(ticket)
        return pin(seq)

    monkeypatch.setattr(lifecycle, 'pin_restore', fail_between_match_and_pin)
    fallback = maker._make_forward_inputs(True)
    assert failed == [(ticket,)]
    assert fallback['inputs'].history_lengths.tolist() == [0]
    assert fallback['inputs'].seq_length.tolist() == [136]
    assert fallback['inputs'].ssm_source_state_ids.tolist() == [-1]
    assert not reader.prefix_cache.restore.is_selected
    assert not lifecycle._failed_nodes
    scheduler.end_session(1)
    scheduler.reset_cache_after_worker_drain()
    assert not owner._units and not owner._accesses


def test_already_admitted_failed_restore_rejects_before_new_forward_submission():
    maker = _maker(width=1, prefix=True, block_size=64, checkpoint_budget=8)
    scheduler, owner, lifecycle = maker.scheduler, maker.compact_state_manager, maker.state_checkpoints
    scheduler.seq_meta.compact_prefill_checkpoint_stride = 0
    scheduler.seq_meta.prefill_save_alignment = 0
    scheduler.add_session(0).add_sequence(list(range(128)))
    producer = maker._make_forward_inputs(True)
    ticket = producer['cache_inputs'].ssm_state_access_id
    lifecycle.publish_forward(ticket)
    scheduler.end_session(0)
    reader = scheduler.add_session(1).add_sequence(list(range(136)))
    assert scheduler.schedule(is_prefill=True).running == [reader]
    assert reader.prefix_cache.restore.pinned
    checkpoint = reader.prefix_cache.restore.node.state_checkpoint
    lifecycle.fail_forward(ticket)
    before = set(owner._accesses)
    with pytest.raises(RuntimeError, match='no longer published'):
        maker._bind_compact_state(_prefix_packet(maker, [reader], (136,)))
    assert set(owner._accesses) == before and checkpoint.pin_count == 1
    scheduler.end_session(1)
    assert checkpoint.pin_count == 0 and checkpoint.state_reference.producing
    scheduler.reset_cache_after_worker_drain()
    assert not lifecycle._failed_nodes and not owner._units and not owner._accesses




@pytest.mark.parametrize('width', [1, 4, 6])
@pytest.mark.parametrize('prefix', [False, True])
def test_loaded_private_input_is_bound_once_without_restore_copy(width, prefix):
    maker = _maker(width, requests=1, prefix=prefix, checkpoint_budget=2 if prefix else 0)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    seq = scheduler.add_session(0).add_sequence([1] * 72)
    scheduler.block_manager.allocate(seq)
    owner.allocate(seq)
    bank = owner.bank_for(seq)
    load = owner.submit_load(bank, 64)
    owner.complete(load)
    source = bank.loaded_input[0]
    seq.set_step(64)
    # This tests the real packet transaction, independently of connector wire.
    inputs = maker.create_model_inputs([seq], True)
    payload = dict(inputs=inputs, delta=None, running=[seq], cache_inputs=None, kv_connector_metadata=None)
    maker._bind_compact_state(payload)
    carrier = payload['cache_inputs']
    assert inputs.ssm_source_state_ids.tolist() == [source]
    assert carrier.state_restore_plan is carrier.state_save_plan is None
    assert bank.loaded_input is None and source in bank.unit_ids
    runtime = SSMStateCursor(owner.capacity.request_slots)
    device_inputs = inputs.to_device('cuda')
    device_inputs.ssm_state_meta = runtime.bind(
        device_inputs.ssm_state_meta, device_inputs.state_offsets, device_inputs.history_lengths,
        source_ids=inputs.ssm_source_state_ids.cuda())
    assert device_inputs.ssm_state_meta.input_state_ids.tolist() == [source]
    assert (source in device_inputs.ssm_state_meta.output_state_ids.flatten().tolist())
    assert len(bank.unit_ids) == (1)
    runtime.advance(device_inputs.ssm_state_meta, device_inputs.state_offsets, torch.tensor([1], device='cuda'))
    # A subsequent packet must retain device current ID, not replay the load.
    follow = maker.create_model_inputs([seq], True)
    next_payload = dict(inputs=follow, delta=None, running=[seq], cache_inputs=None, kv_connector_metadata=None)
    maker._bind_compact_state(next_payload)
    assert next_payload['inputs'].ssm_source_state_ids is None
    next_device = follow.to_device('cuda')
    rebound = runtime.bind(next_device.ssm_state_meta, next_device.state_offsets, next_device.history_lengths)
    assert rebound.input_state_ids.tolist() == device_inputs.ssm_state_meta.output_state_ids[:, 0].tolist()
    assert (rebound.input_state_ids.tolist() == [source])
    torch.cuda.synchronize()
    scheduler.end_session(0)
    owner.acknowledge(carrier.ssm_state_access_id)
    owner.acknowledge(next_payload['cache_inputs'].ssm_state_access_id)
    assert not owner._banks and not owner._units


@pytest.mark.parametrize('failure', ['history', 'layout', 'after_submit'])
def test_loaded_input_packet_rollback_retains_one_shot_binding(monkeypatch, failure):
    maker = _maker(1, requests=1)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    seq = scheduler.add_session(0).add_sequence([1] * 72)
    scheduler.block_manager.allocate(seq)
    owner.allocate(seq)
    bank = owner.bank_for(seq)
    owner.complete(owner.submit_load(bank, 64))
    original = bank.loaded_input
    seq.set_step(63 if failure == 'history' else 64)
    inputs = maker.create_model_inputs([seq], True)
    payload = dict(inputs=inputs, delta=None, running=[seq], cache_inputs=None, kv_connector_metadata=None)
    if failure == 'layout':
        inputs.seq_length = inputs.seq_length[:0]
    elif failure == 'after_submit':
        from lmdeploy.pytorch.engine import inputs_maker as module
        real_replace = module.replace

        def fail_after_submit(*args, **kwargs):
            if 'ssm_state_access_id' in kwargs:
                raise ValueError('injected post-submit preparation failure')
            return real_replace(*args, **kwargs)

        monkeypatch.setattr(module, 'replace', fail_after_submit)
    with pytest.raises(ValueError):
        maker._bind_compact_state(payload)
    assert bank.loaded_input == original
    assert not bank.accesses and not owner._accesses
    assert not bank.failed
    scheduler.end_session(0)
    assert not owner._units


@pytest.mark.parametrize('order', [(0, 1), (1, 0)])
def test_loaded_and_cold_rows_use_independent_device_source_overrides(order):
    maker = _maker(4, requests=2)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    seqs = [scheduler.add_session(i).add_sequence([1] * n) for i, n in enumerate((72, 65))]
    for seq in seqs:
        scheduler.block_manager.allocate(seq)
        owner.allocate(seq)
    bank = owner.bank_for(seqs[0])
    owner.complete(owner.submit_load(bank, 64))
    source = bank.loaded_input[0]
    seqs[0].set_step(64)
    running = [seqs[i] for i in order]
    inputs = maker.create_model_inputs(running, True)
    payload = dict(inputs=inputs, delta=None, running=running, cache_inputs=None, kv_connector_metadata=None)
    maker._bind_compact_state(payload)
    carrier = payload['cache_inputs']
    assert inputs.ssm_source_state_ids.tolist() == [source if i == 0 else -1 for i in order]
    device = inputs.to_device('cuda')
    runtime = SSMStateCursor(owner.capacity.request_slots)
    bound = runtime.bind(device.ssm_state_meta, device.state_offsets, device.history_lengths,
                         source_ids=inputs.ssm_source_state_ids.cuda())
    assert bound.input_state_ids.tolist() == [source if i == 0 else 0 for i in order]
    assert torch.equal(bound.output_state_ids[:, 0], bound.state_ids[:, 0])
    assert bound.output_state_ids[order.index(0), 0] == source
    assert (bound.output_state_ids > 0).all()
    torch.cuda.synchronize()
    owner.acknowledge(carrier.ssm_state_access_id)
    for i in range(2):
        scheduler.end_session(i)
    assert not owner._units and not owner._accesses


def _store_maker(*, apc=False, role='kv_producer', credits=1, requests=1, width=1):
    """Use production Scheduler composition, never replace owners after
    init."""
    from unittest.mock import Mock

    maker = _maker(width, budget=4096, requests=requests, prefix=apc, block_size=64,
                   checkpoint_budget=2 if apc else 0, store_role=role, store_credits=credits)
    store = maker.scheduler.kv_connector.connector_scheduler
    assert store._compact_owner is maker.compact_state_manager
    assert store._state_slots is None
    if store.client is not None:
        store.client.lookup = Mock(return_value=0)
    return maker, store


def _finish_store_maker(maker):
    for session_id in tuple(maker.scheduler.sessions):
        maker.scheduler.end_session(session_id)
    # These host tests never dispatch the model/connector packets to a worker.
    maker.scheduler.reset_cache_after_worker_drain()
    maker.scheduler.shutdown()
    owner = maker.compact_state_manager
    assert not owner._units and not owner._banks and not owner._accesses
    assert owner._store_refs == 0


@pytest.mark.parametrize('width', [1, 4])
@pytest.mark.parametrize('tokens', [1096, 2048, 3072])
def test_store_chunk_prefetch_exports_next_checkpoint_before_previous_ack(width, tokens):
    from lmdeploy.pytorch.kv_connector.base import KVConnectorOutput

    maker = _maker(width, budget=1024, requests=1, block_size=64,
                   store_role='kv_producer', store_credits=8)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    seq = scheduler.add_session(0).add_sequence([1] * tokens)
    packets, saves = [], []
    try:
        for history in range(0, tokens, 1024):
            # Match EngineLoop: prepare the next packet BEFORE acknowledging
            # the prior model output. All Store reads stay outstanding here.
            packet = maker._make_forward_inputs(True)
            inputs, carrier = packet['inputs'], packet['cache_inputs']
            end = min(history + 1024, tokens)
            assert inputs.history_lengths.tolist() == [history]
            assert inputs.seq_length.tolist() == [end - history]
            save, = packet['kv_connector_metadata'].save_requests
            assert save.state.boundary_tokens == end // 64 * 64
            assert save.state.forward_end == end
            assert inputs.ssm_prefill_meta.checkpoint_ids.tolist() == [save.state.unit_id]
            assert save.state.unit_id not in owner.bank_for(seq).unit_ids
            assert all(save.state.unit_id != old.state.unit_id for old in saves)
            if packets:
                old_ticket = packets[-1]['cache_inputs'].ssm_state_access_id
                owner.acknowledge(old_ticket)
                scheduler.state_checkpoints.complete_forward(old_ticket)
                assert all(old.state.unit_id in owner._units for old in saves)
            scheduler.activate_seqs([seq])
            maker.update_running_seqs([seq], inputs)
            scheduler.state_checkpoints.publish_forward(carrier.ssm_state_access_id)
            packets.append(packet)
            saves.append(save)
        last_ticket = packets[-1]['cache_inputs'].ssm_state_access_id
        owner.acknowledge(last_ticket)
        scheduler.state_checkpoints.complete_forward(last_ticket)
        scheduler.end_session(0)
        assert owner._store_refs == len(saves)
        assert all(save.state.unit_id in owner._units for save in saves)
        # Reverse I/O completion must release exact units, independently of
        # producer order and the now-closed request bank.
        for save in reversed(saves):
            scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={save.save_id}))
            assert save.state.unit_id not in owner._units
        assert not owner._accesses and not scheduler.kv_save_coordinator._leases
    finally:
        _finish_store_maker(maker)


@pytest.mark.parametrize('apc', [False, True])
@pytest.mark.parametrize('role', ['kv_producer', 'kv_both'])
@pytest.mark.parametrize('tokens', [64, 72, 136])
def test_store_prefill_has_one_direct_producer_with_independent_trie_lease(apc, role, tokens):
    from lmdeploy.pytorch.kv_connector.base import KVConnectorOutput, KVConnectorOutputAggregator
    from lmdeploy.pytorch.kv_connector.mooncake.store.data import build_prefix_block_hashes

    maker, store = _store_maker(apc=apc, role=role)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    seq = scheduler.add_session(0).add_sequence([1] * tokens)
    payload = maker._make_forward_inputs(True)
    inputs, carrier = payload['inputs'], payload['cache_inputs']
    metadata = payload['kv_connector_metadata']
    assert inputs.seq_length.tolist() == [tokens]
    request, = metadata.save_requests
    boundary = 64 if tokens <= 72 else 128
    assert request.state.boundary_tokens == boundary and request.state.forward_end == tokens
    cp_id, = inputs.ssm_prefill_meta.checkpoint_ids.tolist()
    assert request.state.unit_id == cp_id
    hashes = build_prefix_block_hashes(seq.all_ids[:tokens], 64)
    assert request.state.boundary_hash == hashes[boundary // 64 - 1]
    assert len(request.block_ids) == tokens // 64
    assert metadata.get_state_save_copies() == ()
    assert carrier.state_save_plan is carrier.state_restore_plan is None
    bank = owner.bank_for(seq)
    access = owner._accesses[carrier.ssm_state_access_id]
    assert len(access._writes) == 1 and access._writes[0].unit_id == cp_id
    assert cp_id not in bank.unit_ids and len(bank.unit_ids) == 1
    assert len(owner._units) == 2  # One private destination and one shared producer.
    assert owner._store_refs == 1 and store._state_slots is None
    if tokens == boundary:
        assert bank.input_ref.unit_id == cp_id
    assert seq.prefix_cache.pending_save.is_pending == apc
    scheduler.state_checkpoints.publish_forward(access.access_id)
    owner.acknowledge(access.access_id)
    scheduler.state_checkpoints.complete_forward(access.access_id)
    scheduler.end_session(0)
    # Store completion alone must not release a separately published Trie unit.
    aggregator = KVConnectorOutputAggregator(2)
    scheduler.update_connector_output(aggregator.aggregate([
        KVConnectorOutput(completed_save_ids={request.save_id}), None]))
    assert cp_id in owner._units and owner._store_refs == 1
    scheduler.update_connector_output(aggregator.aggregate([
        None, KVConnectorOutput(completed_save_ids={request.save_id})]))
    assert owner._store_refs == 0 and not scheduler.kv_save_coordinator._leases
    assert (cp_id in owner._units) == apc
    _finish_store_maker(maker)


@pytest.mark.parametrize('apc', [False, True])
def test_store_credit_exhaustion_skips_store_not_local_producer_or_request(apc):
    from lmdeploy.pytorch.kv_connector.base import KVConnectorOutput

    maker, store = _store_maker(apc=apc)
    owner, scheduler = maker.compact_state_manager, maker.scheduler
    scheduler.add_session(0).add_sequence([1] * 72)
    first = maker._make_forward_inputs(True)
    ticket = first['cache_inputs'].ssm_state_access_id
    scheduler.state_checkpoints.publish_forward(ticket)
    owner.acknowledge(ticket)
    scheduler.state_checkpoints.complete_forward(ticket)
    scheduler.end_session(0)
    scheduler.add_session(1).add_sequence([2] * 72)
    second = maker._make_forward_inputs(True)
    assert second['kv_connector_metadata'].save_requests == ()
    assert store._num_skipped_state_saves == 1
    second_cp, = second['inputs'].ssm_prefill_meta.checkpoint_ids.tolist()
    assert (second_cp > 0) == apc
    assert owner._store_refs == 1
    scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={0}))
    assert owner._store_refs == 0
    _finish_store_maker(maker)


@pytest.mark.parametrize('failure', ['hash', 'pin', 'retain', 'attach'])
def test_compact_store_packet_rollback_unwinds_model_io_kv_and_highwater(monkeypatch, failure):
    maker, store = _store_maker(apc=True, credits=2, requests=2)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    for i in range(2):
        scheduler.add_session(i).add_sequence([i + 1] * 72)
    if failure == 'hash':
        original = store._get_request_block_hashes
    elif failure == 'pin':
        original = scheduler.block_manager.pin_logical_blocks
    elif failure == 'retain':
        original = owner.retain
    else:
        original = scheduler.state_checkpoints.attach_forward
    calls = [0]

    def fail(*args, **kwargs):
        calls[0] += 1
        if failure == 'attach' or calls[0] == 2:
            raise RuntimeError('injected local packet failure')
        return original(*args, **kwargs)

    if failure == 'hash':
        monkeypatch.setattr(store, '_get_request_block_hashes', fail)
    elif failure == 'pin':
        monkeypatch.setattr(scheduler.block_manager, 'pin_logical_blocks', fail)
    elif failure == 'retain':
        monkeypatch.setattr(owner, 'retain', fail)
    else:
        monkeypatch.setattr(scheduler.state_checkpoints, 'attach_forward', fail)
    with pytest.raises(RuntimeError, match='injected local'):
        maker._make_forward_inputs(True)
    assert not owner._accesses and owner._store_refs == 0
    assert not store._inflight_saves and not scheduler.kv_save_coordinator._leases
    assert all(value == 0 for value in store._next_save_block.values())
    assert not scheduler.state_checkpoints._forwards
    assert all(bank.pending_checkpoint is None and bank.input_ref is None for bank in owner._banks.values())
    assert len(owner._units) == 2  # One prefill destination per admitted request.
    _finish_store_maker(maker)


def test_compact_store_shutdown_drain_retires_model_before_io_without_lost_ack():
    maker, store = _store_maker(apc=True)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    scheduler.add_session(0).add_sequence([1] * 72)
    payload = maker._make_forward_inputs(True)
    ticket = payload['cache_inputs'].ssm_state_access_id
    producer = owner._accesses[ticket]
    operation, = store._inflight_saves.values()
    io = operation.lease
    scheduler.state_checkpoints.publish_forward(ticket)
    scheduler.end_session(0)
    with pytest.raises(RuntimeError, match='actual I/O completion'):
        scheduler.shutdown()
    # Model-only drain explicitly preserves I/O ownership and cannot recycle it.
    owner.finish_cancelled_forwards_after_worker_drain(preserve_store=True)
    assert producer.finished and not io.finished and owner._store_refs == 1
    assert io._units[0].unit_id in owner._units
    _finish_store_maker(maker)
    assert io.finished


@pytest.mark.parametrize('rollback', [False, True])
def test_compact_packet_mixes_other_request_load_with_save_and_rolls_back_pending_load(monkeypatch, rollback):
    from lmdeploy.pytorch.kv_connector.base import KVConnectorOutput

    maker, store = _store_maker(apc=False, role='kv_both', requests=2)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    loading = scheduler.add_session(0).add_sequence([1] * 73)
    computing = scheduler.add_session(1).add_sequence([2] * 72)
    store.client.lookup.side_effect = lambda request_id, *args, **kwargs: 64 if request_id == loading.seq_id else 0
    if rollback:
        def fail(*args, **kwargs):
            raise RuntimeError('injected mixed packet failure')
        monkeypatch.setattr(scheduler.state_checkpoints, 'attach_forward', fail)
        with pytest.raises(RuntimeError, match='mixed packet'):
            maker._make_forward_inputs(True)
        assert loading.seq_id in store._pending_loads and loading.seq_id not in store._inflight_loads
        assert not store._inflight_saves and not scheduler.kv_save_coordinator._leases
        assert len(owner._accesses) == 1  # prior load lease survives packet rollback
    else:
        payload = maker._make_forward_inputs(True)
        assert payload['running'] == [computing]
        metadata = payload['kv_connector_metadata']
        assert len(metadata.load_requests) == len(metadata.save_requests) == 1
        assert metadata.load_requests[0].request_id == loading.seq_id
        producer = payload['cache_inputs'].ssm_state_access_id
        scheduler.state_checkpoints.publish_forward(producer)
        owner.acknowledge(producer)
        scheduler.state_checkpoints.complete_forward(producer)
        scheduler.update_connector_output(KVConnectorOutput(completed_save_ids={metadata.save_requests[0].save_id}))
    assert owner.bank_for(loading).pending_load is not None
    assert owner.bank_for(computing).pending_checkpoint is None
    assert owner._store_refs == 0
    _finish_store_maker(maker)


def test_compact_save_completion_before_producer_ack_keeps_lease_record():
    from lmdeploy.pytorch.kv_connector.base import KVConnectorOutput

    maker, store = _store_maker()
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    scheduler.add_session(0).add_sequence([1] * 72)
    payload = maker._make_forward_inputs(True)
    save, = payload['kv_connector_metadata'].save_requests
    output = KVConnectorOutput(completed_save_ids={save.save_id})
    with pytest.raises(ValueError, match='Dependency must complete'):
        scheduler.update_connector_output(output)
    assert save.save_id in store._inflight_saves
    assert owner._store_refs == 1 and scheduler.kv_save_coordinator.has_pending()
    ticket = payload['cache_inputs'].ssm_state_access_id
    owner.acknowledge(ticket)
    scheduler.state_checkpoints.complete_forward(ticket)
    scheduler.update_connector_output(output)
    assert not store._inflight_saves and owner._store_refs == 0
    _finish_store_maker(maker)


def _decode_binding_packet(maker, seqs, *, steps=None):
    for seq in seqs:
        assert maker.compact_state_manager.promote_to_decode(maker.compact_state_manager.bank_for(seq))
    delta = SimpleNamespace(max_q_seqlen=maker.engine_strategy.get_num_decode_tokens(),
                            block_offsets=torch.zeros((len(seqs), 1), dtype=torch.long),
                            ssm_state_ids=None, ssm_checkpoint_state_ids=None)
    payload = dict(inputs=None, delta=delta, running=seqs,
                   cache_inputs=None if steps is None else CacheCheckpointInputs(ssm_checkpoint_steps=steps),
                   kv_connector_metadata=None)
    maker._bind_compact_state(payload)
    return payload


def _ready_compact_decode(width=1):
    maker = _maker(width, checkpoint_budget=1)
    seqs = [maker.scheduler.add_session(i).add_sequence([1] * n) for i, n in enumerate((72, 65))]
    prefill = maker._make_forward_inputs(True)
    maker.scheduler.activate_seqs(seqs)
    maker.update_running_seqs(seqs, prefill['inputs'])
    maker.compact_state_manager.acknowledge(prefill['cache_inputs'].ssm_state_access_id)
    return maker, seqs


@pytest.mark.parametrize('width', [1, 4, 6])
def test_unchanged_decode_elides_only_bank_metadata_not_physical_holds(width):
    maker, seqs = _ready_compact_decode(width)
    owner = maker.compact_state_manager
    first = maker._make_forward_inputs(False)
    original = first['delta'].ssm_state_ids.clone()
    second = maker._make_forward_inputs(False)  # real prefetched delta, before first completion
    assert second['delta'].ssm_state_ids is None
    tickets = [p['cache_inputs'].ssm_state_access_id for p in (first, second)]
    assert tickets[0] != tickets[1]
    assert owner._accesses[tickets[0]].bank_ids == owner._accesses[tickets[1]].bank_ids
    for seq in seqs:
        assert len(owner.bank_for(seq).accesses) == 2
        maker.scheduler.end_session(seq.session_id)
    owner.acknowledge(tickets[0])
    assert owner.get_num_free_runtime() == 0
    owner.acknowledge(tickets[1])
    assert owner.get_num_free_runtime() == 2 and not owner._accesses
    torch.testing.assert_close(first['delta'].ssm_state_ids, original, rtol=0, atol=0)


@pytest.mark.parametrize('change', ['reorder', 'shrink', 'checkpoint', 'reset', 'prefill', 'failed_bind'])
def test_decode_bank_elision_is_invalidated_by_changed_context(monkeypatch, change):
    maker, seqs = _ready_compact_decode()
    owner = maker.compact_state_manager
    first = _decode_binding_packet(maker, seqs)
    owner.acknowledge(first['cache_inputs'].ssm_state_access_id)
    same = _decode_binding_packet(maker, seqs)
    assert same['delta'].ssm_state_ids is None
    owner.acknowledge(same['cache_inputs'].ssm_state_access_id)
    ref = None
    if change == 'reorder':
        seqs = seqs[::-1]
    elif change == 'shrink':
        seqs = seqs[:1]
    elif change == 'checkpoint':
        bank = owner.bank_for(seqs[0])
        ref = checkpoint_ready(owner, bank, boundary=64)
    elif change == 'reset':
        maker.reset_runtime_state()
    elif change == 'prefill':
        inputs = maker.create_model_inputs(seqs, True)
        packet = dict(inputs=inputs, delta=None, running=seqs, cache_inputs=None, kv_connector_metadata=None)
        maker._bind_compact_state(packet)
        owner.acknowledge(packet['cache_inputs'].ssm_state_access_id)
    else:
        submit = owner.submit_decode
        def fail(*args, **kwargs):
            raise RuntimeError('injected unchanged-decode bind failure')
        monkeypatch.setattr(owner, 'submit_decode', fail)
        with pytest.raises(RuntimeError, match='unchanged-decode'):
            _decode_binding_packet(maker, seqs)
        monkeypatch.setattr(owner, 'submit_decode', submit)
    changed = _decode_binding_packet(maker, seqs)
    if change == 'checkpoint':
        # An out-of-bank reservation must not invalidate the bank snapshot.
        assert changed['delta'].ssm_state_ids is None
    else:
        assert changed['delta'].ssm_state_ids.tolist() == [list(owner.bank_for(s).unit_ids) for s in seqs]
    owner.acknowledge(changed['cache_inputs'].ssm_state_access_id)
    for session_id in (0, 1):
        maker.scheduler.end_session(session_id)
    if ref is not None:
        owner.release_reference(ref)
    assert not owner._accesses and owner.get_num_free_runtime() == 2


@pytest.mark.parametrize('cancel', [False, True])
def test_failed_dispatch_invalidates_decode_bank_cache_without_releasing_holds(monkeypatch, cancel):
    maker, seqs = _ready_compact_decode()
    owner = maker.compact_state_manager
    first = _decode_binding_packet(maker, seqs)
    owner.acknowledge(first['cache_inputs'].ssm_state_access_id)
    second = _decode_binding_packet(maker, seqs)
    ticket = second['cache_inputs'].ssm_state_access_id
    assert second['delta'].ssm_state_ids is None
    monkeypatch.setattr(maker, '_make_forward_inputs', lambda *args: second)
    async def fail(_):
        if cancel:
            raise asyncio.CancelledError()
        raise RuntimeError('injected dispatch failure')
    maker.executor.forward_async = fail
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        asyncio.run(maker._send_next_inputs_impl())
    assert maker._last_compact_decode_banks is None
    assert ticket in owner._accesses and not owner._accesses[ticket].finished
    # No real GPU was dispatched in this fixture. Production must drain first.
    owner.complete(owner._accesses[ticket], success=False)
    for seq in seqs:
        maker.scheduler.end_session(seq.session_id)
    assert not owner._accesses


@pytest.mark.parametrize('pending', [False, True])
def test_decode_checkpoint_never_uses_unchanged_bank_elision(pending):
    maker, _, add = _prefix_fixture(width=1)
    maker.config.prefix_cache_decode_state_interval = 64
    seq = add(list(range(127 if pending else 128)))
    owner, lifecycle = maker.compact_state_manager, maker.state_checkpoints
    ordinary = _decode_binding_packet(maker, [seq])
    owner.acknowledge(ordinary['cache_inputs'].ssm_state_access_id)
    assert maker._last_compact_decode_banks is not None
    earlier = owner.submit_decode((owner.bank_for(seq),)) if pending else None
    packet = _decode_binding_packet(maker, [seq], steps=(128,))
    carrier = packet['cache_inputs']
    assert packet['delta'].ssm_state_ids is not None and packet['delta'].ssm_checkpoint_state_ids is not None
    assert maker._last_compact_decode_banks is None
    if earlier is not None:
        owner.acknowledge(earlier.access_id)
    # Host-only fixture: retire the unpublished producer, including its holds.
    owner.complete(owner._accesses[carrier.ssm_state_access_id], success=False)
    lifecycle.fail_forward(carrier.ssm_state_access_id)
    owner.free(seq)


@pytest.mark.parametrize('width', [1, 4, 6])
@pytest.mark.parametrize('graph', [False, True])
@torch.inference_mode()
@pytest.mark.usefixtures('release_cuda_graph_cycles')
def test_elided_host_delta_preserves_worker_advance_and_graph_ids(monkeypatch, width, graph):
    """Real host delta/H2D/reindex, all consumed counts, fixed-address staging.

    This checks the omitted metadata path, not model numerical updates or speculative sampling. The forced-rebind
    control keeps the old behavior.
    """
    from lmdeploy.pytorch.compact_state import SSMStateMetadata
    from lmdeploy.pytorch.kernels.cuda.compact_state import stage_compact_state_ids

    maker = _maker(width)
    owner = maker.compact_state_manager
    seqs = [maker.scheduler.add_session(i).add_sequence([1] * n) for i, n in enumerate((72, 65))]
    prefill = maker._make_forward_inputs(True)
    inputs = prefill['inputs'].to_device('cuda')
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.ssm_state_cursor = SSMStateCursor(owner.capacity.request_slots)
    agent._prev_chunk_output = None
    inputs = agent._prepare_inputs_prefill(inputs, None)
    decoding = get_model_inputs_next_decoding(inputs, torch.zeros((1, 2 * width), device='cuda', dtype=torch.long),
                                              width, None, ssm_state_cursor=agent.ssm_state_cursor)
    maker.scheduler.activate_seqs(seqs)
    maker.update_running_seqs(seqs, prefill['inputs'])
    first = maker._make_forward_inputs(False)
    steps = SimpleNamespace(model_inputs=decoding, extra_inputs=None, stopping_criteria=None, sampling_delta=None)

    def reindex(delta):
        steps.model_inputs = (_reindex_model_inputs(steps.model_inputs, delta) if width == 1 else
                              _reindex_model_inputs_arspec(steps.model_inputs, delta, width - 1))

    steps.reindex = reindex
    from lmdeploy.pytorch.strategies.base.step_inputs import StepInputs
    steps.ssm_state_cursor = agent.ssm_state_cursor
    steps.apply_delta = lambda delta: StepInputs.apply_delta(steps, delta)
    agent.step_inputs = steps
    sampling = SimpleNamespace(update_delta=lambda _: None)
    actual, *_ = agent._get_inputs_from_delta(first['delta'].to_device('cuda'), sampling)
    torch.cuda.synchronize()
    for packet in (prefill, first):
        owner.acknowledge(packet['cache_inputs'].ssm_state_access_id)

    # This destination simulates graph-padded metadata. Its address stays fixed
    # across all replays, including padding poison before every iteration.
    source = SSMStateMetadata(actual.ssm_state_meta.state_ids.clone(), actual.ssm_state_meta.input_state_ids.clone(),
                                actual.ssm_state_meta.output_state_ids.clone())
    destination = SSMStateMetadata(torch.empty((8, owner.capacity.bank_width), device='cuda', dtype=torch.long),
                                     torch.empty(8, device='cuda', dtype=torch.long),
                                     torch.empty((8, width), device='cuda', dtype=torch.long))
    stage_compact_state_ids(source, destination)
    captured = None
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured, stream=torch.cuda.Stream()):
            stage_compact_state_ids(source, destination)
    rebind = SSMStateCursor.rebind
    calls = []
    def counted_rebind(self, *args, **kwargs):
        calls.append(1)
        return rebind(self, *args, **kwargs)
    monkeypatch.setattr(SSMStateCursor, 'rebind', counted_rebind)
    for count in range(1, width + 1):
        consumed = torch.tensor([count, width + 1 - count],
                                device='cuda')
        advanced = actual.step(actual.input_ids, consumed, ssm_state_cursor=agent.ssm_state_cursor)
        expected = rebind(agent.ssm_state_cursor, advanced.ssm_state_meta, advanced.ssm_state_meta.state_ids)
        runtime_before = agent.ssm_state_cursor.current_ids.clone()
        steps.model_inputs = advanced
        packet = maker._make_forward_inputs(False)
        assert packet['delta'].ssm_state_ids is None
        delta = packet['delta'].to_device('cuda')
        delta.record_stream(torch.cuda.current_stream())
        actual, *_ = agent._get_inputs_from_delta(delta, sampling)
        assert not calls  # production worker did not invoke the planning operator again
        torch.testing.assert_close(agent.ssm_state_cursor.current_ids, runtime_before, rtol=0, atol=0)
        for field in ('state_ids', 'input_state_ids', 'output_state_ids'):
            torch.testing.assert_close(getattr(actual.ssm_state_meta, field), getattr(expected, field), rtol=0, atol=0)
            getattr(source, field).copy_(getattr(actual.ssm_state_meta, field))
            getattr(destination, field).fill_(999)
        if captured is not None:
            captured.replay()
        else:
            stage_compact_state_ids(source, destination)
        for field in ('state_ids', 'input_state_ids', 'output_state_ids'):
            torch.testing.assert_close(getattr(destination, field)[:2], getattr(expected, field), rtol=0, atol=0)
            assert (getattr(destination, field)[2:] == -1).all()
        torch.cuda.synchronize()
        owner.acknowledge(packet['cache_inputs'].ssm_state_access_id)
    for seq in seqs:
        maker.scheduler.end_session(seq.session_id)
    assert owner.get_num_free_runtime() == 2 and not owner._accesses


@pytest.mark.parametrize('width', [1, 4, 6])
def test_prefetched_chunks_allocate_only_phase_units_before_any_cpu_ack(width):
    maker = _maker(width, budget=64, requests=1)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    seq = scheduler.add_session(0).add_sequence([1] * 137)
    packets = []
    for history in (0, 64, 128):
        count = 1
        packet = maker._make_forward_inputs(True)
        packets.append(packet)
        bank = owner.bank_for(seq)
        ids = packet['inputs'].ssm_state_meta.state_ids[0].tolist()
        assert len(bank.unit_ids) == count
        assert ids == list(bank.unit_ids) + [-1] * (owner.capacity.bank_width - count)
        assert packet['inputs'].history_lengths.tolist() == [history]
        assert owner._accesses[packet['cache_inputs'].ssm_state_access_id].bank_ids == (bank.unit_ids,)
        scheduler.activate_seqs([seq])
        maker.update_running_seqs([seq], packet['inputs'])
    assert len(owner._accesses) == 3
    assert packets[0]['inputs'].ssm_state_meta.state_ids[0, 1:].tolist() == [-1] * (owner.capacity.bank_width - 1)
    decode = maker._make_forward_inputs(False)
    assert len(bank.unit_ids) == owner.capacity.bank_width
    scheduler.end_session(0)
    for packet in (*packets, decode):
        owner.acknowledge(packet['cache_inputs'].ssm_state_access_id)
    assert not owner._units and not owner._banks


def test_lingering_shared_reference_is_not_proof_current_input_is_shared():
    maker, _, add = _prefix_fixture(4)
    _, _, _ = _seed_prefix(maker, add, list(range(160)), 64)
    seq = add(list(range(160)), match=True)
    owner = maker.compact_state_manager
    first = _prefix_packet(maker, [seq], (0,))
    first['inputs'].seq_length = torch.tensor([64])
    maker._bind_compact_state(first)
    bank = owner.bank_for(seq)
    assert bank.input_ref is not None and len(bank.unit_ids) == 1
    # No CPU ack yet, but an ordered next chunk reads the first chunk's
    # private result, not the original checkpoint still held by the owner.
    following = _prefix_packet(maker, [seq], (0,))
    following['inputs'].history_lengths = torch.tensor([128])
    following['inputs'].seq_length = torch.tensor([32])
    maker._bind_compact_state(following)
    assert following['inputs'].ssm_source_state_ids.tolist() == [-1]
    assert bank.input_ref is not None and len(bank.unit_ids) == 1
    for packet in (first, following):
        owner.acknowledge(packet['cache_inputs'].ssm_state_access_id)
        maker.state_checkpoints.complete_forward(packet['cache_inputs'].ssm_state_access_id)
    owner.free(seq)


@pytest.mark.parametrize('reason', ['credit', 'unit'])
def test_store_capacity_skip_counts_only_successful_packet_not_retry(monkeypatch, reason):
    maker, store = _store_maker(credits=1, requests=2)
    owner, scheduler = maker.compact_state_manager, maker.scheduler
    donor = owner.allocate_bank(prefill=True)
    reference = checkpoint_ready(owner, donor, 64)
    store_ref = owner.retain(reference, 'store') if reason == 'credit' else None
    assert owner.get_num_free_optional() == 1  # reserved for the next request admission
    assert owner.get_num_free_store_saves() == (reason == 'unit')
    scheduler.add_session(0).add_sequence([1] * 72)
    original_bind = maker._bind_compact_state
    captured = []
    def bind(packet, **kwargs):
        captured.append((packet, kwargs))
        return original_bind(packet, **kwargs)
    monkeypatch.setattr(maker, '_bind_compact_state', bind)
    attach = scheduler.state_checkpoints.attach_forward
    def fail(*args, **kwargs):
        raise RuntimeError('CPU packet rejected')
    monkeypatch.setattr(scheduler.state_checkpoints, 'attach_forward', fail)
    with pytest.raises(RuntimeError, match='CPU packet rejected'):
        maker._make_forward_inputs(True)
    assert store._num_skipped_state_saves == 0
    assert owner.get_num_free_optional() == 0
    monkeypatch.setattr(scheduler.state_checkpoints, 'attach_forward', attach)
    records = []
    record = store.record_compact_save_skips
    def record_skips(skips):
        records.extend(skips)
        record(skips)
    monkeypatch.setattr(store, 'record_compact_save_skips', record_skips)
    packet, kwargs = captured[0]
    original_bind(packet, **kwargs)
    assert store._num_skipped_state_saves == 1
    assert records == [(int(packet['running'][0].seq_id), 64, reason)]
    assert packet['kv_connector_metadata'] is None
    if store_ref is not None:
        owner.release_reference(store_ref)
    owner.release_reference(reference)
    owner.release_bank(donor)
    _finish_store_maker(maker)


def test_store_policy_ineligible_is_not_capacity_skip(monkeypatch):
    maker, store = _store_maker()
    monkeypatch.setattr(maker.compact_state_manager, 'get_num_free_store_saves', lambda: 0)
    maker.scheduler.add_session(0).add_sequence([1] * 32)  # no aligned checkpoint
    maker._make_forward_inputs(True)
    assert store._num_skipped_state_saves == 0
    _finish_store_maker(maker)


@pytest.mark.parametrize('apc', [False, True])
def test_last_only_policy_keeps_internal_compact_export_and_store_boundary(apc):
    maker, store = _store_maker(apc=apc)
    maker.scheduler.cache_config.prefill_save_last_only = True
    maker.scheduler.seq_meta.prefill_save_last_only = True
    seq = maker.scheduler.add_session(0).add_sequence([1] * 136)
    packet = maker._make_forward_inputs(True)
    assert packet['inputs'].seq_length.tolist() == [136]
    save, = packet['kv_connector_metadata'].save_requests
    assert save.state.boundary_tokens == 128
    assert packet['inputs'].ssm_prefill_meta.checkpoint_offsets.tolist() == [128]
    # Even an explicitly supplied earlier candidate is policy-ineligible, not
    # a capacity skip; a repeated last boundary is not a new save either.
    skipped = []
    assert store.plan_compact_save_steps([seq], [136], [64], skipped=skipped) == (0,)
    assert store.plan_compact_save_steps([seq], [136], [128], skipped=skipped) == (0,)
    assert skipped == []
    _finish_store_maker(maker)


@pytest.mark.parametrize('history,tokens,budget,end,checkpoint', [
    (0, 136, 256, 136, 128), (0, 264, 128, 128, 0),
    (128, 264, 128, 256, 256), (256, 264, 128, 264, 0),
    (8, 136, 256, 128, 128),
])
def test_last_only_compact_chunk_plan(history, tokens, budget, end, checkpoint):
    from lmdeploy.pytorch.long_context import plan_long_context_chunk

    maker = _maker(1, prefix=True, checkpoint_budget=1)
    maker.scheduler.seq_meta.prefill_save_last_only = True
    seq = maker.scheduler.add_session(0).add_sequence([1] * tokens)
    seq.set_step(history)
    plan = plan_long_context_chunk(seq, budget)
    assert (plan.chunk_end, plan.checkpoint_step) == (end, checkpoint)
    maker.scheduler.end_session(0)
    maker.scheduler.shutdown()


@pytest.mark.parametrize('hold', ['checkpoint_pin', 'kv_reference'])
def test_compact_reset_reuses_clear_guards_without_partial_release(hold):
    maker = _maker(width=1, prefix=True, block_size=64, checkpoint_budget=8)
    scheduler, owner = maker.scheduler, maker.compact_state_manager
    seq = scheduler.add_session(0).add_sequence(list(range(128)))
    packet = maker._make_forward_inputs(True)
    ticket = packet['cache_inputs'].ssm_state_access_id
    node = seq.prefix_cache.pending_save.node
    scheduler.state_checkpoints.publish_forward(ticket)
    owner.acknowledge(ticket)
    scheduler.state_checkpoints.complete_forward(ticket)
    scheduler.end_session(0)
    trie = scheduler.block_trie
    allocator = scheduler.block_manager.allocator
    import numpy as np
    blocks = np.array([node.block_id], dtype=np.int64)
    if hold == 'checkpoint_pin':
        reader = scheduler.add_session(1).add_sequence(list(range(136)))
        trie.match(reader)
        scheduler.state_checkpoints.pin_restore(reader)
        assert node.state_checkpoint.pin_count > 0
    else:
        allocator.add_ref_count(blocks, 1)
    before = (owner.get_num_free(), scheduler.block_manager.get_num_free_gpu_blocks())
    checkpoint = node.state_checkpoint
    with pytest.raises(RuntimeError, match='pinned|referenced'):
        trie.reset_compact_after_worker_drain()
    assert node.state_checkpoint is checkpoint
    assert before == (owner.get_num_free(), scheduler.block_manager.get_num_free_gpu_blocks())
    if hold == 'checkpoint_pin':
        scheduler.end_session(1)
    else:
        allocator.free(blocks)
    trie.reset_compact_after_worker_drain()
    assert not owner._units and not trie._roots
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 128


def test_transfer_retirement_leaves_trie_invalidation_to_full_reset(monkeypatch):
    maker = _maker(width=1, prefix=True, block_size=64, checkpoint_budget=8)
    scheduler = maker.scheduler
    scheduler.seq_meta.compact_prefill_checkpoint_stride = 0
    scheduler.seq_meta.prefill_save_alignment = 0
    seq = scheduler.add_session(0).add_sequence(list(range(128)))
    assert scheduler.schedule(is_prefill=True).running == [seq]
    packet = _prefix_packet(maker, [seq], (128,))
    maker._bind_compact_state(packet)
    scheduler.end_session(0)
    clears = []
    clear = scheduler.block_trie.clear

    def tracked_clear():
        clears.append(True)
        return clear()

    monkeypatch.setattr(scheduler.block_trie, 'clear', tracked_clear)
    scheduler.finish_kv_transfers_after_worker_drain()
    assert not clears
    scheduler.reset_cache_after_worker_drain()
    assert clears == [True]
    owner = maker.compact_state_manager
    assert not owner._units and not owner._banks and not owner._accesses
    assert not scheduler.state_checkpoints._forwards
