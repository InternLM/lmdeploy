# Copyright (c) OpenMMLab. All rights reserved.
"""Compact access acknowledgment transport, including real per-rank events.

The real-GPU tests use local RPC adapters, not a Ray cluster or trained model. Engine-loop release is exercised;
scheduler admission/plan binding is separate.
"""
import asyncio
import pickle
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent, BatchedOutputs
from lmdeploy.pytorch.kv_connector import KVConnectorOutput, KVConnectorOutputAggregator
from tests.pytorch.compact_state_reference import StateAccessRow, submit


@pytest.fixture
def ray_class():
    pytest.importorskip('ray')
    from lmdeploy.pytorch.engine.executor.ray_executor import RayExecutor
    return RayExecutor


class _Worker:
    def __init__(self, output):
        self.output = output
        self.calls = 0
        self.get_outputs = SimpleNamespace(remote=self.get)

    async def get(self):
        self.calls += 1
        if isinstance(self.output, Exception):
            raise self.output
        return self.output.to_numpy()


def _executor(ray_class, outputs, *, connector=False, access_id=7):
    executor = ray_class.__new__(ray_class)
    executor.workers = [_Worker(output) for output in outputs]
    executor._output_steps = deque([(connector, access_id)])
    executor._kv_output_aggregator = KVConnectorOutputAggregator(len(outputs))
    return executor


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '7', torch.tensor(7)])
def test_access_identity_requires_positive_host_integer(value):
    with pytest.raises(ValueError, match='positive host integer'):
        CacheCheckpointInputs(ssm_state_access_id=value)


def test_access_identity_survives_transfer_pickle_and_output_conversion():
    packet = pickle.loads(pickle.dumps(CacheCheckpointInputs(ssm_state_access_id=17)))
    assert packet.to_device('meta').ssm_state_access_id == 17
    output = BatchedOutputs(None, None, ssm_state_access_id=17)
    assert pickle.loads(pickle.dumps(output.to_cpu().to_numpy())).to_tensor().ssm_state_access_id == 17
    for field in ('state_restore_plan', 'state_save_plan'):
        with pytest.raises(ValueError, match='legacy state-copy'):
            CacheCheckpointInputs(ssm_state_access_id=17, **{field: ((1,), (2,))})


@pytest.mark.parametrize('connector', [False, True])
def test_ray_returns_compact_identity_only_after_every_worker(ray_class, connector):
    outputs = [BatchedOutputs(torch.tensor([5]), torch.tensor([False]), ssm_state_access_id=7),
               BatchedOutputs(None, None, ssm_state_access_id=7)]
    if connector:
        outputs[0].kv_connector_output = KVConnectorOutput(finished_receiving={11})
        outputs[1].kv_connector_output = KVConnectorOutput(finished_receiving={11}, failed_receiving={11})
    executor = _executor(ray_class, outputs, connector=connector)
    result = asyncio.run(executor.get_output_async())
    assert result.ssm_state_access_id == 7
    assert result.next_token_ids.tolist() == [5]
    assert [worker.calls for worker in executor.workers] == [1, 1]
    if connector:
        assert result.kv_connector_output.failed_receiving == {11}
    else:
        assert result.kv_connector_output is None


@pytest.mark.parametrize('second', [None, 6, 8, 'failure'])
def test_ray_rejects_missing_stale_future_or_failed_worker_completion(ray_class, second):
    output = (RuntimeError('worker failed') if second == 'failure'
              else BatchedOutputs(None, None, ssm_state_access_id=second))
    executor = _executor(ray_class, [BatchedOutputs(None, None, ssm_state_access_id=7), output])
    with pytest.raises(RuntimeError, match='identity mismatch|worker failed'):
        asyncio.run(executor.get_output_async())
    assert [worker.calls for worker in executor.workers] == [1, 1]


@pytest.mark.parametrize('unexpected', [False, True])
def test_ray_legacy_keeps_leader_only_fast_path_and_rejects_stale_compact_output(ray_class, unexpected):
    executor = _executor(ray_class, [BatchedOutputs(None, None, ssm_state_access_id=7 if unexpected else None),
                                     RuntimeError('must not fetch non-output legacy rank')], access_id=None)
    if unexpected:
        with pytest.raises(RuntimeError, match='Unexpected compact'):
            asyncio.run(executor.get_output_async())
    else:
        assert asyncio.run(executor.get_output_async()).ssm_state_access_id is None
    assert [worker.calls for worker in executor.workers] == [1, 0]


@pytest.mark.parametrize('case', ['missing_ticket', 'legacy_ticket', 'dummy_ticket', 'no_model'])
def test_agent_requires_ticket_for_real_compact_work_and_rejects_spurious_tickets(monkeypatch, case):
    from lmdeploy.pytorch.engine.model_agent import agent as module
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.rank = 0
    agent.kv_connector = None
    agent._prepare_inputs_prefill = lambda inputs, delta: inputs
    monkeypatch.setattr(module, 'get_dist_manager', lambda: SimpleNamespace(current_context=lambda: SimpleNamespace(
        dist_config=SimpleNamespace(attn_tp=1, dp=1))))
    inputs = SimpleNamespace(is_dummy=case == 'dummy_ticket',
                              ssm_state_meta=None if case == 'legacy_ticket' else object())
    packet = None if case == 'missing_ticket' else CacheCheckpointInputs(ssm_state_access_id=7)
    with pytest.raises(ValueError, match='Compact'):
        asyncio.run(agent._async_step(None if case == 'no_model' else inputs, cache_inputs=packet))


def test_legacy_mp_dispatch_uses_constructor_validated_mode():
    from lmdeploy.pytorch.engine.executor.mp_executor import MPExecutor
    executor = MPExecutor.__new__(MPExecutor)
    calls = []

    async def dispatch(*args, **kwargs):
        calls.append((args, kwargs))

    executor.collective_rpc_async = dispatch
    packet = dict(inputs=None, cache_inputs=None)
    asyncio.run(executor.forward_async(packet))
    assert calls == [(('forward_async',), dict(args=(packet,), return_mask=0))]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two allowed GPUs')
@pytest.mark.parametrize('slow_rank', [0, 1])
def test_real_two_gpu_events_prevent_owner_recycling_before_all_shards(ray_class, slow_rank):
    from lmdeploy.pytorch.config import SSMStateCapacity
    from lmdeploy.pytorch.paging.compact_state import SSMStateManager

    owner = SSMStateManager(SSMStateCapacity(2, 1, 0, 0))
    bank = owner.allocate_bank()
    access = owner.submit_decode((bank,))
    owner.release_bank(bank)
    agents, pools, streams = [], [], []
    for rank in range(2):
        with torch.cuda.device(rank):
            pool = torch.zeros(owner.capacity.num_units, device='cuda')
            pool.fill_(0)  # Warm allocator/fill before the delayed experiment.
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            agent = BaseModelAgent.__new__(BaseModelAgent)
            agent._out_que = asyncio.Queue()
            agent.out_stream = torch.cuda.Stream()
            agent._release_completed_h2d_transfers = lambda: None
            agents.append(agent)
            pools.append(pool)
            streams.append(stream)
    executor = ray_class.__new__(ray_class)
    executor._output_steps = deque([(False, access.access_id)])
    executor._kv_output_aggregator = KVConnectorOutputAggregator(2)

    async def run():
        fast_done = asyncio.Event()
        completion_events = []
        for rank, (agent, pool, stream) in enumerate(zip(agents, pools, streams)):
            with torch.cuda.device(rank), torch.cuda.stream(stream):
                if rank == slow_rank:
                    torch.cuda._sleep(1_000_000_000)
                pool[bank.unit_ids[0]].fill_(rank + 1)
                agent._push_output(BatchedOutputs(None, None, ssm_state_access_id=access.access_id))
                completion_events.append(agent._out_que._queue[0][1])

        async def get_rank(rank):
            # Real model-agent method queries/waits the real CUDA event, then
            # converts output; only the Ray RPC transport is replaced locally.
            output = await agents[rank].get_output_async()
            if rank != slow_rank:
                fast_done.set()
            return output.to_numpy()

        executor.workers = [SimpleNamespace(get_outputs=SimpleNamespace(remote=lambda r=r: get_rank(r)))
                            for r in range(2)]
        from lmdeploy.pytorch.engine.engine_loop import EngineLoop
        loop = EngineLoop.__new__(EngineLoop)
        loop.compact_state_manager = owner
        loop.executor = executor
        loop.inputs_maker = SimpleNamespace(update_running_seqs=lambda *_: None)
        loop.state_checkpoints = object()  # Legacy dispatch/unpin must not run.
        delivered = []
        loop._finish_forward_output = lambda output, *_: delivered.append(output)

        async def prefetch():
            assert not access.finished
            return None, None

        loop._prefetch_next_inputs = prefetch
        packet = dict(inputs=SimpleNamespace(ssm_prefill_meta=None), delta=None,
                      cache_inputs=CacheCheckpointInputs(ssm_state_access_id=access.access_id))
        task = asyncio.create_task(loop._main_loop_get_outputs([], packet))
        await asyncio.wait_for(fast_done.wait(), timeout=10)
        assert not completion_events[slow_rank].query(), 'GPU delay must cover the fast rank completion'
        assert not task.done()
        assert not access.finished
        assert owner.get_num_free() == 0
        with pytest.raises(RuntimeError, match='complete state bank'):
            owner.allocate_bank()
        assert await asyncio.wait_for(task, timeout=10) == (None, None)
        assert delivered[0].ssm_state_access_id == access.access_id
        assert all(event.query() for event in completion_events)
        assert access.finished  # Real engine-loop acknowledgment released it.

    asyncio.run(run())
    assert owner.get_num_free() == owner.capacity.num_units - 1
    for rank, pool in enumerate(pools):
        assert pool[bank.unit_ids[0]].item() == rank + 1


@pytest.mark.parametrize('result', ['correct', 'missing', 'wrong', 'failure'])
def test_engine_loop_matches_exact_access_and_preserves_prefetched_holds(result):
    from lmdeploy.pytorch.config import SSMStateCapacity
    from lmdeploy.pytorch.engine.engine_loop import EngineLoop
    from lmdeploy.pytorch.paging.compact_state import SSMStateManager

    owner = SSMStateManager(SSMStateCapacity(2, 1, 0, 0))
    bank = owner.allocate_bank()
    first = owner.submit_decode((bank,))
    later = []
    loop = EngineLoop.__new__(EngineLoop)
    loop.compact_state_manager = owner
    loop.inputs_maker = SimpleNamespace(update_running_seqs=lambda *_: None)
    loop.state_checkpoints = object()  # Compact path must never call old hooks.
    published = []
    loop._finish_forward_output = lambda output, *_: published.append(output)

    async def prefetch():
        later.append(owner.submit_decode((bank,)))
        owner.release_bank(bank)
        return None, None

    async def output():
        assert owner.get_num_free() == 0 and not first.finished
        if result == 'failure':
            raise RuntimeError('worker failed')
        if result == 'missing':
            return None
        access_id = first.access_id if result == 'correct' else first.access_id + 1
        return BatchedOutputs(None, None, ssm_state_access_id=access_id)

    loop._prefetch_next_inputs = prefetch
    loop.executor = SimpleNamespace(get_output_async=output)
    packet = dict(inputs=SimpleNamespace(ssm_prefill_meta=None), delta=None,
                  cache_inputs=CacheCheckpointInputs(ssm_state_access_id=first.access_id))
    if result == 'correct':
        assert asyncio.run(loop._main_loop_get_outputs([], packet)) == (None, None)
        assert first.finished and len(published) == 1
    else:
        with pytest.raises(RuntimeError, match='match|worker failed'):
            asyncio.run(loop._main_loop_get_outputs([], packet))
        assert not first.finished and not published
    assert not later[0].finished and owner.get_num_free() == 0
    # Test-only quiescent cleanup; a failed RPC alone must NOT release storage.
    owner.complete(first, success=False)
    owner.acknowledge(later[0].access_id)
    assert owner.get_num_free() == owner.capacity.num_units - 1


@pytest.mark.parametrize('connector', [False, True])
@pytest.mark.parametrize('access_id', [None, 7])
def test_ray_dispatch_records_one_combined_completion_marker(monkeypatch, ray_class, connector, access_id):
    import lmdeploy.pytorch.engine.executor.ray_executor as module
    executor = ray_class.__new__(ray_class)
    executor.dag = object()
    executor._prev_out = None
    executor._output_steps = deque()
    sent = []
    monkeypatch.setattr(module.ray, 'put', lambda value: value)

    async def run():
        def forward(packet):
            sent.append(packet)
            future = asyncio.get_running_loop().create_future()
            future.set_result(None)
            return future

        executor.workers = [SimpleNamespace(forward_async=SimpleNamespace(remote=forward)) for _ in range(2)]
        packet = dict(cache_inputs=CacheCheckpointInputs(ssm_state_access_id=access_id),
                      kv_connector_metadata=object() if connector else None)
        await executor.forward_async(packet)
        await asyncio.gather(*executor._prev_out)
        assert sent == [packet, packet]
        assert list(executor._output_steps) == [(connector, access_id)]

    asyncio.run(run())


@pytest.mark.parametrize('tokens,consumed', [([7], [True]), ([[7, -1, -1, -1]], [True]),
                                             ([[-1, -1, -1, -1]], [False]),
                                             ([[7, 8, -1, -1], [-1, -1, -1, -1]], [True, False]),
                                             (None, [False])])
def test_engine_loop_retires_decode_input_from_completed_target_token_mask(tokens, consumed):
    from lmdeploy.pytorch.config import SSMStateCapacity
    from lmdeploy.pytorch.engine.engine_loop import EngineLoop
    from lmdeploy.pytorch.paging.compact_state import SSMStateManager

    count = len(consumed)
    owner = SSMStateManager(SSMStateCapacity(count + 1, 4, 2))
    banks = tuple(owner.allocate_bank() for _ in range(count))
    checkpoint = owner.reserve_checkpoint(banks[0], 64)
    producer = submit(owner, (StateAccessRow(banks[0], 0, (checkpoint.unit_id,), checkpoint),))
    owner.complete(producer)
    for bank in banks:
        owner.bind_input(bank, checkpoint)
    references = [bank.input_ref for bank in banks]
    owner.release_reference(checkpoint)
    access = owner.submit_decode(banks)
    next_access = []
    loop = EngineLoop.__new__(EngineLoop)
    loop.compact_state_manager = owner
    loop.inputs_maker = SimpleNamespace(update_running_seqs=lambda *_: None)
    loop.state_checkpoints = object()  # This is a decode, not a prefill lifecycle hook.
    loop._finish_forward_output = lambda *_: None

    async def prefetch():
        assert not access.finished and all(ref.active for ref in references)
        next_access.append(owner.submit_decode(banks))
        return None, None

    async def output():
        assert all(ref.active for ref in references)
        return BatchedOutputs(None if tokens is None else torch.tensor(tokens), None,
                              ssm_state_access_id=access.access_id)

    loop._prefetch_next_inputs = prefetch
    loop.executor = SimpleNamespace(get_output_async=output)
    packet = dict(inputs=None, delta=SimpleNamespace(ssm_checkpoint_state_ids=None, max_q_seqlen=4),
                  cache_inputs=CacheCheckpointInputs(ssm_state_access_id=access.access_id))
    asyncio.run(loop._main_loop_get_outputs([object()] * count, packet))
    assert access.finished
    assert [bank.input_ref is None for bank in banks] == consumed
    assert [ref.active for ref in references] == [not value for value in consumed]
    # Even with zero accepted drafts (one target token), clear the logical
    # reference but not the physical unit held by a prefetched consumer.
    assert checkpoint.unit_id in owner._units and not next_access[0].finished
    owner.acknowledge(next_access[0].access_id, consumed_inputs=(True,) * count)
    assert checkpoint.unit_id not in owner._units
    for bank in banks:
        owner.release_bank(bank)


@pytest.mark.parametrize('case', ['wrong_rows', 'cuda', 'rank3', 'zero_width', 'dtype'])
def test_engine_loop_rejects_unproven_consumption_without_ack(case):
    from lmdeploy.pytorch.config import SSMStateCapacity
    from lmdeploy.pytorch.engine.engine_loop import EngineLoop
    from lmdeploy.pytorch.paging.compact_state import SSMStateManager

    owner = SSMStateManager(SSMStateCapacity(2, 1, 1))
    bank = owner.allocate_bank()
    checkpoint = owner.reserve_checkpoint(bank, 64)
    producer = submit(owner, (StateAccessRow(bank, 0, (checkpoint.unit_id,), checkpoint),))
    owner.complete(producer)
    owner.bind_input(bank, checkpoint)
    owner.release_reference(checkpoint)
    access = owner.submit_decode((bank,))
    tokens = torch.ones((2 if case == 'wrong_rows' else 1, 1), dtype=torch.int64)
    if case == 'cuda':
        tokens = tokens.cuda()
    elif case == 'dtype':
        tokens = tokens.float()
    elif case == 'rank3':
        tokens = tokens.unsqueeze(-1)
    loop = EngineLoop.__new__(EngineLoop)
    loop.compact_state_manager = owner
    loop.inputs_maker = SimpleNamespace(update_running_seqs=lambda *_: None)

    async def prefetch():
        return None, None

    async def output():
        return BatchedOutputs(tokens, None, ssm_state_access_id=access.access_id)

    loop._prefetch_next_inputs = prefetch
    loop.executor = SimpleNamespace(get_output_async=output)
    packet = dict(inputs=None, delta=SimpleNamespace(ssm_checkpoint_state_ids=None,
                                                     max_q_seqlen=0 if case == 'zero_width' else 1),
                  cache_inputs=CacheCheckpointInputs(ssm_state_access_id=access.access_id))
    with pytest.raises(ValueError, match='completed host token rows'):
        asyncio.run(loop._main_loop_get_outputs([object()], packet))
    assert not access.finished
    owner.acknowledge(access.access_id)
    owner.release_bank(bank)


def test_decode_without_shared_input_skips_consumption_tensor_work(monkeypatch):
    from lmdeploy.pytorch.config import SSMStateCapacity
    from lmdeploy.pytorch.engine.engine_loop import EngineLoop
    from lmdeploy.pytorch.paging.compact_state import SSMStateManager

    owner = SSMStateManager(SSMStateCapacity(2, 1))
    bank = owner.allocate_bank()
    access = owner.submit_decode((bank,))
    assert not owner.needs_input_consumption(access.access_id)
    assert access.input_references == ()
    loop = EngineLoop.__new__(EngineLoop)
    loop.compact_state_manager = owner
    loop.inputs_maker = SimpleNamespace(update_running_seqs=lambda *_: None)
    loop._finish_forward_output = lambda *_: None
    result = BatchedOutputs(torch.tensor([7]), None, ssm_state_access_id=access.access_id)

    async def prefetch():
        return None, None

    async def output():
        return result

    def forbidden(*args, **kwargs):
        pytest.fail('No shared input: no additional token-mask conversion is needed.')

    monkeypatch.setattr(torch.Tensor, 'tolist', forbidden)
    loop._prefetch_next_inputs = prefetch
    loop.executor = SimpleNamespace(get_output_async=output)
    packet = dict(inputs=None, delta=SimpleNamespace(ssm_checkpoint_state_ids=None, max_q_seqlen=1),
                  cache_inputs=CacheCheckpointInputs(ssm_state_access_id=access.access_id))
    asyncio.run(loop._main_loop_get_outputs([object()], packet))
    assert access.finished
    owner.release_bank(bank)


@pytest.mark.parametrize('cancel', [False, True])
def test_decode_input_retirement_waits_for_real_worker_event(cancel):
    from lmdeploy.pytorch.config import SSMStateCapacity
    from lmdeploy.pytorch.engine.engine_loop import EngineLoop
    from lmdeploy.pytorch.paging.compact_state import SSMStateManager

    owner = SSMStateManager(SSMStateCapacity(2, 1, 1))
    bank = owner.allocate_bank()
    checkpoint = owner.reserve_checkpoint(bank, 64)
    producer = submit(owner, (StateAccessRow(bank, 0, (checkpoint.unit_id,), checkpoint),))
    owner.complete(producer)
    owner.bind_input(bank, checkpoint)
    ref = bank.input_ref
    owner.release_reference(checkpoint)
    access = owner.submit_decode((bank,))
    pool = torch.ones(owner.capacity.num_units, device='cuda')
    stream = torch.cuda.Stream()
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent._out_que = asyncio.Queue()
    agent.out_stream = torch.cuda.Stream()
    agent._release_completed_h2d_transfers = lambda: None
    tokens = torch.tensor([7], device='cuda')
    # Warm the test payload arithmetic before inserting a real execution delay.
    pool[bank.unit_ids[0]].copy_(pool[ref.unit_id] + 1)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(150_000_000)
        pool[bank.unit_ids[0]].copy_(pool[ref.unit_id] + 1)
        agent._push_output(BatchedOutputs(tokens, None, ssm_state_access_id=access.access_id))
    done = agent._out_que._queue[0][1]
    loop = EngineLoop.__new__(EngineLoop)
    loop.compact_state_manager = owner
    loop.inputs_maker = SimpleNamespace(update_running_seqs=lambda *_: None)
    loop._finish_forward_output = lambda *_: None
    loop.executor = agent

    async def prefetch():
        assert not done.query() and ref.active and not access.finished
        if cancel:
            owner.release_bank(bank)
        assert ref.unit_id in owner._units
        return None, None

    loop._prefetch_next_inputs = prefetch
    packet = dict(inputs=None, delta=SimpleNamespace(ssm_checkpoint_state_ids=None, max_q_seqlen=1),
                  cache_inputs=CacheCheckpointInputs(ssm_state_access_id=access.access_id))
    asyncio.run(loop._main_loop_get_outputs([object()], packet))
    assert done.query() and access.finished and not ref.active
    assert ref.unit_id not in owner._units
    assert pool[bank.unit_ids[0]].item() == 2
    if not cancel:
        owner.release_bank(bank)
