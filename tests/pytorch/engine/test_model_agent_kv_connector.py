# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import builtins
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.kv_connector import (
    KVConnectorMetadata,
    KVConnectorOutput,
    KVConnectorOutputAggregator,
    KVConnectorRole,
    build_kv_connector,
)


def _cache_config(transfer_config=None):
    return CacheConfig(
        max_batches=1,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=1,
        kv_transfer_config=transfer_config,
    )


def _enabled_config(connector='MooncakeStoreConnector'):
    return _cache_config(KVTransferConfig(kv_connector=connector, kv_role='kv_both'))


def test_factory_disabled_does_not_import_mooncake(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith('lmdeploy.pytorch.kv_connector.mooncake'):
            raise AssertionError('disabled connector imported Mooncake')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)

    assert build_kv_connector(KVConnectorRole.WORKER, _cache_config()) is None


def test_factory_rejects_unknown_connector():
    with pytest.raises(ValueError, match='Unsupported KV connector'):
        build_kv_connector(KVConnectorRole.WORKER, _enabled_config('UnknownConnector'))


def test_factory_forwards_worker_rank_context(monkeypatch):
    from lmdeploy.pytorch.kv_connector.mooncake.store import connector as connector_module

    captured = {}
    result = object()

    def fake_connector(role, cache_config, **kwargs):
        captured.update(role=role, cache_config=cache_config, **kwargs)
        return result

    monkeypatch.setattr(connector_module, 'MooncakeStoreConnector', fake_connector)
    cache_config = _enabled_config()

    connector = build_kv_connector(
        KVConnectorRole.WORKER,
        cache_config,
        global_rank=7,
        tp_rank=3,
        tp_size=8,
    )

    assert connector is result
    assert captured == {
        'role': KVConnectorRole.WORKER,
        'cache_config': cache_config,
        'global_rank': 7,
        'tp_rank': 3,
        'tp_size': 8,
        'kv_head_replica_num': 1,
    }


def _bare_model_agent():
    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.all_context = nullcontext
    agent.compact_state_plan = None
    agent.cache_config = _enabled_config()
    agent.model_config = SimpleNamespace(num_replicate_key_value_heads=4)
    agent.rank = 7
    agent.cache_stream = object()
    agent.block_cache_plan = object()
    agent.dist_config = SimpleNamespace(attn_tp=8)
    agent.memdecode_agent = None
    agent._checkpoint_engine_zmq_ctx = None
    return agent


def test_build_cache_engine_replaces_connector_and_registers_row_mapping(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    target_rows = {'kv': object(), 'index': object()}
    mtp_rows = {'kv': object(), 'index': object()}
    state_pools = (object(), )
    cache_engine = SimpleNamespace(connector_kv_caches=target_rows)
    mtp_cache_engine = SimpleNamespace(connector_kv_caches=mtp_rows)
    state_cache_engine = SimpleNamespace(compact_plan=None, connector_state_cache_pools=state_pools)

    class _OldConnector:

        def shutdown(self):
            events.append('old-shutdown')

    class _NewConnector:

        def register_kv_caches(self, caches, *, state_cache_pools=()):
            events.append(('register', caches, state_cache_pools))

        def shutdown(self):
            events.append('new-shutdown')

    new_connector = _NewConnector()
    agent = _bare_model_agent()
    agent.kv_connector = _OldConnector()
    agent.cache_engine = object()
    agent.state_cache_engine = object()

    def build_spec_cache_engine(stream):
        events.append(('spec', stream))
        agent.spec_agent.cache_engine = mtp_cache_engine

    agent.spec_agent = SimpleNamespace(
        cache_engine=None,
        specdecode_config=None,
        build_cache_engine=build_spec_cache_engine,
    )

    def fake_cache_engine(*args, **kwargs):
        events.append(('cache', args, kwargs))
        return cache_engine

    def fake_state_cache_engine(*args, **kwargs):
        events.append(('state', args, kwargs))
        return state_cache_engine

    def fake_build_connector(role, cache_config, **kwargs):
        events.append(('factory', role, cache_config, kwargs))
        return new_connector

    dist_ctx = SimpleNamespace(attn_tp_group=SimpleNamespace(rank=3))
    monkeypatch.setattr(agent_module, 'CacheEngine', fake_cache_engine)
    monkeypatch.setattr(agent_module, 'StateCacheEngine', fake_state_cache_engine)
    monkeypatch.setattr(agent_module, 'build_kv_connector', fake_build_connector)
    monkeypatch.setattr(agent_module, 'get_dist_manager',
                        lambda: SimpleNamespace(current_context=lambda: dist_ctx))

    agent.build_cache_engine()

    assert events[0] == 'old-shutdown'
    assert [event[0] for event in events[1:]] == ['cache', 'state', 'spec', 'factory', 'register']
    cache_call = events[1]
    assert cache_call[2]['rank'] == 7
    assert cache_call[2]['tp_rank'] == 3
    assert cache_call[2]['block_cache_plan'] is agent.block_cache_plan
    factory_call = events[4]
    assert factory_call[1] is KVConnectorRole.WORKER
    assert factory_call[3] == {
        'global_rank': 7,
        'tp_rank': 3,
        'tp_size': 8,
        'kv_head_replica_num': 4,
    }
    assert events[5] == (
        'register', {
            'kv': target_rows['kv'],
            'index': target_rows['index'],
            'mtp.kv': mtp_rows['kv'],
            'mtp.index': mtp_rows['index'],
        },
        state_pools)
    assert agent.kv_connector is new_connector
    assert agent.cache_engine is cache_engine
    assert agent.state_cache_engine is state_cache_engine


def test_build_cache_engine_registers_state_pools_with_connector(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    row_mapping = {'kv': object()}
    state_pools = (object(), )
    cache_engine = SimpleNamespace(connector_kv_caches=row_mapping)
    state_cache_engine = SimpleNamespace(compact_plan=None, connector_state_cache_pools=state_pools)

    class _Connector:

        def register_kv_caches(self, caches, *, state_cache_pools=()):
            events.append((caches, state_cache_pools))

        def shutdown(self):
            pass

    connector = _Connector()
    agent = _bare_model_agent()
    agent.cache_config.states_shapes = [((1, ), torch.float32)]
    agent.kv_connector = None
    agent.spec_agent = SimpleNamespace(
        cache_engine=None,
        specdecode_config=None,
        build_cache_engine=lambda stream: None,
    )

    monkeypatch.setattr(agent_module, 'CacheEngine', lambda *args, **kwargs: cache_engine)
    monkeypatch.setattr(agent_module, 'StateCacheEngine', lambda *args, **kwargs: state_cache_engine)
    monkeypatch.setattr(agent_module, 'build_kv_connector', lambda *args, **kwargs: connector)
    monkeypatch.setattr(
        agent_module,
        'get_dist_manager',
        lambda: SimpleNamespace(current_context=lambda: SimpleNamespace(
            attn_tp_group=SimpleNamespace(rank=0))),
    )

    agent.build_cache_engine()

    assert events == [(row_mapping, state_pools)]


def test_build_cache_engine_propagates_registration_error(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []

    class _Connector:

        def register_kv_caches(self, caches, *, state_cache_pools=()):
            events.append('register')
            raise RuntimeError('registration failed')

        def shutdown(self):
            events.append('shutdown')

    agent = _bare_model_agent()
    agent.kv_connector = None
    agent.spec_agent = SimpleNamespace(
        cache_engine=None,
        specdecode_config=None,
        build_cache_engine=lambda stream: events.append('spec'),
    )
    cache_engine = SimpleNamespace(connector_kv_caches={'kv': object()})
    dist_ctx = SimpleNamespace(attn_tp_group=SimpleNamespace(rank=3))
    monkeypatch.setattr(agent_module, 'CacheEngine', lambda *args, **kwargs: cache_engine)
    monkeypatch.setattr(agent_module, 'StateCacheEngine',
                        lambda *args, **kwargs: SimpleNamespace(compact_plan=None, connector_state_cache_pools=()))
    monkeypatch.setattr(agent_module, 'build_kv_connector', lambda *args, **kwargs: _Connector())
    monkeypatch.setattr(agent_module, 'get_dist_manager',
                        lambda: SimpleNamespace(current_context=lambda: dist_ctx))

    with pytest.raises(RuntimeError, match='registration failed'):
        agent.build_cache_engine()

    assert events == ['spec', 'register']
    assert agent.kv_connector is not None


def test_build_cache_engine_propagates_later_initialization_error(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []

    agent = _bare_model_agent()
    agent.kv_connector = None

    def fail_spec_cache_build(stream):
        events.append('spec')
        raise RuntimeError('spec cache failed')

    agent.spec_agent = SimpleNamespace(
        cache_engine=None,
        specdecode_config=None,
        build_cache_engine=fail_spec_cache_build,
    )
    cache_engine = SimpleNamespace(connector_kv_caches={'kv': object()})
    dist_ctx = SimpleNamespace(attn_tp_group=SimpleNamespace(rank=3))
    monkeypatch.setattr(agent_module, 'CacheEngine', lambda *args, **kwargs: cache_engine)
    monkeypatch.setattr(agent_module, 'StateCacheEngine', lambda *args, **kwargs: SimpleNamespace(compact_plan=None))
    monkeypatch.setattr(agent_module, 'build_kv_connector',
                        lambda *args, **kwargs: events.append('factory'))
    monkeypatch.setattr(agent_module, 'get_dist_manager',
                        lambda: SimpleNamespace(current_context=lambda: dist_ctx))

    with pytest.raises(RuntimeError, match='spec cache failed'):
        agent.build_cache_engine()

    assert events == ['spec']
    assert agent.kv_connector is None


def test_build_cache_engine_rejects_mismatched_mtp_parallel_config(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    agent = _bare_model_agent()
    agent.kv_connector = None
    agent.dist_config = SimpleNamespace(attn_tp=8, dp=1, ep=1, world_size=8)
    agent.spec_agent = SimpleNamespace(
        cache_engine=None,
        specdecode_config=SimpleNamespace(
            cache_config=object(),
            dist_config=SimpleNamespace(attn_tp=1, dp=1, ep=1, world_size=1),
        ),
        build_cache_engine=lambda stream: events.append('spec'),
    )

    monkeypatch.setattr(agent_module, 'CacheEngine', lambda *args, **kwargs: events.append('cache'))
    monkeypatch.setattr(agent_module, 'StateCacheEngine', lambda *args, **kwargs: events.append('state'))
    monkeypatch.setattr(agent_module, 'build_kv_connector', lambda *args, **kwargs: events.append('factory'))

    with pytest.raises(ValueError, match='parallel configs to match'):
        agent.build_cache_engine()

    assert events == []
    assert agent.kv_connector is None


def test_sleep_shuts_down_connector_before_dropping_cache(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    agent = _bare_model_agent()
    cache_engine = object()

    class _Connector:

        def shutdown(self):
            assert agent.cache_engine is cache_engine
            events.append('shutdown')

    agent.kv_connector = _Connector()
    agent.cache_engine = cache_engine
    agent.state_cache_engine = object()
    agent.dist_config = SimpleNamespace(dp=1)
    agent.state = SimpleNamespace(is_sleeping=False,
                                  to_sleep=SimpleNamespace(clear=lambda: events.append('sleep-clear')))
    model = SimpleNamespace(to=lambda **kwargs: events.append('model-to'))
    spec_model = SimpleNamespace(to=lambda **kwargs: events.append('spec-model-to'))
    agent.patched_model = SimpleNamespace(get_model=lambda: model)
    agent.spec_agent = SimpleNamespace(get_model=lambda: spec_model, cache_engine=object())
    agent.reset_graph_runner = lambda: events.append('reset-graph')
    agent._drain_queues = lambda: events.append('drain')
    agent._release_completed_h2d_transfers = lambda: events.append('release-h2d')
    agent.reset_runtime_state = lambda: events.append('reset-runtime')
    agent._update_params_ipc_tensor = object()
    agent._update_params_ipc_event = object()
    monkeypatch.setattr(agent_module.torch.cuda, 'synchronize', lambda: events.append('cuda-sync'))
    monkeypatch.setattr(agent_module.torch.cuda, 'empty_cache', lambda: events.append('empty-cache'))

    asyncio.run(agent.sleep())

    assert events == [
        'drain',
        'cuda-sync',
        'release-h2d',
        'shutdown',
        'reset-graph',
        'model-to',
        'spec-model-to',
        'cuda-sync',
        'reset-runtime',
        'empty-cache',
        'sleep-clear',
    ]
    assert agent.kv_connector is None
    assert agent.cache_engine is None
    assert agent.state_cache_engine is None
    assert agent.spec_agent.cache_engine is None


def test_shutdown_kv_connector_drains_local_work_before_close(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    agent = _bare_model_agent()
    cache_engine = object()

    class _Connector:

        def shutdown(self):
            assert agent.cache_engine is cache_engine
            events.append('shutdown')

    agent.kv_connector = _Connector()
    agent.cache_engine = cache_engine
    agent._drain_queues = lambda: events.append('drain')
    agent._release_completed_h2d_transfers = lambda: events.append('release-h2d')
    monkeypatch.setattr(agent_module.torch.cuda, 'synchronize', lambda: events.append('cuda-sync'))

    agent.shutdown_kv_connector()

    assert events == ['drain', 'cuda-sync', 'release-h2d', 'shutdown']
    assert agent.kv_connector is None
    assert agent.cache_engine is cache_engine


def test_connector_output_aggregator_waits_for_every_tp_rank():
    aggregator = KVConnectorOutputAggregator(world_size=2)

    first = aggregator.aggregate([
        KVConnectorOutput(finished_receiving={11}, failed_receiving={11}),
        KVConnectorOutput(failed_receiving={12}),
    ])
    second = aggregator.aggregate([
        KVConnectorOutput(completed_save_ids={23}, failed_save_ids={23}),
        KVConnectorOutput(finished_receiving={11}),
    ])
    third = aggregator.aggregate([
        KVConnectorOutput(),
        KVConnectorOutput(completed_save_ids={23}),
    ])

    assert first.finished_receiving is None
    assert first.failed_receiving == {11, 12}
    assert second.finished_receiving == {11}
    assert second.completed_save_ids is None
    assert second.failed_save_ids == {23}
    assert third.completed_save_ids == {23}


def test_model_agent_connector_only_step_returns_progress_on_nonzero_tp_rank(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    metadata = KVConnectorMetadata()

    class _Connector:

        def bind_connector_metadata(self, value):
            events.append(('bind', value))

        def start_load_kv(self):
            events.append('start_load')

        def start_save_kv(self):
            events.append('start_save')

        def get_finished(self):
            events.append('finished')
            return KVConnectorOutput(
                finished_receiving={11},
                failed_receiving={11},
            )

        def clear_connector_metadata(self):
            events.append('clear')

    outputs = []
    agent = agent_module.BaseModelAgent.__new__(agent_module.BaseModelAgent)
    agent.rank = 1
    agent.kv_connector = _Connector()
    agent._push_output = outputs.append
    dist_context = SimpleNamespace(
        dist_config=SimpleNamespace(attn_tp=2, dp=1),
    )
    monkeypatch.setattr(
        agent_module,
        'get_dist_manager',
        lambda: SimpleNamespace(current_context=lambda: dist_context),
    )

    asyncio.run(agent._async_step(
        inputs=None,
        delta=None,
        kv_connector_metadata=metadata,
    ))

    assert events == [
        ('bind', metadata),
        'start_load',
        'start_save',
        'finished',
        'clear',
    ]
    assert len(outputs) == 1
    assert outputs[0].next_token_ids is None
    assert outputs[0].kv_connector_output == KVConnectorOutput(
        finished_receiving={11},
        failed_receiving={11},
    )


def test_model_agent_dp_connector_only_step_finishes_after_rendezvous(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    metadata = KVConnectorMetadata()
    dummy_inputs = SimpleNamespace(is_dummy=True)

    class _Connector:

        def bind_connector_metadata(self, value):
            events.append(('bind', value))

        def start_load_kv(self):
            events.append('start_load')

        def start_save_kv(self):
            events.append('start_save')

        def get_finished(self):
            events.append('finished')
            return KVConnectorOutput(finished_receiving={11})

        def clear_connector_metadata(self):
            events.append('clear')

    async def prepare_dp(inputs):
        events.append(('prepare_dp', inputs))
        return None, False

    outputs = []
    agent = agent_module.BaseModelAgent.__new__(agent_module.BaseModelAgent)
    agent.rank = 2
    agent.kv_connector = _Connector()
    agent._prepare_dp_v1 = prepare_dp
    agent._push_output = outputs.append
    dist_context = SimpleNamespace(
        dist_config=SimpleNamespace(attn_tp=1, dp=2),
    )
    monkeypatch.setattr(
        agent_module,
        'get_dist_manager',
        lambda: SimpleNamespace(current_context=lambda: dist_context),
    )

    asyncio.run(agent._async_step(
        inputs=dummy_inputs,
        delta=None,
        kv_connector_metadata=metadata,
    ))

    assert events == [
        ('bind', metadata),
        'start_load',
        ('prepare_dp', dummy_inputs),
        'start_save',
        'finished',
        'clear',
    ]
    assert len(outputs) == 1
    assert outputs[0].next_token_ids is None
    assert outputs[0].kv_connector_output == KVConnectorOutput(
        finished_receiving={11},
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('tp', [1, 2])
@pytest.mark.parametrize('compact', [False, True])
def test_decode_snapshot_mask_is_shared_across_tp_ranks(monkeypatch, tp, compact):
    from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    shared_mask = None
    copies = []
    group = object()

    def broadcast(mask, src, group):
        nonlocal shared_mask
        assert src == 0
        if shared_mask is None:
            shared_mask = mask.clone()
        else:
            mask.copy_(shared_mask)

    monkeypatch.setattr(agent_module.dist, 'broadcast', broadcast)
    monkeypatch.setattr(agent_module.dist, 'get_global_rank', lambda group, rank: rank)
    monkeypatch.setattr(agent_module, 'get_dist_manager', lambda: SimpleNamespace(
        current_context=lambda: SimpleNamespace(dist_config=SimpleNamespace(attn_tp=tp),
                                                 attn_tp_group=SimpleNamespace(gpu_group=group))))
    plan = torch.tensor([[0, 1], [256, 256], [1, 2], [21, 22]], device='cuda')
    cache_inputs = CacheCheckpointInputs(decode_save_plan=plan)
    inputs = SimpleNamespace(history_lengths=torch.tensor([254, 254], device='cuda'),
                             seq_length=torch.tensor([4, 4], device='cuda'),
                             is_decoding=True, ssm_state_meta=None)
    if compact:
        inputs.ssm_state_meta = SimpleNamespace(
            output_state_ids=torch.tensor([[9, 8, 7, 6], [5, 4, 3, 2]], device='cuda'))
    output = agent_module.BatchedOutputs(next_token_ids=torch.tensor([[7, 8, 9, 10], [7, 8, 9, 10]], device='cuda'),
                                         stopped=torch.tensor([True, False]),
                                         stop_pos=torch.tensor([0, -1], device='cuda'))
    for rank in range(tp):
        agent = agent_module.BaseModelAgent.__new__(agent_module.BaseModelAgent)
        agent.need_output = rank == 0
        agent.cache_config = SimpleNamespace(prefix_cache_token_lookahead=1)
        agent.state_cache_engine = SimpleNamespace(copy_decode_slots=lambda plan, mask: copies.append(mask.clone()))
        agent._save_decode_checkpoints(cache_inputs, inputs, output if rank == 0 else None)
        assert plan[2].tolist() == ([8, 4] if compact else [1, 2])
    assert [mask.tolist() for mask in copies] == [[False, True]] * tp
    assert output.to_cpu().to_numpy().to_tensor().decode_checkpoint_mask.tolist() == [False, True]


def test_model_agent_connector_save_hook_runs_between_forward_and_progress_poll():
    from lmdeploy.pytorch.engine.model_agent.kv_connector import (
        finish_kv_connector_step,
        start_kv_connector_save,
        start_kv_connector_step,
    )

    events = []
    metadata = KVConnectorMetadata()
    output = KVConnectorOutput(completed_save_ids={7})

    class _Connector:

        def bind_connector_metadata(self, value):
            events.append(('bind', value))

        def start_load_kv(self):
            events.append('load')

        def start_save_kv(self):
            events.append('save')

        def get_finished(self):
            events.append('poll')
            return output

        def clear_connector_metadata(self):
            events.append('clear')

    connector = _Connector()
    connector_step = start_kv_connector_step(connector, metadata)
    events.append('forward')
    start_kv_connector_save(connector, connector_step)
    assert finish_kv_connector_step(connector, connector_step) is output
    assert events == [
        ('bind', metadata),
        'load',
        'forward',
        'save',
        'poll',
        'clear',
    ]


@pytest.mark.parametrize('scoring', [False, True])
def test_model_agent_restores_both_caches_and_saves_after_speculative_forward(monkeypatch, scoring):
    from lmdeploy.pytorch.engine.cache_inputs import CacheCheckpointInputs
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    checkpoint = CacheCheckpointInputs(kv_restore_plan=torch.tensor([[3], [4]]),
                                       kv_save_plan=None if scoring else torch.tensor([[4], [5]]),
                                       decode_save_plan=None if scoring else torch.tensor([[0], [64], [1], [2]]))

    def copy_mtp_blocks(plan):
        if plan is checkpoint.kv_restore_plan:
            events.append('mtp_restore')
        else:
            assert plan is checkpoint.kv_save_plan
            events.append('mtp_save')

    class _Connector:

        def bind_connector_metadata(self, value):
            events.append(('bind', value))

        def start_load_kv(self):
            events.append('load')

        def start_save_kv(self):
            events.append('save')

        def get_finished(self):
            events.append('poll')
            return None

        def clear_connector_metadata(self):
            events.append('clear')

    async def target_forward(_inputs, return_logits, cache_inputs=None):
        events.append('target')
        return {'logits': torch.zeros((1, 1, 1))}

    async def speculative_forward(*args, **kwargs):
        events.append('mtp')
        inputs = args[0]
        return inputs, torch.ones(1, dtype=torch.long), None, None

    monkeypatch.setattr(
        agent_module,
        'get_dist_manager',
        lambda: SimpleNamespace(current_context=lambda: SimpleNamespace(
            dist_config=SimpleNamespace(attn_tp=1, dp=1))),
    )

    agent = agent_module.BaseModelAgent.__new__(agent_module.BaseModelAgent)
    agent.ssm_state_cursor = None
    agent.rank = 0
    agent.kv_connector = _Connector()
    agent.spec_agent = SimpleNamespace(is_enabled=lambda: True,
                                       cache_engine=SimpleNamespace(copy_logical_blocks=copy_mtp_blocks))
    agent.need_output = False
    agent.memdecode_agent = None
    agent.cache_engine = None
    agent.agent_strategy = SimpleNamespace(
        slice_outputs=lambda value, seq_length: value,
        slice_extra_inputs=lambda extra, inputs, output: extra,
    )
    agent._async_model_forward = target_forward
    agent._prepare_inputs_prefill = lambda value, delta: value
    agent._step_postprocess_without_output = speculative_forward
    agent._save_decode_checkpoints = lambda *args: events.append('decode_snapshot')
    agent._push_output = lambda output: pytest.fail('unexpected output')

    inputs = SimpleNamespace(
        ssm_state_meta=None,
        is_dummy=False,
        is_decoding=False,
        input_ids=torch.tensor([1]),
        seq_length=torch.tensor([1]),
        is_chunk=True,
        is_first_chunk=False,
        is_last_chunk=False,
        dp_meta=None,
        logits_indices=torch.tensor([0]) if scoring else None,
        seq_logit_length=torch.tensor([1]) if scoring else None,
    )
    sampling_inputs = SimpleNamespace(get_delta=lambda: None)

    metadata = KVConnectorMetadata()
    asyncio.run(agent._async_step(
        inputs=inputs,
        sampling_inputs=sampling_inputs,
        kv_connector_metadata=metadata,
        cache_inputs=checkpoint,
    ))

    assert events == [('bind', metadata), 'load', 'mtp_restore', 'target'] + (
        [] if scoring else ['mtp', 'decode_snapshot', 'mtp_save']) + ['save', 'poll', 'clear']


def test_release_shuts_down_connector_before_dropping_cache(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    agent = _bare_model_agent()
    cache_engine = object()

    class _Connector:

        def shutdown(self):
            assert agent.cache_engine is cache_engine
            events.append('shutdown')

    agent.kv_connector = _Connector()
    agent.cache_engine = cache_engine
    agent.state_cache_engine = object()
    agent.patched_model = object()
    agent.cache_config.compact_state_capacity = object()
    agent._background_task = None
    agent.compact_state_plan = object()
    agent.ssm_state_cursor = object()
    agent.step_inputs = SimpleNamespace(ssm_state_cursor=agent.ssm_state_cursor)
    agent.reset_graph_runner = lambda: events.append('reset-graph')
    monkeypatch.setattr(agent_module.torch.cuda, 'empty_cache', lambda: events.append('empty-cache'))

    agent.release()

    assert events[:2] == ['shutdown', 'reset-graph']
    assert agent.kv_connector is None
    assert agent.cache_engine is None
    assert agent.state_cache_engine is None
    assert agent.compact_state_plan is agent.ssm_state_cursor is agent.step_inputs is None


def test_wakeup_rebuilds_connector_with_kv_cache():
    events = []
    agent = _bare_model_agent()
    agent.state = SimpleNamespace(is_sleeping=True)
    agent.dist_config = SimpleNamespace(dp=1)
    agent.build_cache_engine = lambda: events.append('build-cache')
    agent.warmup = lambda: events.append('warmup')

    agent.wakeup(tags=['kv_cache'])

    assert events == ['build-cache', 'warmup']
    assert agent.state.is_sleeping is False
