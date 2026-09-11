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
    agent.cache_config = _enabled_config()
    agent.model_config = SimpleNamespace(num_replicate_key_value_heads=4)
    agent.rank = 7
    agent.cache_stream = object()
    agent.dist_config = SimpleNamespace(attn_tp=8)
    agent.memdecode_agent = None
    return agent


def test_build_cache_engine_replaces_connector_and_registers_row_mapping(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []
    target_rows = {'kv': object(), 'index': object()}
    mtp_rows = {'kv': object(), 'index': object()}
    cache_engine = SimpleNamespace(connector_kv_caches=target_rows)
    mtp_cache_engine = SimpleNamespace(connector_kv_caches=mtp_rows)
    state_cache_engine = object()

    class _OldConnector:

        def shutdown(self):
            events.append('old-shutdown')

    class _NewConnector:

        def register_kv_caches(self, caches):
            events.append(('register', caches))

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
    assert cache_call[2]['world_size'] == 8
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
        })
    assert agent.kv_connector is new_connector
    assert agent.cache_engine is cache_engine
    assert agent.state_cache_engine is state_cache_engine


def test_build_cache_engine_propagates_registration_error(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []

    class _Connector:

        def register_kv_caches(self, caches):
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
    monkeypatch.setattr(agent_module, 'StateCacheEngine', lambda *args, **kwargs: object())
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
    monkeypatch.setattr(agent_module, 'StateCacheEngine', lambda *args, **kwargs: object())
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
        KVConnectorOutput(finished_receiving={11}, invalid_block_ids={3}),
        KVConnectorOutput(invalid_block_ids={4}),
    ])
    second = aggregator.aggregate([
        KVConnectorOutput(completed_save_ids={23}),
        KVConnectorOutput(finished_receiving={11}),
    ])
    third = aggregator.aggregate([
        KVConnectorOutput(),
        KVConnectorOutput(completed_save_ids={23}),
    ])

    assert first.finished_receiving is None
    assert first.invalid_block_ids == {3, 4}
    assert second.finished_receiving == {11}
    assert second.completed_save_ids is None
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

        def get_finished(self):
            events.append('finished')
            return KVConnectorOutput(
                finished_receiving={11},
                invalid_block_ids={3},
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
        'finished',
        'clear',
    ]
    assert len(outputs) == 1
    assert outputs[0].next_token_ids is None
    assert outputs[0].kv_connector_output == KVConnectorOutput(
        finished_receiving={11},
        invalid_block_ids={3},
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
        'finished',
        'clear',
    ]
    assert len(outputs) == 1
    assert outputs[0].next_token_ids is None
    assert outputs[0].kv_connector_output == KVConnectorOutput(
        finished_receiving={11},
    )


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


def test_model_agent_defers_connector_save_until_speculative_forward(monkeypatch):
    from lmdeploy.pytorch.engine.model_agent import agent as agent_module

    events = []

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
    agent.rank = 0
    agent.kv_connector = _Connector()
    agent.spec_agent = SimpleNamespace(is_enabled=lambda: True)
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
    agent._push_output = lambda output: pytest.fail('unexpected output')

    inputs = SimpleNamespace(
        is_dummy=False,
        is_decoding=False,
        input_ids=torch.tensor([1]),
        seq_length=torch.tensor([1]),
        is_chunk=True,
        is_first_chunk=False,
        is_last_chunk=False,
        dp_meta=None,
        logits_indices=None,
        seq_logit_length=None,
    )
    sampling_inputs = SimpleNamespace(get_delta=lambda: None)

    metadata = KVConnectorMetadata()
    asyncio.run(agent._async_step(
        inputs=inputs,
        sampling_inputs=sampling_inputs,
        kv_connector_metadata=metadata,
    ))

    assert events == [
        ('bind', metadata),
        'load',
        'target',
        'mtp',
        'save',
        'poll',
        'clear',
    ]


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
    agent.reset_graph_runner = lambda: events.append('reset-graph')
    monkeypatch.setattr(agent_module.torch.cuda, 'empty_cache', lambda: events.append('empty-cache'))

    agent.release()

    assert events[:2] == ['shutdown', 'reset-graph']
    assert agent.kv_connector is None
    assert agent.cache_engine is None
    assert agent.state_cache_engine is None


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
