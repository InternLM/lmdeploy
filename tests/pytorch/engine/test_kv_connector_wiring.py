# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import pytest

from lmdeploy.messages import KVTransferConfig, PytorchEngineConfig
from lmdeploy.pytorch.config import CacheConfig, DistConfig
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder
from lmdeploy.pytorch.engine.engine import Engine
from lmdeploy.pytorch.kv_connector import prepare_kv_connector_config
from lmdeploy.pytorch.paging.scheduler import Scheduler


def _make_cache_config(kv_transfer_config=None):
    return CacheConfig(
        max_batches=1,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=1,
        kv_transfer_config=kv_transfer_config,
    )


def _make_mooncake_cache_config():
    return _make_cache_config(
        KVTransferConfig(
            kv_connector='MooncakeStoreConnector',
            kv_role='kv_both',
        ))


def test_prepare_kv_connector_config_generates_one_short_unique_path():
    first = _make_mooncake_cache_config()
    second = _make_mooncake_cache_config()

    prepare_kv_connector_config(first)
    first_path = first.kv_transfer_config.kv_connector_extra_config['lookup_rpc_path']
    prepare_kv_connector_config(first)
    prepare_kv_connector_config(second)
    second_path = second.kv_transfer_config.kv_connector_extra_config['lookup_rpc_path']

    assert first.kv_transfer_config.kv_connector_extra_config['lookup_rpc_path'] == first_path
    assert first_path.startswith('ipc:///tmp/lmd-mc-lookup-')
    assert first_path.endswith('.sock')
    assert len(first_path.removeprefix('ipc://').encode()) < 100
    assert second_path != first_path


def test_prepare_mooncake_namespace_and_async_constraints():
    cache_config = _make_mooncake_cache_config()

    prepare_kv_connector_config(
        cache_config,
        model_path='/models/tenant/glm-5.2',
        dist_config=DistConfig(tp=8),
    )

    extra_config = cache_config.kv_transfer_config.kv_connector_extra_config
    assert extra_config['model_name'] == 'glm-5.2'
    assert extra_config['lookup_async'] is True

    invalid = _make_mooncake_cache_config()
    invalid.kv_transfer_config.kv_connector_extra_config['lookup_async'] = False
    with pytest.raises(ValueError, match='lookup_async=true'):
        prepare_kv_connector_config(invalid)

    with pytest.raises(ValueError, match='tensor parallelism only'):
        prepare_kv_connector_config(
            _make_mooncake_cache_config(),
            dist_config=DistConfig(tp=2, dp=2),
        )


def test_prepare_mooncake_rejects_invalid_explicit_model_name():
    cache_config = _make_mooncake_cache_config()
    cache_config.kv_transfer_config.kv_connector_extra_config['model_name'] = None

    with pytest.raises(ValueError, match='model_name'):
        prepare_kv_connector_config(
            cache_config,
            model_path='/models/test-model',
        )


def test_runtime_lookup_path_does_not_mutate_reused_engine_config():
    engine_config = PytorchEngineConfig(
        max_batch_size=1,
        kv_transfer_config=KVTransferConfig(
            kv_connector='MooncakeStoreConnector',
            kv_role='kv_both',
        ),
    )
    first = ConfigBuilder.build_cache_config(engine_config)
    second = ConfigBuilder.build_cache_config(engine_config)

    prepare_kv_connector_config(first)
    prepare_kv_connector_config(second)

    first_path = first.kv_transfer_config.kv_connector_extra_config['lookup_rpc_path']
    second_path = second.kv_transfer_config.kv_connector_extra_config['lookup_rpc_path']
    assert first_path != second_path
    assert engine_config.kv_transfer_config.kv_connector_extra_config == {}


@pytest.mark.parametrize(
    'transfer_config',
    [
        None,
        KVTransferConfig(kv_connector_extra_config={'sentinel': 'unchanged'}),
    ],
)
def test_prepare_kv_connector_config_does_not_change_disabled_config(transfer_config):
    cache_config = _make_cache_config(transfer_config)
    original_extra_config = None if transfer_config is None else transfer_config.kv_connector_extra_config.copy()

    prepare_kv_connector_config(cache_config)

    if transfer_config is None:
        assert cache_config.kv_transfer_config is None
    else:
        assert transfer_config.kv_connector_extra_config == original_extra_config


def test_scheduler_shutdown_releases_injected_connector_once():
    connector = Mock()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_connector = connector
    scheduler.kv_load_coordinator = Mock()
    scheduler.kv_save_coordinator = Mock()

    scheduler.shutdown()
    scheduler.shutdown()

    connector.shutdown.assert_called_once_with()
    assert scheduler.kv_load_coordinator.clear.call_count == 2
    assert scheduler.kv_save_coordinator.clear.call_count == 2
    assert scheduler.kv_connector is None
    assert not scheduler._external_lookup_enabled


def test_engine_loop_finally_shuts_down_scheduler_before_executor():
    calls = []
    engine = Engine.__new__(Engine)
    engine.migration_event = object()
    engine.scheduler = Mock()
    engine.executor = Mock()
    engine.scheduler.shutdown.side_effect = lambda: calls.append('scheduler')
    engine.executor.release.side_effect = lambda: calls.append('executor')

    engine._loop_finally()

    assert calls == ['scheduler', 'executor']
    assert engine.migration_event is None


def test_engine_loop_finally_propagates_scheduler_shutdown_error():
    engine = Engine.__new__(Engine)
    engine.migration_event = object()
    engine.scheduler = Mock()
    engine.executor = Mock()
    engine.scheduler.shutdown.side_effect = RuntimeError('scheduler shutdown failed')

    with pytest.raises(RuntimeError, match='scheduler shutdown failed'):
        engine._loop_finally()

    engine.executor.release.assert_not_called()
