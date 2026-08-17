# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.kv_connector.base import KVConnectorOutput


@pytest.fixture(scope='module')
def ray_components():
    ray = pytest.importorskip('ray')
    from lmdeploy.pytorch.engine.executor import ray_executor as ray_executor_module
    return ray, ray_executor_module, ray_executor_module.RayExecutor


def _make_cache_config(kv_transfer_config=None):
    return CacheConfig(
        max_batches=1,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=1,
        kv_transfer_config=kv_transfer_config,
    )


class _RayContext:

    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


def _make_executor(ray_executor_cls, cache_config, *, dp=1):
    executor = ray_executor_cls.__new__(ray_executor_cls)
    executor.cache_config = cache_config
    executor.dp = dp
    executor.workers = [object(), object()]
    executor.ray_ctx = _RayContext()
    executor.collective_rpc = Mock()
    return executor


def test_ray_executor_worker_release_timeout(ray_components):
    _, _, ray_executor_cls = ray_components
    disabled = KVTransferConfig()
    enabled = KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role='kv_both')

    assert ray_executor_cls._get_worker_release_timeout(None) == 5.0
    assert ray_executor_cls._get_worker_release_timeout(_make_cache_config()) == 5.0
    assert ray_executor_cls._get_worker_release_timeout(_make_cache_config(disabled)) == 5.0
    assert ray_executor_cls._get_worker_release_timeout(_make_cache_config(enabled)) == 45.0


@pytest.mark.parametrize(
    ('transfer_config', 'expected_timeout'),
    [
        (None, 5.0),
        (KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role='kv_both'), 45.0),
    ],
)
def test_ray_executor_release_uses_connector_aware_timeout(monkeypatch, ray_components, transfer_config,
                                                           expected_timeout):
    _, ray_executor_module, ray_executor_cls = ray_components
    executor = _make_executor(ray_executor_cls, _make_cache_config(transfer_config))
    kill = Mock()
    monkeypatch.setattr(ray_executor_module.ray, 'kill', kill)
    monkeypatch.setattr(ray_executor_module._envs, 'ray_timeline_enable', False)

    executor.release()

    executor.collective_rpc.assert_called_once_with('release', timeout=expected_timeout)
    kill.assert_not_called()
    assert executor.ray_ctx.shutdown_calls == 1


def test_ray_executor_release_kills_workers_after_timeout(monkeypatch, ray_components):
    ray, ray_executor_module, ray_executor_cls = ray_components
    transfer_config = KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role='kv_both')
    executor = _make_executor(ray_executor_cls, _make_cache_config(transfer_config))
    executor.collective_rpc.side_effect = ray.exceptions.GetTimeoutError
    kill = Mock()
    monkeypatch.setattr(ray_executor_module.ray, 'kill', kill)
    monkeypatch.setattr(ray_executor_module._envs, 'ray_timeline_enable', False)

    executor.release()

    executor.collective_rpc.assert_called_once_with('release', timeout=45.0)
    assert kill.call_args_list == [((worker, ), {}) for worker in executor.workers]
    assert executor.ray_ctx.shutdown_calls == 1


def test_ray_executor_release_dp_workers_without_rpc(monkeypatch, ray_components):
    _, ray_executor_module, ray_executor_cls = ray_components
    executor = _make_executor(ray_executor_cls, _make_cache_config(), dp=2)
    kill = Mock()
    monkeypatch.setattr(ray_executor_module.ray, 'kill', kill)
    monkeypatch.setattr(ray_executor_module._envs, 'ray_timeline_enable', False)

    executor.release()

    executor.collective_rpc.assert_not_called()
    assert kill.call_args_list == [((worker, ), {}) for worker in executor.workers]
    assert executor.ray_ctx.shutdown_calls == 1


def test_ray_executor_polls_all_tp_workers_with_sticky_ack(ray_components):
    _, _, ray_executor_cls = ray_components
    executor = _make_executor(ray_executor_cls, _make_cache_config())
    executor._kv_connector_acknowledged_sending = {5}
    executor._kv_connector_acknowledged_recving = {6}
    executor.collective_rpc_async = AsyncMock(return_value=[
        KVConnectorOutput(completed_save_ids={7, 8}, completed_load_ids={9}),
        KVConnectorOutput(completed_save_ids={8}, completed_load_ids={9}),
    ])
    metadata = object()

    completed = asyncio.run(executor.poll_kv_connector(metadata))

    assert completed.completed_save_ids == {8}
    assert completed.completed_load_ids == {9}
    executor.collective_rpc_async.assert_awaited_once_with(
        'poll_kv_connector',
        (metadata, {5}, {6}),
    )
    assert executor._kv_connector_poll_acknowledgements() == ({8}, {9})
