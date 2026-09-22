# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from collections import deque
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.messages import KVTransferConfig
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.engine.model_agent.agent import BatchedOutputs
from lmdeploy.pytorch.kv_connector import KVConnectorOutput, KVConnectorOutputAggregator


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


class _AsyncRemoteMethod:

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def remote(self):
        self.calls += 1

        async def _result():
            return self.result

        return _result()


class _OutputWorker:

    def __init__(self, output):
        self.get_outputs = _AsyncRemoteMethod(output)


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


@pytest.mark.parametrize('times_out', [False, True])
def test_ray_executor_release_dp_shuts_down_connector_before_kill(
    monkeypatch,
    ray_components,
    times_out,
):
    ray, ray_executor_module, ray_executor_cls = ray_components
    transfer_config = KVTransferConfig(
        kv_connector='MooncakeStoreConnector',
        kv_role='kv_both',
    )
    executor = _make_executor(
        ray_executor_cls,
        _make_cache_config(transfer_config),
        dp=2,
    )
    events = []

    def shutdown_connector(*args, **kwargs):
        events.append('shutdown')
        if times_out:
            raise ray.exceptions.GetTimeoutError

    executor.collective_rpc.side_effect = shutdown_connector
    kill = Mock(side_effect=lambda worker: events.append(('kill', worker)))
    monkeypatch.setattr(ray_executor_module.ray, 'kill', kill)
    monkeypatch.setattr(ray_executor_module._envs, 'ray_timeline_enable', False)

    executor.release()

    executor.collective_rpc.assert_called_once_with(
        'shutdown_kv_connector',
        timeout=45.0,
    )
    assert events == ['shutdown', *(('kill', worker) for worker in executor.workers)]
    assert executor.ray_ctx.shutdown_calls == 1


def test_ray_executor_aggregates_connector_output_from_every_tp_rank(ray_components):
    _, _, ray_executor_cls = ray_components
    rank_zero = BatchedOutputs(
        next_token_ids=torch.tensor([5]),
        stopped=torch.tensor([False]),
        kv_connector_output=KVConnectorOutput(finished_receiving={11}),
    )
    rank_one = BatchedOutputs.connector_only(
        KVConnectorOutput(
            finished_receiving={11},
            invalid_block_ids={3},
        ))
    executor = ray_executor_cls.__new__(ray_executor_cls)
    executor.workers = [_OutputWorker(rank_zero), _OutputWorker(rank_one)]
    executor._connector_steps = deque([True])
    executor._kv_output_aggregator = KVConnectorOutputAggregator(world_size=2)

    output = asyncio.run(executor.get_output_async())

    assert output.next_token_ids.tolist() == [5]
    assert output.kv_connector_output == KVConnectorOutput(
        finished_receiving={11},
        invalid_block_ids={3},
    )
    assert [worker.get_outputs.calls for worker in executor.workers] == [1, 1]


def test_ray_executor_sleep_clears_dropped_connector_steps(ray_components):
    _, _, ray_executor_cls = ray_components
    executor = ray_executor_cls.__new__(ray_executor_cls)
    executor._connector_steps = deque([True])
    executor._kv_output_aggregator = KVConnectorOutputAggregator(world_size=2)
    executor._kv_output_aggregator.aggregate([
        KVConnectorOutput(finished_receiving={11}),
        KVConnectorOutput(),
    ])

    async def sleep_workers(method, args):
        assert (method, args) == ('sleep', (1, ))

    executor.collective_rpc_async = sleep_workers

    asyncio.run(executor.sleep())

    assert not executor._connector_steps
    output = executor._kv_output_aggregator.aggregate([
        KVConnectorOutput(),
        KVConnectorOutput(finished_receiving={11}),
    ])
    assert output.finished_receiving is None
