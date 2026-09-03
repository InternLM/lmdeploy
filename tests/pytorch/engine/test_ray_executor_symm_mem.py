# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(scope='module')
def ray_executor_module():
    pytest.importorskip('ray')
    from lmdeploy.pytorch.engine.executor import ray_executor
    return ray_executor


def _dist_config(*, attn_tp=8):
    return SimpleNamespace(dp=1, ep=1, attn_tp=attn_tp, enable_microbatch=False)


@pytest.mark.parametrize(
    ('enable_allreduce', 'enable_lmhead', 'attn_tp', 'expected'),
    [
        (False, False, 8, False),
        (True, False, 8, True),
        (False, True, 8, True),
        (False, True, 1, False),
    ],
)
def test_needs_symm_mem_device_setup(monkeypatch, ray_executor_module, enable_allreduce, enable_lmhead, attn_tp,
                                     expected):
    monkeypatch.setattr(ray_executor_module._envs, 'enable_symm_mem_allreduce', enable_allreduce)
    monkeypatch.setattr(ray_executor_module._envs, 'enable_symm_mem_lmhead', enable_lmhead)

    actual = ray_executor_module._needs_symm_mem_device_setup(_dist_config(attn_tp=attn_tp))

    assert actual is expected


class _RemoteMethod:

    def __init__(self, result):
        self.remote = Mock(return_value=result)


@pytest.mark.parametrize('required', [False, True])
def test_ray_worker_runtime_env_tracks_device_setup(monkeypatch, ray_executor_module, required):
    executor = ray_executor_module.RayExecutor.__new__(ray_executor_module.RayExecutor)
    executor._needs_symm_mem_device_setup = required
    executor.dist_config = _dist_config(attn_tp=2)

    placement_group = SimpleNamespace(bundle_specs=[{'GPU': 1}, {'GPU': 1}])
    remote_options = []

    class _RemoteActor:

        def remote(self, **kwargs):
            return object()

    def fake_remote(**options):
        remote_options.append(options)
        return lambda actor_cls: _RemoteActor()

    monkeypatch.setattr(ray_executor_module, 'get_device_str', lambda: 'GPU')
    monkeypatch.setattr(ray_executor_module, '_update_runtime_envs', lambda _: {'env_vars': {}})
    monkeypatch.setattr(ray_executor_module, 'PlacementGroupSchedulingStrategy', lambda **kwargs: kwargs)
    monkeypatch.setattr(ray_executor_module.ray, 'remote', fake_remote)
    monkeypatch.setattr(ray_executor_module._envs, 'ray_external_pg_bundles', [])
    monkeypatch.setattr(ray_executor_module._envs, 'ray_nsys_enable', False)

    workers = executor._init_workers_ray(placement_group, worker_kwargs={})

    assert len(workers) == 2
    assert len(remote_options) == 2
    for options in remote_options:
        actual = options['runtime_env']['env_vars'].get('RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES')
        assert actual == ('1' if required else None)


@pytest.mark.parametrize('required', [False, True])
def test_ray_device_binding_tracks_device_setup(monkeypatch, ray_executor_module, required):
    executor = ray_executor_module.RayExecutor.__new__(ray_executor_module.RayExecutor)
    executor._needs_symm_mem_device_setup = required
    reports = [{'node_ip': '127.0.0.1', 'current_device': rank} for rank in range(2)]
    workers = [SimpleNamespace(set_assigned_cuda_device=_RemoteMethod(report)) for report in reports]
    executor.workers = workers
    executor._sort_workers = Mock(return_value=workers)
    ray_get = Mock(return_value=reports)
    monkeypatch.setattr(ray_executor_module, '_get_master_addr', lambda: '127.0.0.1')
    monkeypatch.setattr(ray_executor_module.ray, 'get', ray_get)

    executor._init_distributed_environment_by_device('cuda')

    for worker in workers:
        assert worker.set_assigned_cuda_device.remote.call_count == int(required)
    assert ray_get.call_count == int(required)


def test_ray_device_binding_rejects_duplicate_local_ordinals(monkeypatch, ray_executor_module):
    executor = ray_executor_module.RayExecutor.__new__(ray_executor_module.RayExecutor)
    executor._needs_symm_mem_device_setup = True
    reports = [
        {'node_ip': '127.0.0.1', 'current_device': 0},
        {'node_ip': '127.0.0.1', 'current_device': 0},
    ]
    workers = [SimpleNamespace(set_assigned_cuda_device=_RemoteMethod(report)) for report in reports]
    executor.workers = workers
    executor._sort_workers = Mock(return_value=workers)
    monkeypatch.setattr(ray_executor_module, '_get_master_addr', lambda: '127.0.0.1')
    monkeypatch.setattr(ray_executor_module.ray, 'get', Mock(return_value=reports))

    with pytest.raises(RuntimeError, match='must bind unique CUDA devices'):
        executor._init_distributed_environment_by_device('cuda')


def test_lmhead_only_does_not_enable_symm_mem_allreduce(monkeypatch, ray_executor_module):
    from lmdeploy.pytorch.backends.cuda.comm import communicator

    monkeypatch.setattr(communicator._envs, 'enable_flashinfer_allreduce', False)
    monkeypatch.setattr(communicator._envs, 'enable_symm_mem_allreduce', False)
    monkeypatch.setattr(communicator._envs, 'enable_symm_mem_lmhead', True)

    actual = communicator.build_cuda_communicator(
        cpu_group=object(),
        device_group=object(),
        dist_config=_dist_config(attn_tp=8),
    )

    assert actual is None
