# Copyright (c) OpenMMLab. All rights reserved.
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import ray
import torch
import torch.distributed as dist

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.config import DistConfig
from lmdeploy.pytorch.engine.executor import ray_executor

LEGACY_FLAGS = ('enable_flashinfer_allreduce', 'enable_symm_mem_allreduce', 'enable_symm_mem_lmhead')


@pytest.mark.parametrize('backend,flag,expected', [
    ('nccl', None, False), ('auto', None, True),
    *[('nccl', flag, True) for flag in LEGACY_FLAGS],
])
@pytest.mark.parametrize('tp,dp,ep,microbatch', [(2, 1, 1, False), (1, 1, 1, False),
                                              (2, 2, 1, False), (2, 1, 2, False), (2, 1, 1, True)])
def test_ray_peer_visibility_follows_communication_policy(monkeypatch, backend, flag, expected,
                                                         tp, dp, ep, microbatch):
    for name in LEGACY_FLAGS:
        monkeypatch.setattr(envs, name, name == flag)
    config = SimpleNamespace(attn_tp=tp, dcp=1, dp=dp, ep=ep, enable_microbatch=microbatch,
                             communication_backend=backend)
    enabled = expected and tp > 1 and dp == ep == 1 and not microbatch
    assert ray_executor._needs_symm_mem_device_setup(config) is enabled


class _GatherWorker:
    """Exercise production Ray binding and LM-head gather without model
    weights."""

    set_device = ray_executor.RayWorkerWrapper.set_device
    set_assigned_cuda_device = ray_executor.RayWorkerWrapper.set_assigned_cuda_device
    get_node_ip = ray_executor.RayWorkerWrapper.get_node_ip

    def __init__(self):
        self.node_ip = ray.util.get_node_ip_address()

    def check_gather(self, rank, rendezvous, config):
        from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
        from lmdeploy.pytorch.distributed import DistContext, get_dist_manager
        from lmdeploy.pytorch.nn.embedding import ParallelLMHead

        torch.set_num_threads(1)
        dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=config.world_size,
                                timeout=timedelta(seconds=60))
        context = None
        head = None
        try:
            context = DistContext.build(rank, config, communicator_builder=CudaOpsBackend.build_communicator)
            with get_dist_manager().context(context):
                head = ParallelLMHead(config.tp * 128, 16, dtype=torch.bfloat16, device='cuda')
                workspace = head._logits_gather_workspace
                assert workspace is not None and workspace._state is not None, 'optimized workspace unavailable'
                local = torch.full((8, 128), rank, dtype=torch.bfloat16, device='cuda')
                expected = torch.cat([torch.full_like(local, i) for i in range(config.tp)], dim=-1)
                with patch.object(dist, 'all_gather_into_tensor', wraps=dist.all_gather_into_tensor) as native:
                    for _ in range(2):
                        result = head.all_gather_logits(local)
                        torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    assert native.call_count == 0, 'legacy LM-head opt-in silently fell back to NCCL'
            return dict(rank=rank, device=torch.cuda.current_device(), workspace_ready=True, native_gathers=0)
        finally:
            if head is not None and head._logits_gather_workspace is not None:
                head._logits_gather_workspace.close()
            if context is not None:
                context.close()
            dist.destroy_process_group()


@pytest.mark.parametrize('tp', [2, 8])
@pytest.mark.parametrize('dcp', [1, 2])
def test_ray_legacy_lmhead_gather(tmp_path, monkeypatch, tp, dcp):
    # Ray workers do not inherit pytest's temporary sys.path entries.
    from tests.pytorch.engine.test_ray_communication import _GatherWorker as worker_class

    if torch.cuda.device_count() < tp:
        pytest.skip(f'requires {tp} idle CUDA GPUs')
    if any(torch.cuda.get_device_capability(i)[0] < 9 for i in range(tp)):
        pytest.skip('symmetric-memory gather requires SM90 or newer')
    if ray.is_initialized():
        pytest.skip('requires an isolated Ray runtime')
    for name in LEGACY_FLAGS:
        monkeypatch.setattr(envs, name, name == 'enable_symm_mem_lmhead')
    monkeypatch.setenv('LMDEPLOY_ENABLE_SYMM_MEM_LMHEAD', '1')
    monkeypatch.setenv('LMDEPLOY_ENABLE_SYMM_MEM_ALLREDUCE', '0')
    monkeypatch.setenv('LMDEPLOY_ENABLE_FLASHINFER_ALLREDUCE', '0')
    monkeypatch.setenv('LMDEPLOY_SYMM_MEM_LMHEAD_MAX_MB', '8')
    monkeypatch.setattr(ray_executor, 'RayWorkerWrapper', worker_class)
    monkeypatch.setattr(envs, 'ray_external_pg_bundles', None)
    monkeypatch.setattr(envs, 'ray_nsys_enable', False)
    config = DistConfig(tp=tp, dcp=dcp, communication_backend='nccl')
    executor = object.__new__(ray_executor.RayExecutor)
    executor.dist_config = config
    executor.ray_ctx = SimpleNamespace(owned_pg=True)
    executor._needs_symm_mem_device_setup = ray_executor._needs_symm_mem_device_setup(config)
    assert executor._needs_symm_mem_device_setup
    workers = []
    pg = None
    try:
        ray.init(address='local', num_gpus=tp, num_cpus=tp, include_dashboard=False, log_to_driver=False)
        pg = ray.util.placement_group([{'GPU': 1}] * tp, strategy='STRICT_PACK')
        ray.get(pg.ready(), timeout=60)
        workers = executor._init_workers_ray(pg, {})
        bindings = ray.get([worker.set_assigned_cuda_device.remote() for worker in workers], timeout=60)
        assert len({x['current_device'] for x in bindings}) == tp
        results = ray.get([worker.check_gather.remote(rank, f'file://{tmp_path}/ray_gather', config)
                           for rank, worker in enumerate(workers)], timeout=180)
        assert all(result['workspace_ready'] and result['native_gathers'] == 0 for result in results)
        print(f'TP{tp}/DCP{dcp}, nccl + legacy LM-head: {results}', flush=True)
    finally:
        for worker in workers:
            ray.kill(worker)
        if pg is not None:
            ray.util.remove_placement_group(pg)
        ray.shutdown()
