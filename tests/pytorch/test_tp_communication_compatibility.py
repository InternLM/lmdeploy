# Copyright (c) OpenMMLab. All rights reserved.
from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
from torch import nn

from tests.pytorch.distributed_test_utils import spawn_with_timeout


def _run_promoted_tp(rank, rendezvous):
    from lmdeploy.pytorch.backends.communicator import DeviceCommunicator
    from lmdeploy.pytorch.backends.cuda.comm.communicator import build_cuda_communicator
    from lmdeploy.pytorch.config import DistConfig, TPMode
    from lmdeploy.pytorch.distributed import DistContext, DistGroup, get_dist_manager
    from lmdeploy.pytorch.models.glm5_next import DeepseekV2MoE, Glm5NextMoE
    from lmdeploy.pytorch.nn.linear.base import LinearBase

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    cpu_group = dist.new_group(backend='gloo')
    try:
        for backend in ('nccl', 'auto'):
            config = DistConfig(tp=2, communication_backend=backend)
            communicator = build_cuda_communicator(cpu_group, dist.group.WORLD, config)
            communicator = communicator or DeviceCommunicator(dist.group.WORLD)
            group = DistGroup(rank=rank, gpu_group=dist.group.WORLD, communicator=communicator)
            context = DistContext(dist_config=config, attn_tp_group=group, moe_tp_group=group)
            local = torch.full((7, 128), rank + 1, dtype=torch.bfloat16, device='cuda')
            linear = LinearBase.__new__(LinearBase)
            nn.Module.__init__(linear)
            linear.tp, linear.tp_mode = 2, TPMode.DEFAULT
            linear.all_reduce, linear.tp_reduce_dtype = True, torch.float32
            linear.tp_group, linear.layer_type = dist.group.WORLD, 'attn'
            linear.lora_adapters = nn.ModuleDict()
            linear._forward_default = lambda *args: local.clone()
            moe = Glm5NextMoE.__new__(Glm5NextMoE)
            nn.Module.__init__(moe)
            moe._fp32_tp_reduce = True
            try:
                with (get_dist_manager().context(context),
                      patch.object(DeepseekV2MoE, 'forward', lambda *args, **kwargs: local.clone()),
                      patch.object(dist, 'all_reduce', wraps=dist.all_reduce) as collective):
                    for layer in (linear, moe):
                        local.fill_(rank + 1)
                        result = layer(local)
                        assert collective.call_args.args[0].dtype == torch.float32
                        assert result.dtype == torch.bfloat16
                        torch.testing.assert_close(result, torch.full_like(result, 3), rtol=0, atol=0)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            result = layer(local)
                        assert collective.call_args.args[0].dtype == torch.float32
                        for step in range(3):
                            local.fill_(rank + step)
                            graph.replay()
                            torch.testing.assert_close(result, torch.full_like(result, 1 + 2 * step), rtol=0, atol=0)
                        graph.reset()
            finally:
                communicator.close()
    finally:
        dist.destroy_process_group()


def _run_flashinfer_tp(rank, rendezvous):
    from lmdeploy.pytorch import envs
    from lmdeploy.pytorch.backends.cuda.comm.communicator import build_cuda_communicator
    from lmdeploy.pytorch.config import DistConfig

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    cpu_group = dist.new_group(backend='gloo')
    envs.enable_flashinfer_allreduce = True
    envs.enable_symm_mem_allreduce = False
    communicator = None
    try:
        communicator = build_cuda_communicator(cpu_group, dist.group.WORLD, DistConfig(tp=2))
        assert communicator is not None and communicator._flashinfer is not None
        provider = communicator._flashinfer
        local = torch.full((7, 128), rank + 1, dtype=torch.bfloat16, device='cuda')
        provider.all_reduce_ = Mock(wraps=provider.all_reduce_)
        communicator.all_reduce_(local)
        assert provider._workspace is not None, 'FlashInfer silently fell back before allocating its workspace'
        torch.testing.assert_close(local, torch.full_like(local, 3), rtol=0, atol=0)
        residual = torch.full_like(local, 0.5)
        weight = torch.ones(128, dtype=local.dtype, device=local.device)
        local.fill_(rank + 1)
        fused = communicator.try_fused_all_reduce_residual_rms_norm(local, residual, weight, 1e-6)
        assert fused is not None, 'FlashInfer silently fell back instead of testing the fused operation'
        torch.testing.assert_close(fused[0], torch.ones_like(local), rtol=0.01, atol=0.01)
        torch.testing.assert_close(fused[1], torch.full_like(local, 3.5), rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused = communicator.try_fused_all_reduce_residual_rms_norm(local, residual, weight, 1e-6)
        for step in range(1, 4):
            local.fill_(rank + step)
            graph.replay()
            torch.testing.assert_close(fused[0], torch.ones_like(local), rtol=0.01, atol=0.01)
            torch.testing.assert_close(fused[1], torch.full_like(local, 1.5 + 2 * step), rtol=0, atol=0)
        graph.reset()
    finally:
        if communicator is not None:
            communicator.close()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two CUDA GPUs')
def test_promoted_tp_eager_and_graph(tmp_path):
    spawn_with_timeout(_run_promoted_tp, args=(f'file://{tmp_path}/promoted',))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two CUDA GPUs')
def test_flashinfer_tp_eager_and_graph(tmp_path):
    comm = pytest.importorskip('flashinfer.comm')
    if not hasattr(comm, 'create_allreduce_fusion_workspace'):
        pytest.skip('requires FlashInfer unified communication API')
    # A fresh FlashInfer installation compiles trtllm_comm on first use.
    spawn_with_timeout(_run_flashinfer_tp, args=(f'file://{tmp_path}/flashinfer',), timeout=600)
