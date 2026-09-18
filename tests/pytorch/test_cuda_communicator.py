# Copyright (c) OpenMMLab. All rights reserved.
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends import communicator as base_communicator_module
from lmdeploy.pytorch.backends.cuda.comm import communicator as communicator_module
from lmdeploy.pytorch.backends.dlinfer.op_backend import DlinferOpsBackend
from lmdeploy.pytorch.nn import norm as norm_module


def _run_dcp_query_gather(rank, rendezvous, enabled):
    from datetime import timedelta

    from torch import distributed as dist

    from lmdeploy.pytorch.backends.cuda.attention.cp import gather_dcp_query
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.config import DistConfig
    from lmdeploy.pytorch.distributed import DistContext, get_dist_manager

    torch.cuda.set_device(rank)
    communicator_module._envs.enable_symm_mem_dcp = enabled[rank]
    dist.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    ctx = DistContext.build(rank, DistConfig(tp=2, dcp=2),
                            communicator_builder=CudaOpsBackend.build_communicator)
    try:
        with get_dist_manager().context(ctx):
            comm = ctx.dcp_group.communicator
            comm.prepare_query_gather(32, 576)
            if all(enabled):
                assert comm._query_gatherer._state is not None
            else:
                assert comm._query_gatherer._state is None
            assert comm._all_reduce is None
            query = torch.empty(384, 32, 576, device='cuda', dtype=torch.bfloat16)
            # Ineligible inputs still return correct results through NCCL.
            for fallback in (query[:1], query[..., ::2], query.float(),
                             torch.empty(1024, 32, 576, device='cuda', dtype=query.dtype)):
                fallback.fill_(rank)
                output = comm.gather_query(fallback)
                expected = torch.arange(2, device='cuda', dtype=fallback.dtype)
                expected = expected.repeat_interleave(32)[None, :, None].expand_as(output)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
            rows = (2, 16, 96, 384)
            for count in rows:
                gather_dcp_query(query[:count], dcp_world_size=2)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            outputs = []
            with torch.cuda.graph(graph):
                for step in range(16):
                    query.fill_(rank + step * 8)
                    gathered = gather_dcp_query(query[:rows[step % len(rows)]], dcp_world_size=2)
                    outputs.append(gathered.clone())
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            graph.reset()
            for step, output in enumerate(outputs):
                expected = (torch.arange(2, device='cuda', dtype=query.dtype) + step * 8)
                expected = expected.repeat_interleave(32)[None, :, None].expand_as(output)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        ctx.close()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two CUDA GPUs')
@pytest.mark.parametrize('enabled', [(False, False), (True, True), (True, False)],
                         ids=['nccl', 'symm_mem', 'mixed_flags'])
def test_dcp_query_gather_graph_reuses_arena(tmp_path, enabled):
    if all(enabled):
        from torch.distributed._symmetric_memory import DeviceType, _SymmetricMemory
        if any(torch.cuda.get_device_capability(i)[0] < 9
               or not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, i) for i in range(2)):
            pytest.skip('requires SM90 or newer with multicast support')
        if not torch.cuda.can_device_access_peer(0, 1):
            pytest.skip('requires peer access')
    torch.multiprocessing.spawn(_run_dcp_query_gather,
                                args=((tmp_path / 'rendezvous').as_uri(), enabled), nprocs=2)


class _Collective:

    def __init__(self, *, result=None, handled=False):
        self._result = result
        self._handled = handled

    def is_available(self):
        return True

    def fused_all_reduce_residual_rms_norm(self, **kwargs):
        return self._result

    def all_reduce_(self, input):
        return self._handled


def _build_communicator(monkeypatch,
                        *,
                        backend,
                        fused_result=None,
                        handled=False):
    collective = _Collective(result=fused_result, handled=handled)
    use_flashinfer = backend == 'flashinfer'
    monkeypatch.setattr(communicator_module._envs, 'enable_flashinfer_allreduce', use_flashinfer)
    monkeypatch.setattr(communicator_module._envs, 'enable_symm_mem_allreduce', not use_flashinfer)
    flashinfer_cls = Mock(return_value=collective)
    symm_mem_cls = Mock(return_value=collective)
    monkeypatch.setattr(communicator_module, 'FlashInferAllReduce', flashinfer_cls)
    monkeypatch.setattr(communicator_module, 'SymmetricMemoryAllReduce', symm_mem_cls)
    communicator = communicator_module.CudaCommunicator(cpu_group='cpu', device_group='gpu',
                                                         all_reduce_backend=backend)
    enabled_cls, disabled_cls = ((flashinfer_cls, symm_mem_cls)
                                 if use_flashinfer else (symm_mem_cls, flashinfer_cls))
    enabled_cls.assert_called_once_with('cpu')
    disabled_cls.assert_not_called()
    return communicator, collective


def _build_norm(monkeypatch, *, fused=False):
    impl = Mock()
    backend = Mock()
    backend.build_op.return_value = impl
    group = Mock()
    group.supports_optimized_all_reduce.return_value = True
    group.supports_fused_all_reduce_residual_rms_norm.return_value = fused
    monkeypatch.setattr(norm_module, 'get_backend', lambda: backend)
    monkeypatch.setattr(norm_module, 'get_dist_group', lambda layer_type: group)
    norm = norm_module.RMSNorm(4, dtype=torch.float32, device='cpu', all_reduce_group='attn')
    return norm, impl, group


def test_cuda_communicator_dispatch(monkeypatch):
    fused_result = object()
    communicator, flashinfer = _build_communicator(
        monkeypatch, backend='flashinfer', fused_result=fused_result)
    nccl_all_reduce = Mock()
    monkeypatch.setattr(communicator_module.dist, 'all_reduce', nccl_all_reduce)

    assert communicator.supports_optimized_all_reduce()
    assert communicator.supports_fused_all_reduce_residual_rms_norm()
    assert communicator.try_fused_all_reduce_residual_rms_norm(
        input=torch.ones(1), residual=torch.ones(1), weight=torch.ones(1), eps=1e-6) is fused_result

    input = torch.ones(1)
    communicator.all_reduce_(input)
    nccl_all_reduce.assert_called_once_with(input, group='gpu')

    flashinfer._handled = True
    communicator.all_reduce_(input)
    nccl_all_reduce.assert_called_once()

    communicator, symm_mem = _build_communicator(monkeypatch, backend='symm_mem', handled=True)
    communicator.all_reduce_(input)
    assert communicator.supports_optimized_all_reduce()
    assert not communicator.supports_fused_all_reduce_residual_rms_norm()
    nccl_all_reduce.assert_called_once()


def test_cuda_communicator_rejects_multiple_backends(monkeypatch):
    monkeypatch.setattr(communicator_module._envs, 'enable_flashinfer_allreduce', True)
    monkeypatch.setattr(communicator_module._envs, 'enable_symm_mem_allreduce', True)
    with pytest.raises(ValueError, match='cannot be enabled together'):
        config = SimpleNamespace(dp=1, ep=1, attn_tp=2, dcp=1, enable_microbatch=False)
        communicator_module.build_cuda_communicator('cpu', 'gpu', config)


def test_rms_norm_fuses_or_falls_back(monkeypatch):
    norm, impl, group = _build_norm(monkeypatch, fused=True)
    fused_output = (torch.full((1, 4), 2.0), torch.full((1, 4), 3.0))
    group.try_fused_all_reduce_residual_rms_norm.return_value = fused_output
    input = torch.ones((1, 4), dtype=torch.bfloat16)
    residual = torch.ones_like(input)

    assert norm(input, residual) is fused_output
    group.try_fused_all_reduce_residual_rms_norm.assert_called_once_with(
        input=input,
        residual=residual,
        weight=norm.weight,
        eps=norm.eps,
    )
    group.all_reduce_.assert_not_called()
    impl.forward.assert_not_called()

    group.reset_mock()
    impl.reset_mock()
    group.try_fused_all_reduce_residual_rms_norm.return_value = None
    impl.forward.return_value = object()
    output = norm(input, residual)

    assert output is impl.forward.return_value
    group.try_fused_all_reduce_residual_rms_norm.assert_called_once()
    group.all_reduce_.assert_called_once_with(input)
    impl.forward.assert_called_once_with(input, norm.weight, residual)


def test_flashinfer_allreduce_in_place_and_dtype_guards(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.comm import flashinfer_allreduce as flashinfer_module

    FlashInferAllReduce = flashinfer_module.FlashInferAllReduce

    flashinfer = FlashInferAllReduce.__new__(FlashInferAllReduce)
    flashinfer.is_available = Mock(return_value=True)
    assert flashinfer.supports(torch.float16)
    assert flashinfer.supports(torch.bfloat16)
    assert not flashinfer.supports(torch.float32)

    flashinfer._max_size = 1024
    flashinfer._one_shot_max_size = 1024
    flashinfer._comm = SimpleNamespace(
        AllReduceFusionPattern=SimpleNamespace(kAllReduce=0),
        allreduce_fusion=Mock(
            return_value=torch.full((2, 4), 2.0, dtype=torch.bfloat16)),
    )
    flashinfer._get_workspace = Mock(return_value='workspace')
    flashinfer.supports = Mock(return_value=True)

    input = torch.ones(1, 2, 4, dtype=torch.bfloat16)
    assert flashinfer.all_reduce_(input)
    torch.testing.assert_close(input, torch.full_like(input, 2.0))
    call_kwargs = flashinfer._comm.allreduce_fusion.call_args.kwargs
    assert call_kwargs['input'].shape == (2, 4)
    assert call_kwargs['trigger_completion_at_end']
    assert call_kwargs['use_oneshot']

    fused_calls = flashinfer._comm.allreduce_fusion.call_count
    bf16_input = torch.ones(2, 4, dtype=torch.bfloat16)
    output = flashinfer.fused_all_reduce_residual_rms_norm(
        input=bf16_input,
        residual=torch.ones_like(bf16_input),
        weight=torch.ones(4, dtype=torch.float32),
        eps=1e-6,
    )
    assert output is None
    assert flashinfer._comm.allreduce_fusion.call_count == fused_calls

    unavailable = FlashInferAllReduce.__new__(FlashInferAllReduce)
    unavailable._comm = None
    unavailable._disabled = False
    unavailable._max_size = 1024
    monkeypatch.setitem(sys.modules, 'flashinfer.comm', None)
    assert not unavailable.is_available()
    assert unavailable._disabled

    workspace_error = FlashInferAllReduce.__new__(FlashInferAllReduce)
    workspace_error.group = 'cpu'
    workspace_error._world_size = 2
    workspace_error._max_size = 1024
    workspace_error._one_shot_max_size = 1024
    workspace_error._workspace = None
    workspace_error._hidden_dim = None
    workspace_error._dtype = None
    workspace_error._disabled = False
    create_workspace = Mock(side_effect=RuntimeError('unsupported topology'))
    workspace_error._comm = SimpleNamespace(
        AllReduceFusionPattern=SimpleNamespace(kAllReduce=0),
        create_allreduce_fusion_workspace=create_workspace,
        allreduce_fusion=Mock(),
    )
    monkeypatch.setattr(flashinfer_module.dist, 'get_rank', lambda group: 0)

    input = torch.ones(2, 4, dtype=torch.bfloat16)
    assert not workspace_error.all_reduce_(input)
    assert workspace_error._disabled
    assert not workspace_error.all_reduce_(input)
    create_workspace.assert_called_once()
    workspace_error._comm.allreduce_fusion.assert_not_called()


def test_symm_mem_allreduce_selects_group_algorithm(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.comm.symm_mem_allreduce import SymmetricMemoryAllReduce

    multimem = Mock()
    two_shot = Mock()
    monkeypatch.setattr(
        torch.ops,
        'symm_mem',
        SimpleNamespace(
            multimem_all_reduce_=multimem,
            two_shot_all_reduce_=two_shot,
        ),
    )

    communicator = SymmetricMemoryAllReduce.__new__(SymmetricMemoryAllReduce)
    communicator.group = SimpleNamespace(group_name='group')
    communicator._buffer = torch.empty(8, dtype=torch.bfloat16)
    communicator._enabled = True
    communicator._max_size = communicator._buffer.nbytes
    input = torch.ones(4, dtype=torch.bfloat16)

    communicator._use_multimem = True
    assert communicator.all_reduce_(input)
    multimem.assert_called_once()
    two_shot.assert_not_called()

    communicator._use_multimem = False
    assert communicator.all_reduce_(input)
    two_shot.assert_called_once()


def test_build_cuda_communicator_gates_unsupported_parallelism(monkeypatch):
    monkeypatch.setattr(communicator_module._envs, 'enable_flashinfer_allreduce', True)
    monkeypatch.setattr(communicator_module._envs, 'enable_symm_mem_allreduce', False)
    communicator_cls = Mock(return_value=object())
    monkeypatch.setattr(communicator_module, 'CudaCommunicator', communicator_cls)

    for config in (
            SimpleNamespace(dp=2, ep=1, attn_tp=8, dcp=1, enable_microbatch=False),
            SimpleNamespace(dp=1, ep=2, attn_tp=8, dcp=1, enable_microbatch=False),
            SimpleNamespace(dp=1, ep=1, attn_tp=8, dcp=1, enable_microbatch=True),
    ):
        assert communicator_module.build_cuda_communicator('cpu', 'device', config) is None
    communicator_cls.assert_not_called()

    config = SimpleNamespace(dp=1, ep=1, attn_tp=4, dcp=1, enable_microbatch=False)
    communicator = communicator_module.build_cuda_communicator('cpu', 'device', config)
    assert communicator is communicator_cls.return_value
    communicator_cls.assert_called_once_with(cpu_group='cpu', device_group='device',
                                             all_reduce_backend='flashinfer', symm_mem_query_gather=False)

    monkeypatch.setattr(communicator_module._envs, 'enable_flashinfer_allreduce', False)
    monkeypatch.setattr(communicator_module._envs, 'enable_symm_mem_allreduce', True)
    config.attn_tp = 1
    assert communicator_module.build_cuda_communicator('cpu', 'device', config) is None
    config.attn_tp = 2
    assert communicator_module.build_cuda_communicator('cpu', 'device', config) is communicator_cls.return_value

    # TP all-reduce flags must not allocate all-reduce resources on DCP groups.
    config.dcp = 2
    monkeypatch.setattr(communicator_module._envs, 'enable_symm_mem_dcp', True)
    communicator_module.build_cuda_communicator('cpu', 'device', config, group_roles=('dcp', ))
    communicator_cls.assert_called_with(cpu_group='cpu', device_group='device',
                                        all_reduce_backend=None, symm_mem_query_gather=True)


@pytest.mark.parametrize('shared_dcp_group', [False, True])
def test_communicator_group_roles(shared_dcp_group):
    from lmdeploy.pytorch.distributed import DistContext, DistGroup, _build_communicators

    tp_group = DistGroup(cpu_group='tp_cpu', gpu_group='tp_gpu')
    dcp_group = tp_group if shared_dcp_group else DistGroup(cpu_group='dcp_cpu', gpu_group='dcp_gpu')
    builder = Mock(side_effect=lambda **kwargs: object())
    context = DistContext(attn_tp_group=tp_group, mlp_tp_group=tp_group, moe_tp_group=tp_group,
                          dcp_group=dcp_group, communicator_builder=builder)
    _build_communicators(context)

    roles = [call.kwargs['group_roles'] for call in builder.call_args_list]
    assert roles == ([('tp', 'dcp')] if shared_dcp_group else [('tp', ), ('dcp', )])
    assert (tp_group.communicator is dcp_group.communicator) == shared_dcp_group


def test_dlinfer_communicator_rejects_cuda_options(monkeypatch):
    monkeypatch.setattr(communicator_module._envs, 'enable_flashinfer_allreduce', True)
    monkeypatch.setattr(communicator_module._envs, 'enable_symm_mem_allreduce', False)
    with pytest.raises(AssertionError, match='not supported by DLInfer'):
        DlinferOpsBackend.build_communicator('cpu', 'device', SimpleNamespace())

    monkeypatch.setattr(communicator_module._envs, 'enable_flashinfer_allreduce', False)
    communicator = DlinferOpsBackend.build_communicator('cpu', 'device', SimpleNamespace())
    assert isinstance(communicator, base_communicator_module.DeviceCommunicator)
    assert communicator.device_group == 'device'
