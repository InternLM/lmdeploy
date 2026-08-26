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
    communicator = communicator_module.CudaCommunicator(cpu_group='cpu', device_group='gpu')
    enabled_cls, disabled_cls = ((flashinfer_cls, symm_mem_cls)
                                 if use_flashinfer else (symm_mem_cls, flashinfer_cls))
    enabled_cls.assert_called_once_with('cpu')
    disabled_cls.assert_not_called()
    return communicator, collective


def _build_norm(monkeypatch, *, fused=False):
    impl = Mock()
    builder = SimpleNamespace(build=Mock(return_value=impl))
    backend = SimpleNamespace(get_layer_impl_builder=Mock(return_value=builder))
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
        communicator_module.CudaCommunicator(cpu_group='cpu', device_group='gpu')


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
            SimpleNamespace(dp=2, ep=1, attn_tp=8, enable_microbatch=False),
            SimpleNamespace(dp=1, ep=2, attn_tp=8, enable_microbatch=False),
            SimpleNamespace(dp=1, ep=1, attn_tp=8, enable_microbatch=True),
    ):
        assert communicator_module.build_cuda_communicator('cpu', 'device', config) is None
    communicator_cls.assert_not_called()

    config = SimpleNamespace(dp=1, ep=1, attn_tp=4, enable_microbatch=False)
    communicator = communicator_module.build_cuda_communicator('cpu', 'device', config)
    assert communicator is communicator_cls.return_value
    communicator_cls.assert_called_once_with(cpu_group='cpu', device_group='device')

    monkeypatch.setattr(communicator_module._envs, 'enable_flashinfer_allreduce', False)
    monkeypatch.setattr(communicator_module._envs, 'enable_symm_mem_allreduce', True)
    config.attn_tp = 1
    assert communicator_module.build_cuda_communicator('cpu', 'device', config) is None
    config.attn_tp = 2
    assert communicator_module.build_cuda_communicator('cpu', 'device', config) is communicator_cls.return_value


def test_dlinfer_communicator_rejects_cuda_options(monkeypatch):
    monkeypatch.setattr(communicator_module._envs, 'enable_flashinfer_allreduce', True)
    monkeypatch.setattr(communicator_module._envs, 'enable_symm_mem_allreduce', False)
    with pytest.raises(AssertionError, match='not supported by DLInfer'):
        DlinferOpsBackend.build_communicator('cpu', 'device', SimpleNamespace())

    monkeypatch.setattr(communicator_module._envs, 'enable_flashinfer_allreduce', False)
    communicator = DlinferOpsBackend.build_communicator('cpu', 'device', SimpleNamespace())
    assert isinstance(communicator, base_communicator_module.DeviceCommunicator)
    assert communicator.device_group == 'device'
