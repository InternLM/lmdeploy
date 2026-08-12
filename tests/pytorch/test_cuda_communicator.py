# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from lmdeploy.pytorch.backends.cuda import communicator as communicator_module


class _Collective:

    def __init__(self, group, *, supports=True, result=None, handled=False):
        self.group = group
        self._supports = supports
        self._result = result
        self._handled = handled
        self.closed = False

    def supports(self, dtype):
        return self._supports

    def fused_allreduce_rmsnorm(self, **kwargs):
        return self._result

    def all_reduce_(self, input):
        return self._handled

    def close(self):
        self.closed = True


def _build_communicator(monkeypatch, *, fused_result=None, symm_handled=False):
    flashinfer = _Collective('cpu', result=fused_result)
    symm_mem = _Collective('cpu', handled=symm_handled)
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_flashinfer', True)
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_symm_mem', True)
    monkeypatch.setattr(communicator_module, 'FlashInferAllReduce', lambda group: flashinfer)
    monkeypatch.setattr(communicator_module, 'SymmetricMemoryAllReduce', lambda group: symm_mem)
    communicator = communicator_module.CudaCommunicator(cpu_group='cpu', gpu_group='gpu')
    return communicator, flashinfer, symm_mem


def test_cuda_communicator_dispatch(monkeypatch):
    fused_result = object()
    communicator, _, symm_mem = _build_communicator(
        monkeypatch, fused_result=fused_result, symm_handled=True)
    nccl_all_reduce = Mock()
    monkeypatch.setattr(communicator_module.dist, 'all_reduce', nccl_all_reduce)

    assert communicator.supports_deferred_allreduce(torch.bfloat16)
    assert communicator.try_fused_allreduce_rmsnorm(
        input=torch.ones(1), residual=torch.ones(1), weight=torch.ones(1), eps=1e-6) is fused_result

    input = torch.ones(1)
    communicator.all_reduce_(input)
    nccl_all_reduce.assert_not_called()

    symm_mem._handled = False
    communicator.all_reduce_(input)
    nccl_all_reduce.assert_called_once_with(input, group='gpu')


def test_cuda_communicator_close(monkeypatch):
    communicator, flashinfer, symm_mem = _build_communicator(monkeypatch)
    communicator.close()
    assert flashinfer.closed
    assert symm_mem.closed


def test_build_cuda_communicator_gates_unsupported_parallelism(monkeypatch):
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_flashinfer', True)
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_symm_mem', False)
    communicator_cls = Mock(return_value=object())
    monkeypatch.setattr(communicator_module, 'CudaCommunicator', communicator_cls)

    for config in (
            SimpleNamespace(dp=2, ep=1, enable_microbatch=False),
            SimpleNamespace(dp=1, ep=2, enable_microbatch=False),
            SimpleNamespace(dp=1, ep=1, enable_microbatch=True),
    ):
        assert communicator_module.build_cuda_communicator('cpu', 'gpu', config) is None
    communicator_cls.assert_not_called()

    config = SimpleNamespace(dp=1, ep=1, enable_microbatch=False)
    communicator = communicator_module.build_cuda_communicator('cpu', 'gpu', config)
    assert communicator is communicator_cls.return_value
    communicator_cls.assert_called_once_with(cpu_group='cpu', gpu_group='gpu')
