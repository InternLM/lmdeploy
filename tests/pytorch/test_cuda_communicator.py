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


def _build_communicator(monkeypatch,
                        *,
                        fused_result=None,
                        flashinfer_handled=False,
                        symm_handled=False):
    flashinfer = _Collective('cpu', result=fused_result, handled=flashinfer_handled)
    symm_mem = _Collective('cpu', handled=symm_handled)
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_flashinfer', True)
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_symm_mem', True)
    monkeypatch.setattr(communicator_module, 'FlashInferAllReduce', lambda group: flashinfer)
    monkeypatch.setattr(communicator_module, 'SymmetricMemoryAllReduce', lambda group: symm_mem)
    communicator = communicator_module.CudaCommunicator(cpu_group='cpu', gpu_group='gpu')
    return communicator, flashinfer, symm_mem


def test_cuda_communicator_dispatch(monkeypatch):
    fused_result = object()
    communicator, flashinfer, symm_mem = _build_communicator(
        monkeypatch, fused_result=fused_result, symm_handled=True)
    nccl_all_reduce = Mock()
    monkeypatch.setattr(communicator_module.dist, 'all_reduce', nccl_all_reduce)

    assert communicator.supports_deferred_allreduce(torch.bfloat16)
    assert communicator.try_fused_allreduce_rmsnorm(
        input=torch.ones(1), residual=torch.ones(1), weight=torch.ones(1), eps=1e-6) is fused_result

    input = torch.ones(1)
    communicator.all_reduce_(input)
    nccl_all_reduce.assert_not_called()

    flashinfer._handled = True
    symm_mem._handled = False
    communicator.all_reduce_(input)
    nccl_all_reduce.assert_not_called()

    flashinfer._handled = False
    symm_mem._handled = False
    communicator.all_reduce_(input)
    nccl_all_reduce.assert_called_once_with(input, group='gpu')


def test_flashinfer_allreduce_in_place():
    from lmdeploy.pytorch.backends.cuda.flashinfer_allreduce import FlashInferAllReduce

    flashinfer = FlashInferAllReduce.__new__(FlashInferAllReduce)
    flashinfer._max_size = 1024
    flashinfer._one_shot_max_size = 1024
    flashinfer._comm = SimpleNamespace(
        AllReduceFusionPattern=SimpleNamespace(kAllReduce=0),
        allreduce_fusion=Mock(return_value=torch.full((2, 4), 2.0)),
    )
    flashinfer._get_workspace = Mock(return_value='workspace')
    flashinfer.supports = Mock(return_value=True)

    input = torch.ones(2, 4)
    assert flashinfer.all_reduce_(input)
    torch.testing.assert_close(input, torch.full_like(input, 2.0))
    flashinfer._comm.allreduce_fusion.assert_called_once_with(
        input=input,
        workspace='workspace',
        pattern=0,
        launch_with_pdl=True,
        trigger_completion_at_end=True,
        use_oneshot=True,
    )


def test_symm_mem_allreduce_selects_group_algorithm(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.symm_mem_allreduce import SymmetricMemoryAllReduce

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
            SimpleNamespace(dp=2, ep=1, attn_tp=8, enable_microbatch=False),
            SimpleNamespace(dp=1, ep=2, attn_tp=8, enable_microbatch=False),
            SimpleNamespace(dp=1, ep=1, attn_tp=8, enable_microbatch=True),
    ):
        assert communicator_module.build_cuda_communicator('cpu', 'gpu', config) is None
    communicator_cls.assert_not_called()

    config = SimpleNamespace(dp=1, ep=1, attn_tp=4, enable_microbatch=False)
    communicator = communicator_module.build_cuda_communicator('cpu', 'gpu', config)
    assert communicator is communicator_cls.return_value
    communicator_cls.assert_called_once_with(cpu_group='cpu', gpu_group='gpu')


def test_symm_mem_is_tried_only_for_supported_config(monkeypatch):
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_flashinfer', False)
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_symm_mem', True)
    communicator_cls = Mock(return_value=object())
    monkeypatch.setattr(communicator_module, 'CudaCommunicator', communicator_cls)

    config = SimpleNamespace(dp=1, ep=1, attn_tp=2, enable_microbatch=False)
    assert communicator_module.should_try_symm_mem(config)
    assert communicator_module.build_cuda_communicator('cpu', 'gpu', config) is communicator_cls.return_value

    config.attn_tp = 1
    assert not communicator_module.should_try_symm_mem(config)
    assert communicator_module.build_cuda_communicator('cpu', 'gpu', config) is None

    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_symm_mem', False)
    config.attn_tp = 2
    assert not communicator_module.should_try_symm_mem(config)
    assert communicator_module.build_cuda_communicator('cpu', 'gpu', config) is None
