# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends import communicator as base_communicator_module
from lmdeploy.pytorch.backends.cuda import communicator as communicator_module
from lmdeploy.pytorch.backends.dlinfer.op_backend import DlinferOpsBackend
from lmdeploy.pytorch.nn import norm as norm_module


class _Collective:

    def __init__(self, group, *, supports=True, result=None, handled=False):
        self.group = group
        self._supports = supports
        self._result = result
        self._handled = handled
        self.closed = False

    def supports(self, dtype):
        return self._supports

    def is_available(self):
        return self._supports

    def fused_all_reduce_residual_rms_norm(self, **kwargs):
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
    communicator = communicator_module.CudaCommunicator(cpu_group='cpu', device_group='gpu')
    return communicator, flashinfer, symm_mem


def _build_norm(monkeypatch, *, optimized=True, fused=False):
    impl = Mock()
    builder = SimpleNamespace(build=Mock(return_value=impl))
    backend = SimpleNamespace(get_layer_impl_builder=Mock(return_value=builder))
    group = Mock()
    group.supports_optimized_all_reduce.return_value = optimized
    group.supports_fused_all_reduce_residual_rms_norm.return_value = fused
    monkeypatch.setattr(norm_module, 'get_backend', lambda: backend)
    monkeypatch.setattr(norm_module, 'get_dist_group', lambda layer_type: group)
    norm = norm_module.RMSNorm(4, dtype=torch.float32, device='cpu', all_reduce_group='attn')
    return norm, impl, group


def test_device_communicator_fallback(monkeypatch):
    all_reduce = Mock()
    monkeypatch.setattr(base_communicator_module.dist, 'all_reduce', all_reduce)
    communicator = base_communicator_module.DeviceCommunicator(device_group='device')

    assert not communicator.supports_optimized_all_reduce()
    assert not communicator.supports_fused_all_reduce_residual_rms_norm()
    assert communicator.try_fused_all_reduce_residual_rms_norm(
        input=torch.ones(1), residual=torch.ones(1), weight=torch.ones(1), eps=1e-6) is None

    input = torch.ones(1)
    communicator.all_reduce_(input)
    all_reduce.assert_called_once_with(input, group='device')


def test_cuda_communicator_dispatch(monkeypatch):
    fused_result = object()
    communicator, flashinfer, symm_mem = _build_communicator(
        monkeypatch, fused_result=fused_result, symm_handled=True)
    nccl_all_reduce = Mock()
    monkeypatch.setattr(communicator_module.dist, 'all_reduce', nccl_all_reduce)

    assert communicator.supports_optimized_all_reduce()
    assert communicator.supports_fused_all_reduce_residual_rms_norm()
    assert communicator.try_fused_all_reduce_residual_rms_norm(
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


def test_symm_mem_does_not_handle_rms_norm_fusion(monkeypatch):
    symm_mem = _Collective('cpu')
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_flashinfer', False)
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_symm_mem', True)
    monkeypatch.setattr(communicator_module, 'SymmetricMemoryAllReduce', lambda group: symm_mem)
    communicator = communicator_module.CudaCommunicator(cpu_group='cpu', device_group='gpu')

    assert communicator.supports_optimized_all_reduce()
    assert not communicator.supports_fused_all_reduce_residual_rms_norm()
    assert communicator.try_fused_all_reduce_residual_rms_norm(
        input=torch.ones(1), residual=torch.ones(1), weight=torch.ones(1), eps=1e-6) is None


def test_rms_norm_fuses_pending_all_reduce(monkeypatch):
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


def test_rms_norm_falls_back_to_standalone_all_reduce(monkeypatch):
    norm, impl, group = _build_norm(monkeypatch)
    impl.forward.return_value = object()
    input = torch.ones((1, 4), dtype=torch.bfloat16)
    residual = torch.ones_like(input)
    output = norm(input, residual)

    assert output is impl.forward.return_value
    group.try_fused_all_reduce_residual_rms_norm.assert_not_called()
    group.all_reduce_.assert_called_once_with(input)
    impl.forward.assert_called_once_with(input, norm.weight, residual)

    group.reset_mock()
    impl.reset_mock()
    norm(input)
    group.all_reduce_.assert_not_called()
    impl.forward.assert_called_once_with(input, norm.weight, None)


def test_rms_norm_keeps_normal_path_without_optimized_all_reduce(monkeypatch):
    norm, impl, group = _build_norm(monkeypatch, optimized=False)
    input = torch.ones((1, 4), dtype=torch.bfloat16)
    residual = torch.ones_like(input)
    norm(input, residual)

    group.all_reduce_.assert_not_called()
    impl.forward.assert_called_once_with(input, norm.weight, residual)


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


def test_flashinfer_allreduce_supports_16bit_floating_dtypes():
    from lmdeploy.pytorch.backends.cuda.flashinfer_allreduce import FlashInferAllReduce

    flashinfer = FlashInferAllReduce.__new__(FlashInferAllReduce)
    flashinfer.is_available = Mock(return_value=True)

    assert flashinfer.supports(torch.float16)
    assert flashinfer.supports(torch.bfloat16)
    assert not flashinfer.supports(torch.float32)


def test_flashinfer_fused_allreduce_rejects_mixed_weight_dtype():
    from lmdeploy.pytorch.backends.cuda.flashinfer_allreduce import FlashInferAllReduce

    flashinfer = FlashInferAllReduce.__new__(FlashInferAllReduce)
    flashinfer.supports = Mock(return_value=True)
    flashinfer._comm = Mock()
    input = torch.ones(2, 4, dtype=torch.bfloat16)

    output = flashinfer.fused_all_reduce_residual_rms_norm(
        input=input,
        residual=torch.ones_like(input),
        weight=torch.ones(4, dtype=torch.float32),
        eps=1e-6,
    )

    assert output is None
    flashinfer._comm.allreduce_fusion.assert_not_called()


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
        assert communicator_module.build_cuda_communicator('cpu', 'device', config) is None
    communicator_cls.assert_not_called()

    config = SimpleNamespace(dp=1, ep=1, attn_tp=4, enable_microbatch=False)
    communicator = communicator_module.build_cuda_communicator('cpu', 'device', config)
    assert communicator is communicator_cls.return_value
    communicator_cls.assert_called_once_with(cpu_group='cpu', device_group='device')


def test_dlinfer_communicator_rejects_cuda_options(monkeypatch):
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_flashinfer', True)
    with pytest.raises(AssertionError, match='not supported by DLInfer'):
        DlinferOpsBackend.build_communicator('cpu', 'device', SimpleNamespace())

    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_flashinfer', False)
    communicator = DlinferOpsBackend.build_communicator('cpu', 'device', SimpleNamespace())
    assert isinstance(communicator, base_communicator_module.DeviceCommunicator)
    assert communicator.device_group == 'device'


def test_symm_mem_is_tried_only_for_supported_config(monkeypatch):
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_flashinfer', False)
    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_symm_mem', True)
    communicator_cls = Mock(return_value=object())
    monkeypatch.setattr(communicator_module, 'CudaCommunicator', communicator_cls)

    config = SimpleNamespace(dp=1, ep=1, attn_tp=2, enable_microbatch=False)
    assert communicator_module.should_try_symm_mem(config)
    assert communicator_module.build_cuda_communicator('cpu', 'device', config) is communicator_cls.return_value

    config.attn_tp = 1
    assert not communicator_module.should_try_symm_mem(config)
    assert communicator_module.build_cuda_communicator('cpu', 'device', config) is None

    monkeypatch.setattr(communicator_module._envs, 'allreduce_use_symm_mem', False)
    config.attn_tp = 2
    assert not communicator_module.should_try_symm_mem(config)
    assert communicator_module.build_cuda_communicator('cpu', 'device', config) is None
