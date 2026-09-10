# Copyright (c) OpenMMLab. All rights reserved.
"""CPU regression coverage for the optional LM-head provider and fallback."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from lmdeploy.pytorch.backends.cuda.comm import symm_mem_allgather as comm
from lmdeploy.pytorch.nn.embedding import ParallelLMHead


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(comm._envs, 'enable_symm_mem_lmhead', True)
    monkeypatch.setattr(comm._envs, 'symm_mem_lmhead_max_mb', 1)


@pytest.mark.parametrize('shape', [(1, 8), (3, 8), (2, 3, 8)])
def test_unavailable_provider_uses_nccl(enabled, monkeypatch, shape):
    imports = Mock(side_effect=AssertionError('CPU fallback must not import CUDA kernels'))
    monkeypatch.setattr(comm.importlib, 'import_module', imports)
    gatherer = comm.MultimemAllGatherer(None, 0, 16, torch.device('cpu'), torch.bfloat16)
    model = ParallelLMHead.__new__(ParallelLMHead)
    torch.nn.Module.__init__(model)
    model.all_reduce, model.tp, model.tp_group, model.vocab_size = True, 2, None, 13
    model._symm_mem_gatherer = gatherer
    local = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)

    def nccl(output, value, group):
        assert group is None
        output.copy_(torch.cat((value, value + 100), dim=0))

    collective = Mock(side_effect=nccl)
    monkeypatch.setattr(dist, 'all_gather_into_tensor', collective)
    torch.testing.assert_close(model.all_gather_logits(local), torch.cat((local, local + 100), dim=-1)[..., :13])
    collective.assert_called_once()
    assert gatherer(local) is None  # No lazy setup on subsequent calls either.
    imports.assert_not_called()


@pytest.mark.parametrize('failure', ['device', 'import', 'allocation'])
def test_prepare_failure_disables_provider(enabled, monkeypatch, failure):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (8, 0) if failure == 'device' else (9, 0))
    monkeypatch.setattr(dist, 'get_world_size', lambda _: 2)
    monkeypatch.setattr(dist, 'get_rank', lambda _: 0)
    monkeypatch.setattr(comm.MultimemAllGatherer, '_agree', lambda self, ready, device: ready)
    monkeypatch.setattr(comm.MultimemAllGatherer, '_same_config', lambda *args: True)
    kernels = SimpleNamespace(_allocate_symmetric_buffer=Mock(side_effect=RuntimeError('allocation unavailable')),
                              create_state=Mock())
    imports = Mock(return_value=kernels,
                   side_effect=ImportError('optional dependency') if failure == 'import' else None)
    monkeypatch.setattr(comm.importlib, 'import_module', imports)
    gatherer = comm.MultimemAllGatherer(Mock(spec=dist.ProcessGroup), 0, 16,
                                      torch.device('cuda', 0), torch.bfloat16)
    assert gatherer._state is None
    assert gatherer(torch.empty(2, 8)) is None
    kernels.create_state.assert_not_called()
    if failure == 'device':
        imports.assert_not_called()


def test_admitted_contract_capacity_and_graph_warmup(enabled, monkeypatch):
    gatherer = comm.MultimemAllGatherer(None, 0, 16, torch.device('cpu'), torch.bfloat16)
    gatherer._state = SimpleNamespace(device=torch.device('cpu'), world_size=2, hidden_dim=16, max_token_num=4)
    gatherer._kernels = SimpleNamespace(all_gather_inner=Mock(return_value=torch.empty(2, 16)))
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    assert gatherer(torch.empty(1, 8, dtype=torch.bfloat16)) is None
    assert gatherer(torch.empty(5, 8, dtype=torch.bfloat16)) is None
    with pytest.raises(RuntimeError, match='contract changed'):
        gatherer(torch.empty(2, 8, dtype=torch.float32))
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    assert gatherer(torch.empty(2, 8, dtype=torch.bfloat16)) is None
    gatherer._kernels.all_gather_inner.assert_not_called()
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    assert gatherer(torch.empty(2, 8, dtype=torch.bfloat16)).shape == (2, 16)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    assert gatherer(torch.empty(2, 8, dtype=torch.bfloat16)).shape == (2, 16)
    assert gatherer._kernels.all_gather_inner.call_count == 2


def test_apply_notifies_provider():
    model = ParallelLMHead.__new__(ParallelLMHead)
    torch.nn.Module.__init__(model)
    model.weight = torch.nn.Parameter(torch.empty(8, 4))
    model._symm_mem_gatherer = Mock()
    model.to(dtype=torch.bfloat16)
    model._symm_mem_gatherer.reset_for_weight.assert_called_once_with(model.weight)


@pytest.mark.parametrize('field,value', [
    ('rank', 1), ('world_size', 4), ('multicast_ptr', 0), ('multicast_ptr', -16),
    ('signal_pad_ptrs_dev', 0), ('signal_pad_ptrs_dev', 7), ('signal_pad_size', 255),
])
def test_invalid_handle_rejected(enabled, field, value):
    gatherer = comm.MultimemAllGatherer(None, 0, 16, torch.device('cpu'), torch.bfloat16)
    handle = SimpleNamespace(rank=0, world_size=2, multicast_ptr=16,
                             signal_pad_ptrs_dev=8, signal_pad_size=256)
    state = SimpleNamespace(symm_mem_hdl=handle, world_size=2)
    kernels = SimpleNamespace(_MAX_BLOCKS=32)
    assert gatherer._valid_handle(state, kernels)
    setattr(handle, field, value)
    assert not gatherer._valid_handle(state, kernels)
