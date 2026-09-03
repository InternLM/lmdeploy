# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

comm = pytest.importorskip('lmdeploy.pytorch.backends.cuda.comm.symm_mem_allgather')


def _gatherer():
    return comm.MultimemAllGatherer(group=object(), rank=0, gathered_width=16, max_tokens=4)


def _disable_cuda_capture_probe(monkeypatch):
    monkeypatch.setattr(comm.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(comm.dist, 'get_world_size', lambda group: 2)


def test_prepare_rejects_before_rendezvous_when_any_rank_cannot_allocate(monkeypatch):
    _disable_cuda_capture_probe(monkeypatch)
    gatherer = _gatherer()
    events = []

    def _allocate(*args, **kwargs):
        events.append('allocate')
        raise RuntimeError('local allocation failed')

    def _agree(local_ready, device):
        events.append(('agree', local_ready, device))
        return False

    monkeypatch.setattr(comm, '_allocate_symmetric_buffer', _allocate)
    monkeypatch.setattr(comm, 'create_state', lambda *args, **kwargs: pytest.fail('rendezvous must not be entered'))
    monkeypatch.setattr(gatherer, 'agree', _agree)

    assert gatherer.prepare(torch.device('cuda:3')) is False
    assert events == ['allocate', ('agree', False, torch.device('cuda:3'))]
    assert gatherer._state is None


def test_prepare_agrees_before_rendezvous_and_preserves_device(monkeypatch):
    _disable_cuda_capture_probe(monkeypatch)
    gatherer = _gatherer()
    events = []
    arena = torch.empty(0)

    def _allocate(group, max_tokens, hidden_size, device):
        events.append(('allocate', device))
        return arena

    def _agree(local_ready, device):
        events.append(('agree', local_ready, device))
        return True

    def _create_state(**kwargs):
        events.append(('rendezvous', kwargs['device'], kwargs['comm_buff']))
        return SimpleNamespace(
            symm_mem_hdl=SimpleNamespace(multicast_ptr=1, rank=0),
            world_size=2,
            max_token_num=4,
        )

    monkeypatch.setattr(comm, '_allocate_symmetric_buffer', _allocate)
    monkeypatch.setattr(comm, 'create_state', _create_state)
    monkeypatch.setattr(gatherer, 'agree', _agree)

    assert gatherer.prepare(torch.device('cuda:3')) is True
    assert events == [
        ('allocate', torch.device('cuda:3')),
        ('agree', True, torch.device('cuda:3')),
        ('rendezvous', torch.device('cuda:3'), arena),
        ('agree', True, torch.device('cuda:3')),
    ]


def test_prepare_does_not_turn_rendezvous_failure_into_local_fallback(monkeypatch):
    _disable_cuda_capture_probe(monkeypatch)
    gatherer = _gatherer()
    monkeypatch.setattr(comm, '_allocate_symmetric_buffer', lambda *args, **kwargs: torch.empty(0))
    monkeypatch.setattr(gatherer, 'agree', lambda local_ready, device: True)

    def _rendezvous_failure(**kwargs):
        raise RuntimeError('collective rendezvous failed')

    monkeypatch.setattr(comm, 'create_state', _rendezvous_failure)

    with pytest.raises(RuntimeError, match='collective rendezvous failed'):
        gatherer.prepare(torch.device('cuda:3'))
    assert gatherer._state is gatherer._UNINIT


def test_release_drops_arena_and_resets_graph_admission():
    gatherer = _gatherer()
    gatherer._state = object()
    gatherer._graph_ready = True
    gatherer._runtime_admitted = True

    gatherer.release()

    assert gatherer._state is gatherer._UNINIT
    assert gatherer._graph_ready is False
    assert gatherer._runtime_admitted is False


def test_first_call_collectively_admits_static_input_contract(monkeypatch):
    gatherer = _gatherer()
    gatherer._state = SimpleNamespace(
        max_token_num=4,
        world_size=2,
        hidden_dim=16,
        device=torch.device('cpu'),
    )
    admission = Mock(return_value=True)
    monkeypatch.setattr(gatherer, 'agree', admission)
    monkeypatch.setattr(comm.torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(comm, 'all_gather_inner', lambda state, x, **kwargs: x)
    value = torch.empty((1, 8), dtype=torch.bfloat16)

    assert gatherer(value) is value
    assert gatherer(value) is value
    admission.assert_called_once_with(True, torch.device('cpu'))
    assert gatherer._runtime_admitted is True


def test_direct_path_rejects_unaligned_local_width_before_kernel():
    gatherer = _gatherer()
    gatherer._state = SimpleNamespace(max_token_num=4, world_size=2, hidden_dim=8, device=torch.device('cpu'))
    gatherer.agree = lambda local_ready, device: local_ready

    assert gatherer(torch.empty((1, 4), dtype=torch.bfloat16)) is None
    assert gatherer._state is None
