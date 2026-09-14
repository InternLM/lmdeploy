# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import pickle
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.openai.endpoints.management import register
from lmdeploy.utils import (
    ALLOW_PICKLE_UPDATE_PARAMS_ENV,
    allow_pickle_update_params,
    coerce_update_params_tensor,
    is_pickle_serialized_named_tensors,
    load_pickled_serialized_named_tensors,
    load_safetensors_serialized_named_tensors,
    serialize_named_tensors_safetensors,
)


class _DummyEngine:

    def __init__(self):
        self.requests = []

    def update_params(self, request):
        self.requests.append(request)


def _client():
    engine = _DummyEngine()
    server_context = SimpleNamespace(async_engine=SimpleNamespace(engine=engine))
    router = APIRouter()
    register(router, server_context)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), engine


def _pickle_b64(obj) -> str:
    import pybase64
    return pybase64.b64encode(pickle.dumps(obj)).decode('utf-8')


def test_http_update_weights_rejects_pickle_string():
    client, engine = _client()
    marker = {'executed': False}

    class Boom:
        def __reduce__(self):
            marker['executed'] = True
            return (int, (0, ))

    response = client.post('/update_weights', json={
        'serialized_named_tensors': _pickle_b64(Boom()),
        'finished': False,
    })

    assert response.status_code == 400
    assert 'pickle' in response.json()['message'].lower()
    assert engine.requests == []
    assert marker['executed'] is False


def test_http_update_weights_rejects_pickle_list():
    client, engine = _client()
    response = client.post('/update_weights', json={
        'serialized_named_tensors': [_pickle_b64({'w': 1})],
        'finished': False,
    })

    assert response.status_code == 400
    assert engine.requests == []


def test_http_update_weights_rejects_pickle_even_when_engine_opt_in_is_set(monkeypatch):
    monkeypatch.setenv(ALLOW_PICKLE_UPDATE_PARAMS_ENV, '1')
    client, engine = _client()
    response = client.post('/update_weights', json={
        'serialized_named_tensors': _pickle_b64({'w': 1}),
        'finished': False,
    })

    assert response.status_code == 400
    assert engine.requests == []


def test_http_update_weights_accepts_structured_dict():
    client, engine = _client()
    payload = {
        'layer.weight': {
            'dtype': 'float32',
            'shape': [1, 1],
            'data': 'AAAAAA==',
        }
    }
    response = client.post('/update_weights', json={
        'serialized_named_tensors': payload,
        'finished': True,
    })

    assert response.status_code == 200
    assert len(engine.requests) == 1
    assert engine.requests[0].serialized_named_tensors == payload


def test_http_update_weights_accepts_safetensors_string():
    import torch
    client, engine = _client()
    blob = serialize_named_tensors_safetensors({'w': torch.ones(1, dtype=torch.float32)})
    response = client.post('/update_weights', json={
        'serialized_named_tensors': blob,
        'load_format': 'safetensors',
        'finished': True,
    })

    assert response.status_code == 200
    assert len(engine.requests) == 1
    assert engine.requests[0].load_format == 'safetensors'
    loaded = load_safetensors_serialized_named_tensors(engine.requests[0].serialized_named_tensors)
    assert list(loaded) == ['w']
    assert loaded['w'].tolist() == [1.0]


def test_is_pickle_serialized_named_tensors():
    assert is_pickle_serialized_named_tensors('AAAA') is True
    assert is_pickle_serialized_named_tensors(['AAAA']) is True
    assert is_pickle_serialized_named_tensors(['AAAA'], load_format='safetensors') is False
    assert is_pickle_serialized_named_tensors({'w': 1}) is False
    assert is_pickle_serialized_named_tensors([]) is False


def test_load_pickled_refuses_without_opt_in(monkeypatch):
    monkeypatch.delenv(ALLOW_PICKLE_UPDATE_PARAMS_ENV, raising=False)
    assert allow_pickle_update_params() is False
    with pytest.raises(ValueError, match='disabled by default'):
        load_pickled_serialized_named_tensors(_pickle_b64({'ok': 1}))


def test_load_pickled_opt_in_does_not_run_without_flag_even_if_loads_is_patched(monkeypatch):
    monkeypatch.delenv(ALLOW_PICKLE_UPDATE_PARAMS_ENV, raising=False)
    called = []

    def boom(*args, **kwargs):
        called.append(True)
        raise AssertionError('pickle.loads must not run without opt-in')

    monkeypatch.setattr(ForkingPickler, 'loads', boom)
    with pytest.raises(ValueError, match='disabled by default'):
        load_pickled_serialized_named_tensors(_pickle_b64({'ok': 1}))
    assert called == []


def test_load_pickled_allowed_with_opt_in(monkeypatch):
    monkeypatch.setenv(ALLOW_PICKLE_UPDATE_PARAMS_ENV, '1')
    assert load_pickled_serialized_named_tensors(_pickle_b64({'ok': 1})) == {'ok': 1}


def test_coerce_update_params_tensor_from_json_spec():
    import torch
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    spec = {
        'dtype': 'float32',
        'shape': list(tensor.shape),
        'data': __import__('pybase64').b64encode(tensor.contiguous().cpu().numpy().tobytes()).decode('utf-8'),
    }
    got = coerce_update_params_tensor(spec)
    assert torch.equal(got, tensor)


def test_coerce_update_params_tensor_rejects_reduce_tuple():
    with pytest.raises(TypeError):
        coerce_update_params_tensor((int, (1, )))


def test_pytorch_update_params_rejects_pickle_without_opt_in(monkeypatch):
    from contextlib import nullcontext

    from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
    from lmdeploy.serve.openai.protocol import UpdateParamsRequest

    monkeypatch.delenv(ALLOW_PICKLE_UPDATE_PARAMS_ENV, raising=False)
    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.dist_ctx = SimpleNamespace(tp_group=SimpleNamespace(rank=0))
    agent.patched_model = SimpleNamespace(get_model=lambda: object())
    agent.spec_agent = SimpleNamespace(get_model=lambda: None, is_enabled=lambda: False)
    agent.all_context = lambda: nullcontext()
    request = UpdateParamsRequest(serialized_named_tensors=_pickle_b64({'w': 1}), finished=False)
    with pytest.raises(ValueError, match='disabled by default'):
        agent.update_params(request)
