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
    FlattenedTensorBucket,
    allow_pickle_update_params,
    coerce_update_params_tensor,
    is_pickle_serialized_named_tensors,
    load_pickled_serialized_named_tensors,
    load_safetensors_serialized_named_tensors,
    serialize_named_tensors_safetensors,
    serialize_state_dict,
)

_PICKLE_MARKER = {'executed': False}


def _pickle_reduce_mark():
    _PICKLE_MARKER['executed'] = True
    return 0


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
    _PICKLE_MARKER['executed'] = False

    class Boom:
        def __reduce__(self):
            return (_pickle_reduce_mark, ())

    response = client.post('/update_weights', json={
        'serialized_named_tensors': _pickle_b64(Boom()),
        'finished': False,
    })

    assert response.status_code == 400
    assert 'pickle' in response.json()['message'].lower()
    assert engine.requests == []
    assert _PICKLE_MARKER['executed'] is False


def test_http_update_weights_rejects_pickle_list():
    client, engine = _client()
    response = client.post('/update_weights', json={
        'serialized_named_tensors': [_pickle_b64({'w': 1})],
        'finished': False,
    })

    assert response.status_code == 400
    assert engine.requests == []


def test_http_update_weights_forwards_pickle_when_opt_in(monkeypatch):
    monkeypatch.setenv(ALLOW_PICKLE_UPDATE_PARAMS_ENV, '1')
    blob = _pickle_b64({'w': 1})
    client, engine = _client()
    response = client.post('/update_weights', json={
        'serialized_named_tensors': blob,
        'finished': False,
    })

    assert response.status_code == 200
    assert len(engine.requests) == 1
    assert engine.requests[0].serialized_named_tensors == blob


def test_http_update_weights_forwards_xtuner_ipc_when_opt_in(monkeypatch):
    monkeypatch.setenv(ALLOW_PICKLE_UPDATE_PARAMS_ENV, '1')
    client, engine = _client()

    import torch
    named_tensors = [('layer.weight', torch.ones(2, dtype=torch.float32))]
    bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    metadata_only = serialize_state_dict(dict(metadata=bucket.get_metadata()))
    empty_finalizer = serialize_state_dict({})
    tp_payload = [serialize_state_dict(dict(metadata=bucket.get_metadata()))]

    metadata_resp = client.post('/update_weights', json={
        'serialized_named_tensors': metadata_only,
        'load_format': 'flattened_bucket',
        'finished': False,
    })
    finalizer_resp = client.post('/update_weights', json={
        'serialized_named_tensors': empty_finalizer,
        'finished': True,
    })
    tp_resp = client.post('/update_weights', json={
        'serialized_named_tensors': tp_payload,
        'load_format': 'flattened_bucket',
        'finished': False,
    })

    assert metadata_resp.status_code == 200
    assert finalizer_resp.status_code == 200
    assert tp_resp.status_code == 200
    assert len(engine.requests) == 3
    assert engine.requests[0].serialized_named_tensors == metadata_only
    assert engine.requests[1].serialized_named_tensors == empty_finalizer
    assert engine.requests[1].finished is True
    assert engine.requests[2].serialized_named_tensors == tp_payload


def test_http_update_weights_rejects_xtuner_ipc_without_opt_in(monkeypatch):
    monkeypatch.delenv(ALLOW_PICKLE_UPDATE_PARAMS_ENV, raising=False)
    client, engine = _client()
    response = client.post('/update_weights', json={
        'serialized_named_tensors': serialize_state_dict({}),
        'finished': True,
    })
    assert response.status_code == 400
    assert engine.requests == []


def test_http_update_weights_never_calls_pickle_loads(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError('HTTP /update_weights must not pickle-load')

    monkeypatch.setattr(ForkingPickler, 'loads', boom)
    monkeypatch.setattr(pickle, 'loads', boom)
    client, engine = _client()
    response = client.post('/update_weights', json={
        'serialized_named_tensors': _pickle_b64({'w': 1}),
        'finished': False,
    })
    assert response.status_code == 400
    assert engine.requests == []


def test_http_update_weights_opt_in_still_does_not_unpickle_in_handler(monkeypatch):
    monkeypatch.setenv(ALLOW_PICKLE_UPDATE_PARAMS_ENV, '1')

    def boom(*args, **kwargs):
        raise AssertionError('HTTP /update_weights must not pickle-load')

    monkeypatch.setattr(ForkingPickler, 'loads', boom)
    monkeypatch.setattr(pickle, 'loads', boom)
    blob = _pickle_b64({'w': 1})
    client, engine = _client()
    response = client.post('/update_weights', json={
        'serialized_named_tensors': blob,
        'finished': False,
    })
    assert response.status_code == 200
    assert engine.requests[0].serialized_named_tensors == blob


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


def test_serialize_named_tensors_safetensors_clones_tied_cpu_weights():
    import torch
    from safetensors.torch import save

    weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    state = {
        'model.embed_tokens.weight': weight,
        'lm_head.weight': weight,
    }
    shared_cpu = {name: tensor.detach().contiguous().cpu() for name, tensor in state.items()}
    with pytest.raises(RuntimeError, match='share memory'):
        save(shared_cpu)

    blob = serialize_named_tensors_safetensors(state)
    loaded = load_safetensors_serialized_named_tensors(blob)
    assert torch.equal(loaded['model.embed_tokens.weight'], weight)
    assert torch.equal(loaded['lm_head.weight'], weight)


def test_pickle_opt_in_is_registered_for_ray_workers():
    """Ray copies pytorch.envs.get_all_envs() onto workers."""
    import importlib
    import os
    from pathlib import Path

    envs_path = Path(__file__).resolve().parents[4] / 'lmdeploy' / 'pytorch' / 'envs.py'
    assert "os.getenv('LMDEPLOY_ALLOW_PICKLE_UPDATE_PARAMS'" in envs_path.read_text()

    import lmdeploy.pytorch.envs as pytorch_envs

    old = os.environ.get(ALLOW_PICKLE_UPDATE_PARAMS_ENV)
    try:
        os.environ[ALLOW_PICKLE_UPDATE_PARAMS_ENV] = '1'
        importlib.reload(pytorch_envs)
        assert pytorch_envs.get_all_envs().get(ALLOW_PICKLE_UPDATE_PARAMS_ENV) == '1'

        os.environ.pop(ALLOW_PICKLE_UPDATE_PARAMS_ENV, None)
        importlib.reload(pytorch_envs)
        assert ALLOW_PICKLE_UPDATE_PARAMS_ENV not in pytorch_envs.get_all_envs()
    finally:
        if old is None:
            os.environ.pop(ALLOW_PICKLE_UPDATE_PARAMS_ENV, None)
        else:
            os.environ[ALLOW_PICKLE_UPDATE_PARAMS_ENV] = old
        importlib.reload(pytorch_envs)


def test_serialize_named_tensors_safetensors_clones_overlapping_views():
    import torch

    base = torch.arange(8, dtype=torch.float32)
    state = {
        'a': base[:6],
        'b': base[2:],
    }
    blob = serialize_named_tensors_safetensors(state)
    loaded = load_safetensors_serialized_named_tensors(blob)
    assert torch.equal(loaded['a'], state['a'])
    assert torch.equal(loaded['b'], state['b'])
