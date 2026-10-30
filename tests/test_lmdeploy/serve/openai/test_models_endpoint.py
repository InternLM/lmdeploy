# Copyright (c) OpenMMLab. All rights reserved.
import json
from types import SimpleNamespace

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.openai.endpoints import models
from lmdeploy.serve.openai.protocol import ModelCard, ModelList


@pytest.mark.parametrize('session_len', [8192, 32768, 131072])
def test_models_exposes_resolved_context_length_for_model_and_adapters(session_len):
    context = SimpleNamespace(
        async_engine=SimpleNamespace(model_name='served-model', session_len=session_len),
        engine_config=SimpleNamespace(adapters=['adapter-a', 'adapter-b'], session_len=None),
    )
    router = APIRouter()
    models.register(router, context)
    app = FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        response = client.get('/v1/models')

    assert response.status_code == 200
    payload = response.json()
    assert payload['object'] == 'list'
    assert [card['id'] for card in payload['data']] == ['served-model', 'adapter-a', 'adapter-b']
    for card in payload['data']:
        assert card['max_model_len'] == session_len
        assert card['object'] == 'model'
        assert card['owned_by'] == 'lmdeploy'
        assert card['root'] == card['id']
        assert card['parent'] is None
        assert len(card['permission']) == 1


@pytest.mark.parametrize('limit', [None, 32768])
def test_model_card_serialization_preserves_legacy_fields(limit):
    card = ModelCard(id='model', max_model_len=limit)
    payloads = [card.model_dump(), json.loads(card.model_dump_json()), ModelList(data=[card]).model_dump()['data'][0]]
    for payload in payloads:
        assert payload['parent'] is None
        assert payload['root'] is None
        assert payload['permission'] == []
        if limit is None:
            assert 'max_model_len' not in payload
        else:
            assert payload['max_model_len'] == limit
