# Copyright (c) OpenMMLab. All rights reserved.
"""Decision task boundaries through the production server middleware stack."""
import pytest
from fastapi.testclient import TestClient
from test_decision_startup import server_startup as _server_startup

server_startup = _server_startup


def start_app(server_startup, tmp_path, *, enabled=True, **kwargs):
    api_server, build, run = server_startup
    api_server.serve(str(tmp_path), backend='pytorch', generation_config='lmdeploy',
                     task='intern-decision' if enabled else None,
                     decision_temperature=1.0 if enabled else None,
                     max_concurrent_requests=1, **kwargs)
    app = run.call_args.kwargs['app']
    app.middleware_stack = app.build_middleware_stack()
    return app, build.return_value



@pytest.mark.parametrize('path', [
    '/v1/chat/completions', '/v1/completions', '/generate', '/v1/messages',
    '/v1/responses', '/pooling', '/get_ppl', '/v1/chat/completions/',
])
def test_decision_task_rejects_other_inference(server_startup, tmp_path, path):
    app, engine = start_app(server_startup, tmp_path)
    with TestClient(app) as client:
        response = client.post(path, json={})
        assert response.status_code == 400
        payload = response.json()
        error = payload.get('error', payload)
        assert 'intern-decision' in error['message']
        if path == '/v1/messages':
            assert error['type'] == 'invalid_request_error'
    assert not engine.requests


def test_task_gate_preserves_decisions_models_encoding_and_preflight(server_startup, tmp_path):
    app, engine = start_app(server_startup, tmp_path, api_keys=['secret'])
    with TestClient(app) as client:
        headers = {'authorization': 'Bearer secret'}
        assert client.get('/v1/models', headers=headers).status_code == 200
        assert client.post('/v1/encode', headers=headers, json={'input': 'text'}).status_code == 200
        assert client.post('/v1/systemone', headers=headers, json={
            'state': 'text', 'questions': {'q': {'type': 'noul', 'instructions': 'Check'}},
        }).status_code == 200
        assert client.post('/v1/decisions', headers=headers, json={
            'model': engine.model_name, 'input': 'text',
            'questions': [{'type': 'predicate', 'instructions': 'Check'}],
        }).status_code == 200
        response = client.options('/v1/chat/completions', headers={
            'origin': 'https://client.example', 'access-control-request-method': 'POST',
        })
        assert response.status_code == 200
    assert len(engine.requests) == 2


@pytest.mark.parametrize('path', ['/v1/chat/completions', '/v1/messages', '/v1/responses', '/generate'])
def test_ordinary_server_keeps_normal_endpoint_validation(server_startup, tmp_path, path):
    app, _ = start_app(server_startup, tmp_path, enabled=False)
    with TestClient(app) as client:
        response = client.post(path, json={})
        assert response.status_code in (400, 422)
        assert 'intern-decision' not in response.text
