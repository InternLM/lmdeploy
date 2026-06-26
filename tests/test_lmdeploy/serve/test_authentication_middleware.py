import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_authentication_middleware():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / 'lmdeploy' / 'serve' / 'utils' / 'server_utils.py'
    spec = importlib.util.spec_from_file_location('server_utils_for_test', module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.AuthenticationMiddleware


AuthenticationMiddleware = _load_authentication_middleware()


def _make_client():
    app = FastAPI()

    @app.get('/health')
    def health():
        return {'ok': True}

    @app.post('/nodes/add')
    def add_node():
        return {'added': True}

    @app.get('/nodes/status')
    def node_status():
        return {'nodes': []}

    @app.post('/v1/chat/completions')
    def chat_completions():
        return {'ok': True}

    app.add_middleware(AuthenticationMiddleware, tokens=['secret'])
    return TestClient(app)


def test_auth_middleware_protects_node_management_routes():
    client = _make_client()

    assert client.post('/nodes/add').status_code == 401
    assert client.get('/nodes/status').status_code == 401

    headers = {'Authorization': 'Bearer secret'}
    assert client.post('/nodes/add', headers=headers).status_code == 200
    assert client.get('/nodes/status', headers=headers).status_code == 200


def test_auth_middleware_keeps_passive_health_public():
    client = _make_client()

    assert client.get('/health').status_code == 200
    assert client.post('/v1/chat/completions').status_code == 401
