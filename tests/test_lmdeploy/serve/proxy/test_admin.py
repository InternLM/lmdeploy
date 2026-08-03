# Copyright (c) OpenMMLab. All rights reserved.

from http import HTTPStatus
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from lmdeploy.serve.proxy.app import create_app
from lmdeploy.serve.proxy.core.config import ProxyConfig
from lmdeploy.serve.proxy.core.replica import ReplicaLoad
from lmdeploy.serve.proxy.endpoint import admin as admin_module


def _client():
    return TestClient(create_app(ProxyConfig()))


@patch.object(admin_module, '_discover_models', return_value=['model-a'])
def test_add_success_returns_200(mock_discover_models):
    with _client() as client:
        response = client.post(
            '/nodes/add',
            json={'url': 'http://replica:8000', 'status': {'models': ['model-a'], 'role': 1}},
        )
        assert response.status_code == HTTPStatus.OK
        assert response.json() == {'message': 'Added successfully'}


@patch.object(admin_module, '_discover_models', side_effect=HTTPException(
    status_code=HTTPStatus.BAD_GATEWAY, detail='Failed to reach replica'))
def test_add_unreachable_returns_502(mock_discover_models):
    with _client() as client:
        response = client.post(
            '/nodes/add',
            json={'url': 'http://replica:8000', 'status': {'models': ['model-a'], 'role': 1}},
        )
        assert response.status_code == HTTPStatus.BAD_GATEWAY
        assert 'detail' in response.json()


@patch.object(admin_module, '_discover_models', return_value=['model-a'])
def test_add_model_mismatch_returns_400(mock_discover_models):
    with _client() as client:
        response = client.post(
            '/nodes/add',
            json={'url': 'http://replica:8000', 'status': {'models': ['other'], 'role': 1}},
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert 'detail' in response.json()


@patch('lmdeploy.serve.openai.api_client.get_model_list', return_value=None)
def test_add_empty_models_returns_502(mock_get_model_list):
    with _client() as client:
        response = client.post('/nodes/add', json={'url': 'http://replica:8000'})
        assert response.status_code == HTTPStatus.BAD_GATEWAY


def test_remove_missing_replica_returns_404():
    with _client() as client:
        response = client.post('/nodes/remove', json={'url': 'http://missing:1'})
        assert response.status_code == HTTPStatus.NOT_FOUND
        assert 'not in the pool' in response.json()['detail']


def test_remove_existing_replica_returns_200():
    with _client() as client:
        client.app.state.runtime.pool.add('http://replica:1', ReplicaLoad(models=['m1']))
        response = client.post('/nodes/remove', json={'url': 'http://replica:1'})
        assert response.status_code == HTTPStatus.OK
        assert response.json() == {'message': 'Deleted successfully'}


def test_terminate_missing_replica_returns_404():
    with _client() as client:
        response = client.post('/nodes/terminate', json={'url': 'http://missing:1'})
        assert response.status_code == HTTPStatus.NOT_FOUND


@patch.object(
    admin_module,
    '_request_replica_terminate',
    side_effect=HTTPException(status_code=HTTPStatus.BAD_GATEWAY, detail='Failed to terminate'),
)
def test_terminate_upstream_failure_removes_from_pool(mock_terminate):
    with _client() as client:
        client.app.state.runtime.pool.add('http://replica:1', ReplicaLoad(models=['m1']))
        response = client.post('/nodes/terminate', json={'url': 'http://replica:1'})
        assert response.status_code == HTTPStatus.BAD_GATEWAY
        assert 'http://replica:1' not in client.app.state.runtime.pool.snapshot()
