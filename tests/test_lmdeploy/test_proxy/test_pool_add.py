# Copyright (c) OpenMMLab. All rights reserved.

from unittest.mock import patch

import requests

from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.serve.proxy.core.errors import ErrorCodes
from lmdeploy.serve.proxy.core.replica import ReplicaLoad
from lmdeploy.serve.proxy.registry.pool import ReplicaPool


@patch('lmdeploy.serve.openai.api_client.get_model_list', return_value=['model-a'])
def test_add_registers_when_v1_models_ok(mock_get_model_list):
    pool = ReplicaPool(PDConnectionPool())
    assert pool.add('http://replica:8000', ReplicaLoad(models=['model-a'])) is None
    assert pool.snapshot()['http://replica:8000'].models == ['model-a']
    mock_get_model_list.assert_called_once()
    assert mock_get_model_list.call_args[0][0] == 'http://replica:8000/v1/models'


@patch('lmdeploy.serve.openai.api_client.get_model_list', return_value=None)
def test_add_fails_when_v1_models_empty(mock_get_model_list):
    pool = ReplicaPool(PDConnectionPool())
    res = pool.add('http://replica:8000', ReplicaLoad(models=['model-a']))
    assert res is not None
    assert str(ErrorCodes.API_TIMEOUT.value).encode() in res
    assert pool.snapshot() == {}


@patch('lmdeploy.serve.openai.api_client.get_model_list', return_value=['model-a'])
def test_add_fails_when_declared_models_not_on_server(mock_get_model_list):
    pool = ReplicaPool(PDConnectionPool())
    res = pool.add('http://replica:8000', ReplicaLoad(models=['other-model']))
    assert res is not None
    assert str(ErrorCodes.SERVICE_UNAVAILABLE.value).encode() in res
    assert pool.snapshot() == {}


@patch('lmdeploy.serve.openai.api_client.get_model_list', side_effect=requests.ConnectionError('refused'))
def test_add_fails_on_request_error(mock_get_model_list):
    pool = ReplicaPool(PDConnectionPool())
    res = pool.add('http://replica:8000')
    assert res is not None
    assert str(ErrorCodes.API_TIMEOUT.value).encode() in res
    assert pool.snapshot() == {}
