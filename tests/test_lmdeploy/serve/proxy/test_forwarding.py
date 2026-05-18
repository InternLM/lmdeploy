# Copyright (c) OpenMMLab. All rights reserved.

from unittest.mock import MagicMock


def test_prepare_headers():
    from lmdeploy.serve.proxy.forwarding import prepare_headers
    raw_request = MagicMock()
    raw_request.headers = {'host': 'original:8000', 'content-type': 'application/json', 'authorization': 'Bearer abc'}
    raw_request.client = MagicMock()
    raw_request.client.host = '10.0.0.1'
    raw_request.url.scheme = 'http'

    headers = prepare_headers(raw_request)
    assert 'host' not in headers
    assert headers['X-Forwarded-For'] == '10.0.0.1'
    assert headers['X-Forwarded-Host'] == 'original:8000'
    assert headers['X-Forwarded-Proto'] == 'http'
    assert headers['content-type'] == 'application/json'


def test_prepare_headers_no_client():
    from lmdeploy.serve.proxy.forwarding import prepare_headers
    raw_request = MagicMock()
    raw_request.headers = {'host': 'original:8000'}
    raw_request.client = None
    raw_request.url.scheme = 'https'

    headers = prepare_headers(raw_request)
    assert headers['X-Forwarded-For'] == 'unknown'
