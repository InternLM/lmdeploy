# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import pytest

from lmdeploy.serve.openai.protocol import ErrorResponse
from lmdeploy.serve.proxy.dispatch.base import response_from_api_exception, safe_json_load
from lmdeploy.serve.proxy.dispatch.hybrid import HybridDispatcher
from lmdeploy.serve.proxy.upstream.exceptions import APIServerException


def test_safe_json_load_invalid_raises():
    with pytest.raises(APIServerException) as exc_info:
        safe_json_load('http://127.0.0.1:19020', 'not-json')
    assert exc_info.value.status_code == HTTPStatus.BAD_GATEWAY


def test_hybrid_dispatch_upstream_error():
    selector = MagicMock()
    selector.select.return_value = 'http://127.0.0.1:19020'
    tracker = MagicMock()
    tracker.start.return_value = object()
    body = ErrorResponse(message='bad gateway', type='server_error', code=502).model_dump_json().encode()
    forwarder = MagicMock()
    forwarder.forward_raw_buffer = AsyncMock(side_effect=APIServerException(status_code=502, body=body))
    dispatcher = HybridDispatcher(selector=selector, forwarder=forwarder, tracker=tracker)
    ctx = MagicMock(model='llama', stream=False, raw_request=MagicMock(), endpoint='/v1/chat/completions')
    resp = asyncio.run(dispatcher.dispatch(ctx))
    assert resp.status_code == 502
    assert json.loads(resp.body)['message'] == 'bad gateway'
    tracker.finish.assert_called_once()


def test_response_from_api_exception_openai_body():
    body = ErrorResponse(message='upstream failed', type='server_error', code=502).model_dump_json().encode()
    exc = APIServerException(status_code=502, body=body)
    resp = response_from_api_exception(exc)
    assert resp.status_code == 502
    assert json.loads(resp.body)['message'] == 'upstream failed'
