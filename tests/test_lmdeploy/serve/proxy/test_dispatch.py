# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json
import time
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import pytest

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.openai.protocol import ErrorResponse
from lmdeploy.serve.proxy.core.config import ProxyConfig, RoutingStrategy
from lmdeploy.serve.proxy.core.replica import ReplicaLoad, SelectedReplica
from lmdeploy.serve.proxy.dispatch.base import response_from_api_exception, safe_json_load
from lmdeploy.serve.proxy.dispatch.distserve import DistServeDispatcher
from lmdeploy.serve.proxy.dispatch.hybrid import HybridDispatcher
from lmdeploy.serve.proxy.metrics.load_tracker import InflightTracker
from lmdeploy.serve.proxy.registry.pool import ReplicaPool
from lmdeploy.serve.proxy.routing.selector import ReplicaSelector
from lmdeploy.serve.proxy.upstream.exceptions import APIServerException


def test_safe_json_load_invalid_raises():
    with pytest.raises(APIServerException) as exc_info:
        safe_json_load('http://127.0.0.1:19020', 'not-json')
    assert exc_info.value.status_code == HTTPStatus.BAD_GATEWAY


def test_hybrid_dispatch_upstream_error():
    selected = SelectedReplica(url='http://127.0.0.1:19020', start_time=time.time())
    selector = MagicMock()
    selector.acquire.return_value = selected
    tracker = MagicMock()
    body = ErrorResponse(message='bad gateway', type='server_error', code=502).model_dump_json().encode()
    forwarder = MagicMock()
    forwarder.forward_raw_buffer = AsyncMock(side_effect=APIServerException(status_code=502, body=body))
    dispatcher = HybridDispatcher(selector=selector, forwarder=forwarder, tracker=tracker)
    ctx = MagicMock(model='llama', stream=False, raw_request=MagicMock(), endpoint='/v1/chat/completions')
    resp = asyncio.run(dispatcher.dispatch(ctx))
    assert resp.status_code == 502
    assert json.loads(resp.body)['message'] == 'bad gateway'
    tracker.finish.assert_called_once_with(selected)


def test_distserve_setup_failure_releases_decode_reservation():
    pd_connection_pool = MagicMock()
    pd_connection_pool.is_connected.return_value = False
    pd_connection_pool.connect = AsyncMock(side_effect=TimeoutError)
    pool = ReplicaPool(pd_connection_pool)
    pool.add('http://prefill', ReplicaLoad(role=EngineRole.Prefill, models=['llama']))
    pool.add('http://decode', ReplicaLoad(role=EngineRole.Decode, models=['llama']))
    selector = ReplicaSelector(pool, RoutingStrategy.MIN_EXPECTED_LATENCY)
    forwarder = MagicMock()
    forwarder.forward_json_buffer = AsyncMock(
        return_value='{"id": 1, "cache_block_ids": [2], "remote_token_ids": [3]}')
    dispatcher = DistServeDispatcher(
        config=ProxyConfig(),
        pool=pool,
        selector=selector,
        forwarder=forwarder,
        tracker=InflightTracker(pool),
    )
    ctx = MagicMock(
        model='llama',
        stream=False,
        raw_request=MagicMock(),
        endpoint='/v1/chat/completions',
        request_dict={},
    )

    with pytest.raises(TimeoutError):
        asyncio.run(dispatcher.dispatch(ctx))

    assert pool.snapshot()['http://decode'].unfinished == 0


def test_response_from_api_exception_openai_body():
    body = ErrorResponse(message='upstream failed', type='server_error', code=502).model_dump_json().encode()
    exc = APIServerException(status_code=502, body=body)
    resp = response_from_api_exception(exc)
    assert resp.status_code == 502
    assert json.loads(resp.body)['message'] == 'upstream failed'
