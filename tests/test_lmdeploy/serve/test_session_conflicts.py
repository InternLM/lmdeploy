# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import APIRouter, FastAPI

from lmdeploy.serve.core.async_engine import GenOut
from lmdeploy.serve.managers import SessionManager
from lmdeploy.serve.openai import chat_completions
from lmdeploy.serve.openai.api_server import ServerContext
from lmdeploy.serve.openai.endpoints import completions, generate
from lmdeploy.serve.openai.protocol import DeltaMessage

ENDPOINTS = ['/v1/chat/completions', '/v1/completions', '/generate']


class _ResponseParser:
    tool_parser_cls = None
    reasoning_parser_cls = None

    def __init__(self, request):
        self.request = request
        self.tool_parser = None
        self.reasoning_tokens = 0

    def stream_chunk(self, text, token_ids, **kwargs):
        return [(DeltaMessage(content=text), False)]

    def parse_complete(self, text, token_ids=None, **kwargs):
        return text, None, None

    def validate_complete(self, *args, **kwargs):
        return True


@pytest.fixture
def server():
    started, release = asyncio.Event(), asyncio.Event()
    release.set()

    async def preprocess(messages, session_id, **kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(session=session_id, inputs={'prompt': 'hello', 'input_ids': [1, 2]})

    async def generate_result(*args, **kwargs):
        yield GenOut(response='ok', input_token_len=2, generate_token_len=1,
                     finish_reason='stop', token_ids=[3])

    context = ServerContext()
    context.response_parser_cls = _ResponseParser
    context.async_engine = SimpleNamespace(
        epoch=0,
        model_name='test-model',
        backend_config=SimpleNamespace(adapters=[], logprobs_mode=None),
        tokenizer=SimpleNamespace(model=SimpleNamespace(model=None)),
        session_mgr=SessionManager(),
        preprocess=AsyncMock(side_effect=preprocess),
        generate=Mock(side_effect=generate_result),
    )
    app, router = FastAPI(), APIRouter()
    for module in (chat_completions, completions, generate):
        module.register(router, context)
    app.include_router(router)
    return app, context, started, release


def _payload(endpoint, session_id, stream):
    payload = {'session_id': session_id, 'stream': stream, 'max_tokens': 8}
    if endpoint != '/generate':
        payload['model'] = 'test-model'
    if endpoint == '/v1/chat/completions':
        payload['messages'] = [{'role': 'user', 'content': 'hello'}]
    else:
        payload['prompt'] = 'hello'
    return payload


def _assert_conflict(response, session_id):
    assert response.status_code == 409, response.text
    body = response.json()
    assert body['object'] == 'error'
    assert body['code'] == 409
    assert body['message'] == f'User session id {session_id} already exists'


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('session_id', [0, 42])
def test_duplicate_session_returns_409_without_changing_existing_request(endpoint, stream, session_id, server):
    async def run():
        app, context, _, _ = server
        manager = context.session_manager
        old = context.create_session(session_id)
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
            response = await client.post(endpoint, json=_payload(endpoint, session_id, stream))
            _assert_conflict(response, session_id)
            context.async_engine.preprocess.assert_not_awaited()
            context.async_engine.generate.assert_not_called()
            assert old.abort_version == 0
            assert manager.sessions == {old.session_id: old}
            assert manager.user_session_id_map == {session_id: old.session_id}
            assert manager.session_id_map == {old.session_id: session_id}

            manager.remove(old)
            response = await client.post(endpoint, json=_payload(endpoint, session_id, stream))
            assert response.status_code == 200, response.text
            assert manager.sessions == manager.user_session_id_map == manager.session_id_map == {}

    asyncio.run(run())


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('stream', [False, True])
def test_public_session_id_matching_another_internal_id_is_allowed(endpoint, stream, server):
    async def run():
        app, context, _, _ = server
        manager = context.session_manager
        old = context.create_session(42)
        assert old.session_id != 42
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
            response = await client.post(endpoint, json=_payload(endpoint, old.session_id, stream))
        assert response.status_code == 200, response.text
        assert manager.sessions == {old.session_id: old}
        assert manager.user_session_id_map == {42: old.session_id}
        assert manager.session_id_map == {old.session_id: 42}

    asyncio.run(run())


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('stream', [False, True])
def test_disconnect_during_preprocessing_keeps_id_reserved_until_cleanup(endpoint, stream, server):
    async def run():
        app, context, started, release = server
        manager = context.session_manager
        release.clear()
        disconnected = asyncio.Event()
        delivered = False
        body = json.dumps(_payload(endpoint, 42, True)).encode()

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            await disconnected.wait()
            return {'type': 'http.disconnect'}

        async def send(message):
            pass

        scope = {
            'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
            'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
            'path': endpoint, 'raw_path': endpoint.encode(), 'query_string': b'',
            'root_path': '', 'headers': [(b'content-type', b'application/json')],
            'server': ('testserver', 80), 'client': ('127.0.0.1', 1234),
        }
        old_request = asyncio.create_task(app(scope, receive, send))
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
            try:
                await asyncio.wait_for(started.wait(), 5)
                old = context.find_session(42)
                disconnected.set()
                # TCP disconnect does not cancel a handler still preprocessing.
                assert not old_request.done()
                response = await client.post(endpoint, json=_payload(endpoint, 42, stream))
                _assert_conflict(response, 42)
                assert context.async_engine.preprocess.await_count == 1
                assert context.find_session(42) is old
            finally:
                release.set()
                await asyncio.wait_for(old_request, 5)

            assert manager.sessions == manager.user_session_id_map == manager.session_id_map == {}
            response = await client.post(endpoint, json=_payload(endpoint, 42, stream))
            assert response.status_code == 200, response.text
            assert manager.sessions == manager.user_session_id_map == manager.session_id_map == {}

    asyncio.run(run())
