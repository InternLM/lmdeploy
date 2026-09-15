# Copyright (c) OpenMMLab. All rights reserved.
"""Reject untrusted DistServe fields on the public OpenAI path."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
from lmdeploy.serve.openai.distserve_fields import (
    DISTSERVE_PROXY_HEADER,
    DISTSERVE_PROXY_HEADER_VALUE,
    is_trusted_distserve_replica_request,
    pop_distserve_request_fields,
)
from lmdeploy.serve.openai.endpoints.completions import register as register_completions
from lmdeploy.serve.openai.protocol import CompletionRequest

_POISON_MIGRATION = {
    'protocol': 3,
    'remote_engine_id': 'http://127.0.0.1:19001',
    'remote_session_id': 123456789,
    'remote_token_id': 1,
    'remote_block_ids': [],
    'is_dummy_prefill': False,
}


class _Session:

    async def async_abort(self):
        pass


class _SessionManager:

    def __init__(self):
        self.removed = []

    def has(self, session_id):
        return False

    def remove(self, session):
        self.removed.append(session)


class _RawRequest:

    def __init__(self, payload=None, headers=None):
        self._payload = payload or {}
        self.headers = headers or {}

    async def json(self):
        return dict(self._payload)

    async def is_disconnected(self):
        return False


class _AsyncEngine:
    model_name = 'fake-model'

    def __init__(self):
        self.generate_calls = 0
        self.gen_configs = []

    async def preprocess(self, prompt, session, **kwargs):
        self.gen_configs.append(kwargs.get('gen_config'))
        return SimpleNamespace(inputs={'input_ids': [1]}, input_token_len=1)

    def generate(self, request, **kwargs):
        self.generate_calls += 1

        async def _gen():
            yield SimpleNamespace(
                response='ok',
                token_ids=[1],
                input_token_len=1,
                generate_token_len=1,
                finish_reason='stop',
                logprobs=None,
                cache_block_ids=None,
            )

        return _gen()


class _ServerContext:

    def __init__(self, role=None):
        self.async_engine = _AsyncEngine()
        self.engine_config = SimpleNamespace(logprobs_mode=None, role=role)
        self.session_manager = _SessionManager()
        self.default_gen_config = {}
        self.sessions = []

    def create_session(self, session_id=None):
        session = _Session()
        self.sessions.append(session)
        return session


def _completion_endpoint(context):
    router = APIRouter()
    register_completions(router, context)
    return router.routes[0].endpoint


def _error_body(response):
    assert isinstance(response, JSONResponse)
    return json.loads(response.body)


def test_helper_rejects_untrusted_migration_request():
    fields, error = pop_distserve_request_fields(
        {'migration_request': _POISON_MIGRATION},
        _RawRequest(),
        SimpleNamespace(role=EngineRole.Decode),
    )
    assert error.status_code == 400
    assert 'DistServe-only' in _error_body(error)['message']
    assert fields.migration_request is None


def test_helper_accepts_decode_replica_marked_by_proxy():
    fields, error = pop_distserve_request_fields(
        {'migration_request': _POISON_MIGRATION},
        _RawRequest(headers={DISTSERVE_PROXY_HEADER: DISTSERVE_PROXY_HEADER_VALUE}),
        SimpleNamespace(role=EngineRole.Decode),
    )
    assert error is None
    assert isinstance(fields.migration_request, MigrationRequest)
    assert fields.migration_request.protocol == MigrationProtocol.NVLINK
    assert fields.migration_request.remote_block_ids == []


def test_helper_rejects_hybrid_even_with_proxy_header():
    _, error = pop_distserve_request_fields(
        {'with_cache': True},
        _RawRequest(headers={DISTSERVE_PROXY_HEADER: DISTSERVE_PROXY_HEADER_VALUE}),
        SimpleNamespace(role=EngineRole.Hybrid),
    )
    assert error.status_code == 400


def test_trusted_check_requires_prefill_or_decode_and_header():
    request = _RawRequest(headers={DISTSERVE_PROXY_HEADER: DISTSERVE_PROXY_HEADER_VALUE})
    assert is_trusted_distserve_replica_request(request, SimpleNamespace(role=EngineRole.Decode))
    assert is_trusted_distserve_replica_request(request, SimpleNamespace(role=EngineRole.Prefill))
    assert not is_trusted_distserve_replica_request(request, SimpleNamespace(role=EngineRole.Hybrid))
    assert not is_trusted_distserve_replica_request(_RawRequest(), SimpleNamespace(role=EngineRole.Decode))


def test_completions_rejects_untrusted_migration_request_without_calling_engine():
    async def _run():
        context = _ServerContext(role=EngineRole.Decode)
        response = await _completion_endpoint(context)(
            CompletionRequest(model='fake-model', prompt='hi', max_tokens=1),
            _RawRequest({'migration_request': _POISON_MIGRATION}),
        )
        return response, context

    response, context = asyncio.run(_run())
    body = _error_body(response)
    assert response.status_code == 400
    assert 'DistServe-only' in body['message']
    assert context.async_engine.generate_calls == 0


def test_completions_still_serves_after_untrusted_migration_request():
    async def _run():
        context = _ServerContext(role=EngineRole.Decode)
        endpoint = _completion_endpoint(context)
        rejected = await endpoint(
            CompletionRequest(model='fake-model', prompt='hi', max_tokens=1),
            _RawRequest({'migration_request': _POISON_MIGRATION}),
        )
        ok = await endpoint(
            CompletionRequest(model='fake-model', prompt='hi', max_tokens=1),
            _RawRequest(),
        )
        return rejected, ok, context

    rejected, ok, context = asyncio.run(_run())
    assert rejected.status_code == 400
    assert ok['object'] == 'text_completion'
    assert ok['choices'][0]['text'] == 'ok'
    assert context.async_engine.generate_calls == 1


def test_completions_proxy_header_forwards_migration_request():
    async def _run():
        context = _ServerContext(role=EngineRole.Decode)
        response = await _completion_endpoint(context)(
            CompletionRequest(model='fake-model', prompt='hi', max_tokens=1),
            _RawRequest(
                {'migration_request': _POISON_MIGRATION},
                headers={DISTSERVE_PROXY_HEADER: DISTSERVE_PROXY_HEADER_VALUE},
            ),
        )
        return response, context

    response, context = asyncio.run(_run())
    assert response['object'] == 'text_completion'
    gen_config = context.async_engine.gen_configs[0]
    assert gen_config.migration_request is not None
    assert gen_config.migration_request.remote_block_ids == []
