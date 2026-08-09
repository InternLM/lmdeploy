# Copyright (c) OpenMMLab. All rights reserved.
"""Shared fakes for ``/v1/chat/completions`` handler tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from lmdeploy.serve.openai.endpoints.chat_completions import register
from lmdeploy.serve.openai.protocol import DeltaMessage


class FakeTokenizer:
    model = SimpleNamespace(model='fake-tokenizer')


class FakeAsyncEngine:
    """Engine fake whose ``generate`` returns distinct outputs per call.

    Each call yields a ``GenOut``-like stream whose text encodes the call
    index, so fan-out tests can assert the N choices are distinct.
    """

    model_name = 'fake-model'
    backend_config = SimpleNamespace(adapters=[], logprobs_mode=None)

    def __init__(self):
        self.session_mgr = FakeSessionManager()
        self.tokenizer = SimpleNamespace(model=FakeTokenizer())
        self.call_count = 0
        self.gen_configs = []

    def generate(self, prompt, session, **kwargs):
        self.call_count += 1
        self.gen_configs.append(kwargs.get('gen_config'))
        call_index = self.call_count

        async def _generator():
            yield SimpleNamespace(
                response=f'choice-{call_index}',
                token_ids=[call_index],
                input_token_len=4,
                generate_token_len=call_index,
                finish_reason='stop',
                logprobs=None,
                cached_tokens=0,
                routed_experts=None,
                cache_block_ids=None,
            )

        return _generator()


class PassthroughResponseParser:
    """Stateful passthrough parser mirroring the real ResponseParser API."""

    tool_parser_cls = None

    def __init__(self, request):
        self.request = request
        self.tool_parser = None
        self._chunks = []

    def stream_chunk(self, delta_text, delta_token_ids, **kwargs):
        if not delta_text:
            return []
        return [(DeltaMessage(content=delta_text), False)]

    def parse_complete(self, text, token_ids=None, **kwargs):
        return text, None, None

    def validate_complete(self, raw_text=None):
        return True


class FakeSessionManager:

    def __init__(self):
        self.removed = []
        self._ids = set()

    def has(self, session_id):
        return session_id in self._ids

    def remove(self, session):
        self.removed.append(session)


class FakeSession:

    def __init__(self, session_id):
        self.session_id = session_id
        self.epoch = 0
        self.aborted = False

    async def async_abort(self):
        self.aborted = True


class FakeServerContext:
    response_parser_cls = PassthroughResponseParser

    def __init__(self):
        self.async_engine = FakeAsyncEngine()
        self.default_gen_config = {}

    @property
    def engine_config(self):
        return self.async_engine.backend_config

    @property
    def session_manager(self):
        return self.async_engine.session_mgr

    def create_session(self, session_id):
        return FakeSession(session_id)


class FakeRawRequest:

    def __init__(self, payload=None):
        self._payload = payload or {}

    async def json(self):
        return self._payload

    async def is_disconnected(self):
        return False


@pytest.fixture
def chat_endpoint():
    context = FakeServerContext()
    from fastapi import APIRouter
    r = APIRouter()
    register(r, context)
    return r.routes[0].endpoint, context


@pytest.fixture
def fake_raw_request():
    return FakeRawRequest()
