# Copyright (c) OpenMMLab. All rights reserved.
"""Shared fakes for ``/v1/chat/completions`` handler tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from lmdeploy.serve.openai.chat_completions import register
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

    async def preprocess(self, prompt, session, **kwargs):
        """Return the minimal preprocessed input consumed by the fake."""
        self.gen_configs.append(kwargs.get('gen_config'))
        return SimpleNamespace(prompt=prompt, session=session)

    def generate(self, preprocessed, **kwargs):
        self.call_count += 1
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
        self.reasoning_tokens = 0

    def stream_chunk(self, delta_text, delta_token_ids, **kwargs):
        if not delta_text:
            return []
        return [(DeltaMessage(content=delta_text), False)]

    def parse_complete(self, text, token_ids=None, **kwargs):
        return text, None, None

    def validate_complete(self, raw_text=None):
        return True


class FakeSessionManager:
    """Mimics the real SessionManager's id/mapping semantics closely enough
    to surface fan-out session bugs: explicit user_session_ids are mapped
    one-to-one and a duplicate raises (like map_user_session_id), while
    None/-1 auto-generates a fresh internal id."""

    def __init__(self):
        self.removed = []
        self.sessions = {}
        self.user_session_id_map = {}
        self._next_id = 0

    def map_user_session_id(self, user_session_id):
        if user_session_id in self.user_session_id_map:
            raise ValueError(
                f'User session id {user_session_id} already exists')
        session_id = self._next_id
        self._next_id += 1
        self.user_session_id_map[user_session_id] = session_id
        return session_id

    def get(self, session_id=None, create_if_not_exists=True, **kwargs):
        if not create_if_not_exists:
            return self.sessions.get(session_id, None)
        if session_id is None:
            session_id = self._next_id
            self._next_id += 1
        if session_id in self.sessions:
            return self.sessions[session_id]
        session = FakeSession(session_id)
        self.sessions[session_id] = session
        return session

    def has(self, session_id):
        return session_id in self.sessions

    def remove(self, session):
        if session is None:
            return
        session_id = (session if isinstance(session, int)
                      else session.session_id)
        self.sessions.pop(session_id, None)
        # also drop any user mapping pointing at this session_id
        for uid, sid in list(self.user_session_id_map.items()):
            if sid == session_id:
                self.user_session_id_map.pop(uid, None)
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

    def create_session(self, user_session_id):
        # Mirror ServerContext.create_session: None/-1 auto-generates; an
        # explicit id maps one-to-one and collides on a second use.
        if user_session_id is None or user_session_id == -1:
            session = self.session_manager.get()
        else:
            session_id = self.session_manager.map_user_session_id(
                user_session_id)
            session = self.session_manager.get(session_id)
        session.epoch = 0
        return session


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
