# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lmdeploy.serve.openai.protocol import DeltaMessage
from lmdeploy.serve.openai.responses import create_responses_router


class FakeAsyncEngine:

    model_name = 'fake-model'
    backend_config = SimpleNamespace(adapters=[])
    tokenizer = SimpleNamespace(model=SimpleNamespace(model=None))

    def __init__(self):
        self.generate_kwargs = None
        self.prompt = None

    def generate(self, prompt, session, **kwargs):
        self.prompt = prompt
        self.generate_kwargs = kwargs

        async def _generator():
            yield SimpleNamespace(
                response='ok',
                token_ids=[1],
                input_token_len=1,
                generate_token_len=1,
                finish_reason='stop',
            )

        return _generator()


class PassthroughResponseParser:

    tool_parser_cls = None
    last_request = None

    def __init__(self, request, tokenizer=None):
        self.request = request
        type(self).last_request = request

    def stream_chunk(self, delta_text: str, delta_token_ids: list[int],
                     **kwargs):
        return [(DeltaMessage(content=delta_text),
                 False)] if delta_text else []

    def parse_complete(self,
                       text: str,
                       token_ids: list[int] | None = None,
                       **kwargs):
        return text, None, None


class FakeServerContext:

    response_parser_cls = PassthroughResponseParser

    def __init__(self):
        self.async_engine = FakeAsyncEngine()

    def create_session(self, session_id):
        return FakeSession(session_id)


class FakeSession:

    def __init__(self, session_id):
        self.session_id = session_id

    async def async_abort(self):
        pass


class FakeRawRequest:

    async def is_disconnected(self):
        return False


@pytest.fixture
def fake_raw_request():
    return FakeRawRequest()


@pytest.fixture
def passthrough_response_parser_cls():
    return PassthroughResponseParser


@pytest.fixture
def responses_endpoint():
    context = FakeServerContext()
    router = create_responses_router(context)
    return router.routes[0].endpoint, context


@pytest.fixture
def sse_payloads():

    def _sse_payloads(events: list[str]):
        payloads = []
        for event in events:
            for line in event.splitlines():
                if line.startswith('data: '):
                    payloads.append(json.loads(line.removeprefix('data: ')))
        return payloads

    return _sse_payloads
