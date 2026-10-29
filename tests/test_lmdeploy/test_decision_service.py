# Copyright (c) OpenMMLab. All rights reserved.
"""HTTP service entry points for Intern-Decision."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from PIL import Image

from lmdeploy.intern_decision_encoding import DEFAULT_TEMPERATURE
from lmdeploy.serve.core.async_engine import GenOut
from lmdeploy.serve.parsers.decision_parser import DecisionContext, DecisionParser

QUESTION = {'q': {'type': 'noul'}}
MARKER, IMAGE_PAD = 900, 800
SYMBOLS = {'A': 32, 'B': 33, 'C': 34}


def config(**kwargs):
    values = dict(marker_id=MARKER, symbol_ids=SYMBOLS, image_token_id=IMAGE_PAD,
                  temperature=DEFAULT_TEMPERATURE)
    return DecisionContext(**{**values, **kwargs})


class FakeSessionManager:

    def __init__(self):
        self.sessions = {}
        self.count = 0

    def get(self):
        self.count += 1
        session = SimpleNamespace(session_id=self.count, epoch=None)
        self.sessions[session.session_id] = session
        return session

    def remove(self, session):
        self.sessions.pop(session.session_id, None)


class FakeEngine:
    epoch = 0
    model_name = 'intern-decision'

    def __init__(self, *, expand=4):
        self.session_mgr = FakeSessionManager()
        self.requests = []
        self.expand = expand

    async def preprocess(self, messages, session, *, gen_config, input_ids=None, **kwargs):
        assert input_ids is None and kwargs['do_preprocess']
        content = messages[1]['content']
        images = sum(part['type'] == 'image_url' for part in content) if isinstance(content, list) else 0
        if images:
            assert all(not isinstance(part.get('image_url', {}).get('url'), Image.Image) for part in content)
        fields = messages[-1]['content'].count('<decision>')
        ids = [1] + [IMAGE_PAD] * (self.expand * images) + [2, 3]
        for _ in range(fields):
            ids.extend([2, MARKER, 3])
        prepared = SimpleNamespace(session=session, inputs={'input_ids': ids}, input_token_len=len(ids),
                                   gen_config=gen_config)
        self.requests.append(prepared)
        return prepared

    async def generate(self, request, stream_response=True):
        assert stream_response is False
        targets = request.inputs['input_ids'][request.gen_config.logprob_start_len + 1:]
        rows = [{token: float(token) for token in request.gen_config.logprob_extra_token_ids} for _ in targets]
        yield GenOut(response='', token_ids=[], input_token_len=request.input_token_len, generate_token_len=0,
                     logprobs=rows, finish_reason='length')


@pytest.mark.parametrize('image_count', [0, 2])
def test_native_http_route(monkeypatch, image_count):
    from lmdeploy.serve.openai.api_server import ServerContext, validation_exception_handler
    from lmdeploy.serve.openai.endpoints import create_openai_router
    context = ServerContext()
    app = FastAPI()
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.include_router(create_openai_router(context))
    body = {
        'state': 1,
        'questions': QUESTION,
        'images': [f'/server/image-{i}.png' for i in range(image_count)],
    }
    with TestClient(app) as client:
        assert client.post('/v1/jev', json=body).status_code == 503
        context.async_engine = FakeEngine()
        context.decision_context = config()
        observed = []
        sessions = []
        original_create_session = context.create_session

        def create_session(session_id=None):
            session = original_create_session(session_id)
            sessions.append(session)
            return session

        monkeypatch.setattr(context, 'create_session', create_session)
        original = DecisionParser.prepare_input

        def prepare(parser):
            observed.append(parser.request.state)
            return original(parser)

        monkeypatch.setattr(DecisionParser, 'prepare_input', prepare)
        for _ in range(2):
            assert client.post('/v1/jev', json=body).status_code == 200
        assert client.post('/v1/jev', json={**body, 'model': 'unknown'}).status_code == 400
        assert client.post('/v1/jev', json={'state': 1, 'questions': {}}).status_code == 400
        for images in (123, [123], None):
            response = client.post('/v1/jev', json={**body, 'images': images})
            assert response.status_code == 400
            assert response.json()['object'] == 'error'
            assert 'images' in response.json()['message']
        assert observed == [1, 1]
        assert len(sessions) == 2 and sessions[0] is not sessions[1]
        assert all(session.epoch == context.async_engine.epoch for session in sessions)
        assert not context.session_manager.sessions


def test_text_engine_rejects_images_through_shared_processor(monkeypatch):
    from lmdeploy.serve.openai.api_server import ServerContext
    from lmdeploy.serve.openai.endpoints import create_openai_router
    from lmdeploy.serve.processors.multimodal import MultimodalProcessor

    context = ServerContext()
    context.async_engine = FakeEngine()
    context.decision_context = config()
    processor = MultimodalProcessor(tokenizer=None, chat_template=None, vl_encoder=None)

    async def preprocess(messages, session, *, gen_config, **kwargs):
        return await processor.get_prompt_input(messages, **kwargs)

    monkeypatch.setattr(context.async_engine, 'preprocess', preprocess)
    app = FastAPI()
    app.include_router(create_openai_router(context))
    with TestClient(app) as client:
        requests = [
            ('/v1/jev', dict(state='test', questions=QUESTION, images=['/image.png'])),
            ('/v1/decisions', dict(model=context.async_engine.model_name, input=[dict(role='user', content=[
                dict(type='input_image', image_url='https://example.com/image.png'),
            ])], questions=[dict(type='predicate', instructions='Check the image.')])),
        ]
        for route, body in requests:
            response = client.post(route, json=body)
            assert response.status_code == 400
            assert 'Multimodal input is not supported' in response.json()['message']
    assert not context.session_manager.sessions
