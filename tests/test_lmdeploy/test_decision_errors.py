# Copyright (c) OpenMMLab. All rights reserved.
"""Decision error classification through the production routes and
preprocess."""
from types import MethodType, SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from test_decision_service import MARKER, FakeEngine, config

from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.managers import SessionManager
from lmdeploy.serve.openai.api_server import ServerContext, validation_exception_handler
from lmdeploy.serve.openai.endpoints import create_openai_router
from lmdeploy.vl.media.base import MediaInputError

PRIVATE = 'private tokenizer implementation detail'


class ErrorEngine(FakeEngine):

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.session_mgr = SessionManager()
        if mode == 'missing_tokenizer':
            self.tokenizer = None
        if mode == 'tokenizer_error':
            def encode(*args, **kwargs):
                raise ValueError(PRIVATE)
            self.tokenizer = SimpleNamespace(encode=encode)
        if mode.startswith('shared_'):
            self.arch = 'Qwen3ForCausalLM'
            self.session_len = 512
            self.backend_config = SimpleNamespace(enable_prefix_caching=False)
            self.request_logger = SimpleNamespace(log_prompt=lambda *a, **kw: None,
                                                  log_inputs=lambda *a, **kw: None)
            self._determine_gen_config = lambda ids, gen_config=None: gen_config

            async def get_prompt_input(**kwargs):
                if mode == 'shared_missing_ids':
                    return {'prompt': 'rendered'}
                if mode == 'shared_media_error':
                    raise MediaInputError('Cannot read user-supplied /data/photo.png')
                errors = {'shared_value_error': ValueError, 'shared_type_error': TypeError,
                          'shared_runtime_error': RuntimeError}
                raise errors[mode](PRIVATE)

            self.prompt_processor = SimpleNamespace(get_prompt_input=get_prompt_input)
            self.preprocess = MethodType(AsyncEngine.preprocess, self)

    async def preprocess(self, *args, **kwargs):
        if self.mode == 'direct_value_error':
            raise ValueError(PRIVATE)
        if self.mode == 'direct_type_error':
            raise TypeError(PRIVATE)
        if self.mode == 'context_length':
            raise RequestError(ErrorCode.CONTEXT_LENGTH_EXCEEDED)
        if self.mode == 'unavailable':
            raise RequestError(ErrorCode.ENGINE_UNAVAILABLE)
        prepared = await super().preprocess(*args, **kwargs)
        if self.mode == 'missing_marker':
            prepared.inputs['input_ids'][:] = [token for token in prepared.inputs['input_ids'] if token != MARKER]
        return prepared

    async def generate(self, *args, **kwargs):
        if self.mode == 'no_output':
            return
        if self.mode == 'generation_value_error':
            raise ValueError(PRIVATE)
        async for result in super().generate(*args, **kwargs):
            if self.mode == 'aborted':
                result.finish_reason = 'abort'
            yield result


@pytest.fixture(params=['/v1/systemone', '/v1/decisions'])
def endpoint(request):
    route = request.param
    if route == '/v1/systemone':
        payload = dict(model='intern-decision', state='text', questions={
            'q': dict(type='noul', instructions='Refund?'),
        })
    else:
        payload = dict(model='intern-decision', input='text', questions=[
            dict(type='predicate', instructions='Refund?'),
        ])
    return route, payload


def evaluate(endpoint, engine):
    route, payload = endpoint
    context = ServerContext()
    context.async_engine = engine
    context.decision_context = config()
    app = FastAPI()
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.include_router(create_openai_router(context))
    with TestClient(app) as client:
        response = client.post(route, json=payload)
    assert not engine.session_mgr.sessions
    return response


@pytest.mark.parametrize('mode', [
    'missing_tokenizer', 'tokenizer_error', 'missing_marker', 'direct_value_error', 'direct_type_error',
    'shared_value_error', 'shared_type_error', 'shared_runtime_error', 'shared_missing_ids',
    'no_output', 'generation_value_error',
])
def test_internal_failures_return_500_without_details(endpoint, mode):
    response = evaluate(endpoint, ErrorEngine(mode))
    assert response.status_code == 500, response.text
    assert PRIVATE not in response.text


@pytest.mark.parametrize(('mode', 'status'), [
    ('context_length', 400), ('unavailable', 503), ('aborted', 503), ('shared_media_error', 400),
])
def test_explicit_input_and_availability_errors_keep_their_status(endpoint, mode, status):
    response = evaluate(endpoint, ErrorEngine(mode))
    assert response.status_code == status, response.text


def test_reserved_marker_remains_an_input_error(endpoint):
    route, payload = endpoint
    payload['state' if route == '/v1/systemone' else 'input'] = '<decision>'
    response = evaluate(endpoint, ErrorEngine('normal'))
    assert response.status_code == (422 if route == '/v1/systemone' else 400)
    assert 'Reserved' in response.text
