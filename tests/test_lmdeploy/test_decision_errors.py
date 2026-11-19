# Copyright (c) OpenMMLab. All rights reserved.
"""Decision error classification through the production routes and
preprocess."""
import json
from types import MethodType, SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from test_decision_service import IMAGE_PAD, MARKER, FakeEngine, config, image_asset

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
        if self.mode == 'unexpected_image_placeholder':
            prepared.inputs['input_ids'].insert(1, IMAGE_PAD)
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


def evaluate(endpoint, engine, *, decision_context=None):
    route, payload = endpoint
    context = ServerContext()
    context.async_engine = engine
    context.decision_context = decision_context or config()
    app = FastAPI()
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.include_router(create_openai_router(context))
    with TestClient(app) as client:
        response = client.post(route, json=payload)
    assert not engine.session_mgr.sessions
    return response


@pytest.mark.parametrize('mode', [
    'missing_tokenizer', 'tokenizer_error', 'missing_marker', 'unexpected_image_placeholder',
    'direct_value_error', 'direct_type_error',
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


@pytest.mark.parametrize('location', ['input', 'json_key', 'question_name', 'instructions',
                                      'text_part', 'choice_value', 'choice_description', 'score_label'])
@pytest.mark.parametrize('bad', ['\ud800', '\udfff'])
def test_invalid_unicode_is_rejected_before_preprocessing(endpoint, location, bad):
    route, payload = endpoint
    systemone = route == '/v1/systemone'
    q = payload['questions']['q'] if systemone else payload['questions'][0]
    if location == 'input':
        payload['state' if systemone else 'input'] = bad
    elif location == 'json_key':
        payload['state' if systemone else 'input'] = ({bad: 'text'} if systemone else [
            dict(role='user', content=[dict(type='input_json', data={bad: 'text'})])])
    elif location == 'question_name':
        if systemone:
            payload['questions'] = {bad: q}
        else:
            q['name'] = bad
    elif location in ('instructions', 'text_part'):
        q['instructions'] = bad if systemone or location == 'instructions' else [dict(type='input_text', text=bad)]
    elif location in ('choice_value', 'choice_description'):
        value, desc = (bad, 'A') if location == 'choice_value' else ('a', bad)
        if systemone:
            payload['questions']['q'] = dict(type='choice', instructions='Pick', criteria={value: desc, 'b': 'B'})
        else:
            payload['questions'][0] = dict(type='choice', instructions='Pick', choices=[
                dict(value=value, description=desc), dict(value='b')])
    elif systemone:
        payload['questions']['q'] = dict(type='score', instructions='Rate', criteria=[bad, 'B'])
    else:
        payload['questions'][0] = dict(type='score', instructions='Rate', levels=[dict(label=bad), dict(label='B')])

    engine = ErrorEngine('normal')
    context = ServerContext()
    context.async_engine = engine
    context.decision_context = config()
    app = FastAPI()
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.include_router(create_openai_router(context))
    with TestClient(app) as client:
        response = client.post(route, content=json.dumps(payload, ensure_ascii=True).encode(),
                               headers={'content-type': 'application/json'})
    assert response.status_code == 422, response.text
    assert 'valid Unicode' in response.text
    assert not engine.requests and not engine.session_mgr.sessions


def test_valid_supplementary_unicode_is_accepted(endpoint):
    route, payload = endpoint
    payload['state' if route == '/v1/systemone' else 'input'] = 'hello 😀'
    response = evaluate(endpoint, ErrorEngine('normal'))
    assert response.status_code == 200, response.text


@pytest.mark.parametrize('location', [
    'evidence', 'evidence_key', 'question_name', 'instructions', 'split_instructions',
    'choice_value', 'choice_description', 'score_label',
])
def test_user_image_marker_is_an_input_error(endpoint, location):
    route, payload = endpoint
    systemone = route == '/v1/systemone'
    marker = '<|image_pad|>'
    if location == 'evidence':
        payload['state' if systemone else 'input'] = f'Log: {marker}'
    elif location == 'evidence_key':
        payload['state' if systemone else 'input'] = ({marker: 'text'} if systemone else [
            dict(role='user', content=[dict(type='input_json', data={marker: 'text'})])])
    elif location == 'question_name':
        if systemone:
            payload['questions'] = {marker: payload['questions']['q']}
        else:
            payload['questions'][0]['name'] = marker
    elif location in ('instructions', 'split_instructions'):
        question = payload['questions']['q'] if systemone else payload['questions'][0]
        question['instructions'] = (marker if systemone or location == 'instructions' else [
            dict(type='input_text', text='<|image_'), dict(type='input_text', text='pad|>')])
    elif location in ('choice_value', 'choice_description'):
        if systemone:
            payload['questions']['q'] = dict(type='choice', instructions='Pick one', criteria={
                marker if location == 'choice_value' else 'a': marker if location == 'choice_description' else 'A',
                'b': 'B'})
        else:
            payload['questions'][0] = dict(type='choice', instructions='Pick one', choices=[
                dict(value=marker if location == 'choice_value' else 'a',
                     description=marker if location == 'choice_description' else 'A'), dict(value='b')])
    else:
        if systemone:
            payload['questions']['q'] = dict(type='score', instructions='Rate', criteria=[marker, 'B'])
        else:
            payload['questions'][0] = dict(type='score', instructions='Rate', levels=[
                dict(label=marker), dict(label='B')])
    engine = ErrorEngine('normal')
    response = evaluate(endpoint, engine)
    assert response.status_code == 400, response.text
    assert 'Reserved image marker' in response.text
    assert not engine.requests


def test_user_image_marker_with_an_actual_image_is_rejected(endpoint):
    route, payload = endpoint
    asset = image_asset()
    if route == '/v1/systemone':
        payload.update(state=['photo', '<|image_pad|>'], assets={'photo': asset})
    else:
        payload['input'] = [dict(role='user', content=[
            dict(type='input_image', image_url='data:image/png;base64,' + asset['data_base64']),
            dict(type='input_text', text='<|image_pad|>')])]
    engine = ErrorEngine('normal')
    response = evaluate(endpoint, engine)
    assert response.status_code == 400, response.text
    assert not engine.requests


def test_image_marker_follows_checkpoint_metadata(endpoint):
    route, payload = endpoint
    payload['state' if route == '/v1/systemone' else 'input'] = '<custom-image>'
    response = evaluate(endpoint, ErrorEngine('normal'), decision_context=config(image_token='<custom-image>'))
    assert response.status_code == 400, response.text


def test_marker_in_unused_metadata_is_ignored(endpoint):
    route, payload = endpoint
    payload['ignored' if route == '/v1/systemone' else 'safety_identifier'] = '<|image_pad|>'
    response = evaluate(endpoint, ErrorEngine('normal'))
    assert response.status_code == 200, response.text


def test_image_marker_used_as_an_asset_reference():
    endpoint = ('/v1/systemone', dict(state='<|image_pad|>',
                                     questions={'q': dict(type='noul', instructions='Check')},
                                     assets={'<|image_pad|>': image_asset()}))
    response = evaluate(endpoint, ErrorEngine('normal'))
    assert response.status_code == 200, response.text
