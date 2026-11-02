# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise the production wire models, parser and HTTP routes with fixed engine scores."""
import asyncio
import base64
import copy
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from PIL import Image
from test_decision_service import MARKER, FakeEngine, config

from lmdeploy.intern_decision_encoding import ANSWER_SYMBOLS, compile_row
from lmdeploy.serve.core.async_engine import GenOut
from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.api_server import ServerContext, validation_exception_handler
from lmdeploy.serve.openai.decision_protocol import OpenAIDecisionRequest
from lmdeploy.serve.openai.endpoints import create_openai_router
from lmdeploy.serve.parsers.decision_parser import DecisionParser
from lmdeploy.serve.processors.multimodal import MultimodalProcessor


def request(questions=None, **kwargs):
    return dict(model='intern-decision', input='The export failed.', questions=questions or [
        dict(type='predicate', name='blocked', instructions='Is work blocked?'),
        dict(type='choice', name='team', instructions='Who handles this?', choices=[
            dict(value='support', description='Help'), dict(value='billing', description='Payments')]),
        dict(type='score', name='severity', instructions='How severe?', levels=[
            dict(label='Low'), dict(label='Medium', description='A workaround exists'), dict(label='High')]),
    ], **kwargs)


def prepared_output(parser, probabilities):
    """Substitute only tokenizer/engine work; production compilation and row selection run."""
    inputs = parser.prepare_input(tokenizer=FakeEngine().tokenizer)
    prepared = SimpleNamespace(inputs={'input_ids': [1] + [MARKER, 3] * len(probabilities)},
                               gen_config=inputs.gen_config)
    parser.finalize_input(prepared)
    rows = []
    for distribution in probabilities:
        rows.append({32 + i: math.log(p) for i, p in enumerate(distribution)})
        rows.append({32: -1000., 33: 1000., 34: -1000.})
    return GenOut(response='', input_token_len=37, generate_token_len=0, cached_tokens=11,
                  logprobs=rows, finish_reason='length')


def test_three_types_use_existing_compiler_and_decoder():
    body = request()
    original = copy.deepcopy(body)
    parser = DecisionParser(body, config(temperature=1.))
    legacy = dict(state=body['input'], questions={
        'blocked': dict(type='noul', instructions='Is work blocked?'),
        'team': dict(type='choice', instructions='Who handles this?',
                     criteria={'support': 'Help', 'billing': 'Payments'}),
        'severity': dict(type='score', instructions='How severe?',
                         criteria=['Low', 'Medium: A workaround exists', 'High']),
    })
    assert parser.prepare_input(tokenizer=FakeEngine().tokenizer).messages == compile_row(legacy).messages
    output = prepared_output(parser, [[.8, .2], [.7, .3], [.2, .3, .5]])
    result = parser.parse_output(output, model_name='intern-decision')
    assert body == original
    predicate, choice, score = result['answers']
    assert [a['name'] for a in result['answers']] == ['blocked', 'team', 'severity']
    assert predicate['probability'] == pytest.approx(.2)  # P(yes), not max probability (.8).
    assert choice['choice'] == 'support' and choice['confidence'] == pytest.approx(.7)
    assert [p['value'] for p in choice['probabilities']] == ['support', 'billing']
    assert score['score'] == pytest.approx(1.3)
    assert [p['value'] for p in score['probabilities']] == [0, 1, 2]
    assert [p['label'] for p in score['probabilities']] == ['Low', 'Medium', 'High']
    output_tokens = len(compile_row(legacy).messages[-1]['content'].encode('utf-8'))
    assert result['usage'] == dict(input_tokens=37, output_tokens=output_tokens, total_tokens=37 + output_tokens,
                                  input_tokens_details={'cached_tokens': 11, 'cache_write_tokens': 0},
                                  output_tokens_details={'reasoning_tokens': 0})


def test_typed_choices_and_interleaved_parsers():
    typed = request([dict(type='choice', name='typed', instructions='Choose',
                          choices=[dict(value=True), dict(value='true')])])
    other = request([dict(type='choice', name='other', instructions='Choose',
                          choices=[dict(value='first'), dict(value='second')])])
    first, second = (DecisionParser(body, config(temperature=1.)) for body in (typed, other))
    first_output = prepared_output(first, [[.8, .2]])
    second_output = prepared_output(second, [[.1, .9]])
    assert 'A = true:' in first.compiled.messages[1]['content']
    assert 'B = "true":' in first.compiled.messages[1]['content']
    second_result = second.parse_output(second_output, model_name='intern-decision')
    assert second_result['answers'][0]['choice'] == 'second'
    answer = first.parse_output(first_output, model_name='intern-decision')['answers'][0]
    assert answer['name'] == 'typed' and answer['choice'] is True
    assert [p['value'] for p in answer['probabilities']] == [True, 'true']
    # Reverse the winner through the real decoder, not a hand-built internal answer.
    answer = first.parse_output(prepared_output(first, [[.2, .8]]),
                                model_name='intern-decision')['answers'][0]
    assert answer['choice'] == 'true' and type(answer['choice']) is str


def test_names_preserve_order_without_collisions():
    names = [None, 'question_0', 'same', None, None, '_question_0']
    body = request([dict(type='predicate', instructions='Check', **({'name': name} if name is not None else {}))
                    for name in names])
    parser = DecisionParser(body, config(temperature=1.))
    distributions = [[1 - p, p] for p in (.1, .2, .3, .4, .5, .6)]
    answers = parser.parse_output(prepared_output(parser, distributions),
                                  model_name='intern-decision')['answers']
    assert len(parser.compiled.fields) == len(set(parser.compiled.fields)) == len(names)
    assert [a['name'] for a in answers] == names
    assert [a['probability'] for a in answers] == pytest.approx([.1, .2, .3, .4, .5, .6])
    assert 'question_0' in parser.compiled.fields and '_question_0' in parser.compiled.fields


@pytest.mark.parametrize('name', ['Intern-Decision-0.8B', 'Intern-Decision-2B', 'Intern-Decision-4B'])
def test_model_identity_comes_from_engine(http_service, name):
    client, context = http_service
    context.async_engine.model_name = name
    body = request([dict(type='predicate', instructions='Check')])
    for alias in (name, f'InternLM/{name}', f'internlm/{name.lower()}'):
        body['model'] = alias
        response = client.post('/v1/decisions', json=body)
        assert response.status_code == 200
        assert response.json()['model'] == name
    body['model'] = name
    # A directory basename is not an alias when the service has another name.
    context.async_engine.model_name = '/models/' + name
    assert client.post('/v1/decisions', json=body).status_code == 422
    body['model'] = context.async_engine.model_name
    response = client.post('/v1/decisions', json=body)
    assert response.status_code == 200
    assert response.json()['model'] == context.async_engine.model_name


@pytest.mark.parametrize('count', [10, 11, 62])
def test_score_level_extension_uses_all_candidates(count):
    levels = [dict(label=f'level-{i}') for i in range(count)]
    parser = DecisionParser(request([dict(type='score', instructions='Rate', levels=levels)]),
                            config(temperature=1., symbol_ids={s: 32 + i for i, s in enumerate(ANSWER_SYMBOLS)}))
    output = prepared_output(parser, [[1 / count] * count])
    answer = parser.parse_output(output, model_name='intern-decision')['answers'][0]
    assert answer['score'] == pytest.approx((count - 1) / 2)
    assert [p['label'] for p in answer['probabilities']] == [level['label'] for level in levels]
    assert [p['probability'] for p in answer['probabilities']] == pytest.approx([1 / count] * count)


@pytest.mark.parametrize('identifier', [None, '', 'user-42', '用' * 64])
def test_safety_identifier_is_metadata_only(http_service, identifier):
    client, _ = http_service
    before = client.post('/v1/decisions', json=request()).json()
    response = client.post('/v1/decisions', json=request(safety_identifier=identifier))
    assert response.status_code == 200
    assert response.json() == before
    plain = DecisionParser(request(), config()).prepare_input(tokenizer=FakeEngine().tokenizer)
    tagged = DecisionParser(request(safety_identifier=identifier), config()).prepare_input(
        tokenizer=FakeEngine().tokenizer)
    assert tagged.messages == plain.messages


@pytest.mark.parametrize(('change', 'field'), [
    ({'safety_identifier': 'x' * 65}, 'safety_identifier'),
    ({'safety_identifier': 42}, 'safety_identifier'),
    ({'input': ''}, 'input'),
    ({'model': ''}, 'model'),
    ({'questions': [dict(type='predicate', name=' \t', instructions='Check')]}, 'questions.0.name'),
    ({'questions': [dict(type='predicate', name='duplicate', instructions='Check')] * 2}, 'questions.1.name'),
    ({'questions': [dict(type='choice', instructions='Choose', choices=[dict(value=' '), dict(value=False)])]},
     'questions.0.choices.0.value'),
    ({'questions': [dict(type='unknown', instructions='Check')]}, 'questions.0.type'),
    ({'input': [dict(role='user', content=[dict(type='input_image', image_url='file:///private')])]},
     'input.0.content.0.image_url'),
    ({'input': [dict(role='user', content=[dict(type='unknown')])]}, 'input.0.content.0.type'),
    ({'questions': [dict(type='predicate', instructions='Check', choice='extra')]}, 'questions.0.choice'),
])
def test_validation_details_are_field_specific(http_service, change, field):
    client, context = http_service
    response = client.post('/v1/decisions', json={**request(), **change})
    assert response.status_code == 422, response.text
    result = response.json()
    assert field in [detail['field'] for detail in result['error']['details']]
    assert all(detail['loc'][0] == 'body' for detail in result['detail'])
    assert not context.async_engine.requests and not context.session_manager.sessions


def test_validation_preserves_whitespace_and_boolean_choices():
    body = request([dict(type='choice', name=' named ', instructions='',
                         choices=[dict(value=' value '), dict(value=False)])])
    parsed = OpenAIDecisionRequest.model_validate({**body, 'input': ' '})
    assert parsed.input == ' ' and parsed.questions[0].name == ' named '
    assert [choice.value for choice in parsed.questions[0].choices] == [' value ', False]


def test_decision_error_format_does_not_change_chat_validation(http_service):
    client, _ = http_service
    response = client.post('/v1/chat/completions', json={})
    assert response.status_code == 400
    assert 'detail' not in response.json()


@pytest.mark.parametrize('change', [
    {'input': []},
    {'input': 1}, {'model': None}, {'state': 'mixed protocols'}, {'images': []},
    {'prompt_cache_key': 'unsupported'}, {'questions': []},
    {'questions': [dict(type='predicate', instructions='x')] * 17},
    {'questions': [dict(type='predicate', instructions=[{'type': 'input_text', 'text': 'x'}])]},
    {'questions': [dict(type='predicate', instructions='x', unexpected=True)]},
    {'questions': [dict(type='choice', instructions='x', choices=[{'value': 'a'}])]},
    {'questions': [dict(type='choice', instructions='x', choices=[{'value': 'a'}, {'value': 'a'}])]},
    {'questions': [dict(type='choice', instructions='x', choices=[{'value': 1}, {'value': 2}])]},
    {'questions': [dict(type='choice', instructions='x', choices=[{'value': str(i)} for i in range(63)])]},
    {'questions': [dict(type='score', instructions='x', levels=[{'label': 'one'}])]},
    {'questions': [dict(type='score', instructions='x', levels=[{'label': str(i)} for i in range(63)])]},
])
def test_reject_unsupported_input(change):
    with pytest.raises(ValueError):
        OpenAIDecisionRequest.model_validate({**request(), **change})


@pytest.mark.parametrize('value', [
    [{'role': 'assistant', 'content': 'x'}],
    [{'role': 'user', 'content': []}],
    [{'role': 'user', 'content': [{'type': 'input_audio', 'data': 'x'}]}],
    *[[{'role': 'user', 'content': [dict(type='input_image', **image)]}] for image in [
        {'image_url': '/server/private.png'}, {'image_url': 'file:///image.png'}, {'image_url': 'https://'},
        {'image_url': 'not-base64'}, {'image_url': 'https://example.com/a.png', 'detail': 'high'},
    ]],
])
def test_reject_unsupported_message_content(value):
    with pytest.raises(ValueError):
        OpenAIDecisionRequest.model_validate({**request(), 'input': value})


def test_single_string_message_preserves_text_prompt():
    body = request()
    plain = DecisionParser(body, config()).prepare_input(tokenizer=FakeEngine().tokenizer)
    body['input'] = [dict(type='message', role='user', content=body['input'])]
    assert DecisionParser(body, config()).prepare_input(tokenizer=FakeEngine().tokenizer).messages == plain.messages


@pytest.fixture
def image_input():
    def png(color):
        with Image.new('RGB', (28, 28), color) as image, BytesIO() as buffer:
            image.save(buffer, format='PNG')
            return buffer.getvalue()

    red = 'data:image/png;base64,' + base64.b64encode(png('red')).decode()
    blue, fetched = png('blue'), []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            fetched.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(blue if self.path == '/blue.png' else b'not an image')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/blue.png'
    value = [
        dict(role='user', content=[dict(type='input_text', text='Compare these images.'),
                                   dict(type='input_image', image_url=red, detail='auto')]),
        dict(type='message', role='user', content=[dict(type='input_image', image_url=url, detail=None),
              dict(type='input_text', text='The first image appears again.'),
              dict(type='input_image', image_url=red)]),
    ]
    try:
        yield value, [red, url, red], fetched
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_image_mapping_and_real_loader(http_service, image_input, monkeypatch):
    from lmdeploy.vl.media import connection

    value, images, fetched = image_input
    original_url_check = connection._is_safe_url
    # Allow only this fixture's loopback server; keep real HTTP transport and decoding.
    monkeypatch.setattr(connection, '_is_safe_url',
                        lambda url: (True, 'test server') if url.startswith(images[1]) else original_url_check(url))
    body = {**request(), 'input': value}
    original = copy.deepcopy(body)
    parser = DecisionParser(body, config())
    expected_state = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Compare these images.'},
                                     {'type': 'image', 'image_index': 1}]},
        {'role': 'user', 'content': [{'type': 'image', 'image_index': 2},
                                     {'type': 'text', 'text': 'The first image appears again.'},
                                     {'type': 'image', 'image_index': 3}]},
    ]
    assert parser.request.state == expected_state
    assert parser.request.images == images
    legacy = dict(state=expected_state, images=images, questions=parser.request.questions)
    messages = parser.prepare_input(tokenizer=FakeEngine().tokenizer).messages
    assert messages[1]['content'][-1] == compile_row(legacy).messages[1]['content'][-1]
    assert all(part['image_data']['data'].size == (256, 256) for part in messages[1]['content'][:-1])
    assert messages[1]['content'][0]['image_data']['data'] is messages[1]['content'][2]['image_data']['data']
    assert fetched == ['/blue.png'] and body == original
    fetched.clear()

    client, context = http_service
    preprocess = context.async_engine.preprocess
    colors = []

    async def with_loader(messages, *args, **kwargs):
        decoded = await MultimodalProcessor.async_parse_multimodal_item(messages)
        for part in decoded[1]['content']:
            if part['type'] == 'image':
                colors.append(part['data'].getpixel((0, 0)))
        return await preprocess(messages, *args, **kwargs)

    monkeypatch.setattr(context.async_engine, 'preprocess', with_loader)
    response = client.post('/v1/decisions', json=body)
    assert response.status_code == 200, response.text
    assert colors == [(255, 0, 0), (0, 0, 255), (255, 0, 0)]
    assert fetched == ['/blue.png']
    assert not context.session_manager.sessions
    # A syntactically valid reference must still pass real image decoding.
    body['input'] = [dict(role='user', content=[dict(type='input_image', image_url=images[1] + '?broken')])]
    bad = client.post('/v1/decisions', json=body)
    assert bad.status_code == 422
    assert bad.json()['error']['details'][0]['field'] == 'input.0.content.0.image_url'
    assert len(context.async_engine.requests) == 1
    assert not context.session_manager.sessions


def test_reserved_marker_and_missing_scores():
    parser = DecisionParser({**request(), 'input': '<decision>'}, config())
    with pytest.raises(ValueError, match='Reserved'):
        parser.prepare_input(tokenizer=FakeEngine().tokenizer)
    parser = DecisionParser(request(), config())
    output = prepared_output(parser, [[.8, .2], [.7, .3], [.2, .3, .5]])
    del output.logprobs[0][33]
    with pytest.raises(RequestError, match='lack a requested candidate'):
        parser.parse_output(output, model_name='intern-decision')


@pytest.fixture
def http_service():
    context = ServerContext()
    context.async_engine = FakeEngine()
    context.decision_context = config(temperature=1.)
    app = FastAPI()
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.include_router(create_openai_router(context))
    with TestClient(app) as client:
        yield client, context


def test_http_protocol_dispatch_and_validation(http_service):
    client, context = http_service
    result = client.post('/v1/decisions', json=request())
    assert result.status_code == 200
    assert [a['type'] for a in result.json()['answers']] == ['predicate', 'choice', 'score']
    assert result.json()['usage']['output_tokens'] > 0
    assert result.json()['usage']['total_tokens'] == (
        result.json()['usage']['input_tokens'] + result.json()['usage']['output_tokens'])
    count = len(context.async_engine.requests)
    assert client.post('/v1/systemone', json=request()).status_code == 422
    for change in ({'state': 'ambiguous'}, {'model': 'wrong'}):
        assert client.post('/v1/decisions', json={**request(), **change}).status_code == 422
    assert client.post('/v1/decisions', json={**request(), 'input': '<decision>'}).status_code == 400
    assert len(context.async_engine.requests) == count
    legacy = dict(state='old', questions={'q': {'type': 'noul', 'instructions': 'Check the state.'}})
    assert client.post('/v1/decisions', json=legacy).status_code == 422
    result = client.post('/v1/systemone', json=legacy)
    assert result.status_code == 200
    assert result.json()['answers']['q']['type'] == 'noul'
    assert result.json()['usage']['output_tokens'] > 0
    assert not context.session_manager.sessions
    context.decision_context = None
    assert client.post('/v1/decisions', json=request()).status_code == 503


def test_openapi_exposes_separate_protocols(http_service):
    client, _ = http_service
    schema = client.get('/openapi.json').json()
    decisions = schema['paths']['/v1/decisions']['post']['requestBody']['content']['application/json']['schema']
    systemone = schema['paths']['/v1/systemone']['post']['requestBody']['content']['application/json']['schema']
    assert decisions == {'$ref': '#/components/schemas/OpenAIDecisionRequest'}
    response = schema['paths']['/v1/decisions']['post']['responses']['200']['content']['application/json']['schema']
    assert response == {'$ref': '#/components/schemas/OpenAIDecisionResponse'}
    assert systemone['$ref'].endswith('/SystemoneRequest')
    assert '/v1/jev' not in schema['paths']


@pytest.fixture
def sdk_server(http_service):
    pytest.importorskip('openai.types.decision', reason='Install an OpenAI SDK with Decisions support')
    import uvicorn

    client, _ = http_service
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    server = uvicorn.Server(uvicorn.Config(client.app, log_level='error', access_log=False))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started, 'Local HTTP server did not start'
        yield f'http://127.0.0.1:{sock.getsockname()[1]}/v1'
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive(), 'Local HTTP server did not stop'


@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('with_images', [False, True])
def test_official_sdk_over_http(sdk_server, asynchronous, with_images, image_input):
    import openai

    body = request()
    if with_images:
        body['input'] = image_input[0]
    body['questions'][1]['choices'] = [dict(value=True), dict(value='true')]
    kwargs = dict(api_key='local-test', base_url=sdk_server, max_retries=0, timeout=10,
                  _strict_response_validation=True)
    if asynchronous:
        async def run():
            async with openai.AsyncOpenAI(**kwargs) as client:
                return await client.decisions.create(**body)
        result = asyncio.run(run())
    else:
        with openai.OpenAI(**kwargs) as client:
            result = client.decisions.create(**body)
            with pytest.raises(openai.UnprocessableEntityError):
                client.decisions.create(**{**body, 'model': 'not-loaded'})
    assert [a.type for a in result.answers] == ['predicate', 'choice', 'score']
    assert [p.value for p in result.answers[1].probabilities] == [True, 'true']
    assert result.usage.output_tokens > 0
    assert result.usage.total_tokens == result.usage.input_tokens + result.usage.output_tokens
