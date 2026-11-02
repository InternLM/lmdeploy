# Copyright (c) OpenMMLab. All rights reserved.
# Asset cases adapted from intern-decision-inference a8fe089/tests/test_assets.py.
"""Systemone wire, image positioning and response contracts on the shared engine."""
import base64
import copy
import json
import threading
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from PIL import Image
from test_decision_service import FakeEngine, config

from lmdeploy.serve.openai.api_server import ServerContext, validation_exception_handler
from lmdeploy.serve.openai.decision_protocol import SystemoneRequest
from lmdeploy.serve.openai.endpoints import create_openai_router
from lmdeploy.serve.parsers import decision_assets as assets
from lmdeploy.serve.parsers.decision_parser import DecisionParser


def asset(color='red', size=(320, 320), fmt='PNG'):
    stream = BytesIO()
    Image.new('RGB', size, color).save(stream, format=fmt)
    return dict(type='image', format='png' if fmt == 'PNG' else 'jpg',
                data_base64=base64.b64encode(stream.getvalue()).decode())


def body():
    return dict(state={'photo': 'photo-ref'}, assets={'photo-ref': asset()}, questions={
        'urgent': dict(type='noul', instructions='Is photo-ref red?'),
        'team': dict(type='choice', instructions='Team?', criteria={'billing': 'Payments', 'support': None}),
        'severity': dict(type='score', instructions='Severity?', criteria=['Low', 'High']),
    })


class SystemoneEngine(FakeEngine):

    def __init__(self):
        super().__init__()
        self.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode('utf-8')))
        self.image_inputs = []
        self.preprocess_thread = None

    async def preprocess(self, messages, session, **kwargs):
        self.preprocess_thread = threading.get_ident()
        converted = []
        for message in messages:
            message = dict(message)
            if isinstance(message['content'], list):
                content = []
                for part in message['content']:
                    if part['type'] == 'image_data':
                        self.image_inputs.append(part['image_data']['data'])
                        content.append(dict(type='image_url', image_url={'url': 'prepared-image'}))
                    else:
                        content.append(part)
                message['content'] = content
            converted.append(message)
        return await super().preprocess(converted, session, **kwargs)


@pytest.fixture
def service():
    context = ServerContext()
    context.async_engine = SystemoneEngine()
    context.decision_context = config()
    app = FastAPI()
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.include_router(create_openai_router(context))
    with TestClient(app) as client:
        yield client, context.async_engine


def validate(row):
    parsed = SystemoneRequest.model_validate(row).model_dump(exclude={'model'}, exclude_unset=True)
    for image in assets.prepare_assets(parsed).values():
        image.close()
    return parsed


def test_nested_references_keep_order_keys_and_original_response_values(service):
    row = body()
    row['assets']['blue-ref'] = asset('blue')
    row['state'] = {'photo-ref': ['before photo-ref after', {'key': 'blue-ref'}]}
    row['questions']['urgent']['instructions'] = {'check': ['photo-ref']}
    row['questions']['team']['criteria'] = {'photo-ref': {'example': 'blue-ref'}, 'other': None}
    row['questions']['severity']['criteria'] = [{'example': 'photo-ref'}, ['blue-ref']]
    before = copy.deepcopy(row)
    parser = DecisionParser(SystemoneRequest.model_validate(row), config())
    parser.prepare_input(tokenizer=FakeEngine().tokenizer)
    compiled = parser.compiled
    content = compiled.messages[1]['content']
    text = ''.join(part['text'] for part in content if part['type'] == 'text')
    assert '"photo-ref":' in text and 'A = photo-ref:' in text
    assert 'ASSET_' not in text and row['assets']['photo-ref']['data_base64'] not in text
    assert 'None' not in text
    client, engine = service
    response = client.post('/v1/systemone', json=row)
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == {'model', 'answers', 'usage'}
    assert set(result['answers']['urgent']) == {'type', 'noul'}
    assert result['answers']['severity']['legend'] == {'0': {'example': 'photo-ref'}, '1': ['blue-ref']}
    assert result['usage']['output_tokens'] == len(compiled.messages[-1]['content'].encode('utf-8'))
    assert [im.getpixel((0, 0)) for im in engine.image_inputs] == [(255, 0, 0), (0, 0, 255)] * 3
    assert row == before and not engine.session_mgr.sessions


def test_longest_reference_and_original_string_escaping():
    row = body()
    row['assets'] = {'cat': asset(), 'category': asset('blue')}
    row['state'] = ['category', 'cat', 'xxcategorycat']
    row['questions'] = {'q': dict(type='noul', instructions='Red?')}
    parser = DecisionParser(SystemoneRequest.model_validate(row), config())
    parser.prepare_input(tokenizer=FakeEngine().tokenizer)
    compiled = parser.compiled
    images = [part['image_data']['data'] for part in compiled.messages[1]['content'] if part['type'] == 'image_data']
    assert len(images) == 4 and images[0] is images[2] and images[1] is images[3]
    assert [im.getpixel((0, 0)) for im in images] == [(0, 0, 255), (255, 0, 0)] * 2
    row['assets'] = {'a"b\\c': asset()}
    row['state'] = ['a"b\\c']
    parser = DecisionParser(SystemoneRequest.model_validate(row), config())
    content = parser.prepare_input(tokenizer=FakeEngine().tokenizer).messages[1]['content']
    assert sum(p['type'] == 'image_data' for p in content) == 1


@pytest.mark.parametrize('ref', ['State', 'instructions', 'noul', 'team'])
def test_prompt_scaffolding_and_keys_do_not_count_as_references(ref):
    row = body()
    row['state'] = {'team': 'no image reference'}
    row['questions'] = {'q': dict(type='noul', instructions='Red?')}
    row['assets'] = {ref: asset()}
    with pytest.raises(assets.AssetError, match='Every asset'):
        validate(row)


@pytest.mark.parametrize(('size', 'status'), [((256, 256), 400), ((255, 400), 400), ((1569, 320), 413)])
def test_image_dimensions_rejected_before_engine_work(service, size, status):
    client, engine = service
    row = body()
    row['assets']['photo-ref'] = asset(size=size)
    assert client.post('/v1/systemone', json=row).status_code == status
    assert not engine.requests and engine.session_mgr.count == 0


@pytest.mark.parametrize('size', [(280, 280), (280, 307), (1568, 1568)])
def test_inclusive_dimension_limits(size):
    validate({**body(), 'assets': {'photo-ref': asset(size=size)}})


@pytest.mark.parametrize('change', [
    {'format': None}, {'part': 'upload'}, {'data_base64': '!'}, {'format': 'jpg'},
    {'data_base64': base64.b64encode(b'not an image').decode()},
])
def test_bad_asset_contents_return_400(service, change):
    client, engine = service
    row = body()
    row['assets']['photo-ref'].update(change)
    assert client.post('/v1/systemone', json=row).status_code == 400
    assert not engine.requests


def test_base64_requires_canonical_bare_payload_and_static_matching_file(service):
    client, _ = service
    row = body()
    for value in ('data:image/png;base64,' + row['assets']['photo-ref']['data_base64'],
                  row['assets']['photo-ref']['data_base64'] + '\n'):
        invalid = copy.deepcopy(row)
        invalid['assets']['photo-ref']['data_base64'] = value
        assert client.post('/v1/systemone', json=invalid).status_code == 400
    stream = BytesIO()
    Image.new('RGB', (320, 320), 'red').save(stream, format='PNG', save_all=True,
        append_images=[Image.new('RGB', (320, 320), 'blue')])
    with pytest.raises(assets.AssetError):
        assets.decode_asset(dict(format='png', data_base64=base64.b64encode(stream.getvalue()).decode()))
    raw = base64.b64decode(row['assets']['photo-ref']['data_base64'])[:-20]
    with pytest.raises(assets.AssetError):
        assets.decode_asset(dict(format='png', data_base64=base64.b64encode(raw).decode()))


def test_repeated_references_and_byte_limits(service, monkeypatch):
    client, engine = service
    row = body()
    row['questions'] = {'q': dict(type='noul', instructions='Red?')}
    row['state'] = ['photo-ref'] * 8
    assert client.post('/v1/systemone', json=row).status_code == 200
    assert len(engine.image_inputs) == 8
    row['state'].append('photo-ref')
    assert client.post('/v1/systemone', json=row).status_code == 400
    with monkeypatch.context() as patch:
        patch.setattr(assets, 'MAX_FILE_BYTES', 1)
        assert client.post('/v1/systemone', json=body()).status_code == 413
    monkeypatch.setattr(assets, 'MAX_TOTAL_BYTES', 1)
    assert client.post('/v1/systemone', json=body()).status_code == 413


def test_prepared_images_are_reused_only_within_one_request(monkeypatch):
    row = body()
    row['questions'] = {'q': dict(type='noul', instructions='Red?')}
    row['state'] = ['photo-ref'] * 8
    decoded, resized = [], []
    decode, resize = assets.decode_asset, assets.resize_image

    def track_decode(value):
        image = decode(value)
        decoded.append(image)
        return image

    def track_resize(image, count):
        resized.append(count)
        return resize(image, count)

    monkeypatch.setattr(assets, 'decode_asset', track_decode)
    monkeypatch.setattr(assets, 'resize_image', track_resize)
    first = DecisionParser(SystemoneRequest.model_validate(row), config())
    messages = first.prepare_input(tokenizer=FakeEngine().tokenizer).messages
    images = [p['image_data']['data'] for p in messages[1]['content'] if p['type'] == 'image_data']
    assert len(decoded) == 1 and resized == [8]
    assert len(images) == 8 and all(image is images[0] for image in images)
    second = DecisionParser(SystemoneRequest.model_validate(row), config())
    other = second.prepare_input(tokenizer=FakeEngine().tokenizer).messages
    other_image = next(p['image_data']['data'] for p in other[1]['content'] if p['type'] == 'image_data')
    assert other_image is not images[0] and len(decoded) == 2 and resized == [8, 8]


def test_asset_budget_precedes_resize_and_closes_failed_results(monkeypatch):
    row = body()
    row['state'] = ['photo-ref'] * 8
    row['questions'] = {'q': dict(type='noul', instructions='Red?')}
    decoded = []
    decode = assets.decode_asset

    def track_decode(value):
        image = decode(value)
        decoded.append(image)
        return image

    def unexpected_resize(*args):
        pytest.fail('Resizing must not bypass the expanded byte limit')

    # One asset fits, but its eight appearances exceed the budget.
    monkeypatch.setattr(assets, 'MAX_TOTAL_BYTES', 4000)
    monkeypatch.setattr(assets, 'decode_asset', track_decode)
    monkeypatch.setattr(assets, 'resize_image', unexpected_resize)
    with pytest.raises(assets.AssetError, match='expanded request'):
        assets.prepare_assets(row)
    assert len(decoded) == 1
    with pytest.raises(ValueError, match='closed image'):
        decoded[0].getpixel((0, 0))


def test_null_normalization_preserves_request_and_default_noul_descriptions():
    row = dict(state='text', questions={
        'q': dict(type='noul', instructions=None, criteria={'no': None, 'yes': 'Allowed'}),
        'c': dict(type='choice', instructions=None, criteria={'a': None, 'b': 'Other'}),
    })
    request = SystemoneRequest.model_validate(row)
    original = request.model_dump()
    parser = DecisionParser(request, config())
    text = parser.prepare_input(tokenizer=FakeEngine().tokenizer).messages[1]['content']
    assert 'None' not in text and 'q: \n' in text and 'c: \n' in text
    assert 'A = no: The answer is no (negative, or disagree with the claim).' in text
    assert 'B = yes: Allowed' in text
    assert 'A = a: \n' in text
    assert request.model_dump() == original


@pytest.mark.parametrize(('ref', 'status'), [('', 400), (' ', 400), ('图' * 342, 413)])
def test_reference_name_limits(service, ref, status):
    client, _ = service
    row = body()
    row['state'] = ref
    row['questions'] = {'q': dict(type='noul', instructions='Red?')}
    row['assets'] = {ref: asset()}
    assert client.post('/v1/systemone', json=row).status_code == status


def test_multipart_matches_json_and_part_overrides_base64(service):
    client, _ = service
    row = body()
    expected = client.post('/v1/systemone', json=row).json()
    raw = base64.b64decode(row['assets']['photo-ref']['data_base64'])
    row['assets']['photo-ref'] = {'part': 'upload', 'data_base64': 'ignored'}

    def parts(request=row):
        return [('request', (None, json.dumps(request), 'application/json')),
                ('upload', ('photo.png', raw, 'image/png'))]

    response = client.post('/v1/systemone', files=parts())
    assert response.status_code == 200, response.text
    assert response.json() == expected
    invalid_parts = [list(reversed(parts())), parts() + [parts()[1]], parts()[:1],
                     parts() + [('unused', ('x.png', raw, 'image/png'))],
                     [('request', (None, json.dumps(row), 'text/plain')), parts()[1]],
                     [parts()[0], ('upload', ('x.png', raw, 'text/plain'))]]
    shared = copy.deepcopy(row)
    shared['assets']['second'] = {'part': 'upload'}
    shared['state'] = ['photo-ref', 'second']
    invalid_parts.append(parts(shared))
    for parts_value in invalid_parts:
        assert client.post('/v1/systemone', files=parts_value).status_code == 400
    assert client.post('/v1/systemone', json=row).status_code == 400
    assert client.post('/v1/systemone', content=b'--x\r\n',
                       headers={'Content-Type': 'multipart/form-data; boundary=x'}).status_code == 400


def test_schema_errors_are_422_and_protocols_use_separate_routes(service):
    client, engine = service
    for images in ([], ['x'], None):
        response = client.post('/v1/systemone', json={**body(), 'images': images})
        assert response.status_code == 422
        assert 'assets' in response.json()['error']['details'][0]['message']
    response = client.post('/v1/systemone', json={**body(), 'questions': {
        'where': dict(type='box', instructions='Locate it.')}})
    assert response.status_code == 422
    assert response.json()['error']['details'][0]['field'] == 'questions.where.type'
    assert 'not supported' in response.json()['error']['details'][0]['message']
    assert not engine.requests
    assert client.post('/v1/jev', json=body()).status_code == 404
    assert client.post('/v1/decisions', json=body()).status_code == 422
    assert client.post('/v1/systemone', json={
        'state': 'text', 'questions': {'q': {'type': 'noul', 'instructions': 'Check the state.'}},
    }).status_code == 200


def test_model_aliases_use_the_loaded_model(service):
    client, engine = service
    engine.model_name = 'Intern-Decision-0.8B'
    for model in (None, 'Intern-Decision-0.8B', 'InternLM/Intern-Decision-0.8B', 'internlm/intern-decision-0.8b'):
        row = body()
        if model is not None:
            row['model'] = model
        assert client.post('/v1/systemone', json=row).status_code == 200
    assert client.post('/v1/systemone', json={**body(), 'model': 'other'}).status_code == 422


def test_body_limits_and_malformed_json(service, monkeypatch):
    from lmdeploy.serve.openai.endpoints import decisions
    client, engine = service
    assert client.post('/v1/systemone', content=b'{',
                       headers={'content-type': 'application/json'}).status_code == 400
    monkeypatch.setattr(decisions, 'MAX_TOTAL_BYTES', 8)
    assert client.post('/v1/systemone', json=body()).status_code == 413
    assert client.post('/v1/systemone', content=iter([b'12345', b'67890']),
                       headers={'content-type': 'application/json'}).status_code == 413
    assert not engine.requests


def test_image_work_runs_outside_the_event_loop(service, monkeypatch):
    client, engine = service
    threads = []
    decode = assets.decode_asset

    def checked_decode(value):
        threads.append(threading.get_ident())
        return decode(value)

    monkeypatch.setattr(assets, 'decode_asset', checked_decode)
    assert client.post('/v1/systemone', json=body()).status_code == 200
    assert threads and engine.preprocess_thread not in threads


def test_openapi_declares_json_and_multipart_without_changing_decisions(service):
    client, _ = service
    spec = client.get('/openapi.json').json()
    upload = spec['paths']['/v1/systemone']['post']
    assert set(upload['requestBody']['content']) == {'application/json', 'multipart/form-data'}
    assert upload['requestBody']['content']['application/json']['schema'] == {
        '$ref': '#/components/schemas/SystemoneRequest'}
    decide = spec['paths']['/v1/decisions']['post']
    assert decide['requestBody']['content']['application/json']['schema'] == {
        '$ref': '#/components/schemas/OpenAIDecisionRequest'}


def test_reference_resize_budget_matches_original_policy():
    assert assets.resized_size(1568, 1568, 1) == (1024, 1024)
    assert assets.resized_size(1024, 1024, 8) == (704, 704)
    assert assets.resized_size(280, 280, 1) == (288, 288)


def test_systemone_is_rejected_while_engine_sleeps():
    from lmdeploy.serve.utils.server_utils import EngineSleepingMiddleware

    context = ServerContext()
    context.async_engine = SystemoneEngine()
    context.decision_context = config()
    app = FastAPI()
    app.include_router(create_openai_router(context))
    app.add_middleware(EngineSleepingMiddleware, is_sleeping=lambda: True)
    with TestClient(app) as client:
        assert client.post('/v1/systemone', json=body()).status_code == 503
    assert not context.async_engine.requests
