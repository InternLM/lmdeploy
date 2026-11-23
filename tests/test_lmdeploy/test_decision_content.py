# Copyright (c) OpenMMLab. All rights reserved.
"""OpenAI content blocks through the real schema, parser and HTTP endpoint."""
import base64
import copy
import json
import threading
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError
from test_decision_openai import http_service as _http_service
from test_decision_openai import request
from test_decision_service import FakeEngine, config

from lmdeploy.serve.openai.decision_protocol import DecisionInputJSON, OpenAIDecisionRequest
from lmdeploy.serve.parsers.decision_assets import DecisionImageIO
from lmdeploy.serve.parsers.decision_parser import DecisionParser

http_service = _http_service


def image_part(color='red', size=(320, 320)):
    with BytesIO() as buffer, Image.new('RGB', size, color) as image:
        image.save(buffer, format='PNG')
        return dict(type='input_image',
                    image_url='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode())


def text(value):
    return dict(type='input_text', text=value)


def body(instructions, input='Evidence'):
    return dict(model='intern-decision', input=input,
                questions=[dict(type='predicate', name='check', instructions=instructions)])


def prepared(value):
    return DecisionParser(value, config()).prepare_input(tokenizer=FakeEngine().tokenizer)


def test_text_and_json_instructions_are_compiler_text():
    value = {'订单': [1, True, None], 'url': 'https://example.com/not-an-image'}
    rendered = 'Inspect ' + json.dumps(value, ensure_ascii=False, separators=(',', ':')) + ' now.'
    blocks = body([text('Inspect '), dict(type='input_json', data=value), text(' now.')])
    original = copy.deepcopy(blocks)
    assert prepared(blocks).messages == prepared(body(rendered)).messages
    assert blocks == original


def test_input_json_matches_explicit_json_text():
    value = [{'nested': {'items': [1, False, None]}}, {}]
    compact = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    left = body('Check', input=[dict(role='user', content=[dict(type='input_json', data=value)])])
    right = body('Check', input=[dict(role='user', content=[text(compact)])])
    assert prepared(left).messages == prepared(right).messages


def test_images_preserve_instruction_order_and_literal_urls(monkeypatch):
    red, blue = image_part(), image_part('blue')
    # A literal URL in JSON must stay text even when the same URL is an image part.
    value = body([text('first '), blue, text(' second '), red,
                  dict(type='input_json', data={'literal': blue['image_url']})],
                 input=[dict(role='user', content=[red, text('input evidence')])])
    original = copy.deepcopy(value)
    decoded = []
    load = DecisionImageIO.load_bytes

    def record(self, raw):
        decoded.append(raw)
        return load(self, raw)

    monkeypatch.setattr(DecisionImageIO, 'load_bytes', record)
    content = prepared(value).messages[1]['content']
    images = [p['image_data']['data'] for p in content if p['type'] == 'image_data']
    assert [im.getpixel((0, 0)) for im in images] == [(255, 0, 0), (0, 0, 255), (255, 0, 0)]
    assert images[0] is images[2] and len(decoded) == 2
    inline = ''.join('<image>' if p['type'] == 'image_data' else p['text'] for p in content)
    assert 'check: first <image> second <image>{"literal":' in inline
    assert blue['image_url'] in inline and 'ASSET_' not in inline
    assert value == original


def test_input_and_question_occurrences_share_resize_budget():
    picture = image_part(size=(1568, 1568))
    value = body([picture] * 4, input=[dict(role='user', content=[picture] * 4)])
    images = [p['image_data']['data'] for p in prepared(value).messages[1]['content'] if p['type'] == 'image_data']
    assert len(images) == 8 and len({id(im) for im in images}) == 1
    assert images[0].size == (704, 704)


def test_ninth_occurrence_across_questions_rejected_before_decode(http_service, monkeypatch):
    client, context = http_service
    picture = image_part()
    value = body([picture] * 4, input=[dict(role='user', content=[picture] * 4)])
    value['questions'].append(dict(type='predicate', name='last', instructions=[picture]))
    monkeypatch.setattr(DecisionImageIO, 'load_bytes', lambda *args: pytest.fail('Must validate before decoding'))
    response = client.post('/v1/decisions', json=value)
    assert response.status_code == 422
    assert response.json()['error']['details'][0]['field'] == 'questions.1.instructions.0.image_url'
    assert not context.async_engine.requests and not context.session_manager.sessions


@pytest.mark.parametrize('part,field', [
    (dict(type='input_image', image_url='data:image/png;base64,AAAA'), 'image_url'),
    (dict(type='input_image', image_url='file:///private.png'), 'image_url'),
    (dict(type='input_image', image_url='https://example.com/a.png', detail='high'), 'detail'),
    (dict(type='input_text', text=42), 'text'),
    (dict(type='input_json', data={'n': float('inf')}), 'data'),
])
def test_question_content_errors_keep_wire_location(http_service, part, field):
    if field == 'data':
        # HTTP clients reject nonfinite JSON themselves; exercise the wire model directly.
        with pytest.raises(ValidationError):
            OpenAIDecisionRequest.model_validate(body([part]))
        return
    client, context = http_service
    response = client.post('/v1/decisions', json=body([part]))
    assert response.status_code == 422, response.text
    assert f'questions.0.instructions.0.{field}' in [d['field'] for d in response.json()['error']['details']]
    assert not context.async_engine.requests and not context.session_manager.sessions


def test_question_images_run_off_loop_and_return_normal_answers(http_service, monkeypatch):
    client, context = http_service
    workers, loops = [], []
    load, preprocess = DecisionImageIO.load_bytes, context.async_engine.preprocess

    def record(self, raw):
        workers.append(threading.get_ident())
        return load(self, raw)

    async def record_loop(*args, **kwargs):
        loops.append(threading.get_ident())
        return await preprocess(*args, **kwargs)

    monkeypatch.setattr(DecisionImageIO, 'load_bytes', record)
    monkeypatch.setattr(context.async_engine, 'preprocess', record_loop)
    value = request()
    for question in value['questions']:
        question['instructions'] = [text(question['instructions']), image_part()]
    response = client.post('/v1/decisions', json=value)
    assert response.status_code == 200, response.text
    assert len(workers) == len(loops) == 1 and workers[0] != loops[0]
    result = response.json()
    assert [a['type'] for a in result['answers']] == ['predicate', 'choice', 'score']
    assert result['usage']['output_tokens'] > 0
    assert result['usage']['total_tokens'] == result['usage']['input_tokens'] + result['usage']['output_tokens']
    assert not context.session_manager.sessions


def test_question_http_images_keep_domain_policy(http_service):
    client, context = http_service
    context.async_engine.prompt_processor = SimpleNamespace(allowed_media_domains=['allowed.example'])
    value = body([dict(type='input_image', image_url='https://blocked.example/image.png')])
    response = client.post('/v1/decisions', json=value)
    assert response.status_code == 422
    detail = response.json()['error']['details'][0]
    assert detail['field'] == 'questions.0.instructions.0.image_url' and 'allowed domains' in detail['message']
    assert not context.async_engine.requests


@pytest.mark.parametrize('value', [None, 1, 'text', [1], [{'n': float('nan')}], {'surrogate': '\ud800'}])
def test_invalid_json_evidence(value):
    with pytest.raises(ValidationError):
        DecisionInputJSON(type='input_json', data=value)


def nested(count):
    value = 'leaf'
    for _ in range(count):
        value = {'child': value}
    return value


@pytest.mark.parametrize('valid,invalid', [
    (nested(32), nested(33)),
    ({'values': [None] * 9998}, {'values': [None] * 9999}),
    ({'x': 'a' * (256 * 1024 - 8)}, {'x': 'a' * (256 * 1024 - 7)}),
])
def test_json_evidence_boundaries(valid, invalid):
    DecisionInputJSON(type='input_json', data=valid)
    with pytest.raises(ValidationError):
        DecisionInputJSON(type='input_json', data=invalid)


def test_openapi_exposes_content_blocks(http_service):
    client, _ = http_service
    schemas = client.get('/openapi.json').json()['components']['schemas']
    instruction = schemas['DecisionPredicateQuestion']['properties']['instructions']['anyOf']
    assert any(branch.get('type') == 'string' for branch in instruction)
    parts = next(branch for branch in instruction if branch.get('type') == 'array')['items']
    assert set(parts['discriminator']['mapping']) == {'input_text', 'input_image', 'input_json'}
    assert 'DecisionInputJSON' in schemas


def test_sentinels_do_not_replace_concatenated_literal_text(monkeypatch):
    from lmdeploy.serve.parsers import decision_assets

    prefixes = iter([SimpleNamespace(hex='collision'), SimpleNamespace(hex='unique')])
    monkeypatch.setattr(decision_assets, 'uuid4', lambda: next(prefixes))
    value = body([text('ASSET_'), text('collision_0_END'), image_part()])
    content = prepared(value).messages[1]['content']
    assert sum(p['type'] == 'image_data' for p in content) == 1
    assert 'ASSET_collision_0_END' in ''.join(p['text'] for p in content if p['type'] == 'text')
