# Copyright (c) OpenMMLab. All rights reserved.
"""Decisions image limits and preparation, including HTTP field errors."""
import base64
import threading
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image
from test_decision_openai import http_service as _http_service
from test_decision_openai import request
from test_decision_service import FakeEngine, config

from lmdeploy.serve.parsers.decision_assets import DecisionImageIO
from lmdeploy.serve.parsers.decision_parser import DecisionParser

http_service = _http_service


def image_bytes(size=(320, 320), fmt='PNG', animated=False):
    with BytesIO() as buffer, Image.new('RGB', size, 'red') as image:
        if animated:
            with Image.new('RGB', size, 'blue') as other:
                image.save(buffer, format=fmt, save_all=True, append_images=[other])
        else:
            image.save(buffer, format=fmt)
        return buffer.getvalue()


def body_with_images(data, count=1):
    url = 'data:image/png;base64,' + base64.b64encode(data).decode()
    return {**request(), 'input': [dict(role='user', content=[
        dict(type='input_image', image_url=url) for _ in range(count)])]}


@pytest.mark.parametrize('fmt', ['PNG', 'JPEG', 'WEBP', 'GIF'])
def test_static_formats(fmt):
    with DecisionImageIO().load_bytes(image_bytes(fmt=fmt)) as image:
        assert image.mode == 'RGB' and image.size == (320, 320)


@pytest.mark.parametrize('data', [
    b'', b'not an image', image_bytes(fmt='BMP'), image_bytes(fmt='GIF', animated=True),
    image_bytes(size=(20_001, 1)), image_bytes(size=(201, 1)), image_bytes(size=(5000, 4001)),
])
def test_invalid_image_returns_field_error_before_engine(http_service, data):
    client, context = http_service
    response = client.post('/v1/decisions', json=body_with_images(data))
    assert response.status_code == 422, response.text
    assert response.json()['error']['details'][0]['field'] == 'input.0.content.0.image_url'
    assert not context.async_engine.requests
    assert not context.session_manager.sessions


def test_byte_limit():
    raw = image_bytes()
    boundary = raw + b'\0' * (10 * 1024**2 - len(raw))
    with DecisionImageIO().load_bytes(boundary) as image:
        assert image.size == (320, 320)
    with pytest.raises(ValueError, match='10 MiB'):
        DecisionImageIO().load_bytes(boundary + b'\0')
    encoded = base64.b64encode(boundary + b'\0').decode()
    with pytest.raises(ValueError, match='10 MiB'):
        DecisionImageIO().load_base64('image/png', encoded)


@pytest.mark.parametrize('size', [(20_000, 100), (200, 1), (5000, 4000)])
def test_dimension_boundaries(size):
    with DecisionImageIO().load_bytes(image_bytes(size)) as image:
        assert image.size == size


def test_strict_base64(http_service):
    client, context = http_service
    body = body_with_images(image_bytes())
    body['input'][0]['content'][0]['image_url'] += '!'
    response = client.post('/v1/decisions', json=body)
    assert response.status_code == 422
    assert not context.async_engine.requests


@pytest.mark.parametrize('count,expected_size', [(1, (1024, 1024)), (8, (704, 704))])
def test_total_occurrences_control_resize_and_decode_reuse(monkeypatch, count, expected_size):
    decoded = []
    original = DecisionImageIO.load_bytes

    def record(self, data):
        decoded.append(1)
        return original(self, data)

    monkeypatch.setattr(DecisionImageIO, 'load_bytes', record)
    parser = DecisionParser(body_with_images(image_bytes((1568, 1568)), count), config())
    content = parser.prepare_input(tokenizer=FakeEngine().tokenizer).messages[1]['content']
    images = [part['image_data']['data'] for part in content if part['type'] == 'image_data']
    assert len(images) == count
    assert all(image.size == expected_size for image in images)
    assert len({id(image) for image in images}) == 1
    assert decoded == [1]


def test_ninth_image_rejected_before_decode(http_service, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Nine images must fail before downloading or decoding')

    monkeypatch.setattr(DecisionImageIO, 'load_bytes', unexpected)
    client, context = http_service
    body = body_with_images(image_bytes(), 8)
    body['input'].append(dict(role='user', content=body['input'][0]['content'][:1]))
    response = client.post('/v1/decisions', json=body)
    assert response.status_code == 422
    assert not context.async_engine.requests


def test_decode_runs_off_event_loop(http_service, monkeypatch):
    client, context = http_service
    workers, loops = [], []
    decode = DecisionImageIO.load_bytes
    preprocess = context.async_engine.preprocess

    def record(self, data):
        workers.append(threading.get_ident())
        return decode(self, data)

    async def record_loop(*args, **kwargs):
        loops.append(threading.get_ident())
        return await preprocess(*args, **kwargs)

    monkeypatch.setattr(DecisionImageIO, 'load_bytes', record)
    monkeypatch.setattr(context.async_engine, 'preprocess', record_loop)
    response = client.post('/v1/decisions', json=body_with_images(image_bytes()))
    assert response.status_code == 200, response.text
    assert len(workers) == len(loops) == 1 and workers[0] != loops[0]


def test_http_domain_policy_preserved(http_service):
    client, context = http_service
    context.async_engine.prompt_processor = SimpleNamespace(allowed_media_domains=['allowed.example'])
    body = request()
    body['input'] = [dict(role='user', content=[
        dict(type='input_image', image_url='https://blocked.example/image.png')])]
    response = client.post('/v1/decisions', json=body)
    assert response.status_code == 422
    assert 'allowed domains' in response.json()['error']['details'][0]['message']
    assert not context.async_engine.requests


@pytest.mark.parametrize('data', [image_bytes(fmt='BMP'), b'x' * (10 * 1024**2 + 1)])
def test_downloaded_images_use_same_limits(http_service, monkeypatch, data):
    from lmdeploy.vl.media import connection

    monkeypatch.setattr(connection, '_is_safe_url', lambda url: (True, 'test transport'))
    monkeypatch.setattr(connection.requests.Session, 'get', lambda *args, **kwargs: SimpleNamespace(
        is_redirect=False, content=data, raise_for_status=lambda: None))
    client, context = http_service
    body = request()
    body['input'] = [dict(role='user', content=[
        dict(type='input_image', image_url='https://media.example/image')])]
    response = client.post('/v1/decisions', json=body)
    assert response.status_code == 422
    assert response.json()['error']['details'][0]['field'] == 'input.0.content.0.image_url'
    assert not context.async_engine.requests


def test_download_failure_is_a_field_error(http_service, monkeypatch):
    from requests import HTTPError

    from lmdeploy.serve.parsers import decision_assets as assets

    def fail(*args, **kwargs):
        raise HTTPError('Remote server returned 404')

    monkeypatch.setattr(assets, 'load_from_url', fail)
    client, context = http_service
    body = request()
    body['input'] = [dict(role='user', content=[
        dict(type='input_image', image_url='https://media.example/missing.png')])]
    response = client.post('/v1/decisions', json=body)
    assert response.status_code == 422
    assert 'Could not download image' in response.json()['error']['details'][0]['message']
    assert not context.async_engine.requests


def test_partial_preparation_closes_images(monkeypatch):
    from pydantic import ValidationError

    images = []
    decode = DecisionImageIO.load_bytes

    def record(self, data):
        image = decode(self, data)
        images.append(image)
        return image

    monkeypatch.setattr(DecisionImageIO, 'load_bytes', record)
    body = body_with_images(image_bytes())
    body['input'][0]['content'].append(dict(type='input_image', image_url='data:image/png;base64,AAAA'))
    parser = DecisionParser(body, config())
    with pytest.raises(ValidationError):
        parser.prepare_input(tokenizer=FakeEngine().tokenizer)
    assert len(images) == 1
    with pytest.raises(ValueError, match='closed'):
        images[0].getpixel((0, 0))
