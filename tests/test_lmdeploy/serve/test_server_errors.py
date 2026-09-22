import asyncio
import json

from fastapi import Request
from fastapi.exceptions import RequestValidationError

from lmdeploy.serve.openai.api_server import validation_exception_handler
from lmdeploy.serve.utils.server_utils import AuthenticationMiddleware, EngineSleepingMiddleware


def _scope(path: str, *, authorization: str | None = None):
    headers = []
    if authorization is not None:
        headers.append((b'authorization', authorization.encode()))
    return {
        'type': 'http',
        'http_version': '1.1',
        'method': 'POST',
        'scheme': 'http',
        'path': path,
        'raw_path': path.encode(),
        'query_string': b'',
        'root_path': '',
        'headers': headers,
        'server': ('testserver', 80),
        'client': ('127.0.0.1', 1234),
    }


async def _call_middleware(middleware, path: str):
    messages = []

    async def receive():
        return {'type': 'http.request', 'body': b'', 'more_body': False}

    async def send(message):
        messages.append(message)

    await middleware(_scope(path), receive, send)
    start = next(message for message in messages if message['type'] == 'http.response.start')
    body = b''.join(message.get('body', b'') for message in messages
                    if message['type'] == 'http.response.body')
    return start['status'], json.loads(body)


async def _unused_app(scope, receive, send):
    raise AssertionError('request should have been rejected by middleware')


def test_authentication_errors_use_stable_protocol_codes():
    middleware = AuthenticationMiddleware(_unused_app, tokens=['secret'])

    status_code, openai_body = asyncio.run(
        _call_middleware(middleware, '/v1/chat/completions'))
    assert status_code == 401
    assert openai_body['code'] == 401
    assert openai_body['message'] == 'Unauthorized.'

    status_code, anthropic_body = asyncio.run(
        _call_middleware(middleware, '/v1/messages'))
    assert status_code == 401
    assert anthropic_body == {
        'type': 'error',
        'error': {
            'type': 'authentication_error',
            'message': 'Unauthorized.',
        },
    }


def test_sleeping_engine_protects_anthropic_inference_route():
    middleware = EngineSleepingMiddleware(_unused_app, is_sleeping=lambda: True)

    status_code, body = asyncio.run(
        _call_middleware(middleware, '/v1/messages'))
    assert status_code == 503
    assert body['error']['type'] == 'overloaded_error'
    assert 'sleeping' in body['error']['message']


def test_validation_errors_use_route_protocol_envelope():
    exc = RequestValidationError(errors=[{
        'type': 'missing',
        'loc': ('body', 'model'),
        'msg': 'Field required',
        'input': {},
    }])

    async def _run(path):
        return await validation_exception_handler(Request(_scope(path)), exc)

    openai_response = asyncio.run(_run('/v1/chat/completions'))
    assert openai_response.status_code == 400
    openai_body = json.loads(openai_response.body)
    assert openai_body['code'] == 400
    assert openai_body['message'] == 'model: Field required'

    responses_response = asyncio.run(_run('/v1/responses'))
    assert responses_response.status_code == 400
    responses_body = json.loads(responses_response.body)
    assert responses_body['error']['code'] == 400

    anthropic_response = asyncio.run(_run('/v1/messages'))
    assert anthropic_response.status_code == 400
    anthropic_body = json.loads(anthropic_response.body)
    assert anthropic_body['type'] == 'error'
    assert anthropic_body['error']['type'] == 'invalid_request_error'
