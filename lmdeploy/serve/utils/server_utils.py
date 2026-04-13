# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/server_utils.py
import hashlib
import secrets
from collections.abc import Awaitable, Callable
from http import HTTPStatus

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import URL, Headers
from starlette.types import ASGIApp, Receive, Scope, Send


def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get('content-type', '').lower()
    media_type = content_type.split(';', maxsplit=1)[0]
    if media_type != 'application/json':
        raise RequestValidationError(errors=["Unsupported Media Type: Only 'application/json' is allowed"])


class EngineSleepingMiddleware:
    """Pure ASGI middleware that returns 503 for configured inference routes
    when ``is_sleeping()`` is true (after ``POST /sleep``, until ``POST
    /wakeup``).

    Notes
    -----
    - Skips non-http scopes (except ``http``/``websocket`` are passed through
      to the app; only ``http`` requests are gated).
    - HTTP ``OPTIONS`` is passed through so CORS preflight is unaffected.
    """

    # POST routes rejected while sleeping (see POST /sleep, /wakeup).
    DEFAULT_PROTECTED_INFERENCE_ROUTES = frozenset({
        ('POST', '/v1/chat/completions'),
        ('POST', '/v1/completions'),
        ('POST', '/generate'),
    })

    def __init__(
        self,
        app: ASGIApp,
        is_sleeping: Callable[[], bool],
        protected_routes: frozenset[tuple[str, str]] | None = None,
    ) -> None:
        self.app = app
        self.is_sleeping = is_sleeping
        self.protected_routes = protected_routes or type(self).DEFAULT_PROTECTED_INFERENCE_ROUTES

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope['type'] not in ('http', 'websocket'):
            return self.app(scope, receive, send)
        if scope['type'] == 'http' and scope['method'] == 'OPTIONS':
            return self.app(scope, receive, send)
        if scope['type'] == 'http':
            root_path = scope.get('root_path', '')
            url_path = URL(scope=scope).path.removeprefix(root_path)
            key = (scope['method'], url_path)
            if key in self.protected_routes and self.is_sleeping():
                response = JSONResponse(
                    content={
                        'error': (
                            'Engine is sleeping; call POST /wakeup before inference '
                            '(e.g. tags=weights&tags=kv_cache).'
                        ),
                    },
                    status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return response(scope, receive, send)
        return self.app(scope, receive, send)


class AuthenticationMiddleware:
    """Pure ASGI middleware that authenticates each request by checking if the
    Authorization Bearer token exists and equals anyof "{api_key}".

    Notes
    -----
    There are two cases in which authentication is skipped:
        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with /v1 (e.g. /health).
    """

    def __init__(self, app: ASGIApp, tokens: list[str]) -> None:
        self.app = app
        self.api_tokens = [hashlib.sha256(t.encode('utf-8')).digest() for t in tokens]
        # Path prefixes that bypass authentication
        self.skip_prefixes = [
            '/health',  # Health check endpoints
            '/docs',  # Swagger UI documentation
            '/redoc',  # ReDoc documentation
            '/nodes',  # Endpoints about node operation between proxy and api_server
        ]

    def verify_token(self, headers: Headers) -> bool:
        authorization_header_value = headers.get('Authorization')
        if not authorization_header_value:
            return False

        scheme, _, param = authorization_header_value.partition(' ')
        if scheme.lower() != 'bearer':
            return False

        param_hash = hashlib.sha256(param.encode('utf-8')).digest()

        token_match = False
        for token_hash in self.api_tokens:
            token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope['type'] not in ('http', 'websocket'):
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        if scope['type'] == 'http' and scope['method'] == 'OPTIONS':
            return self.app(scope, receive, send)

        root_path = scope.get('root_path', '')
        url_path = URL(scope=scope).path.removeprefix(root_path)
        headers = Headers(scope=scope)
        if not any(url_path.startswith(path) for path in self.skip_prefixes) and not self.verify_token(headers):
            response = JSONResponse(content={'error': 'Unauthorized'}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)
