# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/server_utils.py
import hashlib
import secrets
from collections.abc import Awaitable

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import URL, Headers
from starlette.types import ASGIApp, Receive, Scope, Send


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get('content-type', '').lower()
    media_type = content_type.split(';', maxsplit=1)[0]
    if media_type != 'application/json':
        raise RequestValidationError(errors=["Unsupported Media Type: Only 'application/json' is allowed"])


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
