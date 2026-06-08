# Copyright (c) OpenMMLab. All rights reserved.

import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from lmdeploy.serve.openai.api_server import create_error_response
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest, ErrorResponse
from lmdeploy.serve.proxy.registry.pool import ReplicaPool
from lmdeploy.serve.proxy.upstream.exceptions import APIServerException


@dataclass
class ProxyContext:
    model: str
    stream: bool
    endpoint: str
    raw_request: Request
    parsed_request: ChatCompletionRequest | CompletionRequest
    request_dict: dict[str, Any]


def model_not_found_response(model_name: str) -> JSONResponse:
    from lmdeploy.utils import get_logger

    logger = get_logger('lmdeploy')
    logger.warning(f'no model name: {model_name}')
    return create_error_response(HTTPStatus.NOT_FOUND, f'The model {model_name!r} does not exist.')


def replica_unavailable_response(replica_url: str) -> JSONResponse:
    return create_error_response(
        HTTPStatus.SERVICE_UNAVAILABLE,
        f'The replica {replica_url!r} is no longer available. May retry later.',
    )


def response_from_api_exception(exc: APIServerException) -> JSONResponse:
    try:
        content = json.loads(exc.body)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
        message = exc.body.decode() if isinstance(exc.body, bytes) else str(exc.body)
        content = ErrorResponse(message=message, type='server_error', code=exc.status_code).model_dump()
    return JSONResponse(content=content, status_code=exc.status_code, headers=exc.headers)


def safe_json_load(replica_url: str, response_text: str | bytes) -> dict:
    if isinstance(response_text, bytes):
        response_text = response_text.decode()
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        from lmdeploy.utils import get_logger

        logger = get_logger('lmdeploy')
        logger.error(f'failed to parse response from {replica_url}, {e}')
        body = ErrorResponse(
            message=f'Invalid JSON response from replica {replica_url!r}.',
            type='server_error',
            code=HTTPStatus.BAD_GATEWAY.value,
        ).model_dump_json().encode()
        raise APIServerException(status_code=HTTPStatus.BAD_GATEWAY.value, body=body) from e


async def check_model(pool: ReplicaPool, model_name: str) -> JSONResponse | None:
    if model_name in pool.model_list:
        return None
    return model_not_found_response(model_name)
