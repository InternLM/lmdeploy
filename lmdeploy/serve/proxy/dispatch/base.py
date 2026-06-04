# Copyright (c) OpenMMLab. All rights reserved.

import json
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest
from lmdeploy.serve.proxy.core.errors import ErrorCodes, err_msg
from lmdeploy.serve.proxy.registry.pool import ReplicaPool


@dataclass
class ProxyContext:
    model: str
    stream: bool
    endpoint: str
    raw_request: Request
    parsed_request: ChatCompletionRequest | CompletionRequest
    request_dict: dict[str, Any]


def safe_json_load(replica_url: str, response_text: str | bytes) -> dict:
    if isinstance(response_text, bytes):
        try:
            return json.loads(response_text.decode())
        except json.JSONDecodeError:
            pass
        return {
            'error_code': ErrorCodes.SERVICE_UNAVAILABLE,
            'text': err_msg[ErrorCodes.SERVICE_UNAVAILABLE],
            'replica_url': replica_url,
        }
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        from lmdeploy.utils import get_logger

        logger = get_logger('lmdeploy')
        logger.error(f'failed to parse response from {replica_url}, {e}')
        return {
            'error_code': ErrorCodes.SERVICE_UNAVAILABLE,
            'text': err_msg[ErrorCodes.SERVICE_UNAVAILABLE],
            'replica_url': replica_url,
        }


def unavailable_model_bytes(model_name: str) -> bytes:
    from lmdeploy.utils import get_logger

    logger = get_logger('lmdeploy')
    logger.warning(f'no model name: {model_name}')
    ret = {
        'error_code': ErrorCodes.MODEL_NOT_FOUND,
        'text': err_msg[ErrorCodes.MODEL_NOT_FOUND],
    }
    return json.dumps(ret).encode() + b'\n'


async def check_model(pool: ReplicaPool, model_name: str) -> JSONResponse | None:
    from http import HTTPStatus

    from lmdeploy.serve.openai.api_server import create_error_response

    if model_name in pool.model_list:
        return None
    return create_error_response(HTTPStatus.NOT_FOUND, f'The model {model_name!r} does not exist.')
