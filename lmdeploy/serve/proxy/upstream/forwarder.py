# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
from collections.abc import AsyncIterator
from enum import Enum
from http import HTTPStatus

import aiohttp
from fastapi import Request

from lmdeploy.serve.openai.protocol import ErrorResponse
from lmdeploy.serve.proxy.utils import APIServerException
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class ForwardMode(str, Enum):
    JSON = 'json'
    RAW = 'raw'


class UpstreamForwarder:
    """Forward requests to api_server replicas using a shared aiohttp
    session."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    @staticmethod
    def _prepare_headers(raw_request: Request) -> dict[str, str]:
        headers = dict((name, value) for name, value in raw_request.headers.items() if name.lower() != 'host')
        client_ip = raw_request.client.host if raw_request.client else 'unknown'
        headers.update({
            'X-Forwarded-For': client_ip,
            'X-Forwarded-Host': raw_request.headers.get('host', ''),
            'X-Forwarded-Proto': raw_request.url.scheme,
        })
        return headers

    @staticmethod
    def _gateway_error(replica_url: str, cause: Exception | None = None) -> APIServerException:
        if cause is not None:
            logger.error(f'caught an exception: {cause}')
        message = f'Failed to get response from replica {replica_url}.'
        body = ErrorResponse(
            message=message,
            type='server_error',
            code=HTTPStatus.BAD_GATEWAY.value,
        ).model_dump_json().encode()
        return APIServerException(status_code=HTTPStatus.BAD_GATEWAY.value, body=body)

    def _target_url(self, replica_url: str, endpoint: str) -> str:
        return replica_url.rstrip('/') + endpoint

    async def forward_json_buffer(self, payload: dict, replica_url: str, endpoint: str) -> str:
        try:
            target_url = self._target_url(replica_url, endpoint)
            async with self._session.post(target_url, json=payload) as response:
                body = await response.read()
                if response.status != HTTPStatus.OK:
                    raise APIServerException(status_code=response.status, body=body)
                return body.decode()
        except APIServerException:
            raise
        except (Exception, GeneratorExit, aiohttp.ClientError, asyncio.CancelledError) as e:
            raise self._gateway_error(replica_url, cause=e) from e

    async def forward_json_stream(self, payload: dict, replica_url: str,
                                  endpoint: str) -> AsyncIterator[bytes]:
        try:
            target_url = self._target_url(replica_url, endpoint)
            async with self._session.post(target_url, json=payload) as response:
                if response.status != HTTPStatus.OK:
                    error_body = await response.read()
                    raise APIServerException(status_code=response.status, body=error_body)
                async for line in response.content:
                    if line.strip():
                        yield line + b'\n\n'
        except APIServerException:
            raise
        except (Exception, GeneratorExit, aiohttp.ClientError) as e:
            raise self._gateway_error(replica_url, cause=e) from e

    async def forward_raw_buffer(self, raw_request: Request, replica_url: str, endpoint: str) -> str:
        try:
            target_url = self._target_url(replica_url, endpoint)
            headers = self._prepare_headers(raw_request)
            body_bytes = await raw_request.body()
            async with self._session.post(target_url, headers=headers, data=body_bytes) as response:
                body = await response.read()
                if response.status != HTTPStatus.OK:
                    raise APIServerException(status_code=response.status, body=body)
                return body.decode()
        except APIServerException:
            raise
        except (Exception, GeneratorExit, aiohttp.ClientError, asyncio.CancelledError) as e:
            raise self._gateway_error(replica_url, cause=e) from e

    async def forward_raw_stream(self, raw_request: Request, replica_url: str,
                                 endpoint: str) -> AsyncIterator[bytes]:
        try:
            target_url = self._target_url(replica_url, endpoint)
            headers = self._prepare_headers(raw_request)
            body_bytes = await raw_request.body()
            async with self._session.post(target_url, headers=headers, data=body_bytes) as response:
                if response.status != HTTPStatus.OK:
                    error_body = await response.read()
                    raise APIServerException(status_code=response.status, body=error_body)
                async for line in response.content:
                    if line.strip():
                        yield line + b'\n\n'
        except APIServerException:
            raise
        except (Exception, GeneratorExit, aiohttp.ClientError) as e:
            raise self._gateway_error(replica_url, cause=e) from e
