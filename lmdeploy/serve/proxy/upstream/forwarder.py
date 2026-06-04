# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json
from collections.abc import AsyncIterator
from enum import Enum
from http import HTTPStatus

import aiohttp
from fastapi import Request

from lmdeploy.serve.proxy.core.errors import ErrorCodes, err_msg
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
    def api_timeout_bytes(url: str) -> bytes:
        logger.warning(f'api timeout: {url}')
        ret = {
            'error_code': ErrorCodes.API_TIMEOUT.value,
            'text': err_msg[ErrorCodes.API_TIMEOUT],
        }
        return json.dumps(ret).encode() + b'\n'

    def _target_url(self, replica_url: str, endpoint: str) -> str:
        return replica_url.rstrip('/') + endpoint

    async def forward_json_buffer(self, payload: dict, replica_url: str, endpoint: str) -> str | bytes:
        try:
            target_url = self._target_url(replica_url, endpoint)
            async with self._session.post(target_url, json=payload) as response:
                if response.status != HTTPStatus.OK:
                    return self.api_timeout_bytes(replica_url)
                return await response.text()
        except (Exception, GeneratorExit, aiohttp.ClientError, asyncio.CancelledError) as e:
            logger.error(f'caught an exception: {e}')
            return self.api_timeout_bytes(replica_url)

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
            logger.error(f'caught an exception: {e}')
            yield self.api_timeout_bytes(replica_url)

    async def forward_raw_buffer(self, raw_request: Request, replica_url: str, endpoint: str) -> str | bytes:
        try:
            target_url = self._target_url(replica_url, endpoint)
            headers = self._prepare_headers(raw_request)
            body_bytes = await raw_request.body()
            async with self._session.post(target_url, headers=headers, data=body_bytes) as response:
                if response.status != HTTPStatus.OK:
                    return self.api_timeout_bytes(replica_url)
                return await response.text()
        except (Exception, GeneratorExit, aiohttp.ClientError, asyncio.CancelledError) as e:
            logger.error(f'caught an exception: {e}')
            return self.api_timeout_bytes(replica_url)

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
            logger.error(f'caught an exception: {e}')
            yield self.api_timeout_bytes(replica_url)
