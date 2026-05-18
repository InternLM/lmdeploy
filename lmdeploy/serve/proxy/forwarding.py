# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json

import aiohttp

from lmdeploy.serve.proxy.config import ErrorCodes, err_msg
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def prepare_headers(raw_request) -> dict[str, str]:
    """Prepare forwarding headers from the original request."""
    headers = {name: value for name, value in raw_request.headers.items() if name.lower() != 'host'}
    client_ip = raw_request.client.host if raw_request.client else 'unknown'
    headers.update({
        'X-Forwarded-For': client_ip,
        'X-Forwarded-Host': raw_request.headers.get('host', ''),
        'X-Forwarded-Proto': raw_request.url.scheme,
    })
    return headers


def handle_api_timeout(node_url: str) -> bytes:
    """Handle the api timeout."""
    logger.warning(f"api timeout: {node_url}")
    ret = {
        'error_code': ErrorCodes.API_TIMEOUT.value,
        'text': err_msg[ErrorCodes.API_TIMEOUT],
    }
    return json.dumps(ret).encode() + b'\n'


async def forward_request_stream(client: aiohttp.ClientSession, node_url: str,
                                  raw_request, endpoint: str):
    """Forward a raw HTTP request as a streaming response.

    Yields response chunks. On error, yields an error payload.
    """
    from lmdeploy.serve.proxy.config import APIServerException
    try:
        target_url = node_url.rstrip('/') + endpoint
        headers = prepare_headers(raw_request)
        body_bytes = await raw_request.body()
        async with client.post(target_url, headers=headers, data=body_bytes) as response:
            if response.status != 200:
                error_body = await response.read()
                raise APIServerException(status_code=response.status, body=error_body)
            async for line in response.content:
                if line.strip():
                    yield line + b'\n\n'
    except APIServerException:
        raise
    except (Exception, GeneratorExit, aiohttp.ClientError) as e:
        logger.error(f"caught an exception: {e}")
        yield handle_api_timeout(node_url)


async def forward_request(client: aiohttp.ClientSession, node_url: str,
                           raw_request, endpoint: str) -> str:
    """Forward a raw HTTP request and return the response text."""
    try:
        target_url = node_url.rstrip('/') + endpoint
        headers = prepare_headers(raw_request)
        body_bytes = await raw_request.body()
        async with client.post(target_url, headers=headers, data=body_bytes) as response:
            return await response.text()
    except (Exception, GeneratorExit, aiohttp.ClientError, asyncio.CancelledError) as e:
        logger.error(f"caught an exception: {e}")
        return handle_api_timeout(node_url).decode()


async def generate(client: aiohttp.ClientSession, request: dict,
                    node_url: str, endpoint: str) -> str:
    """Forward a parsed dict request and return the response text."""
    try:
        async with client.post(node_url + endpoint, json=request) as response:
            return await response.text()
    except (Exception, GeneratorExit, aiohttp.ClientError, asyncio.CancelledError) as e:
        logger.error(f"caught an exception: {e}")
        return handle_api_timeout(node_url).decode()


async def stream_generate(client: aiohttp.ClientSession, request: dict,
                           node_url: str, endpoint: str):
    """Forward a parsed dict request as a streaming response."""
    try:
        async with client.post(node_url + endpoint, json=request) as response:
            async for line in response.content:
                if line.strip():
                    yield line + b'\n\n'
    except (Exception, GeneratorExit, aiohttp.ClientError) as e:
        logger.error(f"caught an exception: {e}")
        yield handle_api_timeout(node_url)
