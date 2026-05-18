# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from http import HTTPStatus

import aiohttp
import requests
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from lmdeploy.serve.openai.api_server import create_error_response
from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
    ModelPermission,
)
from lmdeploy.serve.proxy.config import AIOHTTP_TIMEOUT, ErrorCodes, ProxyConfig, err_msg
from lmdeploy.serve.proxy.forwarding import (
    forward_request,
    forward_request_stream,
)
from lmdeploy.serve.proxy.node import Node, NodeRegistry
from lmdeploy.serve.proxy.streaming import ProxyStreamingResponse
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

CONTROLLER_HEART_BEAT_EXPIRATION = int(os.getenv('LMDEPLOY_CONTROLLER_HEART_BEAT_EXPIRATION', 90))


def _heart_beat_controller(registry: NodeRegistry):
    """Background thread that removes stale nodes."""
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        logger.info('Start heart beat check')
        _remove_stale_nodes(registry)


def _remove_stale_nodes(registry: NodeRegistry):
    to_be_deleted = []
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return
    if loop.is_closed():
        return
    nodes = asyncio.run_coroutine_threadsafe(registry.all_nodes(), loop).result()
    for node in nodes:
        url = f"{node.url}/health"
        headers = {'accept': 'application/json'}
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                to_be_deleted.append(node.url)
        except Exception:
            to_be_deleted.append(node.url)
    for node_url in to_be_deleted:
        asyncio.run_coroutine_threadsafe(registry.remove(node_url), loop).result()
        logger.info(f"Removed node_url: {node_url} due to heart beat expiration")


def _handle_unavailable_model(model_name: str) -> bytes:
    logger.warning(f"no model name: {model_name}")
    ret = {
        'error_code': ErrorCodes.MODEL_NOT_FOUND,
        'text': err_msg[ErrorCodes.MODEL_NOT_FOUND],
    }
    return json.dumps(ret).encode() + b'\n'


def create_app(config: ProxyConfig, registry: NodeRegistry, strategy) -> FastAPI:
    """Create the FastAPI application with all routes.

    Args:
        config: Proxy configuration.
        registry: Node registry.
        strategy: Routing strategy instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: create a shared aiohttp session
        aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT) if AIOHTTP_TIMEOUT else None
        session = aiohttp.ClientSession(timeout=aiotimeout)
        strategy.client = session
        app.state.client = session
        await strategy.start()
        if not config.disable_cache_status:
            await registry.load()
        heart_beat_thread = threading.Thread(
            target=_heart_beat_controller, args=(registry,), daemon=True)
        heart_beat_thread.start()
        yield
        # Shutdown
        await strategy.stop()
        await session.close()

    app = FastAPI(docs_url='/', lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    # DistServe router (conditionally created)
    distserve = None
    if config.serving_strategy == config.serving_strategy.DIST_SERVE:
        from lmdeploy.serve.proxy.distserve import DistServeRouter
        distserve = DistServeRouter(registry, config)

    @app.get('/v1/models')
    async def available_models():
        model_cards = []
        model_names = await registry.list_models()
        for model_name in model_names:
            model_cards.append(
                ModelCard(id=model_name, root=model_name, permission=[ModelPermission()]))
        return ModelList(data=model_cards)

    @app.get('/nodes/status')
    async def node_status():
        try:
            nodes = await registry.all_nodes()
            return {n.url: n for n in nodes}
        except Exception:
            return False

    @app.post('/nodes/add', dependencies=[Depends(validate_json_request)])
    async def add_node(node: Node, raw_request: Request = None):
        try:
            await registry.add(
                node.url,
                role=node.role,
                models=node.models if node.models else None,
                status=node if node.models else None,
            )
            logger.info(f"add node {node.url} successfully")
            return 'Added successfully'
        except Exception:
            return 'Failed to add, please check the input url.'

    @app.post('/nodes/remove', dependencies=[Depends(validate_json_request)])
    async def remove_node(node: Node):
        try:
            await registry.remove(node.url)
            logger.info(f"delete node {node.url} successfully")
            return 'Deleted successfully'
        except Exception:
            logger.error(f"delete node {node.url} failed.")
            return 'Failed to delete, please check the input url.'

    @app.post('/nodes/terminate', dependencies=[Depends(validate_json_request)])
    async def terminate_node(node: Node):
        try:
            node_url = node.url
            success = True
            existing = await registry.get_by_url(node_url)
            if existing:
                await registry.remove(node_url)
                headers = {'accept': 'application/json'}
                try:
                    response = requests.get(f"{node_url}/terminate", headers=headers)
                    if response.status_code != 200:
                        success = False
                        logger.error(f"Failed to terminate node {node_url}, "
                                     f"error_code={response.status_code}, "
                                     f"error_msg={response.text}")
                except Exception as e:
                    logger.error(f"exception happened when terminating node {node_url}, {e}")
                    success = False
            else:
                logger.error(f"terminating node {node_url} failed since it does not exist.")
                success = False
            if not success:
                return f"Failed to terminate node {node_url}"
            return 'Terminated successfully'
        except Exception:
            logger.error(f"Terminate node {node.url} failed.")
            return f"Failed to terminate node {node.url}, please check the input url."

    @app.get('/nodes/terminate_all', dependencies=[Depends(validate_json_request)])
    async def terminate_node_all():
        try:
            nodes = await registry.all_nodes()
            all_success = True
            for node in nodes:
                try:
                    await registry.remove(node.url)
                    headers = {'accept': 'application/json'}
                    try:
                        response = requests.get(f"{node.url}/terminate", headers=headers)
                        if response.status_code != 200:
                            all_success = False
                    except Exception:
                        all_success = False
                except Exception:
                    all_success = False
            if not all_success:
                return 'Failed to terminate all nodes'
            return 'All nodes terminated successfully'
        except Exception:
            logger.error('Failed to terminate all nodes')
            return 'Failed to terminate all nodes.'

    # DistServe-specific endpoints
    if distserve is not None:
        @app.post('/distserve/connection_warmup', dependencies=[Depends(validate_json_request)])
        async def connection_warmup():
            await distserve.connection_warmup()
            return JSONResponse({'SUCCESS': True})

        @app.post('/distserve/gc', dependencies=[Depends(validate_json_request)])
        async def cache_block_gc_to_be_migrated():
            raise NotImplementedError

    async def _check_request_model(model_name: str) -> JSONResponse | None:
        models = await registry.list_models()
        if model_name in models:
            return None
        return create_error_response(HTTPStatus.NOT_FOUND, f"The model {model_name!r} does not exist.")

    @app.post('/v1/chat/completions', dependencies=[Depends(validate_json_request)])
    async def chat_completions_v1(request: ChatCompletionRequest, raw_request: Request = None):
        check_response = await _check_request_model(request.model)
        if check_response is not None:
            return check_response

        if distserve is not None:
            return await distserve.handle_chat_completions(request, raw_request, '/v1/chat/completions')

        node = await strategy.select_node(request.model)
        logger.info(f"A request is dispatched to {node.url}")
        await strategy.on_request_start(node)
        start = time.time()

        client = raw_request.app.state.client
        if request.stream is True:
            response = forward_request_stream(client, node.url, raw_request, '/v1/chat/completions')
            background_task = BackgroundTasks()
            background_task.add_task(strategy.on_request_end, node, time.time() - start)
            return ProxyStreamingResponse(response, background=background_task, media_type='text/event-stream')
        else:
            response = await forward_request(client, node.url, raw_request, '/v1/chat/completions')
            await strategy.on_request_end(node, time.time() - start)
            return JSONResponse(json.loads(response))

    @app.post('/v1/completions', dependencies=[Depends(validate_json_request)])
    async def completions_v1(request: CompletionRequest, raw_request: Request = None):
        check_response = await _check_request_model(request.model)
        if check_response is not None:
            return check_response

        if distserve is not None:
            return await distserve.handle_completions(request, raw_request, '/v1/completions')

        node = await strategy.select_node(request.model)
        logger.info(f"A request is dispatched to {node.url}")
        await strategy.on_request_start(node)
        start = time.time()

        client = raw_request.app.state.client
        if request.stream is True:
            response = forward_request_stream(client, node.url, raw_request, '/v1/completions')
            background_task = BackgroundTasks()
            background_task.add_task(strategy.on_request_end, node, time.time() - start)
            return ProxyStreamingResponse(response, background=background_task, media_type='text/event-stream')
        else:
            response = await forward_request(client, node.url, raw_request, '/v1/completions')
            await strategy.on_request_end(node, time.time() - start)
            return JSONResponse(json.loads(response))

    return app
