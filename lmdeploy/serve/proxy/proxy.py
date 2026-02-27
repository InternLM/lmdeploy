# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import copy
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from .node_manager import Node

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from lmdeploy.pytorch.disagg.config import DistServeRDMAConfig, EngineRole, RDMALinkType, ServingStrategy
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage
from lmdeploy.serve.openai.api_server import check_api_key
from lmdeploy.serve.openai.protocol import ModelCard  # noqa: E501
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest, ModelList, ModelPermission
from lmdeploy.serve.proxy.constants import LATENCY_DEQUE_LEN, RoutingStrategy
from lmdeploy.serve.proxy.node_manager import NodeManager, connector
from lmdeploy.utils import get_logger

from .constants import ErrorCodes


class Status(BaseModel):
    """Status protocol consists of models' information."""
    role: EngineRole = EngineRole.Hybrid
    models: Optional[list[str]] = Field(default_factory=list, examples=[[]])
    unfinished: int = 0
    latency: Deque = Field(default_factory=lambda: deque(maxlen=LATENCY_DEQUE_LEN), examples=[[]])
    speed: Optional[int] = Field(default=None, examples=[None])


class NodeModel(BaseModel):
    """Node protocol for API requests (Pydantic model)."""
    url: str
    status: Optional[Status] = None


@dataclass
class AppSettings:
    serving_strategy: ServingStrategy = ServingStrategy.Hybrid
    routing_strategy: RoutingStrategy = RoutingStrategy.MIN_EXPECTED_LATENCY
    migration_protocol: MigrationProtocol = MigrationProtocol.RDMA
    dummy_prefill: bool = False
    api_keys: Optional[Union[list[str], str]] = None
    rdma_config: DistServeRDMAConfig = DistServeRDMAConfig(
        link_type=RDMALinkType.RoCE,
        with_gdr=True,
    )
    pd_connection_pool: PDConnectionPool = PDConnectionPool()


app = FastAPI(docs_url='/')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
app_settings = AppSettings()
node_manager = NodeManager()
logger = get_logger('lmdeploy')


def report_model_not_found(content: str):
    """Report model not found error."""
    return JSONResponse(status_code=404, content={'error_code': ErrorCodes.MODEL_NOT_FOUND.value, 'error_msg': content})


async def _generate_response(node: 'Node', request_dict: dict, endpoint: str,
                             stream: bool) -> Union[StreamingResponse, JSONResponse]:
    """Generate streaming or non-streaming response.

    Args:
        node: The node to handle the request.
        request_dict: The request dictionary.
        endpoint: The API endpoint.
        stream: Whether to stream the response.

    Returns:
        StreamingResponse or JSONResponse.
    """
    start = node.pre_call()
    if stream:
        response = node.stream_generate(request_dict, endpoint)
        background_task = BackgroundTasks()
        background_task.add_task(node.post_call, start)
        return StreamingResponse(response, background=background_task, media_type='text/event-stream')
    else:
        response = await node.generate(request_dict, endpoint)
        node.post_call(start)
        return JSONResponse(json.loads(response))


async def _handle_hybrid_request(request: Union[ChatCompletionRequest, CompletionRequest],
                                 endpoint: str) -> Union[StreamingResponse, JSONResponse]:
    """Handle request with Hybrid serving strategy.

    Args:
        request: The request object (ChatCompletionRequest or CompletionRequest).
        endpoint: The API endpoint.

    Returns:
        StreamingResponse or JSONResponse.
    """
    node = node_manager.get_node(request.model, EngineRole.Hybrid)
    if not node:
        return report_model_not_found(f'The model {request.model} is not available.')

    logger.info(f'A request is dispatched to {node.url}')
    request_dict = request.model_dump()
    return await _generate_response(node, request_dict, endpoint, request.stream)


async def _handle_distserve_prefill(request: Union[ChatCompletionRequest, CompletionRequest],
                                    endpoint: str) -> Tuple[dict, str, Optional['Node']]:
    """Handle DistServe prefill phase.

    Returns:
        Tuple of (prefill_info, p_url, p_node). Returns (None, None, None) if no prefill node.
    """
    if app_settings.dummy_prefill:
        return {}, 'dummy:dummy', None

    p_node = node_manager.get_node(request.model, EngineRole.Prefill)
    if not p_node:
        return None, None, None

    prefill_request_dict = copy.deepcopy(request.model_dump())
    prefill_request_dict.update({'max_tokens': 1, 'stream': False, 'with_cache': True, 'preserve_cache': True})
    if endpoint == '/v1/chat/completions':
        prefill_request_dict['max_completion_tokens'] = 1

    logger.info(f'A Prefill request is dispatched to {p_node.url}')
    start = p_node.pre_call()
    response_text = await p_node.generate(prefill_request_dict, endpoint)
    prefill_info = json.loads(response_text)
    p_node.post_call(start)

    return prefill_info, p_node.url, p_node


async def _ensure_pd_connection(p_url: str, d_url: str, with_error_handling: bool = False) -> Optional[JSONResponse]:
    """Ensure PD connection is established.

    Returns:
        None if successful, JSONResponse if error and with_error_handling is True.
        Raises exception if error and with_error_handling is False.
    """
    if app_settings.dummy_prefill or app_settings.pd_connection_pool.is_connected(p_url, d_url):
        return None

    try:
        await app_settings.pd_connection_pool.connect(
            PDConnectionMessage(
                p_url=p_url,
                d_url=d_url,
                protocol=app_settings.migration_protocol,
                rdma_config=app_settings.rdma_config,
            ))
    except Exception as e:
        if with_error_handling:
            logger.error(f'Connection error: {e}')
            return JSONResponse(status_code=500,
                                content={
                                    'error': 'Connection error',
                                    'message': f'Cannot establish connection {(p_url, d_url)}'
                                })
        raise
    return None


def _build_migration_request(prefill_info: dict, p_url: str):
    """Build migration request from prefill info.

    Args:
        prefill_info: Prefill response information.
        p_url: Prefill node URL.

    Returns:
        MigrationRequest dictionary.
    """
    remote_session_id = int(prefill_info.get('id')) if prefill_info.get('id') else 0
    remote_block_ids = prefill_info.get('cache_block_ids') or []
    remote_token_ids = prefill_info.get('remote_token_ids', [])
    remote_token_id = remote_token_ids[-1] if remote_token_ids else 0

    return MigrationRequest(protocol=app_settings.migration_protocol,
                            remote_engine_id=p_url,
                            remote_session_id=remote_session_id,
                            remote_block_ids=remote_block_ids,
                            remote_token_id=remote_token_id,
                            is_dummy_prefill=app_settings.dummy_prefill).model_dump(mode='json')


async def _handle_distserve_decode(request: Union[ChatCompletionRequest, CompletionRequest],
                                   endpoint: str,
                                   prefill_info: dict,
                                   p_url: str,
                                   ensure_connection: bool = False,
                                   handle_shelf: bool = False) -> Union[StreamingResponse, JSONResponse]:
    """Handle DistServe decode phase."""
    d_node = node_manager.get_node(request.model, EngineRole.Decode)
    if not d_node:
        return report_model_not_found(f'The decode node for model {request.model} is not available.')

    logger.info(f'A Decode request is dispatched to {d_node.url}')

    # Ensure connection if needed
    if ensure_connection:
        conn_error = await _ensure_pd_connection(p_url, d_node.url, with_error_handling=True)
        if conn_error:
            return conn_error
        if handle_shelf and not app_settings.dummy_prefill:
            prefill_id = prefill_info.get('id')
            if prefill_id:
                app_settings.pd_connection_pool.shelf_prefill_session((p_url, d_node.url), prefill_id)

    request_dict = request.model_dump()
    request_dict['migration_request'] = _build_migration_request(prefill_info, p_url)
    resp = await _generate_response(d_node, request_dict, endpoint, request.stream)

    # Cleanup
    if not app_settings.dummy_prefill:
        prefill_id = prefill_info.get('id')
        if prefill_id:
            app_settings.pd_connection_pool.unshelf_prefill_session((p_url, d_node.url), prefill_id)

    return resp


@app.on_event('startup')
async def startup_event():
    """Initialize session when application starts."""
    await connector.get_session()


@app.on_event('shutdown')
async def shutdown_event():
    """Cleanup resources when application shuts down."""
    await connector.cleanup()


@app.get('/v1/models', dependencies=[Depends(check_api_key)])
def available_models():
    """Show available models."""
    return ModelList(
        data=[ModelCard(id=name, root=name, permission=[ModelPermission()]) for name in node_manager.model_list])


@app.get('/nodes/status', dependencies=[Depends(check_api_key)])
def node_status():
    """Show nodes status."""
    try:
        return node_manager.nodes
    except Exception as e:
        logger.error(f'Failed to get node status: {e}')
        return JSONResponse(status_code=500, content={'error': 'Failed to get node status', 'message': str(e)})


@app.post('/nodes/add', dependencies=[Depends(check_api_key)])
async def add_node(node: NodeModel):
    """Add a node to the manager."""
    try:
        if node.status is None:
            from lmdeploy.serve.openai.api_client import APIClient
            node.status = Status(models=APIClient(api_server_url=node.url).available_models)
        await node_manager.add(node.url, node.status)
        logger.info(f'add node {node.url} successfully')
        return JSONResponse(status_code=200, content={'message': 'Added successfully', 'url': node.url})
    except Exception as e:
        logger.error(f'add node {node.url} failed: {e}')
        return JSONResponse(status_code=500, content={'error': 'Failed to add node', 'message': str(e)})


@app.post('/nodes/remove', dependencies=[Depends(check_api_key)])
async def remove_node(node: NodeModel):
    """Remove a node."""
    try:
        await node_manager.remove(node.url)
        app_settings.pd_connection_pool.dereg_instance(node.url)
        logger.info(f'removed node {node.url} successfully')
        return JSONResponse(status_code=200, content={'message': 'Removed successfully', 'url': node.url})
    except Exception as e:
        logger.error(f'remove node {node.url} failed: {e}')
        return JSONResponse(status_code=500, content={'error': 'Failed to remove node', 'message': str(e)})


@app.post('/nodes/terminate', dependencies=[Depends(check_api_key)])
async def terminate_node(node: NodeModel):
    """Terminate a node."""
    try:
        await node_manager.terminate_node(node.url)
        logger.info(f'Terminated node {node.url} successfully')
        return JSONResponse(status_code=200, content={'message': 'Terminated successfully', 'url': node.url})
    except Exception as e:
        logger.error(f'Failed to terminate node {node.url}: {e}')
        return JSONResponse(status_code=500, content={'error': 'Failed to terminate node', 'message': str(e)})


@app.get('/nodes/terminate_all', dependencies=[Depends(check_api_key)])
async def terminate_node_all():
    """Terminate all nodes."""
    try:
        await node_manager.terminate_all_nodes()
        return JSONResponse(status_code=200, content={'message': 'All nodes terminated successfully'})
    except Exception as e:
        logger.error(f'Failed to terminate all nodes: {e}')
        return JSONResponse(status_code=500, content={'error': 'Failed to terminate all nodes', 'message': str(e)})


@app.post('/distserve/connection_warmup')
async def connection_warmup():
    await asyncio.gather(*[
        app_settings.pd_connection_pool.connect(
            PDConnectionMessage(
                p_url=p_url,
                d_url=d_url,
                protocol=app_settings.migration_protocol,
                rdma_config=app_settings.rdma_config,
            )) for p_url in node_manager.prefill_nodes.keys() for d_url in node_manager.decode_nodes.keys()
    ])
    return JSONResponse({'SUCCESS': True})


@app.post('/distserve/gc')
async def cache_block_gc_to_be_migrated():
    # TODO (JimyMa): add garbage collection of to be migrated request
    raise NotImplementedError


async def _handle_request(request: Union[ChatCompletionRequest, CompletionRequest],
                          endpoint: str,
                          is_chat: bool = False) -> Union[StreamingResponse, JSONResponse]:
    """Handle completion request (unified for chat and completions)."""
    if app_settings.serving_strategy == ServingStrategy.Hybrid:
        return await _handle_hybrid_request(request, endpoint)
    elif app_settings.serving_strategy == ServingStrategy.DistServe:
        prefill_info, p_url, p_node = await _handle_distserve_prefill(request, endpoint)
        if p_node is None and not app_settings.dummy_prefill:
            return report_model_not_found(f'The prefill node for model {request.model} is not available.')

        # For chat_completions, ensure connection without error handling
        if is_chat and not app_settings.dummy_prefill:
            d_node = node_manager.get_node(request.model, EngineRole.Decode)
            if d_node:
                await _ensure_pd_connection(p_url, d_node.url, with_error_handling=False)

        return await _handle_distserve_decode(request,
                                              endpoint,
                                              prefill_info,
                                              p_url,
                                              ensure_connection=not is_chat,
                                              handle_shelf=not is_chat)
    else:
        raise ValueError(f'No serving strategy named {app_settings.serving_strategy}')


@app.post('/v1/chat/completions', dependencies=[Depends(check_api_key)])
async def chat_completions_v1(request: ChatCompletionRequest):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/chat/create
    """
    return await _handle_request(request, '/v1/chat/completions', is_chat=True)


@app.post('/v1/completions', dependencies=[Depends(check_api_key)])
async def completions_v1(request: CompletionRequest):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    """
    return await _handle_request(request, '/v1/completions', is_chat=False)


def proxy(server_name: str = '0.0.0.0',
          server_port: int = 8000,
          serving_strategy: Literal['Hybrid', 'DistServe'] = 'Hybrid',
          routing_strategy: Literal['random', 'min_expected_latency', 'min_observed_latency'] = 'min_expected_latency',
          api_keys: Optional[Union[list[str], str]] = None,
          ssl: bool = False,
          log_level: str = 'INFO',
          link_type: Literal['RoCE', 'IB'] = 'RoCE',
          migration_protocol: Literal['RDMA'] = 'RDMA',
          dummy_prefill: bool = False,
          **kwargs):
    """To launch the proxy server.

    Args:
        server_name (str): the server name of the proxy. Default to '0.0.0.0'.
        server_port (str): the server port. Default to 8000.
        serving_strategy ('Hybrid' | 'DistServe'):  the strategy to serving. Hybrid default.
            DistServe for PD Disaggregation.
        route_strategy ('random' | 'min_expected_latency' | 'min_observed_latency'):
            the strategy to dispatch requests to nodes. Default to
            'min_expected_latency'
        api_keys (List[str] | str | None): Optional list of API keys. Accepts string type as
            a single api_key. Default to None, which means no api key applied.
        ssl (bool): Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.
        log_level (str): Set the log level. Default to INFO.
        migration_protocol: migration protocol when PD disaggregation. RDMA default.
    """  # noqa
    app_settings.serving_strategy = ServingStrategy[serving_strategy]
    app_settings.migration_protocol = MigrationProtocol[migration_protocol]
    app_settings.dummy_prefill = dummy_prefill
    app_settings.rdma_config = DistServeRDMAConfig(
        link_type=RDMALinkType[link_type],
        with_gdr=True,
    )
    node_manager.routing_strategy = RoutingStrategy.from_str(routing_strategy)
    if api_keys is not None:
        if isinstance(api_keys, str):
            api_keys = api_keys.split(',')
        app_settings.api_keys = api_keys
    ssl_keyfile, ssl_certfile = None, None
    if ssl:
        ssl_keyfile = os.environ['SSL_KEYFILE']
        ssl_certfile = os.environ['SSL_CERTFILE']
    logger.setLevel(log_level)
    uvicorn_log_level = os.getenv('UVICORN_LOG_LEVEL', 'info').lower()
    uvicorn.run(app=app,
                host=server_name,
                port=server_port,
                log_level=uvicorn_log_level,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile)


if __name__ == '__main__':
    import fire

    fire.Fire(proxy)
