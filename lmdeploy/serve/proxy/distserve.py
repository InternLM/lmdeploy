# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import copy
import json
import time

import aiohttp

from lmdeploy.pytorch.disagg.config import DistServeRDMAConfig, EngineRole, RDMALinkType
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage
from lmdeploy.serve.proxy.config import AIOHTTP_TIMEOUT, ErrorCodes, ProxyConfig, err_msg
from lmdeploy.serve.proxy.node import NodeRegistry
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class DistServeRouter:
    """Handles DistServe (prefill-decode disaggregation) routing."""

    def __init__(self, registry: NodeRegistry, config: ProxyConfig):
        self.registry = registry
        self.config = config
        self.migration_protocol = MigrationProtocol[config.migration_protocol]
        self.rdma_config = DistServeRDMAConfig(
            with_gdr=not config.disable_gdr,
            link_type=RDMALinkType[config.link_type],
        )
        self.pd_connection_pool = PDConnectionPool()
        self.dummy_prefill = config.dummy_prefill
        self._aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT) if AIOHTTP_TIMEOUT else None

    async def handle_chat_completions(self, request, raw_request, endpoint: str):
        """Handle a chat completion request in DistServe mode."""
        request_dict = request.model_dump()
        return await self._handle_request(request_dict, request, raw_request, endpoint, request.stream)

    async def handle_completions(self, request, raw_request, endpoint: str):
        """Handle a completion request in DistServe mode."""
        request_dict = request.model_dump()
        return await self._handle_request(request_dict, request, raw_request, endpoint, request.stream)

    async def _handle_request(self, request_dict, request, raw_request, endpoint: str, stream: bool):
        from fastapi import BackgroundTasks
        from fastapi.responses import JSONResponse, StreamingResponse

        from lmdeploy.serve.proxy.forwarding import generate, stream_generate

        model_name = request.model

        # Prefill
        prefill_request_dict = copy.deepcopy(request_dict)
        prefill_request_dict['max_tokens'] = 1
        prefill_request_dict['max_completion_tokens'] = 1
        prefill_request_dict['stream'] = False
        prefill_request_dict['with_cache'] = True
        prefill_request_dict['preserve_cache'] = True

        prefill_info = {}
        p_url = 'dummy:dummy'
        if not self.dummy_prefill:
            p_nodes = await self.registry.get(model_name, role=EngineRole.Prefill)
            if not p_nodes:
                return self._handle_unavailable_model(model_name)
            p_url = p_nodes[0].url
            logger.info(f"A Prefill request is dispatched to {p_url}")

            node = await self.registry.get_by_url(p_url)
            node.unfinished += 1
            start = time.time()

            async with aiohttp.ClientSession(timeout=self._aiotimeout) as client:
                prefill_info = json.loads(await generate(client, prefill_request_dict, p_url, endpoint))

            node.unfinished -= 1
            node.latency.append(time.time() - start)

        # Decode
        d_nodes = await self.registry.get(model_name, role=EngineRole.Decode)
        if not d_nodes:
            return self._handle_unavailable_model(model_name)
        d_url = d_nodes[0].url
        logger.info(f"A Decode request is dispatched to {d_url}")

        if not self.dummy_prefill:
            if not self.pd_connection_pool.is_connected(p_url, d_url):
                await self.pd_connection_pool.connect(
                    PDConnectionMessage(
                        p_url=p_url,
                        d_url=d_url,
                        protocol=self.migration_protocol,
                        rdma_config=self.rdma_config,
                    ))

        remote_session_id = int(prefill_info.get('id')) if prefill_info.get('id') else 0
        remote_block_ids = prefill_info.get('cache_block_ids') or []
        remote_token_id = prefill_info.get('remote_token_ids')[-1] if prefill_info.get('remote_token_ids') else 0

        request_dict['migration_request'] = MigrationRequest(
            protocol=self.migration_protocol,
            remote_engine_id=p_url,
            remote_session_id=remote_session_id,
            remote_block_ids=remote_block_ids,
            remote_token_id=remote_token_id,
            is_dummy_prefill=self.dummy_prefill,
        ).model_dump(mode='json')

        d_node = await self.registry.get_by_url(d_url)
        d_node.unfinished += 1
        start = time.time()

        if not self.dummy_prefill:
            self.pd_connection_pool.shelf_prefill_session((p_url, d_url), prefill_info['id'])

        async with aiohttp.ClientSession(timeout=self._aiotimeout) as client:
            if stream:
                response = stream_generate(client, request_dict, d_url, endpoint)
                background_task = BackgroundTasks()
                resp = StreamingResponse(response, background=background_task, media_type='text/event-stream')
            else:
                response_text = await generate(client, request_dict, d_url, endpoint)
                d_node.unfinished -= 1
                d_node.latency.append(time.time() - start)
                resp = JSONResponse(json.loads(response_text))

        if not self.dummy_prefill:
            self.pd_connection_pool.unshelf_prefill_session((p_url, d_url), prefill_info.get('id'))

        return resp

    def _handle_unavailable_model(self, model_name: str) -> bytes:
        logger.warning(f"no model name: {model_name}")
        ret = {
            'error_code': ErrorCodes.MODEL_NOT_FOUND,
            'text': err_msg[ErrorCodes.MODEL_NOT_FOUND],
        }
        return json.dumps(ret).encode() + b'\n'

    async def connection_warmup(self):
        """Warm up all PD connections."""
        p_nodes = await self.registry.get_nodes_by_role(EngineRole.Prefill)
        d_nodes = await self.registry.get_nodes_by_role(EngineRole.Decode)
        await asyncio.gather(*[
            self.pd_connection_pool.connect(
                PDConnectionMessage(
                    p_url=p_url,
                    d_url=d_url,
                    protocol=self.migration_protocol,
                    rdma_config=self.rdma_config,
                )) for p_url in p_nodes for d_url in d_nodes
        ])
