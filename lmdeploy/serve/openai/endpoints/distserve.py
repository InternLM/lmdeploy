# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from lmdeploy.pytorch.disagg.config import DistServeEngineConfig
from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeCacheFreeRequest,
    DistServeConnectionRequest,
    DistServeDropConnectionRequest,
    DistServeInitRequest,
)


def register(router: APIRouter, server_context) -> None:
    """PD Disaggregation API Begin."""

    @router.get('/distserve/engine_info')
    async def engine_info():
        engine_config = server_context.async_engine.backend_config

        response = DistServeEngineConfig(
            tp_size=engine_config.tp,
            dp_size=engine_config.dp,
            pp_size=None,
            ep_size=engine_config.ep,
            dp_rank=engine_config.dp_rank,
            block_size=engine_config.block_size,
            num_cpu_blocks=engine_config.num_cpu_blocks,
            num_gpu_blocks=engine_config.num_gpu_blocks)

        return response.model_dump_json()

    @router.post('/distserve/p2p_initialize')
    async def p2p_initialize(init_request: DistServeInitRequest):
        return server_context.async_engine.p2p_initialize(init_request)

    @router.post('/distserve/p2p_connect')
    async def p2p_connect(conn_request: DistServeConnectionRequest):
        return server_context.async_engine.p2p_connect(conn_request)

    @router.post('/distserve/p2p_drop_connect')
    async def p2p_drop_connect(
            drop_conn_request: DistServeDropConnectionRequest):
        return server_context.async_engine.p2p_drop_connect(drop_conn_request)

    @router.post('/distserve/free_cache')
    async def free_cache(
            cache_free_request: DistServeCacheFreeRequest) -> JSONResponse:
        session_id = cache_free_request.remote_session_id
        server_context.async_engine.free_cache(session_id)
        return {'status': 'SUCCESS'}
