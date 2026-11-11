# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from lmdeploy.pytorch.disagg.conn.protocol import (
    DistServeCacheFreeRequest,
    DistServeConnectionRequest,
    DistServeDropConnectionRequest,
    DistServeInitRequest,
    PDControlRequest,
)


def register(router: APIRouter, server_context) -> None:
    """PD Disaggregation API Begin."""

    @router.get('/distserve/engine_info')
    async def engine_info():
        if server_context.async_engine.backend != 'pytorch':
            raise HTTPException(400, 'Version 2 PD requires the PyTorch backend.')
        return server_context.async_engine.engine.get_pd_engine_info()

    @router.post('/distserve/pd_control')
    async def pd_control(request: PDControlRequest):
        try:
            server_context.async_engine.engine.pd_control(request)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {'status': 'SUCCESS'}

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
