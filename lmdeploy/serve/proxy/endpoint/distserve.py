# Copyright (c) OpenMMLab. All rights reserved.

import asyncio

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage
from lmdeploy.serve.proxy.runtime import ProxyRuntime
from lmdeploy.serve.utils.server_utils import validate_json_request


def get_runtime(request: Request) -> ProxyRuntime:
    return request.app.state.runtime


router = APIRouter()


@router.post('/distserve/connection_warmup', dependencies=[Depends(validate_json_request)])
async def connection_warmup(runtime: ProxyRuntime = Depends(get_runtime)):
    await asyncio.gather(*[
        runtime.pool.pd_connection_pool.connect(
            PDConnectionMessage(
                p_url=p_url,
                d_url=d_url,
                protocol=runtime.config.migration_protocol,
                rdma_config=runtime.config.rdma_config,
            )) for p_url in runtime.pool.get_by_role(EngineRole.Prefill)
        for d_url in runtime.pool.get_by_role(EngineRole.Decode)
    ])
    return JSONResponse({'SUCCESS': True})


@router.post('/distserve/gc', dependencies=[Depends(validate_json_request)])
async def cache_block_gc_to_be_migrated():
    raise NotImplementedError
