# Copyright (c) OpenMMLab. All rights reserved.

from fastapi import APIRouter, Depends, Request

from lmdeploy.pytorch.disagg.config import ServingStrategy
from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
    ModelPermission,
)
from lmdeploy.serve.proxy.dispatch.base import ProxyContext, check_model
from lmdeploy.serve.proxy.runtime import ProxyRuntime
from lmdeploy.serve.utils.server_utils import validate_json_request


def get_runtime(request: Request) -> ProxyRuntime:
    return request.app.state.runtime


router = APIRouter()


@router.get('/v1/models')
def available_models(runtime: ProxyRuntime = Depends(get_runtime)):
    model_cards = [
        ModelCard(id=name, root=name, permission=[ModelPermission()])
        for name in runtime.pool.model_list
    ]
    return ModelList(data=model_cards)


async def _dispatch(ctx: ProxyContext, runtime: ProxyRuntime):
    check_response = await check_model(runtime.pool, ctx.model)
    if check_response is not None:
        return check_response
    if runtime.config.serving_strategy == ServingStrategy.Hybrid:
        return await runtime.hybrid.dispatch(ctx)
    if runtime.config.serving_strategy == ServingStrategy.DistServe:
        return await runtime.distserve.dispatch(ctx)
    raise ValueError(f'No serving strategy named {runtime.config.serving_strategy}')


@router.post('/v1/chat/completions', dependencies=[Depends(validate_json_request)])
async def chat_completions_v1(request: ChatCompletionRequest, raw_request: Request,
                              runtime: ProxyRuntime = Depends(get_runtime)):
    ctx = ProxyContext(
        model=request.model,
        stream=request.stream is True,
        endpoint='/v1/chat/completions',
        raw_request=raw_request,
        parsed_request=request,
        request_dict=request.model_dump(),
    )
    return await _dispatch(ctx, runtime)


@router.post('/v1/completions', dependencies=[Depends(validate_json_request)])
async def completions_v1(request: CompletionRequest, raw_request: Request,
                         runtime: ProxyRuntime = Depends(get_runtime)):
    ctx = ProxyContext(
        model=request.model,
        stream=request.stream is True,
        endpoint='/v1/completions',
        raw_request=raw_request,
        parsed_request=request,
        request_dict=request.model_dump(),
    )
    return await _dispatch(ctx, runtime)
