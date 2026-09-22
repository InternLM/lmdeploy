# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

from fastapi import FastAPI

from lmdeploy.serve.openai.api_server import ServerContext
from lmdeploy.serve.openai.endpoints import create_openai_router
from lmdeploy.serve.openai.responses import create_responses_router


def test_responses_openapi_router_is_included_with_openai_router():
    context = ServerContext()
    app = FastAPI()
    app.include_router(create_openai_router(context))
    app.include_router(create_responses_router(context))

    assert '/v1/responses' in app.openapi()['paths']
