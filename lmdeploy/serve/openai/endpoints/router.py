# Copyright (c) OpenMMLab. All rights reserved.
"""Router assembly for OpenAI-compatible and LMDeploy endpoints."""

from fastapi import APIRouter

from . import auxiliary, chat_completions, completions, distserve, generate, management, models


def create_openai_router(server_context) -> APIRouter:
    """Create a router containing all legacy API-server endpoints."""
    router = APIRouter()
    models.register(router, server_context)
    management.register(router, server_context)
    chat_completions.register(router, server_context)
    completions.register(router, server_context)
    generate.register(router, server_context)
    auxiliary.register(router, server_context)
    distserve.register(router, server_context)
    return router
