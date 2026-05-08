# Copyright (c) OpenMMLab. All rights reserved.
"""Router assembly for Anthropic-compatible endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from .endpoints import messages, messages_count_tokens, models


def create_anthropic_router(server_context) -> APIRouter:
    """Create router with all Anthropic endpoints."""

    router = APIRouter(tags=['anthropic'])
    messages.register(router, server_context)
    messages_count_tokens.register(router, server_context)
    models.register(router, server_context)
    return router
