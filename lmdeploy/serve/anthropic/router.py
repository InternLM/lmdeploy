# Copyright (c) OpenMMLab. All rights reserved.
"""Router assembly for Anthropic-compatible endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from .adapter import detect_inline_system_support
from .endpoints import messages, messages_count_tokens, models


def create_anthropic_router(server_context) -> APIRouter:
    """Create router with all Anthropic endpoints."""

    router = APIRouter(tags=['anthropic'])
    merge_inline_system = not detect_inline_system_support(server_context.async_engine.chat_template)
    messages.register(router, server_context, merge_inline_system=merge_inline_system)
    messages_count_tokens.register(router, server_context, merge_inline_system=merge_inline_system)
    models.register(router, server_context)
    return router
