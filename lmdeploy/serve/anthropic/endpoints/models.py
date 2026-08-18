# Copyright (c) OpenMMLab. All rights reserved.
"""Endpoint for Anthropic-scoped model listing."""

from __future__ import annotations

from fastapi import APIRouter

from ..adapter import get_model_list
from ..protocol import AnthropicModel, AnthropicModelList


def register(router: APIRouter, server_context) -> None:
    """Register endpoint onto router."""

    @router.get('/anthropic/v1/models')
    async def list_models():
        models = [AnthropicModel(id=name, display_name=name) for name in get_model_list(server_context)]
        first_id = models[0].id if models else None
        last_id = models[-1].id if models else None
        return AnthropicModelList(data=models, first_id=first_id, last_id=last_id).model_dump()
