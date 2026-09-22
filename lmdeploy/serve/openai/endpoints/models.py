# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from fastapi import APIRouter

from lmdeploy.serve.openai.protocol import (
    ModelCard,
    ModelList,
    ModelPermission,
)
from lmdeploy.serve.openai.utils import get_model_list


def register(router: APIRouter, server_context) -> None:

    @router.get('/v1/models')
    def available_models():
        """Show available models."""
        model_cards = []
        for model_name in get_model_list(server_context):
            model_cards.append(
                ModelCard(id=model_name,
                          root=model_name,
                          permission=[ModelPermission()]))
        return ModelList(data=model_cards)
