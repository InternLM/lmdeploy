# Copyright (c) OpenMMLab. All rights reserved.
"""Shared helpers for OpenAI-compatible endpoint handlers."""

from http import HTTPStatus

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.core.generation_config import build_generation_config
from lmdeploy.serve.openai.errors import create_error_response
from lmdeploy.serve.openai.utils import get_model_list


def build_serving_generation_config(request, server_context,
                                    **extra_kwargs) -> GenerationConfig:
    """Merge request sampling values with server generation defaults."""
    max_new_tokens = getattr(request, 'max_completion_tokens', None)
    if max_new_tokens is None:
        max_new_tokens = getattr(request, 'max_tokens', None)
    return build_generation_config(
        request,
        server_context.default_gen_config,
        max_new_tokens=max_new_tokens,
        **extra_kwargs,
    )


def validate_request(request, server_context, request_validator,
                     **validator_kwargs):
    """Validate the selected model and endpoint-specific request contract."""
    if hasattr(
            request,
            'model') and request.model not in get_model_list(server_context):
        return create_error_response(
            HTTPStatus.NOT_FOUND,
            f'The model {request.model!r} does not exist.')

    error_message = request_validator(request, server_context,
                                      **validator_kwargs)
    if error_message:
        return create_error_response(HTTPStatus.BAD_REQUEST, error_message)
    return None
