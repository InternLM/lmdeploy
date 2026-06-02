# Copyright (c) OpenMMLab. All rights reserved.
"""OpenAI Responses-compatible endpoint."""

from .protocol import (
    ResponsesRequest,
    ResponsesResponse,
)
from .serving import (
    OpenAIServingResponses,
    create_responses_router,
)

__all__ = [
    'ResponsesRequest',
    'ResponsesResponse',
    'OpenAIServingResponses',
    'create_responses_router',
]
