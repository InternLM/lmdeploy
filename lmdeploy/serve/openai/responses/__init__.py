# Copyright (c) OpenMMLab. All rights reserved.
"""OpenAI Responses-compatible endpoint."""

from openai.types.responses import (
    ResponseFunctionToolCall as ResponseOutputFunctionCall,
)
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails as ResponseIncompleteDetails
from openai.types.responses.response_usage import InputTokensDetails as ResponseInputTokensDetails
from openai.types.responses.response_usage import OutputTokensDetails as ResponseOutputTokensDetails

from .protocol import (
    ResponseInputOutputItem,
    ResponsesRequest,
    ResponsesResponse,
)
from .request import (
    _messages_from_input,
    _openai_tools_from_responses,
    _to_generation_config,
    _tool_choice_from_responses,
    _validate_text_v1_request,
)
from .response import _make_response
from .serving import (
    OpenAIServingResponses,
    create_responses_router,
)
from .streaming import _stream_response

__all__ = [
    'ResponseIncompleteDetails',
    'ResponseInputTokensDetails',
    'ResponseInputOutputItem',
    'ResponseOutputTokensDetails',
    'ResponseOutputFunctionCall',
    'ResponseOutputMessage',
    'ResponseOutputText',
    'ResponsesRequest',
    'ResponsesResponse',
    'ResponseUsage',
    'OpenAIServingResponses',
    'create_responses_router',
    '_make_response',
    '_messages_from_input',
    '_openai_tools_from_responses',
    '_stream_response',
    '_to_generation_config',
    '_tool_choice_from_responses',
    '_validate_text_v1_request',
]
