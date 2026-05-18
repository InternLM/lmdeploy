# Copyright (c) OpenMMLab. All rights reserved.
"""OpenAI Responses-compatible endpoint."""

from .protocol import (
    ResponseIncompleteDetails,
    ResponseInputOutputItem,
    ResponseInputTokensDetails,
    ResponseOutputFunctionCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseOutputTokensDetails,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsage,
)
from .serving import (
    _generation_messages_from_parser,
    _make_response,
    _messages_from_input,
    _openai_tools_from_responses,
    _stream_response,
    _to_generation_config,
    _tool_choice_from_responses,
    _validate_text_v1_request,
    create_responses_router,
)

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
    'create_responses_router',
    '_generation_messages_from_parser',
    '_make_response',
    '_messages_from_input',
    '_openai_tools_from_responses',
    '_stream_response',
    '_to_generation_config',
    '_tool_choice_from_responses',
    '_validate_text_v1_request',
]
