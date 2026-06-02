# Copyright (c) OpenMMLab. All rights reserved.
"""Anthropic-compatible protocol models."""

from __future__ import annotations

import time
from typing import Any, Literal

import shortuuid
from pydantic import BaseModel, ConfigDict, Field

RoutedExperts = list[list[list[int]]] | str | None
MessageStopReason = Literal['end_turn', 'max_tokens', 'stop_sequence', 'tool_use', 'parse_error']


class AnthropicError(BaseModel):
    """Anthropic-style error body."""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Anthropic-style error response."""

    type: Literal['error'] = 'error'
    error: AnthropicError


class TextContentBlockParam(BaseModel):
    """Input text content block."""

    type: Literal['text'] = 'text'
    text: str


class ContentBlockParam(BaseModel):
    """Permissive Anthropic input content block.

    Claude Code may replay beta conversation history containing blocks such as
    ``tool_use`` and ``tool_result``. The adapter decides how to render each
    block into LMDeploy's text-only chat history.
    """

    model_config = ConfigDict(extra='allow')

    type: str
    text: str | None = None
    thinking: str | None = None
    id: str | None = None
    name: str | None = None
    input: Any | None = None
    tool_use_id: str | None = None
    content: str | list[Any] | dict[str, Any] | None = None
    is_error: bool | None = None


class MessageParam(BaseModel):
    """Anthropic input message."""

    role: Literal['user', 'assistant', 'system']
    content: str | list[ContentBlockParam]


class ToolParam(BaseModel):
    """Anthropic tool definition in request body."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class ToolChoiceAutoParam(BaseModel):
    """Let model decide whether to call tools."""

    type: Literal['auto'] = 'auto'


class ToolChoiceAnyParam(BaseModel):
    """Require model to call at least one tool."""

    type: Literal['any'] = 'any'


class ToolChoiceToolParam(BaseModel):
    """Require one concrete tool by name."""

    type: Literal['tool'] = 'tool'
    name: str


ToolChoiceParam = ToolChoiceAutoParam | ToolChoiceAnyParam | ToolChoiceToolParam


class MessagesRequest(BaseModel):
    """Request body for ``POST /v1/messages``."""

    model: str
    messages: list[MessageParam]
    max_tokens: int = Field(gt=0)
    system: str | list[ContentBlockParam] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = 1.0
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[ToolParam] | None = None
    tool_choice: ToolChoiceParam | Literal['auto', 'any'] | None = None
    service_tier: Literal['auto', 'standard_only'] | None = None
    # Extended input fields from /generate endpoint.
    # input_ids and image_data are fallback inputs — they are only used when
    # messages is empty. When messages is non-empty, it takes priority.
    input_ids: list[int] | None = Field(
        default=None,
        description=('Token IDs as input. Only used when messages is empty. '
                     'Mutually exclusive with non-empty messages.'),
    )
    # Mirrors ImageDataFormat from lmdeploy/serve/openai/protocol.py
    image_data: str | dict | list[str | dict] | None = Field(
        default=None,
        description=('Image data for multimodal input. Only used alongside input_ids '
                     'when messages is empty. Mutually exclusive with non-empty messages. '
                     'Can be a URL/base64 string, a dict, or a list of these.'),
    )
    return_routed_experts: bool | None = Field(
        default=False,
        description=('Whether to return MoE routed expert indices in the response.'),
    )
    return_token_ids: bool | None = Field(
        default=False,
        description=('Whether to include output token IDs in the response.'),
    )
    return_logprob: bool | None = Field(
        default=False,
        description=('Whether to return log probabilities for output tokens.'),
    )


class MessageTextBlock(BaseModel):
    """Output text content block."""

    type: Literal['text'] = 'text'
    text: str


class MessageThinkingBlock(BaseModel):
    """Output thinking content block."""

    type: Literal['thinking'] = 'thinking'
    thinking: str


class MessageToolUseBlock(BaseModel):
    """Output tool use content block."""

    type: Literal['tool_use'] = 'tool_use'
    id: str
    name: str
    input: dict[str, Any]


class MessageUsage(BaseModel):
    """Token usage in Anthropic style."""

    input_tokens: int = 0
    output_tokens: int = 0


class MessageDeltaUsage(BaseModel):
    """Cumulative token usage carried by ``message_delta`` stream events."""

    output_tokens: int = 0


class MessagesResponse(BaseModel):
    """Response body for ``POST /v1/messages``."""

    id: str = Field(default_factory=lambda: f'msg_{shortuuid.random()}')
    type: Literal['message'] = 'message'
    role: Literal['assistant'] = 'assistant'
    content: list[MessageTextBlock | MessageThinkingBlock | MessageToolUseBlock]
    model: str
    stop_reason: MessageStopReason | None = None
    stop_sequence: str | None = None
    usage: MessageUsage
    output_ids: list[int] | None = None
    output_token_logprobs: list[tuple[float, int]] | None = None  # (logprob, token_id)
    routed_experts: RoutedExperts = None


class StreamTextBlock(BaseModel):
    """Streaming text content block."""

    type: Literal['text'] = 'text'
    text: str


class StreamThinkingBlock(BaseModel):
    """Streaming thinking content block."""

    type: Literal['thinking'] = 'thinking'
    thinking: str


class StreamToolUseBlock(BaseModel):
    """Streaming tool-use content block."""

    type: Literal['tool_use'] = 'tool_use'
    id: str | None = None
    name: str
    input: dict[str, Any]


StreamContentBlock = StreamTextBlock | StreamThinkingBlock | StreamToolUseBlock


class MessageStartMessage(BaseModel):
    """Message object carried by a ``message_start`` stream event."""

    id: str
    type: Literal['message'] = 'message'
    role: Literal['assistant'] = 'assistant'
    content: list[StreamContentBlock] = Field(default_factory=list)
    model: str
    stop_reason: MessageStopReason | None = None
    stop_sequence: str | None = None
    usage: MessageUsage


class MessageStartEvent(BaseModel):
    """Anthropic ``message_start`` stream event."""

    type: Literal['message_start'] = 'message_start'
    message: MessageStartMessage


class ContentBlockStartEvent(BaseModel):
    """Anthropic ``content_block_start`` stream event."""

    type: Literal['content_block_start'] = 'content_block_start'
    index: int
    content_block: StreamContentBlock


class TextDelta(BaseModel):
    """Anthropic text content delta."""

    type: Literal['text_delta'] = 'text_delta'
    text: str


class ThinkingDelta(BaseModel):
    """Anthropic thinking content delta."""

    type: Literal['thinking_delta'] = 'thinking_delta'
    thinking: str


class InputJsonDelta(BaseModel):
    """Anthropic tool input JSON delta."""

    type: Literal['input_json_delta'] = 'input_json_delta'
    partial_json: str


class SignatureDelta(BaseModel):
    """Anthropic thinking signature delta."""

    type: Literal['signature_delta'] = 'signature_delta'
    signature: str


StreamDelta = TextDelta | ThinkingDelta | InputJsonDelta | SignatureDelta


class ContentBlockDeltaEvent(BaseModel):
    """Anthropic ``content_block_delta`` stream event."""

    type: Literal['content_block_delta'] = 'content_block_delta'
    index: int
    delta: StreamDelta
    output_ids: list[int] | None = None
    output_token_logprobs: list[tuple[float, int]] | None = None


class ContentBlockStopEvent(BaseModel):
    """Anthropic ``content_block_stop`` stream event."""

    type: Literal['content_block_stop'] = 'content_block_stop'
    index: int


class MessageDelta(BaseModel):
    """Top-level message delta carried by ``message_delta``."""

    stop_reason: MessageStopReason | None = None
    stop_sequence: str | None = None


class MessageDeltaEvent(BaseModel):
    """Anthropic ``message_delta`` stream event."""

    type: Literal['message_delta'] = 'message_delta'
    delta: MessageDelta
    usage: MessageDeltaUsage
    routed_experts: RoutedExperts = None


class MessageStopEvent(BaseModel):
    """Anthropic ``message_stop`` stream event."""

    type: Literal['message_stop'] = 'message_stop'


class PingEvent(BaseModel):
    """Anthropic ``ping`` stream event."""

    type: Literal['ping'] = 'ping'


class ErrorEvent(BaseModel):
    """Anthropic ``error`` stream event."""

    type: Literal['error'] = 'error'
    error: AnthropicError


AnthropicStreamEvent = (
    MessageStartEvent
    | ContentBlockStartEvent
    | ContentBlockDeltaEvent
    | ContentBlockStopEvent
    | MessageDeltaEvent
    | MessageStopEvent
    | PingEvent
    | ErrorEvent
)


class CountTokensRequest(BaseModel):
    """Request body for ``POST /v1/messages/count_tokens``."""

    model: str
    messages: list[MessageParam]
    system: str | list[ContentBlockParam] | None = None
    tools: list[ToolParam] | None = None


class CountTokensResponse(BaseModel):
    """Response body for ``POST /v1/messages/count_tokens``."""

    input_tokens: int


class AnthropicModel(BaseModel):
    """Anthropic-like model metadata."""

    id: str
    type: Literal['model'] = 'model'
    display_name: str
    created_at: int = Field(default_factory=lambda: int(time.time()))


class AnthropicModelList(BaseModel):
    """Anthropic-like model listing response."""

    data: list[AnthropicModel]
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None
