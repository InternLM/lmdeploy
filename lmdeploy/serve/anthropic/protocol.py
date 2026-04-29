# Copyright (c) OpenMMLab. All rights reserved.
"""Anthropic-compatible protocol models."""

from __future__ import annotations

import time
from typing import Any, Literal

import shortuuid
from pydantic import BaseModel, Field


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


class MessageParam(BaseModel):
    """Anthropic input message."""

    role: Literal['user', 'assistant']
    content: str | list[TextContentBlockParam]


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
    system: str | list[TextContentBlockParam] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = 1.0
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[ToolParam] | None = None
    tool_choice: ToolChoiceParam | Literal['auto', 'any'] | None = None
    service_tier: Literal['auto', 'standard_only'] | None = None


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


class MessagesResponse(BaseModel):
    """Response body for ``POST /v1/messages``."""

    id: str = Field(default_factory=lambda: f'msg_{shortuuid.random()}')
    type: Literal['message'] = 'message'
    role: Literal['assistant'] = 'assistant'
    content: list[MessageTextBlock | MessageThinkingBlock | MessageToolUseBlock]
    model: str
    stop_reason: Literal['end_turn', 'max_tokens', 'stop_sequence', 'tool_use'] | None = None
    stop_sequence: str | None = None
    usage: MessageUsage


class CountTokensRequest(BaseModel):
    """Request body for ``POST /v1/messages/count_tokens``."""

    model: str
    messages: list[MessageParam]
    system: str | list[TextContentBlockParam] | None = None
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


class TextDelta(BaseModel):
    """Delta payload for text content blocks."""

    type: Literal['text_delta'] = 'text_delta'
    text: str


class ThinkingDelta(BaseModel):
    """Delta payload for thinking content blocks."""

    type: Literal['thinking_delta'] = 'thinking_delta'
    thinking: str


class InputJsonDelta(BaseModel):
    """Delta payload for tool_use input JSON fragments."""

    type: Literal['input_json_delta'] = 'input_json_delta'
    partial_json: str


class ContentBlockDeltaEvent(BaseModel):
    """SSE content block delta event payload."""

    type: Literal['content_block_delta'] = 'content_block_delta'
    index: int
    delta: TextDelta | ThinkingDelta | InputJsonDelta
