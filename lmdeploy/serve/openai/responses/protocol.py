# Copyright (c) OpenMMLab. All rights reserved.
"""Protocol models for the OpenAI Responses-compatible endpoint."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import shortuuid
from openai.types.responses import (
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponsePrompt,
    ResponseTextConfig,
)
from openai.types.responses.response import ToolChoice as ResponseToolChoice
from openai.types.responses.tool import Tool as ResponseTool
from openai.types.shared import Metadata, Reasoning
from pydantic import BaseModel, ConfigDict, Field

ResponseInputOutputItem: TypeAlias = ResponseInputItemParam | ResponseOutputItem


class ResponseInputTokensDetails(BaseModel):
    """Input token details in Responses API shape."""

    cached_tokens: int = 0
    input_tokens_per_turn: list[int] = Field(default_factory=list)
    cached_tokens_per_turn: list[int] = Field(default_factory=list)


class ResponseOutputTokensDetails(BaseModel):
    """Output token details in Responses API shape."""

    reasoning_tokens: int = 0
    tool_output_tokens: int = 0
    output_tokens_per_turn: list[int] = Field(default_factory=list)
    tool_output_tokens_per_turn: list[int] = Field(default_factory=list)


class ResponseIncompleteDetails(BaseModel):
    """Details about why a response is incomplete."""

    reason: Literal['max_output_tokens', 'content_filter'] | None = None


class ResponsesRequest(BaseModel):
    """Request body for ``POST /v1/responses``.

    Fields are ordered to follow the OpenAI Responses create API reference, with LMDeploy-specific generation knobs kept
    after the official fields.
    Reference: https://developers.openai.com/api/reference/resources/responses/methods/create
    """

    model_config = ConfigDict(extra='allow')

    background: bool | None = False
    context_management: list[dict[str, Any]] | None = None
    conversation: str | dict[str, Any] | None = None
    include: list[str] | None = None
    input: str | list[ResponseInputOutputItem | dict[str, Any]] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = None
    metadata: Metadata | None = None
    model: str | None = None
    logit_bias: dict[str, float] | None = None
    parallel_tool_calls: bool | None = True
    previous_response_id: str | None = None
    prompt: ResponsePrompt | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal['in-memory', '24h'] | None = None
    reasoning: Reasoning | None = None
    safety_identifier: str | None = Field(default=None, max_length=64)
    service_tier: Literal['auto', 'default', 'flex', 'scale', 'priority'] | None = 'auto'
    store: bool | None = True
    stream: bool | None = False
    stream_options: dict[str, Any] | None = None
    temperature: float | None = 1.0
    text: ResponseTextConfig | dict[str, Any] | None = None
    tool_choice: ResponseToolChoice = 'auto'
    tools: list[ResponseTool | dict[str, Any]] = Field(default_factory=list)
    top_logprobs: int | None = None
    top_p: float | None = 1.0
    truncation: Literal['auto', 'disabled'] | None = 'disabled'
    user: str | None = None

    # LMDeploy-compatible generation extensions.
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    top_k: int | None = 40
    stop: str | list[str] | None = None
    seed: int | None = None
    min_p: float = 0.0
    ignore_eos: bool | None = False
    skip_special_tokens: bool | None = True
    include_stop_str_in_output: bool | None = False
    request_id: str = Field(default_factory=lambda: f'resp_{shortuuid.random()}')


class ResponseUsage(BaseModel):
    """Token usage in Responses API shape."""

    input_tokens: int = 0
    input_tokens_details: ResponseInputTokensDetails = Field(default_factory=ResponseInputTokensDetails)
    output_tokens: int = 0
    output_tokens_details: ResponseOutputTokensDetails = Field(default_factory=ResponseOutputTokensDetails)
    total_tokens: int = 0


class ResponseOutputText(BaseModel):
    """Text content part in a Responses output message."""

    type: Literal['output_text'] = 'output_text'
    text: str
    annotations: list[Any] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    """Assistant output item."""

    id: str = Field(default_factory=lambda: f'msg_{shortuuid.random()}')
    type: Literal['message'] = 'message'
    role: Literal['assistant'] = 'assistant'
    status: Literal['in_progress', 'completed', 'incomplete'] = 'completed'
    content: list[ResponseOutputText]


class ResponseOutputFunctionCall(BaseModel):
    """Function call output item in Responses API shape."""

    id: str
    type: Literal['function_call'] = 'function_call'
    call_id: str
    name: str
    arguments: str
    status: Literal['in_progress', 'completed', 'incomplete'] = 'completed'


class ResponsesResponse(BaseModel):
    """Response body for Text V1 ``POST /v1/responses``.

    Fields are ordered to follow the OpenAI Responses return schema.
    Reference: https://developers.openai.com/api/reference/resources/responses/methods/create
    """

    id: str
    created_at: int
    error: dict[str, Any] | None = None
    incomplete_details: ResponseIncompleteDetails | None = None
    instructions: str | None = None
    metadata: Metadata | None = None
    model: str
    object: Literal['response'] = 'response'
    output: list[ResponseOutputMessage | ResponseOutputFunctionCall] = Field(default_factory=list)
    parallel_tool_calls: bool | None = True
    temperature: float | None = None
    tool_choice: ResponseToolChoice | None = None
    tools: list[ResponseTool | dict[str, Any]] = Field(default_factory=list)
    top_p: float | None = None
    background: bool = False
    completed_at: int | None = None
    conversation: str | dict[str, Any] | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    output_text: str = ''
    previous_response_id: str | None = None
    prompt: ResponsePrompt | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal['in-memory', '24h'] | None = None
    reasoning: Reasoning | None = None
    safety_identifier: str | None = None
    service_tier: Literal['auto', 'default', 'flex', 'scale', 'priority'] | None = 'auto'
    status: Literal['in_progress', 'completed', 'incomplete', 'failed', 'cancelled', 'queued'] = 'completed'
    text: ResponseTextConfig | dict[str, Any] | None = None
    top_logprobs: int | None = None
    truncation: Literal['auto', 'disabled'] | None = 'disabled'
    usage: ResponseUsage | None = None
    user: str | None = None

    # Existing compatibility echo for clients that expect the request store flag.
    store: bool = False
