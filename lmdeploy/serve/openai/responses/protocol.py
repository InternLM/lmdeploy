# Copyright (c) OpenMMLab. All rights reserved.
"""Protocol models for the OpenAI Responses-compatible endpoint."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import shortuuid
from openai.types.responses import (
    ResponseError,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponsePrompt,
    ResponseStatus,
    ResponseTextConfig,
    ResponseUsage,
)
from openai.types.responses.response import IncompleteDetails as ResponseIncompleteDetails
from openai.types.responses.response import ToolChoice as ResponseToolChoice
from openai.types.responses.response_create_params import StreamOptions as ResponseStreamOptions
from openai.types.responses.tool import Tool as ResponseTool
from openai.types.shared import Metadata, Reasoning
from pydantic import BaseModel, ConfigDict, Field

ResponseInputOutputItem: TypeAlias = ResponseInputItemParam | ResponseOutputItem


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
    include: (
        list[
            Literal[
                'code_interpreter_call.outputs',
                'computer_call_output.output.image_url',
                'file_search_call.results',
                'message.input_image.image_url',
                'message.output_text.logprobs',
                'reasoning.encrypted_content',
            ],
        ]
        | None
    ) = None
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
    stream_options: ResponseStreamOptions | None = None
    temperature: float | None = None
    text: ResponseTextConfig | dict[str, Any] | None = None
    tool_choice: ResponseToolChoice = 'auto'
    tools: list[ResponseTool | dict[str, Any]] = Field(default_factory=list)
    top_logprobs: int | None = None
    top_p: float | None = None
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


class ResponsesResponse(BaseModel):
    """Response body for Text V1 ``POST /v1/responses``.

    Fields are ordered to follow the OpenAI Responses return schema.
    Reference: https://developers.openai.com/api/reference/resources/responses/methods/create
    """

    id: str
    created_at: int
    error: ResponseError | None = None
    incomplete_details: ResponseIncompleteDetails | None = None
    instructions: str | None = None
    metadata: Metadata | None = None
    model: str
    object: Literal['response'] = 'response'
    output: list[ResponseOutputItem] = Field(default_factory=list)
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
    status: ResponseStatus = 'completed'
    text: ResponseTextConfig | dict[str, Any] | None = None
    top_logprobs: int | None = None
    truncation: Literal['auto', 'disabled'] | None = 'disabled'
    usage: ResponseUsage | None = None
    user: str | None = None

    # Existing compatibility echo for clients that expect the request store flag.
    store: bool = False
