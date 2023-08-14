# Copyright (c) OpenMMLab. All rights reserved.
# Modified from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Any, Dict, List, Literal, Optional, Union

import shortuuid
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Error responses."""
    object: str = 'error'
    message: str
    code: int


class ModelPermission(BaseModel):
    """Model permissions."""
    id: str = Field(default_factory=lambda: f'modelperm-{shortuuid.random()}')
    object: str = 'model_permission'
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = '*'
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    """Model cards."""
    id: str
    object: str = 'model'
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = 'lmdeploy'
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = []


class ModelList(BaseModel):
    """Model list consists of model cards."""
    object: str = 'list'
    data: List[ModelCard] = []


class UsageInfo(BaseModel):
    """Usage information."""
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""
    model: str
    messages: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    # additional argument of lmdeploy
    repetition_penalty: Optional[float] = 1.0
    renew_session: Optional[bool] = False
    ignore_eos: Optional[bool] = False


class ChatMessage(BaseModel):
    """Chat messages."""
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    """Chat completion response choices."""
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal['stop', 'length']]


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    object: str = 'chat.completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    """Delta messages."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    """Chat completion response stream choice."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal['stop', 'length']]


class ChatCompletionStreamResponse(BaseModel):
    """Chat completion stream response."""
    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    object: str = 'chat.completion.chunk'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]


class CompletionRequest(BaseModel):
    """Completion request."""
    model: str
    prompt: Union[str, List[Any]]
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    max_tokens: Optional[int] = 16
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class CompletionResponseChoice(BaseModel):
    """Completion response choices."""
    index: int
    text: str
    logprobs: Optional[int] = None
    finish_reason: Optional[Literal['stop', 'length']]


class CompletionResponse(BaseModel):
    """Completion response."""
    id: str = Field(default_factory=lambda: f'cmpl-{shortuuid.random()}')
    object: str = 'text_completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[float] = None
    finish_reason: Optional[Literal['stop', 'length']] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f'cmpl-{shortuuid.random()}')
    object: str = 'text_completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
