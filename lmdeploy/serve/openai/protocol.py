# Copyright (c) OpenMMLab. All rights reserved.
# Modified from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Any, Dict, List, Literal, Optional, Union

import shortuuid
from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Error responses."""
    message: str
    type: str
    code: int
    param: Optional[str] = None
    object: str = 'error'


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
    is_blocking: bool = False


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


class Function(BaseModel):
    """Function descriptions."""
    description: Optional[str] = Field(default=None, examples=[None])
    name: str
    parameters: Optional[object] = None


class Tool(BaseModel):
    """Function wrapper."""
    type: str = Field(default='function', examples=['function'])
    function: Function


class ToolChoiceFuncName(BaseModel):
    """The name of tool choice function."""
    name: str


class ToolChoice(BaseModel):
    """The tool choice definition."""
    function: ToolChoiceFuncName
    type: Literal['function'] = Field(default='function', examples=['function'])


class StreamOptions(BaseModel):
    """The stream options."""
    include_usage: Optional[bool] = False


class JsonSchema(BaseModel):
    name: str
    # description is not used since it depends on model
    description: Optional[str] = None
    # `schema` is a reserved field in Pydantic BaseModel
    # use alias since pydantic does not support the OpenAI key `schema`
    json_schema: Optional[Dict[str, Any]] = Field(default=None, alias='schema', examples=[None])
    # strict is not used
    strict: Optional[bool] = False
    model_config = ConfigDict(serialize_by_alias=True)


class ResponseFormat(BaseModel):
    # regex_schema is extended by lmdeploy to support regex output
    type: Literal['text', 'json_object', 'json_schema', 'regex_schema']
    json_schema: Optional[JsonSchema] = None
    regex_schema: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""
    model: str

    messages: Union[str, List[Dict[str, Any]]] = Field(examples=[[{'role': 'user', 'content': 'hi'}]])
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])
    tool_choice: Union[ToolChoice, Literal['auto', 'required', 'none']] = Field(default='auto', examples=['none'])
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    n: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = Field(default=None, examples=[None])
    max_completion_tokens: Optional[int] = Field(
        default=None,
        examples=[None],
        description=('An upper bound for the number of tokens that can be generated for a completion, '
                     'including visible output tokens and reasoning tokens'),
    )
    max_tokens: Optional[int] = Field(
        default=None,
        examples=[None],
        deprecated='max_tokens is deprecated in favor of the max_completion_tokens field',
    )
    stop: Optional[Union[str, List[str]]] = Field(default=None, examples=[None])

    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = Field(default=None, examples=[None])
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    reasoning_effort: Optional[Literal['low', 'medium', 'high']] = None
    response_format: Optional[ResponseFormat] = Field(default=None, examples=[None])
    # additional argument of lmdeploy
    do_preprocess: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.0
    session_id: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    top_k: Optional[int] = 40
    seed: Optional[int] = None
    min_new_tokens: Optional[int] = Field(default=None, examples=[None])
    min_p: float = 0.0
    enable_thinking: Optional[bool] = None
    return_token_ids: Optional[bool] = False
    include_stop_str_in_output: Optional[bool] = False


class FunctionCall(BaseModel):
    """Function response."""
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call response."""
    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    type: Literal['function'] = 'function'
    function: FunctionCall


class ExtractedToolCallInformation(BaseModel):
    # modified from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/entrypoints/openai/protocol.py#L1199
    # indicate if tools were called
    tools_called: bool
    # extracted tool calls
    tool_calls: List[ToolCall]
    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: Optional[str] = None


class ChatMessage(BaseModel):
    """Chat messages."""
    role: str
    content: Optional[str] = None
    gen_tokens: Optional[List[int]] = None
    reasoning_content: Optional[str] = Field(default=None, examples=[None])
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None


class TopLogprob(BaseModel):
    token: str
    bytes: Optional[List[int]] = None
    logprob: float


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    bytes: Optional[List[int]] = None
    logprob: float
    top_logprobs: List[TopLogprob]


class ChoiceLogprobs(BaseModel):
    content: Optional[List[ChatCompletionTokenLogprob]] = None


class ChatCompletionResponseChoice(BaseModel):
    """Chat completion response choices."""
    index: int
    message: ChatMessage
    logprobs: Optional[ChoiceLogprobs] = None
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls', 'error']] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    object: str = 'chat.completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


# a tool call delta where everything is optional
class DeltaToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f'chatcmpl-tool-{shortuuid.random()}')
    type: Literal['function'] = 'function'
    index: int
    function: Optional[DeltaFunctionCall] = None


class DeltaMessage(BaseModel):
    """Delta messages."""
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    gen_tokens: Optional[List[int]] = None
    tool_calls: List[DeltaToolCall] = Field(default_factory=list)


class ChatCompletionResponseStreamChoice(BaseModel):
    """Chat completion response stream choice."""
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChoiceLogprobs] = None
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls', 'error', 'abort']] = None


class ChatCompletionStreamResponse(BaseModel):
    """Chat completion stream response."""
    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    object: str = 'chat.completion.chunk'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class CompletionRequest(BaseModel):
    """Completion request."""
    model: str
    prompt: Union[str, List[Any]]
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    logprobs: Optional[int] = None
    max_completion_tokens: Optional[int] = Field(
        default=None,
        examples=[None],
        description=('An upper bound for the number of tokens that can be generated for a completion, '
                     'including visible output tokens and reasoning tokens'),
    )
    max_tokens: Optional[int] = Field(
        default=16,
        examples=[16],
        deprecated='max_tokens is deprecated in favor of the max_completion_tokens field',
    )
    stop: Optional[Union[str, List[str]]] = Field(default=None, examples=[None])
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = Field(default=None, examples=[None])
    top_p: Optional[float] = 1.0
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    # additional argument of lmdeploy
    repetition_penalty: Optional[float] = 1.0
    session_id: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    top_k: Optional[int] = 40  # for opencompass
    seed: Optional[int] = None
    min_p: float = 0.0
    return_token_ids: Optional[bool] = False


class CompletionResponseChoice(BaseModel):
    """Completion response choices."""
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    gen_tokens: Optional[List[int]] = None
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls', 'error', 'abort']] = None


class CompletionResponse(BaseModel):
    """Completion response."""
    id: str = Field(default_factory=lambda: f'cmpl-{shortuuid.random()}')
    object: str = 'text_completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    """Completion response stream choice."""
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    gen_tokens: Optional[List[int]] = None
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls', 'error', 'abort']] = None


class CompletionStreamResponse(BaseModel):
    """Completion stream response."""
    id: str = Field(default_factory=lambda: f'cmpl-{shortuuid.random()}')
    object: str = 'text_completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class EmbeddingsRequest(BaseModel):
    """Embedding request."""
    model: str = None
    input: Union[str, List[str]]
    user: Optional[str] = None


class EmbeddingsResponse(BaseModel):
    """Embedding response."""
    object: str = 'list'
    data: List[Dict[str, Any]]
    model: str
    usage: UsageInfo


class PoolingRequest(BaseModel):
    """Pooling request.

    Currently we follow vLLM API protocol,
    https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/protocol.py#L1174

    Notice that ideally we should reuse the input format of embedding API
    https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/protocol.py#L1174
    https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py#L383
    """
    model: Optional[str] = None
    input: Union[List[int], List[List[int]], str, List[str]]
    encoding_format: Literal['float', 'base64'] = 'float'
    dimensions: Optional[int] = None
    user: Optional[str] = None


class PoolingResponse(BaseModel):
    """Pooling response."""
    id: str = Field(default_factory=lambda: f'pool-{shortuuid.random()}')
    object: str = 'list'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = None
    data: List[Dict[str, Any]]
    usage: UsageInfo


class EncodeRequest(BaseModel):
    """Encode request."""
    input: Union[str, List[str]]
    do_preprocess: Optional[bool] = False
    add_bos: Optional[bool] = True


class EncodeResponse(BaseModel):
    """Encode response."""
    input_ids: Union[List[int], List[List[int]]]
    length: Union[int, List[int]]


class GenerateRequest(BaseModel):
    """Generate request."""
    prompt: Union[str, List[Dict[str, Any]]]
    image_url: Optional[Union[str, List[str]]] = Field(default=None, examples=[None])
    session_id: int = -1
    interactive_mode: bool = False
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = Field(default=None, examples=[None])
    request_output_len: Optional[int] = Field(default=None, examples=[None])  # noqa
    top_p: float = 0.8
    top_k: int = 40
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    cancel: Optional[bool] = False  # cancel a responding request
    adapter_name: Optional[str] = Field(default=None, examples=[None])
    seed: Optional[int] = None
    min_new_tokens: Optional[int] = Field(default=None, examples=[None])
    min_p: float = 0.0


class GenerateResponse(BaseModel):
    """Generate response."""
    text: str
    tokens: int
    input_tokens: int
    history_tokens: int
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls', 'error', 'abort']] = None


class UpdateParamsRequest(BaseModel):
    """Update weights request."""
    serialized_named_tensors: Union[str, List[str], Dict]
    finished: bool = False


# str for url/base64, base64 should be data:image/jpeg;base64, dict should be {'url': url/base64, 'options': ...}
ImageDataInputItem = Union[str, Dict]
ImageDataFormat = Union[ImageDataInputItem, List[ImageDataInputItem]]


# /generate input
class GenerateReqInput(BaseModel):
    session_id: Optional[int] = -1
    prompt: Optional[str] = None
    input_ids: Optional[List[int]] = None
    image_data: Optional[ImageDataFormat] = None
    return_logprob: Optional[bool] = None
    max_tokens: int = 128
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    stream: Optional[bool] = False
    temperature: float = 1.0
    repetition_penalty: Optional[float] = 1.0
    ignore_eos: Optional[bool] = False
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    include_stop_str_in_output: Optional[bool] = False


class GenerateReqMetaOutput(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[Dict[str, Any]] = None
    output_token_logprobs: Optional[List[tuple[float, int]]] = None  # (logprob, token_id)


# /generate output
class GenerateReqOutput(BaseModel):
    text: str
    output_ids: List[int]
    meta_info: GenerateReqMetaOutput


class AbortRequest(BaseModel):
    # Whether to abort all requests
    abort_all: bool = False
    # The finished reason data
    finished_reason: Optional[Dict[str, Any]] = None
    abort_message: Optional[str] = None
    # The session ID to abort. If `abort_all` is True, this field is ignored.
    session_id: Optional[int] = -1
