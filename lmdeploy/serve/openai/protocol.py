# Copyright (c) OpenMMLab. All rights reserved.
# Modified from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Any, Literal

import shortuuid
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Error responses."""
    message: str
    type: str
    code: int
    param: str | None = None
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
    group: str | None = None
    is_blocking: bool = False


class ModelCard(BaseModel):
    """Model cards."""
    id: str
    object: str = 'model'
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = 'lmdeploy'
    root: str | None = None
    parent: str | None = None
    permission: list[ModelPermission] = []


class ModelList(BaseModel):
    """Model list consists of model cards."""
    object: str = 'list'
    data: list[ModelCard] = []


class UsageInfo(BaseModel):
    """Usage information."""
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0


class Function(BaseModel):
    """Function descriptions."""
    description: str | None = Field(default=None, examples=[None])
    name: str
    parameters: BaseModel | None = None


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
    include_usage: bool | None = False


class JsonSchema(BaseModel):
    name: str
    # description is not used since it depends on model
    description: str | None = None
    # `schema` is a reserved field in Pydantic BaseModel
    # use alias since pydantic does not support the OpenAI key `schema`
    json_schema: dict[str, Any] | None = Field(default=None, alias='schema', examples=[None])
    # strict is not used
    strict: bool | None = False
    model_config = ConfigDict(serialize_by_alias=True)


class ResponseFormat(BaseModel):
    # regex_schema is extended by lmdeploy to support regex output
    type: Literal['text', 'json_object', 'json_schema', 'regex_schema']
    json_schema: JsonSchema | None = None
    regex_schema: str | None = None


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""
    model: str

    # messages: str | list[dict[str, Any]] = Field(examples=[[{'role': 'user', 'content': 'hi'}]])
    messages: list[ChatCompletionMessageParam] = Field(examples=[[{'role': 'user', 'content': 'hi'}]])
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    tools: list[Tool] | None = Field(default=None, examples=[None])
    tool_choice: ToolChoice | Literal['auto', 'required', 'none'] = Field(default='auto', examples=['none'])
    logprobs: bool | None = False
    top_logprobs: int | None = None
    n: int | None = 1
    logit_bias: dict[str, float] | None = Field(default=None, examples=[None])
    max_completion_tokens: int | None = Field(
        default=None,
        examples=[None],
        description=('An upper bound for the number of tokens that can be generated for a completion, '
                     'including visible output tokens and reasoning tokens'),
    )
    max_tokens: int | None = Field(
        default=None,
        examples=[None],
        deprecated='max_tokens is deprecated in favor of the max_completion_tokens field',
    )
    stop: str | list[str] | None = Field(default=None, examples=[None])

    stream: bool | None = False
    stream_options: StreamOptions | None = Field(default=None, examples=[None])
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    user: str | None = None
    reasoning_effort: Literal['low', 'medium', 'high'] | None = None
    response_format: ResponseFormat | None = Field(default=None, examples=[None])
    # additional argument of lmdeploy
    repetition_penalty: float | None = 1.0
    session_id: int | None = -1
    ignore_eos: bool | None = False
    skip_special_tokens: bool | None = True
    spaces_between_special_tokens: bool | None = True
    top_k: int | None = 40
    seed: int | None = None
    min_new_tokens: int | None = Field(default=None, examples=[None])
    min_p: float = 0.0
    enable_thinking: bool | None = None  # will be deprecated in the future
    return_token_ids: bool | None = False
    include_stop_str_in_output: bool | None = False
    chat_template_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=('Additional keyword args to pass to the template renderer. '
                     'Will be accessible by the chat template.'),
    )
    # kwargs for hf processor
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=('Additional kwargs to pass to the HF processor'),
    )


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
    tool_calls: list[ToolCall]
    # content - per OpenAI spec, content AND tool calls can be returned rarely
    # But some models will do this intentionally
    content: str | None = None


class ChatMessage(BaseModel):
    """Chat messages."""
    role: str
    content: str | None = None
    gen_tokens: list[int] | None = None
    reasoning_content: str | None = Field(default=None, examples=[None])
    tool_calls: list[ToolCall] | None = Field(default=None, examples=[None])


class LogProbs(BaseModel):
    text_offset: list[int] = Field(default_factory=list)
    token_logprobs: list[float | None] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    top_logprobs: list[dict[str, float] | None] | None = None


class TopLogprob(BaseModel):
    token: str
    bytes: list[int] | None = None
    logprob: float


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    bytes: list[int] | None = None
    logprob: float
    top_logprobs: list[TopLogprob]


class ChoiceLogprobs(BaseModel):
    content: list[ChatCompletionTokenLogprob] | None = None


class ChatCompletionResponseChoice(BaseModel):
    """Chat completion response choices."""
    index: int
    message: ChatMessage
    logprobs: ChoiceLogprobs | None = None
    finish_reason: Literal['stop', 'length', 'tool_calls', 'error'] | None = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    object: str = 'chat.completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


# a tool call delta where everything is optional
class DeltaToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f'chatcmpl-tool-{shortuuid.random()}')
    type: Literal['function'] = 'function'
    index: int
    function: DeltaFunctionCall | None = None


class DeltaMessage(BaseModel):
    """Delta messages."""
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    gen_tokens: list[int] | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class ChatCompletionResponseStreamChoice(BaseModel):
    """Chat completion response stream choice."""
    index: int
    delta: DeltaMessage
    logprobs: ChoiceLogprobs | None = None
    finish_reason: Literal['stop', 'length', 'tool_calls', 'error', 'abort'] | None = None


class ChatCompletionStreamResponse(BaseModel):
    """Chat completion stream response."""
    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    object: str = 'chat.completion.chunk'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None = None


class CompletionRequest(BaseModel):
    """Completion request."""
    model: str
    prompt: str | list[Any]
    suffix: str | None = None
    temperature: float | None = 0.7
    n: int | None = 1
    logprobs: int | None = None
    max_completion_tokens: int | None = Field(
        default=None,
        examples=[None],
        description=('An upper bound for the number of tokens that can be generated for a completion, '
                     'including visible output tokens and reasoning tokens'),
    )
    max_tokens: int | None = Field(
        default=16,
        examples=[16],
        deprecated='max_tokens is deprecated in favor of the max_completion_tokens field',
    )
    stop: str | list[str] | None = Field(default=None, examples=[None])
    stream: bool | None = False
    stream_options: StreamOptions | None = Field(default=None, examples=[None])
    top_p: float | None = 1.0
    echo: bool | None = False
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    user: str | None = None
    # additional argument of lmdeploy
    repetition_penalty: float | None = 1.0
    session_id: int | None = -1
    ignore_eos: bool | None = False
    skip_special_tokens: bool | None = True
    spaces_between_special_tokens: bool | None = True
    top_k: int | None = 40  # for opencompass
    seed: int | None = None
    min_p: float = 0.0
    return_token_ids: bool | None = False


class CompletionResponseChoice(BaseModel):
    """Completion response choices."""
    index: int
    text: str
    logprobs: LogProbs | None = None
    gen_tokens: list[int] | None = None
    finish_reason: Literal['stop', 'length', 'tool_calls', 'error', 'abort'] | None = None


class CompletionResponse(BaseModel):
    """Completion response."""
    id: str = Field(default_factory=lambda: f'cmpl-{shortuuid.random()}')
    object: str = 'text_completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    """Completion response stream choice."""
    index: int
    text: str
    logprobs: LogProbs | None = None
    gen_tokens: list[int] | None = None
    finish_reason: Literal['stop', 'length', 'tool_calls', 'error', 'abort'] | None = None


class CompletionStreamResponse(BaseModel):
    """Completion stream response."""
    id: str = Field(default_factory=lambda: f'cmpl-{shortuuid.random()}')
    object: str = 'text_completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseStreamChoice]
    usage: UsageInfo | None = None


class EmbeddingsRequest(BaseModel):
    """Embedding request."""
    model: str = None
    input: str | list[str]
    user: str | None = None


class EmbeddingsResponse(BaseModel):
    """Embedding response."""
    object: str = 'list'
    data: list[dict[str, Any]]
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
    model: str | None = None
    input: list[int] | list[list[int]] | str | list[str]
    encoding_format: Literal['float', 'base64'] = 'float'
    dimensions: int | None = None
    user: str | None = None


class PoolingResponse(BaseModel):
    """Pooling response."""
    id: str = Field(default_factory=lambda: f'pool-{shortuuid.random()}')
    object: str = 'list'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = None
    data: list[dict[str, Any]]
    usage: UsageInfo


class EncodeRequest(BaseModel):
    """Encode request."""
    input: str | list[str]
    do_preprocess: bool | None = False
    add_bos: bool | None = True


class EncodeResponse(BaseModel):
    """Encode response."""
    input_ids: list[int] | list[list[int]]
    length: int | list[int]


class GenerateResponse(BaseModel):
    """Generate response."""
    text: str
    tokens: int
    input_tokens: int
    history_tokens: int
    finish_reason: Literal['stop', 'length', 'tool_calls', 'error', 'abort'] | None = None


class UpdateParamsRequest(BaseModel):
    """Update weights request."""
    serialized_named_tensors: str | list[str] | dict
    load_format: str | None = None  # 'flattened_bucket' or None
    finished: bool = False


# str for url/base64, base64 should be data:image/jpeg;base64, dict should be {'url': url/base64, 'options': ...}
ImageDataInputItem = str | dict
ImageDataFormat = ImageDataInputItem | list[ImageDataInputItem]


# /generate input
class GenerateReqInput(BaseModel):
    session_id: int | None = -1
    prompt: str | None = None
    input_ids: list[int] | None = None
    image_data: ImageDataFormat | None = None
    return_logprob: bool | None = None
    max_tokens: int = 128
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    stream: bool | None = False
    temperature: float = 1.0
    repetition_penalty: float | None = 1.0
    ignore_eos: bool | None = False
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    skip_special_tokens: bool | None = True
    spaces_between_special_tokens: bool | None = True
    include_stop_str_in_output: bool | None = False
    return_routed_experts: bool | None = False
    # kwargs for hf processor
    mm_processor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=('Additional kwargs to pass to the HF processor'),
    )


class GenerateReqMetaOutput(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: dict[str, Any] | None = None
    output_token_logprobs: list[tuple[float, int]] | None = None  # (logprob, token_id)
    routed_experts: list[list[list[int]]] | str | None = None  # (num_token, num_layer, topk_expert)


# /generate output
class GenerateReqOutput(BaseModel):
    text: str
    output_ids: list[int]
    meta_info: GenerateReqMetaOutput


class AbortRequest(BaseModel):
    # Whether to abort all requests
    abort_all: bool = False
    # The finished reason data
    finished_reason: dict[str, Any] | None = None
    abort_message: str | None = None
    # The session ID to abort. If `abort_all` is True, this field is ignored.
    session_id: int | None = -1
