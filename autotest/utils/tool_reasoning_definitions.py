import asyncio
import json
import os
import re
import time
import uuid

import aiohttp
import requests
from openai import OpenAI
from utils.constant import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_PORT

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
)
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.response_parser import (
    BaseResponseParser,
    ResponseParserManager,
    _normalize_request_messages,
    _parse_tool_call_arguments_dict,
)

BASE_HTTP_URL = f"http://{os.getenv('MASTER_ADDR', 'localhost')}"
PORT = os.getenv('LMDEPLOY_PORT', str(DEFAULT_PORT))
BASE_URL = f'{BASE_HTTP_URL}:{PORT}'

LMDEPLOY_DECODE_DEFAULTS = {'spaces_between_special_tokens': False}

def get_reasoning_open_close_tags(reasoning_parser_name: str = 'default') -> tuple[str, str]:
    """Reasoning tag pair from ``ReasoningParser`` registry."""
    parser_cls = ReasoningParserManager.get(reasoning_parser_name)
    return parser_cls.get_reasoning_open_tag(), parser_cls.get_reasoning_close_tag()


THINK_START_TOKEN, THINK_END_TOKEN = get_reasoning_open_close_tags('default')

# -- Basic tools (English) --------------------------------------------------

CONCURRENT_WEATHER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get the current weather in a given location',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The location to find the weather for, '
                    'e.g. London or Tokyo, Japan',
                },
            },
            'required': ['location'],
        },
    },
}

WEATHER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_current_weather',
        'description': 'Get the current weather in a given location',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': 'The city to find the weather for, '
                    'e.g. San Francisco',
                },
                'state': {
                    'type': 'string',
                    'description': 'The state abbreviation, e.g. CA',
                },
                'unit': {
                    'type': 'string',
                    'description': 'The unit for temperature',
                    'enum': ['celsius', 'fahrenheit'],
                },
            },
            'required': ['city', 'state'],
        },
    },
}

SEARCH_TOOL = {
    'type': 'function',
    'function': {
        'name': 'web_search',
        'description': 'Search the web for information',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The search query string',
                },
            },
            'required': ['query'],
        },
    },
}

CALCULATOR_TOOL = {
    'type': 'function',
    'function': {
        'name': 'calculate',
        'description': 'Perform a mathematical calculation',
        'parameters': {
            'type': 'object',
            'properties': {
                'expression': {
                    'type': 'string',
                    'description': 'The math expression to evaluate, e.g. 2+2',
                },
            },
            'required': ['expression'],
        },
    },
}

# -- Complex-parameter tools -------------------------------------------------

NESTED_PARAM_TOOL = {
    'type': 'function',
    'function': {
        'name': 'create_event',
        'description': 'Create a calendar event with nested location details',
        'parameters': {
            'type': 'object',
            'properties': {
                'title': {
                    'type': 'string',
                    'description': 'Event title',
                },
                'location': {
                    'type': 'object',
                    'description': 'Event location details',
                    'properties': {
                        'venue': {
                            'type': 'string',
                            'description': 'Venue name',
                        },
                        'address': {
                            'type': 'string',
                            'description': 'Street address',
                        },
                        'city': {
                            'type': 'string',
                            'description': 'City name',
                        },
                    },
                    'required': ['venue', 'city'],
                },
                'attendees': {
                    'type': 'array',
                    'description': 'List of attendee emails',
                    'items': {
                        'type': 'string'
                    },
                },
                'priority': {
                    'type': 'string',
                    'description': 'Event priority level',
                    'enum': ['low', 'medium', 'high'],
                },
            },
            'required': ['title', 'location'],
        },
    },
}

ALL_OPTIONAL_TOOL = {
    'type': 'function',
    'function': {
        'name': 'log_message',
        'description': 'Log a message with optional metadata',
        'parameters': {
            'type': 'object',
            'properties': {
                'message': {
                    'type': 'string',
                    'description': 'The log message',
                },
                'level': {
                    'type': 'string',
                    'description': 'Log level',
                    'enum': ['debug', 'info', 'warning', 'error'],
                },
                'timestamp': {
                    'type': 'string',
                    'description': 'Optional ISO timestamp',
                },
            },
        },
    },
}


def get_client_and_model(base_url=None):
    url = base_url or BASE_URL
    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{url}/v1')
    models = client.models.list().data
    if not models:
        raise RuntimeError(f'No model returned from GET {url}/v1/models')
    return client, models[0].id


# -- Logging / client helpers ------------------------------------------------


def _append_log_repr(log_file: str, obj) -> None:
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'{obj!r}\n')
    except OSError:
        pass


def _merge_create_kwargs_defaults(kwargs: dict) -> None:
    """Apply ``LMDEPLOY_DECODE_DEFAULTS`` into OpenAI SDK ``extra_body``."""
    extra_body = kwargs.setdefault('extra_body', {})
    if not isinstance(extra_body, dict):
        raise TypeError(f'extra_body must be dict, got {type(extra_body).__name__}')
    for key, value in LMDEPLOY_DECODE_DEFAULTS.items():
        extra_body.setdefault(key, value)


class StreamTee:
    """Transparent iterator proxy: yields every chunk unchanged while
    recording each ``repr(chunk)`` to the log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def __iter__(self):
        for chunk in self._stream:
            _append_log_repr(self._log_file, chunk)
            yield chunk


def setup_log_file(config, test_name, category):
    """Compute log-file path and ensure the directory exists.

    Parameters
    ----------
    config : dict
        Test configuration (must contain ``log_path`` or defaults to
        ``./logs``).
    test_name : str
        Raw test node name (will be sanitised for filesystem safety).
    category : str
        Subdirectory under *log_path*, e.g. ``'tool_calls'`` or
        ``'reasoning'``.

    Returns
    -------
    str
        Full path to the log file.
    """
    safe_test_name = re.sub(r'[^\w\.-]', '_', test_name)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_base = config['log_path']
    log_dir = os.path.join(log_base, category)
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f'{safe_test_name}_{timestamp}.log')


def make_logged_client(log_file):
    """Return an OpenAI client whose ``chat.completions.create`` logs I/O."""
    client, model_name = get_client_and_model()
    original_create = client.chat.completions.create

    def _logged_create(*args, **kwargs):
        _merge_create_kwargs_defaults(kwargs)
        stream = 'stream' in kwargs and kwargs['stream']
        result = original_create(*args, **kwargs)
        if stream:
            return StreamTee(result, log_file)
        _append_log_repr(log_file, result)
        return result

    client.chat.completions.create = _logged_create
    return client, model_name


# -- Assertion helpers (delegate to lmdeploy protocol / response_parser) -----


def assert_tool_call_fields(tool_call):
    """Validate tool call fields (OpenAI tool_calls schema)."""
    data = tool_call if isinstance(tool_call, dict) else tool_call.model_dump()
    assert data['type'] == 'function'
    assert isinstance(data['id'], str) and data['id'].strip()
    fn = data['function']
    assert isinstance(fn['name'], str) and fn['name'].strip()
    assert isinstance(fn['arguments'], str)


def assert_arguments_parseable(arguments_str: str) -> dict:
    """Validate tool arguments via ``_parse_tool_call_arguments_dict`` (raises
    on invalid JSON)."""
    try:
        parsed = _parse_tool_call_arguments_dict(arguments_str)
    except ValueError as exc:
        raise AssertionError(f'tool call arguments are not valid JSON object: {exc}') from exc
    assert parsed is not None, 'tool call arguments must be a JSON object string'
    return parsed


def assert_tool_call_dict_fields(tc: dict) -> dict:
    """Validate aggregated tool-call dict from stream collectors."""
    assert_tool_call_fields({
        'type': 'function',
        'id': tc['id'],
        'function': {
            'name': tc['name'],
            'arguments': tc['args_str'],
        },
    })
    return assert_arguments_parseable(tc['args_str'])


def assert_tool_name_single_delta(stream, expected_name: str) -> None:
    """Function name must arrive in a single SSE delta (not split across
    chunks)."""
    name_events = []
    for chunk in stream:
        if not chunk.choices:
            continue
        tool_calls_delta = _stream_delta_field(chunk.choices[0].delta, 'tool_calls')
        if tool_calls_delta:
            for tc in tool_calls_delta:
                if tc.function and tc.function.name:
                    name_events.append(tc.function.name)
    assert len(name_events) == 1, f'Expected one function-name delta, got {name_events!r}'
    assert name_events[0] == expected_name, (
        f'Expected function name {expected_name!r}, got {name_events[0]!r}')


def _assert_stream_finish_reason(result: dict, expected_finish_reason: str | None) -> None:
    """Shared finish_reason / finish_reason_count checks for stream
    aggregators."""
    if expected_finish_reason is not None:
        assert result['finish_reason'] == expected_finish_reason, (
            f"finish_reason={result['finish_reason']!r}, expected {expected_finish_reason!r}")

    if result['finish_reason_count'] > 0:
        assert result['finish_reason_count'] == 1, (
            f'Expected exactly 1 finish_reason chunk, got {result["finish_reason_count"]}')


def validate_stream_reasoning_result(
    result: dict,
    *,
    expected_finish_reason: str | None = None,
    require_tool_calls: bool = False,
) -> None:
    """Validate ``collect_stream_reasoning`` aggregation (protocol
    counters)."""
    _assert_stream_finish_reason(result, expected_finish_reason)

    if result['role_count'] > 0:
        assert result['role'] == 'assistant', f'Expected role assistant, got {result["role"]!r}'
        assert not result['role_inconsistent'], 'Inconsistent role across stream chunks'

    if require_tool_calls:
        assert result['tool_calls'], 'Expected at least one streamed tool call'
        for tc in result['tool_calls'].values():
            assert_tool_call_dict_fields(tc)


# -- Tokenizer helpers -------------------------------------------------------

_TOKENIZER_CACHE: dict[str, object] = {}


def resolve_tokenizer_model_path(config: dict, model_case: str) -> str:
    """Local HF path for tokenizer: ``{model_path}/{model_case}``."""
    if os.path.isabs(model_case):
        return model_case
    model_root = config['model_path']
    return os.path.join(model_root, model_case)


def get_tokenizer(tokenizer_path: str):
    if tokenizer_path not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[tokenizer_path] = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True)
    return _TOKENIZER_CACHE[tokenizer_path]


def build_input_ids_and_prompt_tokens(
    messages: list,
    tokenizer_path: str,
    tools: list | None,
) -> tuple[list[int], int]:
    tokenizer = get_tokenizer(tokenizer_path)
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    return prompt_token_ids, len(prompt_token_ids)


def resolve_tool_parser_name(model_case: str) -> str:
    """Map model id/path to ``--tool-call-parser`` registry name."""
    name = model_case.lower().replace('_', '-')
    if 'llama' in name:
        return 'llama3'
    if 'glm' in name:
        return 'glm47'
    if 'intern-s2' in name or 'interns2' in name:
        return 'interns2-preview'
    if 'qwen2.5' in name:
        return 'qwen2d5'
    if 'qwen3.5' in name:
        return 'qwen3coder'
    return 'qwen3'


def make_response_parser(
    tokenizer_path: str,
    *,
    tool_parser_name: str,
    tools: list | None = None,
    reasoning_parser_name: str | None = 'default',
    tool_choice: str = 'auto',
    enable_thinking: bool | None = None,
):
    """Build ``BaseResponseParser`` aligned with server parser
    configuration."""
    BaseResponseParser.set_parsers(reasoning_parser_name, tool_parser_name)
    parser_cls = ResponseParserManager.get('default')
    tokenizer = get_tokenizer(tokenizer_path)
    chat_template_kwargs = None
    if enable_thinking is not None:
        chat_template_kwargs = {'enable_thinking': enable_thinking}
    request = ChatCompletionRequest(
        model='autotest',
        messages=[{'role': 'user', 'content': 'test'}],
        tools=tools,
        tool_choice=tool_choice,
        chat_template_kwargs=chat_template_kwargs,
    )
    return parser_cls(request=request, tokenizer=tokenizer)


def supports_raw_reasoning_decode_validate(model_case: str) -> bool:
    """Whether raw ``output_ids`` decode should run reasoning markup checks."""
    return 'llama' not in model_case.lower()


# -- Stream consumption helpers ----------------------------------------------

DEFAULT_TOOL_CALL_CONCURRENCY = int(os.getenv('TOOL_CALL_CONCURRENCY', '50'))
DEFAULT_TOOL_CALL_HTTP_ERROR_WORKERS = int(os.getenv('TOOL_CALL_HTTP_ERROR_WORKERS', '5'))


def format_error_response(status: int, body_text: str) -> str:
    if not body_text.strip():
        return f'HTTP {status} (empty body)'

    try:
        payload = json.loads(body_text)
    except json.JSONDecodeError:
        return f'HTTP {status}: {body_text[:500]}'

    if isinstance(payload, dict):
        if 'message' in payload:
            parts = [f'HTTP {status}: {payload["message"]}']
            if 'type' in payload:
                parts.append(f"type={payload['type']}")
            if 'code' in payload:
                parts.append(f"code={payload['code']}")
            return ', '.join(parts)
        if 'detail' in payload:
            detail = payload['detail']
            if isinstance(detail, list):
                detail = '; '.join(str(item) for item in detail)
            return f'HTTP {status}: {detail}'
        return f'HTTP {status}: {json.dumps(payload, ensure_ascii=False)[:500]}'

    return f'HTTP {status}: {body_text[:500]}'


def _merge_stream_tool_call_delta(tool_calls: dict, tc, default_idx: int = 0) -> None:
    idx = tc.index if tc.index is not None else default_idx
    if idx not in tool_calls:
        tool_calls[idx] = {
            'name': '',
            'args_str': '',
            'id': f'call_{uuid.uuid4().hex[:8]}',
        }
    if tc.id:
        tool_calls[idx]['id'] = tc.id
    if tc.function:
        if tc.function.name:
            tool_calls[idx]['name'] += tc.function.name
        if tc.function.arguments is not None:
            assert isinstance(tc.function.arguments, str), (
                'tool call arguments must be str per protocol.DeltaFunctionCall, '
                f'got {type(tc.function.arguments).__name__}')
            tool_calls[idx]['args_str'] += tc.function.arguments


def _new_stream_tool_call_result() -> dict:
    return {
        'function_name': None,
        'args_str': '',
        'tool_call_id': None,
        'finish_reason': None,
        'role': None,
        'finish_reason_count': 0,
        'tool_calls': {},
        'reasoning_content': '',
        'content': '',
        'chunk_count': 0,
        'raw_text': '',
        'output_ids': [],
        'routed_experts': None,
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'prompt_tokens_computed': 0,
        'stream_complete': False,
        'decoded_str': '',
    }


def _stream_object_field(obj, field: str):
    """Read a field from OpenAI SDK pydantic chunks or lmdeploy protocol
    models."""
    model_fields = getattr(obj, 'model_fields', None)
    if model_fields is not None and field in model_fields:
        return getattr(obj, field)

    for extra_attr in ('__pydantic_extra__', 'model_extra'):
        extra = getattr(obj, extra_attr, None)
        if extra and field in extra:
            return extra[field]

    try:
        data = obj.model_dump(exclude_unset=True)
    except AttributeError:
        return None
    if field in data:
        return data[field]
    return None


def _stream_choice_extension(choice, field: str):
    """Read lmdeploy-only stream fields on ``choices[]`` (e.g.
    ``output_ids``)."""
    return _stream_object_field(choice, field)


def _stream_delta_field(delta, field: str):
    """Read lmdeploy-only stream fields on ``delta`` (e.g.
    ``reasoning_content``)."""
    return _stream_object_field(delta, field)


def _merge_stream_choice(
    choice,
    result: dict,
    tool_calls: dict,
    *,
    track_reasoning_stats: bool = False,
) -> None:
    """Merge one stream choice (OpenAI SDK chunk or protocol model)."""
    if choice.finish_reason is not None:
        result['finish_reason'] = choice.finish_reason
        result['finish_reason_count'] += 1

    delta = choice.delta
    role = _stream_delta_field(delta, 'role')
    if track_reasoning_stats:
        if role:
            if result['role'] is None:
                result['role'] = role
            elif role != result['role']:
                result['role_inconsistent'] = True
            result['role_count'] += 1
    elif role is not None:
        result['role'] = role

    reasoning_content = _stream_delta_field(delta, 'reasoning_content')
    if reasoning_content:
        result['reasoning_content'] += reasoning_content
        result['raw_text'] += reasoning_content
        if track_reasoning_stats:
            result['reasoning_chunks'] += 1

    content = _stream_delta_field(delta, 'content')
    if content:
        result['content'] += content
        result['raw_text'] += content
        if track_reasoning_stats:
            result['content_chunks'] += 1

    tool_calls_delta = _stream_delta_field(delta, 'tool_calls')
    if tool_calls_delta:
        for tc in tool_calls_delta:
            _merge_stream_tool_call_delta(tool_calls, tc)

    output_ids = _stream_choice_extension(choice, 'output_ids')
    if output_ids:
        result['output_ids'].extend(output_ids)

    routed_experts = _stream_choice_extension(choice, 'routed_experts')
    if routed_experts is not None:
        result['routed_experts'] = routed_experts


def _merge_stream_choice_dict(choice: dict, result: dict, tool_calls: dict) -> None:
    """Merge one SSE ``choices[]`` element via ``protocol`` models (no soft
    dict access)."""
    stream_choice = ChatCompletionResponseStreamChoice.model_validate(choice)
    _merge_stream_choice(stream_choice, result, tool_calls)


def _apply_stream_chunk(item: dict, result: dict, tool_calls: dict) -> None:
    """Merge one SSE JSON chunk (``chat.completion.chunk`` schema)."""
    chunk = ChatCompletionStreamResponse.model_validate(item)
    if chunk.usage is not None:
        if chunk.usage.prompt_tokens:
            result['prompt_tokens'] = chunk.usage.prompt_tokens
        if chunk.usage.completion_tokens:
            result['completion_tokens'] = chunk.usage.completion_tokens
    if not chunk.choices:
        return
    result['chunk_count'] += 1
    _merge_stream_choice(chunk.choices[0], result, tool_calls)


def _finalize_stream_tool_call_result(result: dict, tool_calls: dict) -> dict:
    result['tool_calls'] = tool_calls
    if tool_calls:
        first = tool_calls[min(tool_calls)]
        if not first['id']:
            first['id'] = f'call_{uuid.uuid4().hex[:8]}'
        result['function_name'] = first['name']
        result['args_str'] = first['args_str']
        result['tool_call_id'] = first['id']
    return result


def collect_stream_tool_call(stream):
    tool_calls = {}
    result = _new_stream_tool_call_result()

    for chunk in stream:
        result['chunk_count'] += 1
        if chunk.usage is not None:
            if chunk.usage.prompt_tokens:
                result['prompt_tokens'] = chunk.usage.prompt_tokens
            if chunk.usage.completion_tokens:
                result['completion_tokens'] = chunk.usage.completion_tokens

        if not chunk.choices:
            continue
        _merge_stream_choice(chunk.choices[0], result, tool_calls)

    result['stream_complete'] = True
    return _finalize_stream_tool_call_result(result, tool_calls)


def collect_stream_tool_call_http(
    api_model_name: str,
    messages: list | None = None,
    tools=None,
    base_url: str | None = None,
    timeout: int = 600,
    log_file: str | None = None,
    *,
    use_input_ids: bool = False,
    tokenizer_path: str | None = None,
    reference_payload: bool = False,
    **payload_extra,
) -> dict:
    tok_path = tokenizer_path or api_model_name
    url = f'{base_url or BASE_URL}/v1/chat/completions'
    payload, prompt_tokens_computed = _build_stream_tool_call_payload(
        api_model_name,
        messages,
        tools,
        use_input_ids=use_input_ids,
        tokenizer_path=tok_path,
        reference_payload=reference_payload,
        **payload_extra,
    )

    headers = {
        'Authorization': 'Bearer YOUR_API_KEY',
        'Content-Type': 'application/json',
    }

    tool_calls = {}
    result = _new_stream_tool_call_result()
    raw_lines: list[str] = []

    with requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout) as resp:
        if resp.status_code != 200:
            body = resp.text
            if resp.status_code == 400 and 'routed experts' in body.lower():
                raise RoutedExpertsNotSupported(body)
            raise HttpToolCallError(resp.status_code, format_error_response(resp.status_code, body))

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            raw_lines.append(line)
            if not line.startswith('data: '):
                continue

            data = line[6:]
            if data == '[DONE]':
                result['stream_complete'] = True
                break

            try:
                item = json.loads(data)
            except json.JSONDecodeError:
                continue

            _apply_stream_chunk(item, result, tool_calls)

    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(raw_lines) + '\n')
        except Exception:
            pass

    result['prompt_tokens_computed'] = prompt_tokens_computed
    if prompt_tokens_computed and not result['prompt_tokens']:
        result['prompt_tokens'] = prompt_tokens_computed

    result = _finalize_stream_tool_call_result(result, tool_calls)
    return attach_decoded_validation(
        result,
        tok_path,
        tool_parser_name=resolve_tool_parser_name(tok_path),
        tools=tools,
    )


def _build_stream_tool_call_payload(
    api_model_name: str,
    messages: list | None,
    tools,
    *,
    use_input_ids: bool,
    tokenizer_path: str,
    reference_payload: bool = False,
    **payload_extra,
) -> tuple[dict, int]:
    prompt_tokens_computed = 0
    if use_input_ids:
        if messages is None:
            raise ValueError('messages required when use_input_ids=True')
        input_ids, prompt_tokens_computed = build_input_ids_and_prompt_tokens(
            messages, tokenizer_path, tools)
        if reference_payload:
            payload = {
                'model': api_model_name,
                'input_ids': input_ids,
                'messages': [],
                'stream': True,
                'return_token_ids': True,
                'return_routed_experts': True,
                'stream_options': {'include_usage': True},
            }
        else:
            payload = {
                'model': api_model_name,
                'input_ids': input_ids,
                'messages': [],
                'stream': True,
                'temperature': 0,
                'max_completion_tokens': DEFAULT_MAX_COMPLETION_TOKENS,
                'return_token_ids': True,
                'return_routed_experts': True,
                'stream_options': {'include_usage': True},
                **LMDEPLOY_DECODE_DEFAULTS,
            }
    elif reference_payload:
        payload = {
            'model': api_model_name,
            'messages': messages,
            'stream': True,
            'return_token_ids': True,
            'return_routed_experts': True,
            'stream_options': {'include_usage': True},
        }
    else:
        payload = {
            'model': api_model_name,
            'messages': messages,
            'stream': True,
            'temperature': 0,
            'max_completion_tokens': DEFAULT_MAX_COMPLETION_TOKENS,
            'return_token_ids': True,
            'return_routed_experts': True,
            'stream_options': {'include_usage': True},
            **LMDEPLOY_DECODE_DEFAULTS,
        }
    if tools is not None and not use_input_ids:
        payload['tools'] = tools
    payload.update(payload_extra)
    return payload, prompt_tokens_computed


async def _read_stream_tool_call_error_body(resp: aiohttp.ClientResponse) -> str:
    try:
        return await resp.text()
    except Exception as exc:
        return f'<failed to read response body: {exc}>'


async def collect_stream_tool_call_http_async(
    session: aiohttp.ClientSession,
    api_model_name: str,
    messages: list | None = None,
    tools=None,
    base_url: str | None = None,
    timeout: int = 600,
    log_file: str | None = None,
    *,
    use_input_ids: bool = False,
    tokenizer_path: str | None = None,
    reference_payload: bool = False,
    **payload_extra,
) -> dict:
    """Async SSE collector for streaming tool-call HTTP responses."""
    tok_path = tokenizer_path or api_model_name
    url = f'{base_url or BASE_URL}/v1/chat/completions'
    payload, prompt_tokens_computed = _build_stream_tool_call_payload(
        api_model_name,
        messages,
        tools,
        use_input_ids=use_input_ids,
        tokenizer_path=tok_path,
        reference_payload=reference_payload,
        **payload_extra,
    )
    headers = {
        'Authorization': 'Bearer YOUR_API_KEY',
        'Content-Type': 'application/json',
    }

    tool_calls: dict = {}
    result = _new_stream_tool_call_result()
    raw_lines: list[str] = []
    buffer = b''
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    async with session.post(url, json=payload, headers=headers, timeout=client_timeout) as resp:
        if resp.status != 200:
            body = await _read_stream_tool_call_error_body(resp)
            if resp.status == 400 and 'routed experts' in body.lower():
                raise RoutedExpertsNotSupported(body)
            raise HttpToolCallError(resp.status, format_error_response(resp.status, body))

        async for raw_chunk in resp.content.iter_any():
            buffer += raw_chunk
            while b'\n' in buffer:
                line_bytes, buffer = buffer.split(b'\n', 1)
                line = line_bytes.decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                raw_lines.append(line)
                if not line.startswith('data: '):
                    continue

                data = line[6:]
                if data == '[DONE]':
                    result['stream_complete'] = True
                    break

                try:
                    item = json.loads(data)
                except json.JSONDecodeError:
                    continue

                _apply_stream_chunk(item, result, tool_calls)

            if result['stream_complete']:
                break

    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(raw_lines) + '\n')
        except Exception:
            pass

    result['prompt_tokens_computed'] = prompt_tokens_computed
    if prompt_tokens_computed and not result['prompt_tokens']:
        result['prompt_tokens'] = prompt_tokens_computed

    result = _finalize_stream_tool_call_result(result, tool_calls)
    return attach_decoded_validation(
        result,
        tok_path,
        tool_parser_name=resolve_tool_parser_name(tok_path),
        tools=tools,
    )


class RoutedExpertsNotSupported(Exception):
    pass


class HttpToolCallError(Exception):

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def collect_stream_parallel_tool_calls(stream):
    tool_calls_data = {}
    finish_reason_count = 0

    for chunk in stream:
        if not chunk.choices:
            continue
        if chunk.choices[0].finish_reason:
            finish_reason_count += 1

        tool_calls_delta = _stream_delta_field(chunk.choices[0].delta, 'tool_calls')
        if tool_calls_delta:
            for stc in tool_calls_delta:
                _merge_stream_tool_call_delta(tool_calls_data, stc)

    return tool_calls_data, finish_reason_count


def _has_parsed_tool_calls(tool_calls) -> bool:
    if not tool_calls:
        return False
    if isinstance(tool_calls, dict):
        return any(data['name'] or data['args_str'] for data in tool_calls.values())
    if isinstance(tool_calls, list):
        return len(tool_calls) > 0
    return bool(tool_calls)


def assert_raw_decode_validate_complete(
    text: str,
    tokenizer_path: str,
    *,
    tool_parser_name: str,
    tools: list | None = None,
    reasoning_parser_name: str | None = None,
    enable_thinking: bool | None = None,
) -> None:
    """Run ``ResponseParser.validate_complete`` on decoded output.

    ``reasoning_parser_name=None`` validates tool markup only (tool-call path).
    Pass ``reasoning_parser_name`` (e.g. ``'default'``) and ``enable_thinking``
    for reasoning-suite raw decode checks.
    """
    if not text.strip():
        return
    parser = make_response_parser(
        tokenizer_path,
        tool_parser_name=tool_parser_name,
        tools=tools,
        reasoning_parser_name=reasoning_parser_name,
        tool_choice='auto' if tools else 'none',
        enable_thinking=enable_thinking,
    )
    assert parser.validate_complete(text), (
        'ResponseParser.validate_complete failed: incomplete or malformed decoded markup '
        f'in output snippet: {text[:300]!r}')


def attach_decoded_validation(
    result: dict,
    tokenizer_path: str,
    *,
    tool_parser_name: str | None = None,
    tools: list | None = None,
    reasoning_parser_name: str | None = None,
    enable_thinking: bool | None = None,
    model_case: str | None = None,
    validate_decoded: bool = True,
) -> dict:
    """Decode ``output_ids`` and run ``validate_complete`` on raw decoded text.

    Tool-call path: ``tool_parser_name`` set, ``reasoning_parser_name`` omitted.
    Reasoning path: also pass ``reasoning_parser_name``, ``enable_thinking``,
    and optional ``model_case`` to gate decode validation per model.
    """
    if model_case is not None and not supports_raw_reasoning_decode_validate(model_case):
        return result
    if enable_thinking is False:
        return result
    output_ids = result.get('output_ids') or []
    if not output_ids:
        return result
    tokenizer = get_tokenizer(tokenizer_path)
    result['decoded_str'] = tokenizer.decode(output_ids, skip_special_tokens=False)
    if not result['decoded_str'].strip():
        return result
    if not validate_decoded:
        return result
    if tool_parser_name is None:
        return result
    assert_raw_decode_validate_complete(
        result['decoded_str'],
        tokenizer_path,
        tool_parser_name=tool_parser_name,
        tools=tools,
        reasoning_parser_name=reasoning_parser_name,
        enable_thinking=enable_thinking,
    )
    return result


def assert_parser_drop_decoded_only(
    decoded_str: str,
    tool_calls,
    tokenizer_path: str,
    *,
    tool_parser_name: str,
    tools: list | None = None,
) -> None:
    if _has_parsed_tool_calls(tool_calls) or not decoded_str.strip():
        return
    parser = make_response_parser(
        tokenizer_path,
        tool_parser_name=tool_parser_name,
        tools=tools,
        reasoning_parser_name=None,
    )
    open_tag = parser.profile.tool_open_tag
    if not open_tag or open_tag not in decoded_str:
        return
    assert_raw_decode_validate_complete(
        decoded_str,
        tokenizer_path,
        tool_parser_name=tool_parser_name,
        tools=tools,
    )
    raise AssertionError(
        'Parser dropped tool call: decoded output contains complete tool markup '
        'but streamed tool_calls are empty')


def assert_no_parser_drop(
    raw_text: str,
    tool_calls,
    decoded_str: str,
    *,
    tokenizer_path: str,
    tool_parser_name: str,
    tools: list | None = None,
) -> None:
    if _has_parsed_tool_calls(tool_calls):
        return
    combined = f'{raw_text}\n{decoded_str}'.strip()
    if not combined:
        return
    assert_parser_drop_decoded_only(
        combined,
        tool_calls,
        tokenizer_path,
        tool_parser_name=tool_parser_name,
        tools=tools,
    )


def validate_stream_tool_call_result(
    result: dict,
    *,
    expected_finish_reason: str = 'tool_calls',
    expected_function_name: str | None = None,
    require_tool_call: bool = True,
    tokenizer_path: str,
    tool_parser_name: str,
    tools: list | None = None,
) -> None:
    _assert_stream_finish_reason(result, expected_finish_reason)

    if require_tool_call:
        assert result['function_name'], 'stream ended without function name'
        assert result['args_str'], f'stream ended without arguments for {result["function_name"]!r}'
        assert result['tool_call_id'], 'stream ended without tool_call id'
        assert isinstance(result['tool_call_id'], str) and len(result['tool_call_id']) >= 1
        assert result['tool_call_id'].strip() == result['tool_call_id'], (
            'tool_call_id has leading/trailing whitespace')
        assert_arguments_parseable(result['args_str'])

    if expected_function_name is not None and result['function_name']:
        assert result['function_name'] == expected_function_name, (
            f"function_name={result['function_name']!r}, expected {expected_function_name!r}")

    assert_no_parser_drop(
        result['raw_text'],
        result['tool_calls'],
        result['decoded_str'],
        tokenizer_path=tokenizer_path,
        tool_parser_name=tool_parser_name,
        tools=tools,
    )


def _resolve_prompt_tokens(result: dict) -> int:
    """Prompt token count from stream usage or precomputed tokenizer count."""
    computed = result['prompt_tokens_computed']
    if computed:
        return computed
    usage_prompt = result['prompt_tokens']
    if usage_prompt:
        return usage_prompt
    raise AssertionError(
        'prompt_tokens missing: enable stream_options.include_usage or use_input_ids')


def validate_output_ids_present(result: dict) -> None:
    output_ids = result['output_ids']
    assert isinstance(output_ids, list), f'output_ids should be list, got {type(output_ids)}'
    assert len(output_ids) > 0, (
        'return_token_ids=True but no output_ids in stream '
        f'(finish_reason={result["finish_reason"]!r})')


def validate_output_ids_match_usage(result: dict) -> None:
    """return_token_ids stream must emit one output_ids entry per completion
    token."""
    output_id_count = len(result['output_ids'])
    usage_completion = result['completion_tokens']
    assert usage_completion > 0, (
        'stream_options.include_usage=True but usage.completion_tokens missing '
        'from stream (cannot verify return_token_ids parity)')
    assert output_id_count == usage_completion, (
        f'return_token_ids=True but stream output_ids count ({output_id_count}) '
        f'!= usage.completion_tokens ({usage_completion}); '
        'server dropped output_ids on some SSE chunks (see api_server continue path)')


def validate_routed_experts_length(result: dict, prompt_tokens: int | None = None) -> None:
    routed_experts = result['routed_experts']
    if routed_experts is None:
        raise RoutedExpertsNotSupported(
            'return_routed_experts=True but routed_experts missing in final stream chunk')

    if prompt_tokens is None:
        prompt_tokens = _resolve_prompt_tokens(result)
    completion_tokens = len(result['output_ids'])
    expected_len = prompt_tokens + completion_tokens - 1 if prompt_tokens > 0 else 0
    actual_len = len(routed_experts)
    usage_completion = result['completion_tokens']

    assert prompt_tokens > 0, (
        'prompt_tokens missing from stream usage; enable stream_options.include_usage')
    assert completion_tokens > 0, 'completion_tokens (len output_ids) must be > 0'
    assert expected_len == actual_len, (
        f'Routed experts length mismatch: expected {expected_len} '
        f'(prompt_tokens={prompt_tokens} + len(output_ids)={completion_tokens} - 1), '
        f'got {actual_len}'
        + (f'; usage.completion_tokens={usage_completion} '
           f'(routed_experts aligns with usage, not with streamed output_ids — server bug)'
           if usage_completion and usage_completion != completion_tokens else ''))


def validate_stream_tool_call_with_tokens(
    result: dict,
    prompt_tokens: int | None = None,
    *,
    tokenizer_path: str | None = None,
    tool_parser_name: str | None = None,
    tools: list | None = None,
    **kwargs,
) -> None:
    validate_stream_tool_call_result(
        result,
        tokenizer_path=tokenizer_path,
        tool_parser_name=tool_parser_name,
        tools=tools,
        **kwargs,
    )
    assert result['stream_complete'], 'stream ended before data: [DONE]'
    validate_output_ids_present(result)
    validate_output_ids_match_usage(result)
    validate_routed_experts_length(result, prompt_tokens=prompt_tokens)


def _tool_calls_for_reference_validation(tool_calls) -> list[dict]:
    if not tool_calls:
        return []
    if isinstance(tool_calls, dict):
        return [{
            'id': data['id'],
            'type': 'function',
            'function': {
                'name': data['name'],
                'arguments': data['args_str'],
            },
        } for idx in sorted(tool_calls) for data in [tool_calls[idx]]]
    return list(tool_calls)


def validate_reference_turn_result(
    result: dict,
    prompt_tokens: int,
    *,
    expected_function_name: str | None = None,
    tokenizer_path: str | None = None,
    tool_parser_name: str | None = None,
    tools: list | None = None,
) -> None:
    """Per-turn validation for concurrent input_ids streaming tool calls."""
    assert result['stream_complete'], 'stream ended before data: [DONE]'
    _assert_stream_finish_reason(result, 'tool_calls')

    tool_calls = _tool_calls_for_reference_validation(result['tool_calls'])
    if not tool_calls:
        raise AssertionError('no tool_calls after stream completed')

    for tc in tool_calls:
        assert_tool_call_fields(tc)
        assert_arguments_parseable(tc['function']['arguments'])

    if expected_function_name is not None:
        first_name = tool_calls[0]['function']['name']
        assert first_name == expected_function_name, (
            f'function_name={first_name!r}, expected {expected_function_name!r}')

    if tokenizer_path is not None and tool_parser_name is not None:
        assert_parser_drop_decoded_only(
            result['decoded_str'],
            result['tool_calls'],
            tokenizer_path,
            tool_parser_name=tool_parser_name,
            tools=tools,
        )

    if result['routed_experts'] is not None and prompt_tokens > 0:
        validate_routed_experts_length(result, prompt_tokens=prompt_tokens)


def append_concurrent_turn_to_messages(messages: list, result: dict) -> None:
    ast_msg = {'role': 'assistant', 'content': result['content']}
    if result['reasoning_content']:
        ast_msg['reasoning_content'] = result['reasoning_content']

    tool_calls_out = []
    if result['tool_calls'] and isinstance(result['tool_calls'], dict):
        for idx in sorted(result['tool_calls']):
            data = result['tool_calls'][idx]
            tool_calls_out.append({
                'id': data['id'],
                'type': 'function',
                'function': {'name': data['name'], 'arguments': data['args_str']},
            })
    elif result['function_name']:
        tool_calls_out.append({
            'id': result['tool_call_id'],
            'type': 'function',
            'function': {
                'name': result['function_name'],
                'arguments': result['args_str'],
            },
        })

    if tool_calls_out:
        ast_msg['tool_calls'] = tool_calls_out
    messages.append(ast_msg)

    for tc in tool_calls_out:
        fn = tc['function']
        messages.append({
            'role': 'tool',
            'tool_call_id': tc['id'],
            'name': fn['name'],
            'content': json.dumps({'weather': 'Sunny', 'temperature': 25}),
        })

    normalized = _normalize_request_messages(messages)
    if normalized is not None:
        messages[:] = normalized


async def _async_concurrent_worker_turns(
    session: aiohttp.ClientSession,
    worker_id: int,
    api_model_name: str,
    tokenizer_path: str,
    num_turns: int,
    tools: list,
    cities: list[str],
    use_input_ids: bool,
    log_file: str | None,
    reference_payload: bool,
) -> bool:
    messages: list = []
    expected_name = tools[0]['function']['name'] if tools else None

    for turn in range(num_turns):
        city = cities[turn % len(cities)]
        messages.append({'role': 'user', 'content': f'What is the weather in {city}?'})
        try:
            result = await collect_stream_tool_call_http_async(
                session,
                api_model_name,
                messages,
                tools=tools,
                use_input_ids=use_input_ids,
                log_file=log_file,
                tokenizer_path=tokenizer_path,
                reference_payload=reference_payload,
            )
        except HttpToolCallError as exc:
            raise AssertionError(
                f'worker {worker_id} turn {turn + 1}: HTTP tool-call request failed: {exc}'
            ) from exc

        prompt_tokens = _resolve_prompt_tokens(result)
        try:
            validate_reference_turn_result(
                result,
                prompt_tokens,
                expected_function_name=expected_name,
                tokenizer_path=tokenizer_path,
                tool_parser_name=resolve_tool_parser_name(tokenizer_path),
                tools=tools,
            )
        except AssertionError as exc:
            raise AssertionError(f'worker {worker_id} turn {turn + 1}: {exc}') from exc

        append_concurrent_turn_to_messages(messages, result)

    return True


async def _run_concurrent_tool_call_workers_async(
    api_model_name: str,
    *,
    tokenizer_path: str | None = None,
    num_workers: int | None = None,
    num_turns: int = 3,
    tools: list | None = None,
    cities: list[str] | None = None,
    use_input_ids: bool = True,
    log_file: str | None = None,
    reference_payload: bool = True,
) -> tuple[int, int]:
    tok_path = tokenizer_path or api_model_name
    if num_workers is None:
        num_workers = DEFAULT_TOOL_CALL_CONCURRENCY
    if tools is None:
        tools = [CONCURRENT_WEATHER_TOOL]
    if cities is None:
        cities = ['Tokyo', 'London', 'Paris', 'New York']

    timeout = aiohttp.ClientTimeout(total=600)
    read_bufsize = 1024 * 1024 * 100
    connector = aiohttp.TCPConnector(limit=num_workers)
    async with aiohttp.ClientSession(
            timeout=timeout,
            read_bufsize=read_bufsize,
            connector=connector,
    ) as session:
        tasks = [
            _async_concurrent_worker_turns(
                session,
                i,
                api_model_name,
                tok_path,
                num_turns,
                tools,
                cities,
                use_input_ids,
                log_file,
                reference_payload,
            ) for i in range(num_workers)
        ]
        await asyncio.gather(*tasks)

    return num_workers, num_workers


async def _async_concurrent_http_error_worker(
    session: aiohttp.ClientSession,
    worker_id: int,
    invalid_model_name: str,
) -> bool:
    """One-shot request expected to fail with formatted
    ``HttpToolCallError``."""
    try:
        await collect_stream_tool_call_http_async(
            session,
            invalid_model_name,
            messages=[{'role': 'user', 'content': 'What is the weather in Tokyo?'}],
            use_input_ids=False,
        )
        raise AssertionError(f'worker {worker_id}: expected HTTP error, request succeeded')
    except HttpToolCallError as exc:
        if 'HTTP' not in exc.message:
            raise AssertionError(
                f'worker {worker_id}: error message not formatted: {exc.message!r}'
            ) from exc
        return True


async def _run_concurrent_http_error_workers_async(
    api_model_name: str,
    *,
    num_workers: int | None = None,
    invalid_model_name: str | None = None,
) -> tuple[int, int]:
    if num_workers is None:
        num_workers = DEFAULT_TOOL_CALL_HTTP_ERROR_WORKERS
    invalid_name = invalid_model_name or f'{api_model_name}__invalid_for_http_error_test__'

    timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=num_workers)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        results = await asyncio.gather(*[
            _async_concurrent_http_error_worker(session, i, invalid_name)
            for i in range(num_workers)
        ])

    successes = sum(results)
    if successes != num_workers:
        raise AssertionError(
            f'concurrent HTTP error probe: {successes}/{num_workers} workers '
            'received formatted HttpToolCallError')
    return successes, num_workers


def run_concurrent_http_error_workers(
    api_model_name: str,
    *,
    num_workers: int | None = None,
    invalid_model_name: str | None = None,
) -> tuple[int, int]:
    """Concurrent invalid requests; each worker must get
    ``HttpToolCallError``."""
    return asyncio.run(_run_concurrent_http_error_workers_async(
        api_model_name,
        num_workers=num_workers,
        invalid_model_name=invalid_model_name,
    ))


def run_concurrent_tool_call_workers(
    api_model_name: str,
    *,
    tokenizer_path: str | None = None,
    num_workers: int | None = None,
    num_turns: int = 3,
    tools: list | None = None,
    cities: list[str] | None = None,
    use_input_ids: bool = True,
    log_file: str | None = None,
    reference_payload: bool = True,
) -> tuple[int, int]:
    """Run N asyncio workers for multi-turn concurrent tool-call stress."""
    return asyncio.run(_run_concurrent_tool_call_workers_async(
        api_model_name,
        tokenizer_path=tokenizer_path,
        num_workers=num_workers,
        num_turns=num_turns,
        tools=tools,
        cities=cities,
        use_input_ids=use_input_ids,
        log_file=log_file,
        reference_payload=reference_payload,
    ))


def _new_stream_reasoning_result() -> dict:
    return {
        'reasoning_content': '',
        'content': '',
        'raw_text': '',
        'tool_calls': {},
        'finish_reason': None,
        'finish_reason_count': 0,
        'role': None,
        'role_count': 0,
        'role_inconsistent': False,
        'chunk_count': 0,
        'reasoning_chunks': 0,
        'content_chunks': 0,
        'output_ids': [],
        'decoded_str': '',
    }


def collect_stream_reasoning(stream):
    """Consume a streaming response, collecting reasoning + content + tool
    calls.

    Returns a dict with keys:
        reasoning_content   – aggregated reasoning string
        content             – aggregated final content string
        tool_calls          – dict  index -> {name, args_str, id}
        finish_reason       – last non-None finish_reason
        finish_reason_count – how many chunks carried a non-None finish_reason
        role                – first non-None role value (all chunks must match)
        role_count          – how many chunks carried a non-None role
        role_inconsistent   – True if any chunk role differed from the first
        chunk_count         – total number of chunks received
        reasoning_chunks    – number of chunks containing reasoning
        content_chunks      – number of chunks containing content
        output_ids          – aggregated token ids when ``return_token_ids=True``
        decoded_str         – filled by ``attach_decoded_validation``
    """
    result = _new_stream_reasoning_result()

    for chunk in stream:
        result['chunk_count'] += 1
        if not chunk.choices:
            continue
        _merge_stream_choice(
            chunk.choices[0],
            result,
            result['tool_calls'],
            track_reasoning_stats=True,
        )

    return result


# -- Message-building helpers -----------------------------------------------


def build_messages_with_tool_response(
    tool_call_id='call_test_001',
    function_name='get_current_weather',
):
    """Build message list: user ask → assistant tool_call → tool result."""
    return [
        {
            'role': 'system',
            'content': 'You are a helpful assistant that can use tools.',
        },
        {
            'role': 'user',
            'content': "What's the weather like in Dallas, TX?",
        },
        {
            'role':
            'assistant',
            'content':
            None,
            'tool_calls': [{
                'id': tool_call_id,
                'type': 'function',
                'function': {
                    'name': function_name,
                    'arguments': '{"city": "Dallas", "state": "TX"}',
                },
            }],
        },
        {
            'role': 'tool',
            'tool_call_id': tool_call_id,
            'name': function_name,
            'content': json.dumps({
                'temperature': 98,
                'unit': 'fahrenheit',
                'description': 'Sunny with clear skies',
            }),
        },
    ]


def build_messages_with_parallel_tool_responses():
    """Build message list simulating two parallel tool calls + results."""
    return [
        {
            'role': 'system',
            'content': 'You are a helpful assistant that can use tools.',
        },
        {
            'role': 'user',
            'content': "What's the weather in Dallas, TX and San Francisco, CA?",
        },
        {
            'role':
            'assistant',
            'content':
            None,
            'tool_calls': [
                {
                    'id': 'call_001',
                    'type': 'function',
                    'function': {
                        'name': 'get_current_weather',
                        'arguments': '{"city": "Dallas", "state": "TX"}',
                    },
                },
                {
                    'id': 'call_002',
                    'type': 'function',
                    'function': {
                        'name': 'get_current_weather',
                        'arguments': '{"city": "San Francisco", "state": "CA"}',
                    },
                },
            ],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_001',
            'name': 'get_current_weather',
            'content': json.dumps({
                'temperature': 98,
                'unit': 'fahrenheit',
                'description': 'Sunny',
            }),
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_002',
            'name': 'get_current_weather',
            'content': json.dumps({
                'temperature': 65,
                'unit': 'fahrenheit',
                'description': 'Foggy',
            }),
        },
    ]


def build_reasoning_tool_roundtrip_messages(tool_call_id='call_reason_001'):
    """Build multi-turn messages: user → assistant (reasoning+tool_call) → tool → continue."""
    return [
        {
            'role': 'system',
            'content': 'You are a helpful assistant that can use tools. '
            'Think through problems step by step.',
        },
        {
            'role': 'user',
            'content': "I'm visiting Dallas, TX. Should I bring an umbrella?",
        },
        {
            'role':
            'assistant',
            'content':
            'Let me think about this. To answer whether you need '
            'an umbrella, I should check the current weather in '
            'Dallas, TX.',
            'tool_calls': [{
                'id': tool_call_id,
                'type': 'function',
                'function': {
                    'name': 'get_current_weather',
                    'arguments': '{"city": "Dallas", "state": "TX"}',
                },
            }],
        },
        {
            'role':
            'tool',
            'tool_call_id':
            tool_call_id,
            'name':
            'get_current_weather',
            'content':
            json.dumps({
                'temperature': 95,
                'unit': 'fahrenheit',
                'description': 'Sunny with clear skies',
                'precipitation': '0%',
            }),
        },
    ]
