import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from openai import OpenAI
from utils.constant import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_PORT

BASE_HTTP_URL = f"http://{os.getenv('MASTER_ADDR', 'localhost')}"
PORT = os.getenv('LMDEPLOY_PORT', str(DEFAULT_PORT))
BASE_URL = f'{BASE_HTTP_URL}:{PORT}'

#: Think-tag delimiters used by DeepSeek-R1 and QwenQwQ parsers
THINK_START_TOKEN = '<think>'
THINK_END_TOKEN = '</think>'

# -- Basic tools (English) --------------------------------------------------

CONCURRENT_WEATHER_TOOL = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get current weather',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                },
            },
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

# -- Chinese tool ------------------------------------------------------------

WEATHER_TOOL_CN = {
    'type': 'function',
    'function': {
        'name': 'get_current_weather',
        'description': '获取指定城市的当前天气信息',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string',
                    'description': '城市名称，例如：北京',
                },
                'unit': {
                    'type': 'string',
                    'description': '温度单位',
                    'enum': ['摄氏度', '华氏度'],
                },
            },
            'required': ['city'],
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
            # NOTE: no 'required' key — all params are optional
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


class StreamTee:
    """Transparent iterator proxy: yields every chunk unchanged while
    recording each ``repr(chunk)`` to the log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def __iter__(self):
        try:
            for chunk in self._stream:
                try:
                    with open(self._log_file, 'a', encoding='utf-8') as f:
                        f.write(repr(chunk) + '\n')
                except Exception:
                    pass
                yield chunk
        except Exception:
            raise


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
    log_base = config.get('log_path', './logs')
    log_dir = os.path.join(log_base, category)
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f'{safe_test_name}_{timestamp}.log')


def make_logged_client(log_file):
    client, model_name = get_client_and_model()
    _original_create = client.chat.completions.create

    def _logged_create(*args, **kwargs):
        extra_body = kwargs.get('extra_body')
        if extra_body is None:
            kwargs['extra_body'] = {'spaces_between_special_tokens': False}
        elif isinstance(extra_body, dict) and 'spaces_between_special_tokens' not in extra_body:
            extra_body['spaces_between_special_tokens'] = False
        is_stream = kwargs.get('stream', False)
        result = _original_create(*args, **kwargs)
        if is_stream:
            return StreamTee(result, log_file)
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(repr(result) + '\n')
        except Exception:
            pass
        return result

    client.chat.completions.create = _logged_create
    return client, model_name


# -- Assertion helpers -------------------------------------------------------


def assert_tool_call_fields(tool_call):
    """Assert a single tool call object has all required fields."""
    assert tool_call.type == 'function', (f'tool_call.type should be "function", got {tool_call.type}')
    assert tool_call.function is not None, ('tool_call.function should not be None')
    assert isinstance(tool_call.id, str), (f'tool_call.id should be a string, got {type(tool_call.id)}')
    assert len(tool_call.id) >= 1, (f'tool_call.id should be non-empty, got "{tool_call.id}"')
    assert isinstance(tool_call.function.name,
                      str), (f'function.name should be a string, got {type(tool_call.function.name)}')
    assert len(tool_call.function.name) > 0, ('function.name should be non-empty')
    assert isinstance(tool_call.function.arguments, str), (f'function.arguments should be a string, '
                                                           f'got {type(tool_call.function.arguments)}')


def assert_arguments_parseable(arguments_str):
    """Assert *arguments_str* is valid JSON dict; return the parsed dict."""
    parsed = json.loads(arguments_str)
    assert isinstance(parsed, dict), (f'Parsed arguments should be a dict, got {type(parsed)}')
    return parsed


# -- Tokenizer helpers -------------------------------------------------------

_TOKENIZER_CACHE: dict[str, object] = {}


def resolve_tokenizer_model_path(config: dict, model_case: str) -> str:
    """Local HF path for tokenizer: ``{model_path}/{model_case}``."""
    if os.path.isabs(model_case):
        return model_case
    model_root = config.get('model_path', './model')
    return os.path.join(model_root, model_case)


def get_chat_tokenizer(tokenizer_path: str):
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
    tokenizer = get_chat_tokenizer(tokenizer_path)
    prompt_str = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    return prompt_token_ids, len(prompt_token_ids)


def attach_decoded_output_ids(result: dict, tokenizer_path: str) -> dict:
    output_ids = result.get('output_ids') or []
    if output_ids:
        try:
            tokenizer = get_chat_tokenizer(tokenizer_path)
            result['decoded_str'] = tokenizer.decode(output_ids, skip_special_tokens=False)
        except Exception:
            result.setdefault('decoded_str', '')
    else:
        result.setdefault('decoded_str', '')
    return result


# -- Stream consumption helpers ----------------------------------------------

_TOOL_CALL_RAW_MARKERS = ('<tool_call>', '<function=')
DEFAULT_TOOL_CALL_CONCURRENCY = int(os.getenv('TOOL_CALL_CONCURRENCY', '50'))


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
            if payload.get('type'):
                parts.append(f"type={payload['type']}")
            if payload.get('code') is not None:
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
        if tc.function.arguments:
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


def _merge_stream_choice_json(choice: dict, result: dict, tool_calls: dict) -> None:
    if choice.get('finish_reason'):
        result['finish_reason'] = choice['finish_reason']
        result['finish_reason_count'] += 1

    delta = choice.get('delta') or {}
    if delta.get('role'):
        result['role'] = delta['role']

    rc = delta.get('reasoning_content')
    if rc:
        result['reasoning_content'] += rc
        result['raw_text'] += rc
    content = delta.get('content')
    if content:
        result['content'] += content
        result['raw_text'] += content

    for tc in delta.get('tool_calls') or []:
        _merge_stream_tool_call_delta(tool_calls, _DictToolCallDelta(tc))

    output_ids = choice.get('output_ids')
    if isinstance(output_ids, list) and output_ids:
        result['output_ids'].extend(output_ids)

    if choice.get('routed_experts') is not None:
        result['routed_experts'] = choice['routed_experts']


class _DictToolCallDelta:

    def __init__(self, data: dict):
        self.index = data.get('index')
        self.id = data.get('id')
        fn = data.get('function') or {}
        self.function = type('Fn', (), {
            'name': fn.get('name'),
            'arguments': fn.get('arguments'),
        })()


def _finalize_stream_tool_call_result(result: dict, tool_calls: dict) -> dict:
    result['tool_calls'] = tool_calls
    if tool_calls:
        first = tool_calls[min(tool_calls)]
        if not first.get('id'):
            first['id'] = f'call_{uuid.uuid4().hex[:8]}'
        result['function_name'] = first['name'] or None
        result['args_str'] = first['args_str']
        result['tool_call_id'] = first['id']
    return result


def collect_stream_tool_call(stream):
    tool_calls = {}
    result = _new_stream_tool_call_result()

    for chunk in stream:
        result['chunk_count'] += 1
        usage = getattr(chunk, 'usage', None)
        if usage is not None:
            prompt_tokens = getattr(usage, 'prompt_tokens', None)
            if prompt_tokens:
                result['prompt_tokens'] = prompt_tokens
            completion_tokens = getattr(usage, 'completion_tokens', None)
            if completion_tokens:
                result['completion_tokens'] = completion_tokens

        if not chunk.choices:
            continue
        choice = chunk.choices[0]

        if choice.finish_reason:
            result['finish_reason'] = choice.finish_reason
            result['finish_reason_count'] += 1

        output_ids = getattr(choice, 'output_ids', None)
        if isinstance(output_ids, list) and output_ids:
            result['output_ids'].extend(output_ids)

        routed_experts = getattr(choice, 'routed_experts', None)
        if routed_experts is not None:
            result['routed_experts'] = routed_experts

        delta = choice.delta
        if delta.role:
            result['role'] = delta.role

        rc = getattr(delta, 'reasoning_content', None)
        if rc:
            result['reasoning_content'] += rc
            result['raw_text'] += rc

        if delta.content:
            result['content'] += delta.content
            result['raw_text'] += delta.content

        if delta.tool_calls:
            for tc in delta.tool_calls:
                _merge_stream_tool_call_delta(tool_calls, tc)

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
    **payload_extra,
) -> dict:
    tok_path = tokenizer_path or api_model_name
    url = f'{base_url or BASE_URL}/v1/chat/completions'
    prompt_tokens_computed = 0
    if use_input_ids:
        if not messages:
            raise ValueError('messages required when use_input_ids=True')
        input_ids, prompt_tokens_computed = build_input_ids_and_prompt_tokens(messages, tok_path, tools)
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
            'spaces_between_special_tokens': False,
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
            'spaces_between_special_tokens': False,
        }
    if tools is not None and not use_input_ids:
        payload['tools'] = tools
    payload.update(payload_extra)

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

            usage = item.get('usage')
            if isinstance(usage, dict):
                if usage.get('prompt_tokens'):
                    result['prompt_tokens'] = usage['prompt_tokens']
                if usage.get('completion_tokens'):
                    result['completion_tokens'] = usage['completion_tokens']

            choices = item.get('choices') or []
            if not choices:
                continue

            result['chunk_count'] += 1
            _merge_stream_choice_json(choices[0], result, tool_calls)

    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(raw_lines) + '\n')
        except Exception:
            pass

    result['prompt_tokens_computed'] = prompt_tokens_computed
    if prompt_tokens_computed and not result.get('prompt_tokens'):
        result['prompt_tokens'] = prompt_tokens_computed

    result = _finalize_stream_tool_call_result(result, tool_calls)
    return attach_decoded_output_ids(result, tok_path)


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

        streamed_tool_calls = chunk.choices[0].delta.tool_calls
        if streamed_tool_calls:
            for stc in streamed_tool_calls:
                _merge_stream_tool_call_delta(tool_calls_data, stc)

    return tool_calls_data, finish_reason_count


def _has_parsed_tool_calls(tool_calls) -> bool:
    if not tool_calls:
        return False
    if isinstance(tool_calls, dict):
        return any((data.get('name') or data.get('args_str')) for data in tool_calls.values())
    if isinstance(tool_calls, list):
        return len(tool_calls) > 0
    return bool(tool_calls)


def assert_parser_drop_decoded_only(decoded_str: str, tool_calls) -> None:
    if '<tool_call>' in (decoded_str or '').lower() and not _has_parsed_tool_calls(tool_calls):
        raise AssertionError(
            'Parser dropped tool call: model emitted <tool_call> in decoded output '
            'but parser returned nothing')


def assert_no_parser_drop(
    raw_text: str,
    tool_calls,
    decoded_str: str = '',
) -> None:
    combined = f'{raw_text}\n{decoded_str}'.lower()
    if not combined.strip():
        return
    model_emitted_tool = any(marker in combined for marker in _TOOL_CALL_RAW_MARKERS)
    assert not (model_emitted_tool and not _has_parsed_tool_calls(tool_calls)), (
        'Parser dropped tool call: model emitted raw tool markup but parser returned nothing')


def validate_stream_tool_call_result(
    result: dict,
    *,
    expected_finish_reason: str = 'tool_calls',
    expected_function_name: str | None = None,
    require_tool_call: bool = True,
) -> None:
    if expected_finish_reason is not None:
        assert result['finish_reason'] == expected_finish_reason, (
            f"finish_reason={result['finish_reason']!r}, expected {expected_finish_reason!r}")

    if result.get('finish_reason_count', 0) > 0:
        assert result['finish_reason_count'] == 1, (
            f'Expected exactly 1 finish_reason chunk, got {result["finish_reason_count"]}')

    if require_tool_call:
        assert result.get('function_name'), 'stream ended without function name'
        assert result.get('args_str'), f'stream ended without arguments for {result["function_name"]!r}'
        assert result.get('tool_call_id'), 'stream ended without tool_call id'
        assert isinstance(result['tool_call_id'], str) and len(result['tool_call_id']) >= 1
        assert result['tool_call_id'].strip() == result['tool_call_id'], (
            'tool_call_id has leading/trailing whitespace')
        assert_arguments_parseable(result['args_str'])

    if expected_function_name is not None and result.get('function_name'):
        assert result['function_name'] == expected_function_name, (
            f"function_name={result['function_name']!r}, expected {expected_function_name!r}")

    assert_no_parser_drop(
        result.get('raw_text', ''),
        result.get('tool_calls'),
        result.get('decoded_str', ''),
    )


def validate_output_ids_present(result: dict) -> None:
    output_ids = result.get('output_ids') or []
    assert isinstance(output_ids, list), f'output_ids should be list, got {type(output_ids)}'
    assert len(output_ids) > 0, (
        'return_token_ids=True but no output_ids in stream '
        f'(finish_reason={result.get("finish_reason")!r})')


def validate_output_ids_match_usage(result: dict) -> None:
    """return_token_ids stream must emit one output_ids entry per completion
    token."""
    output_id_count = len(result.get('output_ids') or [])
    usage_completion = result.get('completion_tokens') or 0
    assert usage_completion > 0, (
        'stream_options.include_usage=True but usage.completion_tokens missing '
        'from stream (cannot verify return_token_ids parity)')
    assert output_id_count == usage_completion, (
        f'return_token_ids=True but stream output_ids count ({output_id_count}) '
        f'!= usage.completion_tokens ({usage_completion}); '
        'server dropped output_ids on some SSE chunks (see api_server continue path)')


def validate_routed_experts_length(result: dict, prompt_tokens: int | None = None) -> None:
    routed_experts = result.get('routed_experts')
    if routed_experts is None:
        raise RoutedExpertsNotSupported(
            'return_routed_experts=True but routed_experts missing in final stream chunk')

    if prompt_tokens is None:
        prompt_tokens = result.get('prompt_tokens_computed') or result.get('prompt_tokens') or 0
    completion_tokens = len(result.get('output_ids') or [])
    expected_len = prompt_tokens + completion_tokens - 1 if prompt_tokens > 0 else 0
    actual_len = len(routed_experts)
    usage_completion = result.get('completion_tokens') or 0

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


def validate_stream_tool_call_with_tokens(result: dict, prompt_tokens: int | None = None, **kwargs) -> None:
    validate_stream_tool_call_result(result, **kwargs)
    assert result.get('stream_complete'), 'stream ended before data: [DONE]'
    validate_output_ids_present(result)
    validate_output_ids_match_usage(result)
    validate_routed_experts_length(result, prompt_tokens=prompt_tokens)


def validate_concurrent_turn_result(
    result: dict,
    prompt_tokens: int,
    *,
    expected_function_name: str | None = None,
) -> None:
    assert result.get('stream_complete'), 'stream ended before data: [DONE]'
    validate_stream_tool_call_result(
        result,
        expected_function_name=expected_function_name,
    )
    validate_output_ids_present(result)
    validate_output_ids_match_usage(result)
    assert_parser_drop_decoded_only(result.get('decoded_str', ''), result.get('tool_calls'))
    if result.get('routed_experts') is not None and prompt_tokens > 0:
        validate_routed_experts_length(result, prompt_tokens=prompt_tokens)


def append_concurrent_turn_to_messages(messages: list, result: dict) -> None:
    ast_msg = {'role': 'assistant', 'content': result.get('content') or None}
    if result.get('reasoning_content'):
        ast_msg['reasoning_content'] = result['reasoning_content']

    tool_calls_out = []
    if result.get('tool_calls') and isinstance(result['tool_calls'], dict):
        for idx in sorted(result['tool_calls']):
            data = result['tool_calls'][idx]
            args = data.get('args_str', '')
            tool_calls_out.append({
                'id': data.get('id') or f'call_{uuid.uuid4().hex[:8]}',
                'type': 'function',
                'function': {'name': data['name'], 'arguments': args},
            })
    elif result.get('function_name'):
        tool_calls_out.append({
            'id': result.get('tool_call_id') or f'call_{uuid.uuid4().hex[:8]}',
            'type': 'function',
            'function': {
                'name': result['function_name'],
                'arguments': result.get('args_str', ''),
            },
        })

    if tool_calls_out:
        ast_msg['tool_calls'] = tool_calls_out
    messages.append(ast_msg)

    for tc in tool_calls_out:
        fn = tc['function']
        args = fn['arguments']
        if isinstance(args, dict):
            args = json.dumps(args)
        messages.append({
            'role': 'tool',
            'tool_call_id': tc['id'],
            'name': fn['name'],
            'content': json.dumps({'weather': 'Sunny', 'temperature': 25}),
        })


def _concurrent_worker_turns(
    worker_id: int,
    api_model_name: str,
    tokenizer_path: str,
    num_turns: int,
    tools: list,
    cities: list[str],
    use_input_ids: bool,
    log_file: str | None,
) -> bool:
    messages = [{
        'role': 'system',
        'content': 'You are a helpful assistant that can use tools. '
        'When asked about weather, use the get_weather tool.',
    }]
    expected_name = tools[0]['function']['name'] if tools else None

    for turn in range(num_turns):
        city = cities[turn % len(cities)]
        messages.append({'role': 'user', 'content': f'What is the weather in {city}?'})
        try:
            result = collect_stream_tool_call_http(
                api_model_name,
                messages,
                tools=tools,
                use_input_ids=use_input_ids,
                log_file=log_file,
                tokenizer_path=tokenizer_path,
            )
        except RoutedExpertsNotSupported:
            raise
        except HttpToolCallError as exc:
            raise AssertionError(
                f'worker {worker_id} turn {turn + 1}: HTTP tool-call request failed: {exc}'
            ) from exc

        prompt_tokens = result.get('prompt_tokens_computed') or result.get('prompt_tokens') or 0
        try:
            validate_concurrent_turn_result(
                result,
                prompt_tokens,
                expected_function_name=expected_name,
            )
        except AssertionError as exc:
            raise AssertionError(f'worker {worker_id} turn {turn + 1}: {exc}') from exc

        append_concurrent_turn_to_messages(messages, result)

    return True


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
) -> tuple[int, int]:
    tok_path = tokenizer_path or api_model_name
    if num_workers is None:
        num_workers = DEFAULT_TOOL_CALL_CONCURRENCY
    if tools is None:
        tools = [CONCURRENT_WEATHER_TOOL]
    if cities is None:
        cities = ['Tokyo', 'London', 'Paris', 'New York']

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _concurrent_worker_turns,
                i,
                api_model_name,
                tok_path,
                num_turns,
                tools,
                cities,
                use_input_ids,
                log_file,
            ) for i in range(num_workers)
        ]
        for fut in as_completed(futures):
            fut.result()

    return num_workers, num_workers


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
    """
    result = {
        'reasoning_content': '',
        'content': '',
        'tool_calls': {},
        'finish_reason': None,
        'finish_reason_count': 0,
        'role': None,
        'role_count': 0,
        'role_inconsistent': False,
        'chunk_count': 0,
        'reasoning_chunks': 0,
        'content_chunks': 0,
    }

    for chunk in stream:
        result['chunk_count'] += 1
        if not chunk.choices:
            continue
        choice = chunk.choices[0]

        if choice.finish_reason is not None:
            result['finish_reason'] = choice.finish_reason
            result['finish_reason_count'] += 1

        delta = choice.delta
        if delta.role:
            if result['role'] is None:
                result['role'] = delta.role
            elif delta.role != result['role']:
                result['role_inconsistent'] = True
            result['role_count'] += 1

        # -- reasoning_content (lmdeploy extension field) -------------------
        rc = getattr(delta, 'reasoning_content', None)
        if rc:
            result['reasoning_content'] += rc
            result['reasoning_chunks'] += 1

        # -- regular content ------------------------------------------------
        if delta.content:
            result['content'] += delta.content
            result['content_chunks'] += 1

        # -- tool calls -----------------------------------------------------
        if delta.tool_calls:
            for stc in delta.tool_calls:
                _merge_stream_tool_call_delta(result['tool_calls'], stc)

    return result


# -- Reasoning extraction helpers -------------------------------------------


def get_reasoning_content(message):
    """Extract reasoning_content from a chat completion message.

    Different backends may expose reasoning in different ways:
      - ``message.reasoning_content``  (lmdeploy / OpenAI extension)
      - Wrapped inside ``<think>...</think>`` tags in ``message.content``
    This helper tries both and returns the reasoning string (or *None*).
    """
    reasoning = getattr(message, 'reasoning_content', None)
    if reasoning:
        return reasoning

    content = message.content or ''
    if THINK_START_TOKEN in content and THINK_END_TOKEN in content:
        start = content.index(THINK_START_TOKEN) + len(THINK_START_TOKEN)
        end = content.index(THINK_END_TOKEN)
        extracted = content[start:end].strip()
        if extracted:
            return extracted

    return None


def get_reasoning_tokens(response):
    """Extract reasoning_tokens from usage, handling various response shapes.

    Returns int or *None* if not available.
    """
    usage = response.usage
    if usage is None:
        return None

    # completion_tokens_details.reasoning_tokens  (OpenAI style)
    details = getattr(usage, 'completion_tokens_details', None)
    if details is not None:
        rt = getattr(details, 'reasoning_tokens', None)
        if rt is not None:
            return rt

    # Direct attribute
    rt = getattr(usage, 'reasoning_tokens', None)
    if rt is not None:
        return rt

    return None


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
