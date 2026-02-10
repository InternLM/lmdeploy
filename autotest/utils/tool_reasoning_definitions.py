"""Shared definitions for tool-call and reasoning tests.

This module centralises tool schemas, message templates, connection settings,
assertion helpers and stream-consumption helpers so they can be imported by
both ``test_restful_tool_calls.py`` and ``test_restful_reasoning.py`` (and
future test modules) without duplication.
"""

import json

from openai import OpenAI
from utils.constant import DEFAULT_PORT

BASE_HTTP_URL = 'http://localhost'
BASE_URL = ':'.join([BASE_HTTP_URL, str(DEFAULT_PORT)])

#: Supported reasoning parser names (from lmdeploy ReasoningParserManager)
REASONING_PARSER_NAMES = ['deepseek-r1', 'qwen-qwq', 'intern-s1']

#: Think-tag delimiters used by DeepSeek-R1 and QwenQwQ parsers
THINK_START_TOKEN = '<think>'
THINK_END_TOKEN = '</think>'

#: Supported tool parser names (from lmdeploy ToolParserManager)
TOOL_PARSER_NAMES = ['qwen', 'qwen3', 'qwen2d5', 'internlm', 'intern-s1', 'llama3']

#: Tool-call tag delimiters — Qwen family (qwen, qwen3, qwen2d5)
TOOL_CALL_START_TOKEN = '<tool_call>'
TOOL_CALL_END_TOKEN = '</tool_call>'

#: Tool-call tag delimiters — InternLM family
INTERNLM_ACTION_START = '<|action_start|><|plugin|>'
INTERNLM_ACTION_END = '<|action_end|>'

#: Llama 3 bot token
LLAMA3_BOT_TOKEN = '<|python_tag|>'

#: Mapping: server 启动时的 --tool-call-parser / --reasoning-parser 值 → pytest -m 表达式
#: 用于根据当前 server 使用的 parser 筛选对应的 case
TOOL_PARSER_MARK_MAP = {
    'qwen': 'qwen3_parser',
    'qwen3': 'qwen3_parser',
    'qwen2d5': 'qwen2d5_parser',
    'internlm': 'internlm2_parser',
    'intern-s1': 'internlm2_parser',
    'llama3': 'llama3_parser',
}

REASONING_PARSER_MARK_MAP = {
    'deepseek-r1': 'deepseek_r1_parser',
    'qwen-qwq': 'qwenqwq_parser',
    'intern-s1': 'qwenqwq_parser',
}

# -- Basic tools (English) --------------------------------------------------

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

# -- Chinese tool (vLLM issue #12869) ---------------------------------------

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
    """Return an ``OpenAI`` client and the first available model name."""
    url = base_url or BASE_URL
    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{url}/v1')
    model_name = client.models.list().data[0].id
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


# -- Stream consumption helpers ----------------------------------------------


def collect_stream_tool_call(stream):
    """Consume a streaming response and return aggregated tool-call data.

    Returns a dict with keys:
        function_name, args_str, tool_call_id, finish_reason, role,
        finish_reason_count
    """
    result = {
        'function_name': None,
        'args_str': '',
        'tool_call_id': None,
        'finish_reason': None,
        'role': None,
        'finish_reason_count': 0,
    }

    for chunk in stream:
        choice = chunk.choices[0]

        if choice.finish_reason:
            result['finish_reason'] = choice.finish_reason
            result['finish_reason_count'] += 1

        delta = choice.delta
        if delta.role:
            result['role'] = delta.role

        if delta.tool_calls and len(delta.tool_calls) > 0:
            tc = delta.tool_calls[0]
            if tc.id:
                result['tool_call_id'] = tc.id
            if tc.function:
                if tc.function.name:
                    result['function_name'] = tc.function.name
                if tc.function.arguments:
                    result['args_str'] += tc.function.arguments

    return result


def collect_stream_parallel_tool_calls(stream):
    """Consume a streaming response that may contain parallel tool calls.

    Returns (tool_calls_data, finish_reason_count) where tool_calls_data is a dict  index -> {name, args_str, id}.
    """
    tool_calls_data = {}
    finish_reason_count = 0

    for chunk in stream:
        if chunk.choices[0].finish_reason:
            finish_reason_count += 1

        streamed_tool_calls = chunk.choices[0].delta.tool_calls
        if streamed_tool_calls:
            for stc in streamed_tool_calls:
                idx = stc.index if stc.index is not None else 0
                if idx not in tool_calls_data:
                    tool_calls_data[idx] = {
                        'name': None,
                        'args_str': '',
                        'id': None,
                    }
                if stc.id:
                    tool_calls_data[idx]['id'] = stc.id
                if stc.function:
                    if stc.function.name:
                        tool_calls_data[idx]['name'] = stc.function.name
                    if stc.function.arguments:
                        tool_calls_data[idx]['args_str'] += (stc.function.arguments)

    return tool_calls_data, finish_reason_count


def collect_stream_content(stream):
    """Consume a streaming response and return (chunks, finish_reason)."""
    chunks = []
    finish_reason = None
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            chunks.append(delta.content)
        if chunk.choices[0].finish_reason is not None:
            finish_reason = chunk.choices[0].finish_reason
    return chunks, finish_reason


def collect_stream_reasoning(stream):
    """Consume a streaming response, collecting reasoning + content + tool
    calls.

    Returns a dict with keys:
        reasoning_content   – aggregated reasoning string
        content             – aggregated final content string
        tool_calls          – dict  index -> {name, args_str, id}
        finish_reason       – last non-None finish_reason
        role                – first non-None role value
        chunk_count         – total number of chunks received
        reasoning_chunks    – number of chunks containing reasoning
        content_chunks      – number of chunks containing content
    """
    result = {
        'reasoning_content': '',
        'content': '',
        'tool_calls': {},
        'finish_reason': None,
        'role': None,
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

        delta = choice.delta
        if delta.role:
            result['role'] = delta.role

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
                idx = stc.index if stc.index is not None else 0
                if idx not in result['tool_calls']:
                    result['tool_calls'][idx] = {
                        'name': None,
                        'args_str': '',
                        'id': None,
                    }
                if stc.id:
                    result['tool_calls'][idx]['id'] = stc.id
                if stc.function:
                    if stc.function.name:
                        result['tool_calls'][idx]['name'] = stc.function.name
                    if stc.function.arguments:
                        result['tool_calls'][idx]['args_str'] += (stc.function.arguments)

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
            'content': json.dumps({
                'temperature': 98,
                'unit': 'fahrenheit',
                'description': 'Sunny',
            }),
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_002',
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
            'content':
            json.dumps({
                'temperature': 95,
                'unit': 'fahrenheit',
                'description': 'Sunny with clear skies',
                'precipitation': '0%',
            }),
        },
    ]
