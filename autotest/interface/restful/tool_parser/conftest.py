import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from utils.constant import BACKEND_LIST, TOOL_REASONING_MODEL_LIST
from utils.tool_reasoning_definitions import (ALL_OPTIONAL_TOOL, CALCULATOR_TOOL, NESTED_PARAM_TOOL, SEARCH_TOOL,
                                              WEATHER_TOOL, WEATHER_TOOL_CN, assert_arguments_parseable,
                                              assert_tool_call_fields, build_messages_with_parallel_tool_responses,
                                              build_messages_with_tool_response, collect_stream_parallel_tool_calls,
                                              collect_stream_tool_call, make_logged_client, setup_log_file)

# ---------------------------------------------------------------------------
# sys.path manipulation (needed for lmdeploy imports in parser unit tests)
# ---------------------------------------------------------------------------

_LMDEPLOY_ROOT = str(Path(__file__).resolve().parents[4] / 'lmdeploy')
_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
for _p in (_PROJECT_ROOT, _LMDEPLOY_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

_CLASS_MARKS = [
    pytest.mark.order(8),
    pytest.mark.tool_call,
    pytest.mark.flaky(reruns=2),
    pytest.mark.parametrize('backend', BACKEND_LIST),
    pytest.mark.parametrize('model_case', TOOL_REASONING_MODEL_LIST),
]


def _apply_marks(cls):
    """Apply the shared set of marks to *cls* and return it."""
    for m in _CLASS_MARKS:
        cls = m(cls)
    return cls


# ---------------------------------------------------------------------------
# Logging helpers – uses shared StreamTee / setup_log_file / make_logged_client
# from utils.tool_reasoning_definitions.
# ---------------------------------------------------------------------------


class _ToolCallTestBase:
    """Mixin providing per-test API request/response logging to *log_path*."""

    @pytest.fixture(autouse=True)
    def _setup_logging(self, request, config, backend, model_case):
        """Create the log directory and compute the log-file path."""
        self._log_file = setup_log_file(config, request.node.name, 'tool_calls')

    def _get_client(self):
        """Return *(client, model_name)* with transparent logging."""
        return make_logged_client(self._log_file)


# ---------------------------------------------------------------------------
# Message constants
# ---------------------------------------------------------------------------

MESSAGES_ASKING_FOR_WEATHER = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant that can use tools. '
        'When asked about weather, use the get_current_weather tool.',
    },
    {
        'role': 'user',
        'content': "What's the weather like in Dallas, TX?",
    },
]

MESSAGES_ASKING_FOR_SEARCH = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant with access to tools. '
        'Use the web_search tool when asked to look something up.',
    },
    {
        'role': 'user',
        'content': 'Search the web for the latest news about AI.',
    },
]

MESSAGES_ASKING_FOR_CALCULATION = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant. When asked math questions, '
        'use the calculate tool.',
    },
    {
        'role': 'user',
        'content': 'What is 1234 * 5678?',
    },
]

MESSAGES_ASKING_FOR_WEATHER_CN = [
    {
        'role': 'system',
        'content': '你是一个有用的助手，可以使用工具。'
        '当被问到天气时，请使用get_current_weather工具。',
    },
    {
        'role': 'user',
        'content': '北京今天的天气怎么样？',
    },
]

MESSAGES_NO_TOOL_NEEDED = [
    {
        'role': 'user',
        'content': 'Hi, please introduce yourself briefly.',
    },
]

MESSAGES_PARALLEL_WEATHER = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant. When asked about weather '
        'in multiple cities, call the weather tool for each city '
        'separately.',
    },
    {
        'role': 'user',
        'content': "What's the weather in Dallas, TX and also in "
        'San Francisco, CA?',
    },
]

MESSAGES_PARALLEL_MIXED = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant with access to multiple tools. '
        'You can call multiple tools in parallel when needed.',
    },
    {
        'role': 'user',
        'content': "What's the weather in Dallas, TX? "
        'Also calculate 1234 * 5678.',
    },
]


# ===========================================================================
# Parser unit test infrastructure
# ===========================================================================

# ---------------------------------------------------------------------------
# Lazy tool parser imports
# ---------------------------------------------------------------------------

_qwen3_parser_cls = None
_llama3_parser_cls = None
_internlm2_parser_cls = None
_qwen2d5_parser_cls = None


def _get_qwen3_tool_parser_cls():
    global _qwen3_parser_cls
    if _qwen3_parser_cls is None:
        from lmdeploy.serve.openai.tool_parser import Qwen3ToolParser
        _qwen3_parser_cls = Qwen3ToolParser
    return _qwen3_parser_cls


def _get_llama3_tool_parser_cls():
    global _llama3_parser_cls
    if _llama3_parser_cls is None:
        from lmdeploy.serve.openai.tool_parser import Llama3JsonToolParser
        _llama3_parser_cls = Llama3JsonToolParser
    return _llama3_parser_cls


def _get_internlm2_tool_parser_cls():
    global _internlm2_parser_cls
    if _internlm2_parser_cls is None:
        from lmdeploy.serve.openai.tool_parser import Internlm2ToolParser
        _internlm2_parser_cls = Internlm2ToolParser
    return _internlm2_parser_cls


def _get_qwen2d5_tool_parser_cls():
    global _qwen2d5_parser_cls
    if _qwen2d5_parser_cls is None:
        from lmdeploy.serve.openai.tool_parser import Qwen2d5ToolParser
        _qwen2d5_parser_cls = Qwen2d5ToolParser
    return _qwen2d5_parser_cls


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_tool_mock_tokenizer(encode_map=None):
    """Create a mock tokenizer for tool parser unit tests.

    *encode_map*: optional dict ``{text: [token_id, ...]}`` consumed by
    ``tokenizer.encode(text, add_special_tokens=False)``.
    """
    tok = MagicMock()
    tok.get_vocab.return_value = {
        '<tool_call>': 1,
        '</tool_call>': 2,
        '<|python_tag|>': 3,
        '<|action_start|>': 4,
        '<|plugin|>': 5,
        '<|action_end|>': 6,
    }
    default_encode = {'<|python_tag|>': [3]}
    if encode_map:
        default_encode.update(encode_map)

    def _encode(text, add_special_tokens=False):
        return default_encode.get(text, [999])

    tok.encode = _encode
    return tok


def _make_tool_mock_request(tools=None):
    """Create a mock ChatCompletionRequest for tool parser unit tests.

    Important: parser-internal state attributes (e.g. ``_tool_parser_state``)
    must be explicitly set to ``None`` so that ``getattr(req, attr, None)``
    returns ``None`` rather than a MagicMock — otherwise the parser skips
    its own initialisation and uses the MagicMock as state, breaking slicing
    and Pydantic validation.
    """
    req = MagicMock()
    req.model = 'test-model'
    req.tools = tools
    req.tool_choice = 'auto'
    req.skip_special_tokens = True
    # Streaming parsers store their state on the request object;
    # ensure these start as None so the parser creates proper state.
    req._tool_parser_state = None
    req._tool_call_parser_state = None
    return req


# ---------------------------------------------------------------------------
# Streaming simulation helpers
# ---------------------------------------------------------------------------


def _run_tool_streaming(parser, deltas, request=None):
    """Feed *deltas* (list of text chunks) through
    ``parser.extract_tool_calls_streaming`` one by one.

    Returns a list of non-None :class:`DeltaMessage` objects.
    """
    if request is None:
        request = _make_tool_mock_request()

    results = []
    previous_text = ''
    previous_token_ids = []

    for d in deltas:
        current_text = previous_text + d
        current_token_ids = list(range(len(current_text)))
        delta_token_ids = list(range(len(d)))

        result = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=d,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
            request=request,
        )

        if result is not None:
            results.append(result)

        previous_text = current_text
        previous_token_ids = current_token_ids

    return results


def _collect_tool_streaming(delta_messages):
    """Accumulate content and tool call information from a list of
    :class:`DeltaMessage` objects returned by :func:`_run_tool_streaming`.

    Returns ``(content_str_or_None, {index: {id, name, arguments, type}})``.
    """
    content_parts = []
    tool_states = {}

    for delta in delta_messages:
        if delta.content:
            content_parts.append(delta.content)

        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in tool_states:
                tool_states[idx] = {
                    'id': None,
                    'name': None,
                    'arguments': '',
                    'type': None,
                }

            if tc.id:
                tool_states[idx]['id'] = tc.id

            func = tc.function
            if func is not None:
                if isinstance(func, dict):
                    if func.get('name'):
                        tool_states[idx]['name'] = func['name']
                    if func.get('arguments') is not None:
                        tool_states[idx]['arguments'] += func['arguments']
                else:
                    if getattr(func, 'name', None):
                        tool_states[idx]['name'] = func.name
                    if getattr(func, 'arguments', None) is not None:
                        tool_states[idx]['arguments'] += func.arguments

    content = ''.join(content_parts) if content_parts else None
    return content, tool_states


# ---------------------------------------------------------------------------
# Parser unit test marks
# ---------------------------------------------------------------------------

_PARSER_UNIT_MARKS = [
    pytest.mark.order(8),
    pytest.mark.tool_call,
    pytest.mark.tool_parser,
]


def _apply_parser_unit_marks(cls):
    """Apply lightweight marks for tool parser unit-test classes."""
    for m in _PARSER_UNIT_MARKS:
        cls = m(cls)
    return cls
