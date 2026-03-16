import json

import pytest

from .conftest import (
    _apply_parser_unit_marks,
    _collect_tool_streaming,
    _get_qwen2d5_tool_parser_cls,
    _make_tool_mock_request,
    _make_tool_mock_tokenizer,
    _run_tool_streaming,
)


# ===================================================================
# Test data — (model_output, expected_tools_called, expected_tools,
#              expected_content)
#
# Qwen2.5 tool call format:
#   <tool_call>{"name": "func", "arguments": {...}}</tool_call>
# ===================================================================

NO_TOOLS = (
    'Just a normal response.',
    False,
    [],
    'Just a normal response.',
)

SINGLE_TOOL = (
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Dallas", "state": "TX"}}</tool_call>',
    True,
    [{'name': 'get_weather', 'arguments': {'city': 'Dallas', 'state': 'TX'}}],
    None,  # starts with <tool_call> and ends with </tool_call> → content=None
)

TOOL_WITH_CONTENT_PREFIX = (
    'Let me check the weather.'
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Dallas"}}</tool_call>',
    True,
    [{'name': 'get_weather', 'arguments': {'city': 'Dallas'}}],
    'Let me check the weather.',
)

TOOL_WITH_CONTENT_SUFFIX = (
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Dallas"}}</tool_call>'
    'Here is the result.',
    True,
    [{'name': 'get_weather', 'arguments': {'city': 'Dallas'}}],
    'Here is the result.',
)

MULTIPLE_TOOLS = (
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Dallas"}}</tool_call>'
    '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>',
    True,
    [
        {'name': 'get_weather', 'arguments': {'city': 'Dallas'}},
        {'name': 'get_weather', 'arguments': {'city': 'SF'}},
    ],
    None,
)

NESTED_ARGS = (
    '<tool_call>{"name": "create_event", "arguments": '
    '{"title": "Meeting", "location": {"city": "NYC"}}}</tool_call>',
    True,
    [{'name': 'create_event', 'arguments': {
        'title': 'Meeting', 'location': {'city': 'NYC'},
    }}],
    None,
)

# --- Truncated JSON (e.g. max_tokens cut mid-string) ---

TRUNCATED_JSON_IN_TAGS = (
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Dallas</tool_call>',
)


# ===================================================================
# Non-streaming tests
# ===================================================================


@_apply_parser_unit_marks
class TestQwen2d5ToolParserNonStreaming:
    """Qwen2.5 non-streaming extract_tool_calls."""

    @staticmethod
    def _make_parser():
        return _get_qwen2d5_tool_parser_cls()(_make_tool_mock_tokenizer())

    @pytest.mark.parametrize(
        'model_output, expected_tools_called, expected_tools, expected_content',
        [
            pytest.param(*NO_TOOLS, id='no_tools'),
            pytest.param(*SINGLE_TOOL, id='single_tool'),
            pytest.param(*TOOL_WITH_CONTENT_PREFIX, id='content_prefix'),
            pytest.param(*TOOL_WITH_CONTENT_SUFFIX, id='content_suffix'),
            pytest.param(*MULTIPLE_TOOLS, id='multiple_tools'),
            pytest.param(*NESTED_ARGS, id='nested_args'),
        ],
    )
    def test_extract_tool_calls(self, model_output, expected_tools_called,
                                expected_tools, expected_content):
        parser = self._make_parser()
        req = _make_tool_mock_request()

        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called == expected_tools_called
        assert len(result.tool_calls) == len(expected_tools)

        for tc, expected in zip(result.tool_calls, expected_tools):
            assert tc.function.name == expected['name']
            assert json.loads(tc.function.arguments) == expected['arguments']

        assert result.content == expected_content

    def test_truncated_json_should_not_crash(self):
        """Truncated JSON between <tool_call>...</tool_call> must not
        raise JSONDecodeError.

        Root cause: ``json.loads(match_result)`` (line 165 of source) has
        no try/except.  When max_tokens truncates the output mid-string,
        the regex still matches both tags but captures malformed JSON like
        ``{"name": "get_weather", "arguments": {"city": "Dallas``.
        ``json.loads`` then raises ``JSONDecodeError: Unterminated string``
        which propagates uncaught and crashes the request handler.
        """
        parser = self._make_parser()
        req = _make_tool_mock_request()

        model_output = TRUNCATED_JSON_IN_TAGS[0]

        try:
            result = parser.extract_tool_calls(model_output, req)
            # If the parser handles it gracefully, it should fallback
            assert not result.tools_called or result.content is not None
        except json.JSONDecodeError as e:
            pytest.fail(
                f'Truncated JSON crashed extract_tool_calls with '
                f'JSONDecodeError: {e}'
                '\n\nRoot cause: json.loads(match_result) in '
                'extract_tool_calls has no try/except.'
            )


# ===================================================================
# Streaming tests
# ===================================================================


@_apply_parser_unit_marks
class TestQwen2d5ToolParserStreaming:
    """Qwen2.5 streaming extract_tool_calls_streaming.

    Tool calls are detected via ``<tool_call>`` in the accumulated text.
    Text before this token is emitted as content.
    """

    @staticmethod
    def _make_parser():
        return _get_qwen2d5_tool_parser_cls()(_make_tool_mock_tokenizer())

    def test_no_tool_streaming(self):
        """Plain text → all emitted as content."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = ['Hello, ', 'how can I ', 'help?']
        results = _run_tool_streaming(parser, deltas, req)
        content, tools = _collect_tool_streaming(results)

        assert content == 'Hello, how can I help?'
        assert len(tools) == 0

    def test_content_before_tool_tag(self):
        """Content before <tool_call> is emitted as content."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = [
            'Let me check.',
            '<tool_call>',
            '{"name": "get_weather", "parameters": {"city": "Dallas"}}',
        ]
        results = _run_tool_streaming(parser, deltas, req)
        content, tools = _collect_tool_streaming(results)

        assert content is not None
        assert 'Let me check.' in content

    def test_truncated_json_streaming_should_not_crash(self):
        """Streaming with truncated JSON arguments must not raise.

        Simulates max_tokens truncation mid-argument string value.
        The parser should either skip the chunk gracefully (outer
        except catches) or produce partial results — but never raise.
        """
        parser = self._make_parser()
        req = _make_tool_mock_request()

        # JSON gets cut mid-string value, then </tool_call> arrives
        deltas = [
            '<tool_call>',
            '{"name": "get_weather", "arguments": {"city": "Dallas',
            '</tool_call>',
        ]

        # Must not raise any exception
        try:
            results = _run_tool_streaming(parser, deltas, req)
            _collect_tool_streaming(results)
        except Exception as e:
            pytest.fail(
                f'Streaming truncated JSON crashed with {type(e).__name__}: '
                f'{e}\n\nThe parser should handle truncated JSON gracefully.'
            )

    def test_parallel_tools_streaming_should_not_crash(self):
        """Parallel tool calls in streaming must not raise ValueError.

        Root cause: ``self.position`` is never advanced past the first
        ``<tool_call>`` block, so when the second ``<tool_call>`` arrives
        ``new_delta.split(self.tool_start_token)`` returns 3 items and
        the unpacking ``text, action = ...`` raises ``ValueError``.
        Additionally this ``split`` is outside the try/except block, so the
        exception propagates uncaught and crashes the streaming handler.
        """
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = [
            '<tool_call>',
            '{"name": "get_weather", "arguments": {"city": "Dallas"}}',
            '</tool_call>',
            '<tool_call>',
            '{"name": "get_weather", "arguments": {"city": "SF"}}',
            '</tool_call>',
        ]

        # This should not raise — if it does, the server would crash
        # during streaming.
        try:
            results = _run_tool_streaming(parser, deltas, req)
            content, tools = _collect_tool_streaming(results)
        except ValueError as e:
            pytest.fail(
                f'Streaming parallel tool calls crashed with ValueError: {e}'
                '\n\nRoot cause: position not advanced past first tool call, '
                'causing split() on line with multiple <tool_call> to return '
                '>2 items.'
            )
