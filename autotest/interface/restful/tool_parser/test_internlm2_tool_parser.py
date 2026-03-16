import json

import pytest

from .conftest import (
    _apply_parser_unit_marks,
    _collect_tool_streaming,
    _get_internlm2_tool_parser_cls,
    _make_tool_mock_request,
    _make_tool_mock_tokenizer,
    _run_tool_streaming,
)

# ===================================================================
# Test data
#
# InternLM2 tool call format:
#   content<|action_start|><|plugin|>\nJSON\n<|action_end|>
# JSON: {"name": "func", "parameters": {...}}
# ===================================================================

NO_TOOLS = (
    'Just a normal response.',
    False,
    [],
    'Just a normal response.',
)

SINGLE_TOOL = (
    '<|action_start|><|plugin|>\n'
    '{"name": "get_weather", "parameters": {"city": "Dallas", "state": "TX"}}\n'
    '<|action_end|>',
    True,
    [{'name': 'get_weather', 'arguments': {'city': 'Dallas', 'state': 'TX'}}],
    None,
)

TOOL_WITH_CONTENT = (
    'Let me check the weather.<|action_start|><|plugin|>\n'
    '{"name": "get_weather", "parameters": {"city": "Dallas"}}\n'
    '<|action_end|>',
    True,
    [{'name': 'get_weather', 'arguments': {'city': 'Dallas'}}],
    'Let me check the weather.',
)

TOOL_WITH_ARGUMENTS_KEY = (
    '<|action_start|><|plugin|>\n'
    '{"name": "search", "arguments": {"query": "AI news"}}\n'
    '<|action_end|>',
    True,
    [{'name': 'search', 'arguments': {'query': 'AI news'}}],
    None,
)


# ===================================================================
# Non-streaming tests
# ===================================================================


@_apply_parser_unit_marks
class TestInternlm2ToolParserNonStreaming:
    """InternLM2 non-streaming extract_tool_calls."""

    @staticmethod
    def _make_parser():
        return _get_internlm2_tool_parser_cls()(_make_tool_mock_tokenizer())

    @pytest.mark.parametrize(
        'model_output, expected_tools_called, expected_tools, expected_content',
        [
            pytest.param(*NO_TOOLS, id='no_tools'),
            pytest.param(*SINGLE_TOOL, id='single_tool'),
            pytest.param(*TOOL_WITH_CONTENT, id='tool_with_content'),
            pytest.param(*TOOL_WITH_ARGUMENTS_KEY, id='arguments_key'),
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
        """Truncated JSON in tool call tags must not raise JSONDecodeError.

        Root cause: ``json.loads(action)`` (line 169 of source) has no
        try/except.  When max_tokens truncates the output mid-string,
        the split still captures malformed JSON and ``json.loads`` raises
        ``JSONDecodeError`` which propagates uncaught.
        """
        parser = self._make_parser()
        req = _make_tool_mock_request()

        model_output = (
            '<|action_start|><|plugin|>\n'
            '{"name": "get_weather", "parameters": {"city": "Dallas'
            '\n<|action_end|>'
        )

        try:
            result = parser.extract_tool_calls(model_output, req)
            assert not result.tools_called or result.content is not None
        except json.JSONDecodeError as e:
            pytest.fail(
                f'Truncated JSON crashed extract_tool_calls with '
                f'JSONDecodeError: {e}'
                '\n\nRoot cause: json.loads(action) in '
                'extract_tool_calls has no try/except.'
            )


# ===================================================================
# Streaming tests
# ===================================================================


@_apply_parser_unit_marks
class TestInternlm2ToolParserStreaming:
    """InternLM2 streaming extract_tool_calls_streaming.

    Tool calls are detected via ``<|action_start|>`` in the accumulated
    text.  Text before this token is emitted as content.
    """

    @staticmethod
    def _make_parser():
        return _get_internlm2_tool_parser_cls()(_make_tool_mock_tokenizer())

    def test_no_tool_streaming(self):
        """Plain text → all emitted as content."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = ['Hello, ', 'how can I ', 'help?']
        results = _run_tool_streaming(parser, deltas, req)
        content, tools = _collect_tool_streaming(results)

        assert content == 'Hello, how can I help?'
        assert len(tools) == 0

    def test_content_before_action_start(self):
        """Content before <|action_start|> is emitted as content."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = ['Let me check.', '<|action_start|><|plugin|>\n']
        results = _run_tool_streaming(parser, deltas, req)
        content, tools = _collect_tool_streaming(results)

        assert content is not None
        assert 'Let me check.' in content

    def test_truncated_json_streaming_should_not_crash(self):
        """Streaming with truncated JSON arguments must not raise.

        Simulates max_tokens truncation mid-argument string value.
        """
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = [
            '<|action_start|><|plugin|>\n',
            '{"name": "get_weather", "parameters": {"city": "Dallas',
            '\n<|action_end|>',
        ]

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

        Same root cause as Qwen2d5: ``self.position`` is never advanced
        past the first tool block, so when the second
        ``<|action_start|><|plugin|>`` arrives the ``split()`` returns
        3 items and the unpacking crashes.  The ``split`` is outside the
        try/except block, so the exception propagates uncaught.
        """
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = [
            '<|action_start|><|plugin|>\n',
            '{"name": "get_weather", "parameters": {"city": "Dallas"}}\n',
            '<|action_end|>',
            '<|action_start|><|plugin|>\n',
            '{"name": "get_weather", "parameters": {"city": "SF"}}\n',
            '<|action_end|>',
        ]

        try:
            results = _run_tool_streaming(parser, deltas, req)
            content, tools = _collect_tool_streaming(results)
        except ValueError as e:
            pytest.fail(
                f'Streaming parallel tool calls crashed with ValueError: {e}'
                '\n\nRoot cause: position not advanced past first tool call, '
                'causing split() on line with multiple '
                '<|action_start|><|plugin|> to return >2 items.'
            )
