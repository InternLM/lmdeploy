import json

import pytest

from .conftest import (
    _apply_parser_unit_marks,
    _collect_tool_streaming,
    _get_qwen3_tool_parser_cls,
    _make_tool_mock_request,
    _make_tool_mock_tokenizer,
    _run_tool_streaming,
)

# ===================================================================
# Test data — (model_output, expected_tool_calls, expected_content)
#
# Each expected_tool_call is {name, arguments (dict)}.
# ===================================================================

NO_TOOLS = (
    'Just a normal response without any tool calls.',
    [],
    'Just a normal response without any tool calls.',
)

SINGLE_TOOL = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Dallas", "state": "TX"}}\n</tool_call>',
    [{'name': 'get_weather', 'arguments': {'city': 'Dallas', 'state': 'TX'}}],
    '',
)

TOOL_WITH_CONTENT = (
    'Let me check the weather.'
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Dallas"}}\n</tool_call>',
    [{'name': 'get_weather', 'arguments': {'city': 'Dallas'}}],
    'Let me check the weather.',
)

PARALLEL_TOOLS = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Dallas"}}\n</tool_call>'
    '\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>',
    [
        {'name': 'get_weather', 'arguments': {'city': 'Dallas'}},
        {'name': 'get_weather', 'arguments': {'city': 'SF'}},
    ],
    '',
)

NESTED_ARGS = (
    '<tool_call>\n{"name": "calculate_area", "arguments": '
    '{"shape": "rectangle", "dimensions": {"width": 10, "height": 20}}}\n</tool_call>',
    [{'name': 'calculate_area', 'arguments': {
        'shape': 'rectangle', 'dimensions': {'width': 10, 'height': 20},
    }}],
    '',
)


# ===================================================================
# Non-streaming tests
# ===================================================================


@_apply_parser_unit_marks
class TestQwen3ToolParserNonStreaming:
    """Qwen3 non-streaming extract_tool_calls."""

    @staticmethod
    def _make_parser():
        return _get_qwen3_tool_parser_cls()(_make_tool_mock_tokenizer())

    @pytest.mark.parametrize(
        'model_output, expected_tools, expected_content',
        [
            pytest.param(*NO_TOOLS, id='no_tools'),
            pytest.param(*SINGLE_TOOL, id='single_tool'),
            pytest.param(*TOOL_WITH_CONTENT, id='tool_with_content'),
            pytest.param(*PARALLEL_TOOLS, id='parallel_tools'),
            pytest.param(*NESTED_ARGS, id='nested_args'),
        ],
    )
    def test_extract_tool_calls(self, model_output, expected_tools,
                                expected_content):
        parser = self._make_parser()
        req = _make_tool_mock_request()

        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called == bool(expected_tools)
        assert len(result.tool_calls) == len(expected_tools)

        for tc, expected in zip(result.tool_calls, expected_tools):
            assert tc.function.name == expected['name']
            assert json.loads(tc.function.arguments) == expected['arguments']

        assert result.content == expected_content

    def test_parameters_key_alias(self):
        """``parameters`` key in JSON should work identically to
        ``arguments``."""
        parser = self._make_parser()
        req = _make_tool_mock_request()
        output = ('<tool_call>\n{"name": "func", "arguments": '
                  '{"key": "value"}}\n</tool_call>')

        result = parser.extract_tool_calls(output, req)

        assert result.tools_called
        assert result.tool_calls[0].function.name == 'func'
        assert json.loads(
            result.tool_calls[0].function.arguments) == {'key': 'value'}


# ===================================================================
# Streaming tests
# ===================================================================


@_apply_parser_unit_marks
class TestQwen3ToolParserStreaming:
    """Qwen3 streaming extract_tool_calls_streaming.

    Deltas are split at ``<tool_call>`` / ``</tool_call>`` boundaries,
    matching real tokenizer behaviour where special tokens are single
    tokens.
    """

    @staticmethod
    def _make_parser():
        return _get_qwen3_tool_parser_cls()(_make_tool_mock_tokenizer())

    def test_no_tools_streaming(self):
        """Plain text → all emitted as content."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = ['Just ', 'a normal ', 'response.']
        results = _run_tool_streaming(parser, deltas, req)
        content, tools = _collect_tool_streaming(results)

        assert content == 'Just a normal response.'
        assert len(tools) == 0

    def test_single_tool_streaming(self):
        """Single tool call → name + arguments extracted."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = [
            '<tool_call>',
            '\n{"name": "get_weather", "arguments": {"city": "Dallas"}}\n',
            '</tool_call>',
        ]
        results = _run_tool_streaming(parser, deltas, req)
        content, tools = _collect_tool_streaming(results)

        assert len(tools) == 1
        assert tools[0]['name'] == 'get_weather'
        assert json.loads(tools[0]['arguments']) == {'city': 'Dallas'}

    def test_tool_with_content_streaming(self):
        """Content before tool call → content emitted, then tool call."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = [
            'Let me check.',
            '<tool_call>',
            '\n{"name": "get_weather", "arguments": {"city": "Dallas"}}\n',
            '</tool_call>',
        ]
        results = _run_tool_streaming(parser, deltas, req)
        content, tools = _collect_tool_streaming(results)

        assert content == 'Let me check.'
        assert len(tools) == 1
        assert tools[0]['name'] == 'get_weather'

    def test_parallel_tools_streaming(self):
        """Two tool calls → both extracted with distinct indices."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = [
            '<tool_call>',
            '\n{"name": "get_weather", "arguments": {"city": "Dallas"}}\n',
            '</tool_call>',
            '\n',
            '<tool_call>',
            '\n{"name": "get_weather", "arguments": {"city": "SF"}}\n',
            '</tool_call>',
        ]
        results = _run_tool_streaming(parser, deltas, req)
        content, tools = _collect_tool_streaming(results)

        assert len(tools) == 2
        assert tools[0]['name'] == 'get_weather'
        assert tools[1]['name'] == 'get_weather'
        assert json.loads(tools[0]['arguments']) == {'city': 'Dallas'}
        assert json.loads(tools[1]['arguments']) == {'city': 'SF'}
