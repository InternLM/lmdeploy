import json

import pytest

from .conftest import (
    _apply_parser_unit_marks,
    _collect_tool_streaming,
    _get_llama3_tool_parser_cls,
    _make_tool_mock_request,
    _make_tool_mock_tokenizer,
    _run_tool_streaming,
)


# ===================================================================
# Test data — (model_output, expected_name, expected_args_dict)
#
# Llama3 non-streaming format: <function=NAME>{JSON}</function>
# ===================================================================

SINGLE_TOOL = (
    '<function=get_weather>{"city": "Dallas", "state": "TX"}</function>',
    'get_weather',
    {'city': 'Dallas', 'state': 'TX'},
)

SINGLE_TOOL_NESTED = (
    '<function=create_event>{"title": "Meeting", '
    '"location": {"city": "NYC", "room": "A1"}}</function>',
    'create_event',
    {'title': 'Meeting', 'location': {'city': 'NYC', 'room': 'A1'}},
)


# ===================================================================
# Non-streaming tests
# ===================================================================


@_apply_parser_unit_marks
class TestLlama3ToolParserNonStreaming:
    """Llama3 non-streaming extract_tool_calls."""

    @staticmethod
    def _make_parser():
        tok = _make_tool_mock_tokenizer(
            encode_map={'<|python_tag|>': [3]},
        )
        return _get_llama3_tool_parser_cls()(tok)

    @pytest.mark.parametrize(
        'model_output, expected_name, expected_args',
        [
            pytest.param(*SINGLE_TOOL, id='single_tool'),
            pytest.param(*SINGLE_TOOL_NESTED, id='nested_args'),
        ],
    )
    def test_extract_tool_calls(self, model_output, expected_name,
                                expected_args):
        parser = self._make_parser()
        req = _make_tool_mock_request()

        result = parser.extract_tool_calls(model_output, req)

        assert result.tools_called
        assert len(result.tool_calls) == 1

        tc = result.tool_calls[0]
        assert tc.function.name == expected_name
        assert json.loads(tc.function.arguments) == expected_args
        assert result.content is None

    def test_no_tool_fallback(self):
        """Malformed or plain text → tools_called=False, content=text."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        model_output = 'Just a normal text response.'
        result = parser.extract_tool_calls(model_output, req)

        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == model_output

    def test_missing_closing_tag(self):
        """Missing </function> tag → graceful fallback."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        model_output = '<function=get_weather>{"city": "Dallas"}'
        result = parser.extract_tool_calls(model_output, req)

        assert not result.tools_called
        assert result.content == model_output


# ===================================================================
# Streaming tests
# ===================================================================


@_apply_parser_unit_marks
class TestLlama3ToolParserStreaming:
    """Llama3 streaming extract_tool_calls_streaming.

    Llama3 streaming detects tool calls via ``<|python_tag|>`` or ``{``
    prefix.  Text not starting with these tokens is emitted as content.
    """

    @staticmethod
    def _make_parser():
        tok = _make_tool_mock_tokenizer(
            encode_map={'<|python_tag|>': [3]},
        )
        return _get_llama3_tool_parser_cls()(tok)

    def test_no_tool_streaming(self):
        """Plain text → all emitted as content."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = ['Hello, ', 'how can I ', 'help you?']
        results = _run_tool_streaming(parser, deltas, req)
        content, tools = _collect_tool_streaming(results)

        assert content == 'Hello, how can I help you?'
        assert len(tools) == 0

    def test_text_not_starting_with_tag(self):
        """Text that does not start with <|python_tag|> or { is
        always content, even if it contains JSON-like strings."""
        parser = self._make_parser()
        req = _make_tool_mock_request()

        deltas = ['The answer is: ', '{"key": "value"}']
        results = _run_tool_streaming(parser, deltas, req)
        content, tools = _collect_tool_streaming(results)

        assert content == 'The answer is: {"key": "value"}'
        assert len(tools) == 0
