# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for tool-call argument validation and normalization."""
import json

import pytest


def test_parse_tool_call_complete_json_normalizes_arguments():
    """_parse_tool_call_complete_json validates and re-serializes arguments."""
    from lmdeploy.serve.parsers.tool_parser.tool_parser import ToolParser

    class TestToolParser(ToolParser):
        def get_tool_open_tag(self):
            return None

        def get_tool_close_tag(self):
            return None

        def get_tool_payload_format(self):
            return 'json'

        def decode_tool_incremental(self, added_text, *, final):
            return []

        def parse_tool_call_complete(self, payload):
            return None

    payload = '{"name": "get_weather", "arguments": {"city": "NYC" }  }'
    result = TestToolParser._parse_tool_call_complete_json(payload)
    assert result is not None
    assert result.function.name == 'get_weather'
    parsed_args = json.loads(result.function.arguments)
    assert parsed_args == {'city': 'NYC'}


def test_parse_tool_call_complete_json_invalid_returns_none():
    """_parse_tool_call_complete_json returns None for invalid JSON."""
    from lmdeploy.serve.parsers.tool_parser.tool_parser import ToolParser

    class TestToolParser(ToolParser):
        def get_tool_open_tag(self):
            return None

        def get_tool_close_tag(self):
            return None

        def get_tool_payload_format(self):
            return 'json'

        def decode_tool_incremental(self, added_text, *, final):
            return []

        def parse_tool_call_complete(self, payload):
            return None

    result = TestToolParser._parse_tool_call_complete_json(
        '{"name": "get_weather", "arguments": {"city":'
    )
    assert result is None


def test_parse_tool_call_arguments_dict_raises_on_invalid():
    """_parse_tool_call_arguments_dict should raise ValueError on invalid
    JSON."""
    from lmdeploy.serve.parsers.response_parser import _parse_tool_call_arguments_dict

    with pytest.raises(ValueError, match=r'invalid JSON at position \d+ \(line \d+, column \d+\)'):
        _parse_tool_call_arguments_dict('{"city":')


def test_parse_tool_call_arguments_dict_returns_dict_on_valid():
    """_parse_tool_call_arguments_dict returns dict for valid JSON."""
    from lmdeploy.serve.parsers.response_parser import _parse_tool_call_arguments_dict

    result = _parse_tool_call_arguments_dict('{"city": "NYC"}')
    assert result == {'city': 'NYC'}


def test_parse_tool_call_arguments_dict_returns_none_for_non_string():
    """_parse_tool_call_arguments_dict returns None for non-string input (no
    error)."""
    from lmdeploy.serve.parsers.response_parser import _parse_tool_call_arguments_dict

    result = _parse_tool_call_arguments_dict({'city': 'NYC'})
    assert result is None


def test_parse_tool_call_arguments_dict_returns_none_for_non_dict_json():
    """_parse_tool_call_arguments_dict returns None when JSON parses to non-
    dict."""
    from lmdeploy.serve.parsers.response_parser import _parse_tool_call_arguments_dict

    result = _parse_tool_call_arguments_dict('[1, 2, 3]')
    assert result is None


def test_parse_complete_falls_back_to_plain_text_on_invalid_tool_arguments():
    """parse_complete falls back to plain text when tool call arguments are
    invalid."""
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest
    from lmdeploy.serve.parsers import ResponseParserManager
    from lmdeploy.serve.parsers.tool_parser import ToolParserManager

    cls = ResponseParserManager.get('default')
    old_reasoning_cls = cls.reasoning_parser_cls
    old_tool_cls = cls.tool_parser_cls
    try:
        cls.reasoning_parser_cls = None
        cls.tool_parser_cls = ToolParserManager.get('qwen3')
        request = ChatCompletionRequest(
            model='test',
            messages=[],
            tool_choice='auto',
        )
        parser = cls(request=request, tokenizer=object())
        # Feed a tool call with invalid JSON arguments using qwen3 tags
        open_tag = parser.profile.tool_open_tag
        close_tag = parser.profile.tool_close_tag
        text = open_tag + '{"name": "get_weather", "arguments": {"city":' + close_tag
        content, tool_calls, reasoning = parser.parse_complete(text)
        assert tool_calls is None
        assert reasoning is None
        # Falls back to plain text when parsing fails
        assert content is not None
    finally:
        cls.reasoning_parser_cls = old_reasoning_cls
        cls.tool_parser_cls = old_tool_cls
