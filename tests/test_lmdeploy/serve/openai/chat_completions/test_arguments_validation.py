# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for complete tool-call parsing and request normalization."""

import pytest


def test_parse_tool_call_complete_json_preserves_arguments():
    """parse_tool_call_complete preserves argument bytes."""
    from lmdeploy.serve.parsers.tool_parser.json_tool_parser import JsonToolParser

    payload = '{"name": "get_weather", "arguments": {"city": "NYC" }  }'
    result = JsonToolParser().parse_tool_call_complete(payload)
    assert result is not None
    assert result.function.name == 'get_weather'
    assert result.function.arguments == '{"city": "NYC" }'


def test_parse_tool_call_complete_json_preserves_incomplete_arguments():
    """Complete parsing leaves argument validity to the caller."""
    from lmdeploy.serve.parsers.tool_parser.json_tool_parser import JsonToolParser

    result = JsonToolParser().parse_tool_call_complete(
        '{"name": "get_weather", "arguments": {"city":'
    )
    assert result is not None
    assert result.function.name == 'get_weather'
    assert result.function.arguments == '{"city":'


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
