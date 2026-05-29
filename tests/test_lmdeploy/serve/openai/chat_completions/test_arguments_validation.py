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


def test_json_tool_parsers_use_default_validate_complete():
    """JSON tool parsers use ToolParser's default complete-payload
    validation."""
    from lmdeploy.serve.parsers.tool_parser import (
        Internlm2ToolParser,
        Qwen2d5ToolParser,
        Qwen3ToolParser,
    )
    from lmdeploy.serve.parsers.tool_parser.tool_parser import ToolParser

    assert not hasattr(ToolParser, '_validate_complete_json_payload')
    for parser_cls in (Internlm2ToolParser, Qwen2d5ToolParser, Qwen3ToolParser):
        assert parser_cls.validate_complete is ToolParser.validate_complete


def test_xml_tool_parsers_override_payload_validation():
    """XML-like tool parsers keep model-specific complete-payload
    validation."""
    from lmdeploy.serve.parsers.tool_parser import Glm47ToolParser, Qwen3CoderToolParser
    from lmdeploy.serve.parsers.tool_parser.tool_parser import ToolParser

    for parser_cls in (Glm47ToolParser, Qwen3CoderToolParser):
        assert parser_cls.validate_complete is ToolParser.validate_complete
        assert '_validate_tool_payload' in parser_cls.__dict__


def test_json_tool_parsers_validate_complete_text():
    """JSON tool parsers validate tool tags and JSON payloads themselves."""
    from lmdeploy.serve.parsers.tool_parser import (
        Internlm2ToolParser,
        Qwen2d5ToolParser,
        Qwen3ToolParser,
    )

    valid_payload = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
    invalid_payload = '{"name": "get_weather", "arguments": {"city":'

    for parser_cls in (Internlm2ToolParser, Qwen2d5ToolParser, Qwen3ToolParser):
        parser = parser_cls()
        parser.parse_tool_call_complete = _fail_parse_tool_call_complete
        open_tag = parser.get_tool_open_tag()
        close_tag = parser.get_tool_close_tag()

        assert parser.validate_complete(f'before{open_tag}{valid_payload}{close_tag}after') is True
        assert parser.validate_complete(f'before{open_tag}{invalid_payload}{close_tag}after') is False
        assert parser.validate_complete(f'before{open_tag}{valid_payload}') is False
        assert parser.validate_complete(f'before{close_tag}after') is False
        assert parser.validate_complete('plain text') is True


def test_llama3_tool_parser_skips_complete_validation():
    """Llama3 has no close tag, so it does not recheck complete text."""
    from lmdeploy.serve.parsers.tool_parser import Llama3JsonToolParser

    parser = Llama3JsonToolParser()
    open_tag = parser.get_tool_open_tag()

    assert parser.validate_complete('plain text') is True
    assert parser.validate_complete(f'{open_tag}{{"name": "get_weather"}}') is True


def test_xml_tool_parsers_validate_complete_text():
    """XML tool parsers validate tool tags and XML payloads themselves."""
    from lmdeploy.serve.parsers.tool_parser import Glm47ToolParser, InternS2PreviewToolParser, Qwen3CoderToolParser

    glm_parser = Glm47ToolParser()
    glm_parser.parse_tool_call_complete = _fail_parse_tool_call_complete
    glm_open_tag = glm_parser.get_tool_open_tag()
    glm_close_tag = glm_parser.get_tool_close_tag()
    assert glm_parser.validate_complete(
        f'before{glm_open_tag}get_weather<arg_key>location</arg_key><arg_value>Beijing</arg_value>{glm_close_tag}after'
    ) is True
    assert glm_parser.validate_complete(f'{glm_open_tag}{glm_close_tag}') is False
    assert glm_parser.validate_complete(f'{glm_open_tag}get_weather') is False
    assert glm_parser.validate_complete('plain text') is True

    qwen_payload = '<function=get_weather><parameter=city>NYC</parameter></function>'
    qwen_incomplete_payload = '<function=get_weather><parameter=city>NYC</parameter>'
    for parser_cls in (InternS2PreviewToolParser, Qwen3CoderToolParser):
        parser = parser_cls()
        parser.parse_tool_call_complete = _fail_parse_tool_call_complete
        open_tag = parser.get_tool_open_tag()
        close_tag = parser.get_tool_close_tag()

        assert parser.validate_complete(f'before{open_tag}{qwen_payload}{close_tag}after') is True
        assert parser.validate_complete(f'before{open_tag}{qwen_incomplete_payload}{close_tag}after') is False
        assert parser.validate_complete(f'before{open_tag}{qwen_payload}') is False
        assert parser.validate_complete('plain text') is True


def _fail_parse_tool_call_complete(payload):
    raise AssertionError('validate_complete must not call parse_tool_call_complete')


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


def test_validate_complete_reports_invalid_tool_arguments_after_parse_complete():
    """parse_complete falls back, then validate_complete reports bad tool
    text."""
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
        open_tag = parser.profile.tool_open_tag
        close_tag = parser.profile.tool_close_tag
        text = open_tag + '{"name": "get_weather", "arguments": {"city":' + close_tag

        content, tool_calls, reasoning = parser.parse_complete(text)

        assert content is not None
        assert tool_calls is None
        assert reasoning is None
        assert parser.validate_complete(text) is False
    finally:
        cls.reasoning_parser_cls = old_reasoning_cls
        cls.tool_parser_cls = old_tool_cls


def test_validate_complete_checks_each_tool_call_block():
    """Every tool block in the final text must be complete and valid."""
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
        open_tag = parser.profile.tool_open_tag
        close_tag = parser.profile.tool_close_tag
        first_payload = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        second_payload = '{"name": "get_time", "arguments": {"tz": "UTC"}}'
        incomplete_payload = '{"name": "get_time", "arguments": {"tz":'

        valid_text = f'{open_tag}{first_payload}{close_tag}{open_tag}{second_payload}{close_tag}'
        invalid_text = f'{open_tag}{first_payload}{close_tag}{open_tag}{incomplete_payload}{close_tag}'
        unclosed_text = f'{open_tag}{first_payload}{close_tag}{open_tag}{second_payload}'

        content, tool_calls, reasoning = parser.parse_complete(valid_text)

        assert content is None
        assert reasoning is None
        assert [call.function.name for call in tool_calls] == ['get_weather', 'get_time']
        assert parser.validate_complete(valid_text) is True
        assert parser.validate_complete(invalid_text) is False
        assert parser.validate_complete(unclosed_text) is False
    finally:
        cls.reasoning_parser_cls = old_reasoning_cls
        cls.tool_parser_cls = old_tool_cls


def test_validate_complete_reports_invalid_tool_arguments_after_stream_chunk():
    """stream_chunk accumulates text for final validation."""
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
        open_tag = parser.profile.tool_open_tag
        close_tag = parser.profile.tool_close_tag

        parser.stream_chunk(open_tag + '{"name": "get_weather", "arguments": {"city":', [])
        parser.stream_chunk(close_tag, [])

        assert parser.validate_complete() is False
    finally:
        cls.reasoning_parser_cls = old_reasoning_cls
        cls.tool_parser_cls = old_tool_cls


def test_validate_complete_rejects_unpaired_tool_close_tag():
    """A tool close tag without a matching open tag is invalid final text."""
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

        assert parser.validate_complete('plain text</tool_call>') is False
    finally:
        cls.reasoning_parser_cls = old_reasoning_cls
        cls.tool_parser_cls = old_tool_cls


def test_validate_complete_delegates_complete_text_to_tool_parser():
    """Complete text validation is owned by the concrete tool parser."""
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
        texts = []

        def _validate_text(text):
            texts.append(text)
            return True

        parser.tool_parser.validate_complete = _validate_text
        text = 'before<tool_call>{"name": "get_weather"}</tool_call>after'

        assert parser.validate_complete(text) is True
        assert texts == [text]
    finally:
        cls.reasoning_parser_cls = old_reasoning_cls
        cls.tool_parser_cls = old_tool_cls


def test_validate_complete_allows_tool_pair_when_tool_parser_disabled():
    """Complete tool-like text is valid when no tool parser owns it."""
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
        parser.tool_parser = None

        text = '<tool_call>{"name": "get_weather"}</tool_call>'
        assert parser.validate_complete(text) is True
    finally:
        cls.reasoning_parser_cls = old_reasoning_cls
        cls.tool_parser_cls = old_tool_cls


def test_validate_complete_allows_llama3_tool_open_tag_without_close_tag():
    """Llama3 complete validation is skipped because it has no close tag."""
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest
    from lmdeploy.serve.parsers import ResponseParserManager
    from lmdeploy.serve.parsers.tool_parser import ToolParserManager

    cls = ResponseParserManager.get('default')
    old_reasoning_cls = cls.reasoning_parser_cls
    old_tool_cls = cls.tool_parser_cls
    try:
        cls.reasoning_parser_cls = None
        cls.tool_parser_cls = ToolParserManager.get('llama3')
        request = ChatCompletionRequest(
            model='test',
            messages=[],
            tool_choice='auto',
        )
        parser = cls(request=request, tokenizer=object())

        text = '<|python_tag|>{"name": "get_weather", "arguments": {"city": "NYC"}}'
        assert parser.validate_complete(text) is True
    finally:
        cls.reasoning_parser_cls = old_reasoning_cls
        cls.tool_parser_cls = old_tool_cls


def test_validate_complete_is_public_and_does_not_require_token_ids():
    """Complete validation should use final text only, not token ids."""
    from inspect import signature

    from lmdeploy.serve.parsers import ResponseParserManager
    from lmdeploy.serve.parsers.response_parser import ResponseParser

    cls = ResponseParserManager.get('default')

    assert hasattr(ResponseParser, 'validate_complete')
    assert hasattr(cls, 'validate_complete')
    assert 'token_ids' not in signature(ResponseParser.validate_complete).parameters
    assert 'token_ids' not in signature(cls.validate_complete).parameters


def test_stream_chunk_does_not_expose_last_chunk_in_parser_contract():
    """stream_chunk keeps streaming parse output independent of final
    validation."""
    from inspect import signature

    from lmdeploy.serve.parsers import ResponseParserManager
    from lmdeploy.serve.parsers.response_parser import ResponseParser

    cls = ResponseParserManager.get('default')

    assert 'last_chunk' not in signature(ResponseParser.stream_chunk).parameters
    assert 'last_chunk' not in signature(cls.stream_chunk).parameters


def test_response_parser_does_not_own_finish_reason_state():
    """Finish reason reporting is owned by API server, not parser state."""
    from inspect import signature

    from lmdeploy.serve.openai.protocol import ChatCompletionRequest
    from lmdeploy.serve.parsers import ResponseParserManager
    from lmdeploy.serve.parsers.response_parser import ResponseParser

    cls = ResponseParserManager.get('default')

    assert 'finish_reason' not in signature(ResponseParser.stream_chunk).parameters
    assert 'finish_reason' not in signature(ResponseParser.parse_complete).parameters
    assert 'finish_reason' not in signature(cls.stream_chunk).parameters
    assert 'finish_reason' not in signature(cls.parse_complete).parameters
    assert not hasattr(ResponseParser, '_set_finish_reason')
    assert not hasattr(ResponseParser, '_validate_complete')
    assert not hasattr(cls, '_has_complete_parse_error')

    old_reasoning_cls = cls.reasoning_parser_cls
    old_tool_cls = cls.tool_parser_cls
    try:
        cls.reasoning_parser_cls = None
        cls.tool_parser_cls = None
        parser = cls(
            request=ChatCompletionRequest(model='test', messages=[]),
            tokenizer=object(),
        )
        assert not hasattr(parser, 'finish_reason')
    finally:
        cls.reasoning_parser_cls = old_reasoning_cls
        cls.tool_parser_cls = old_tool_cls


def test_stream_chunk_uses_accumulated_text_for_validation_only():
    """stream_chunk returns deltas; validate_complete reads accumulated
    text."""
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest
    from lmdeploy.serve.parsers import ResponseParserManager

    cls = ResponseParserManager.get('default')
    old_reasoning_cls = cls.reasoning_parser_cls
    old_tool_cls = cls.tool_parser_cls
    try:
        cls.reasoning_parser_cls = None
        cls.tool_parser_cls = None
        parser = cls(
            request=ChatCompletionRequest(model='test', messages=[]),
            tokenizer=object(),
        )

        deltas = parser.stream_chunk('hello ', [])
        assert len(deltas) == 1
        delta_message, tool_emitted = deltas[0]
        assert delta_message.content == 'hello '
        assert tool_emitted is False

        deltas = parser.stream_chunk('world', [])
        assert len(deltas) == 1
        delta_message, tool_emitted = deltas[0]
        assert delta_message.content == 'world'
        assert tool_emitted is False
        assert parser.validate_complete() is True
    finally:
        cls.reasoning_parser_cls = old_reasoning_cls
        cls.tool_parser_cls = old_tool_cls
