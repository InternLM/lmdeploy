import json

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import Glm47ToolParser, ToolParserManager

MODEL_ID = 'zai-org/GLM-4.7'


class _ReasoningTokenizerStub:

    def get_vocab(self):
        return {
            '<think>': 1,
            '</think>': 2,
        }


@pytest.fixture(scope='module')
def tokenizer():
    return _ReasoningTokenizerStub()


@pytest.fixture()
def response_parser(tokenizer):
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = None
    cls.tool_parser_cls = ToolParserManager.get('glm47')
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=True,
        tool_choice='auto',
    )
    return cls(request=request, tokenizer=tokenizer)


@pytest.fixture()
def response_parser_with_reasoning(tokenizer):
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('default')
    cls.tool_parser_cls = ToolParserManager.get('glm47')
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=True,
        tool_choice='auto',
        chat_template_kwargs={'enable_thinking': True},
    )
    return cls(request=request, tokenizer=tokenizer)


REFERENCE_CHUNKS = [
    # (delta_text, emitted_delta_msg, content, tool_emitted, function_name, function_arguments, tool_call_type)
    ('prefix ', True, 'prefix ', False, None, None, None),
    ('<tool_', False, None, False, None, None, None),
    ('call>', False, None, False, None, None, None),
    ('get_weather', True, None, True, 'get_weather', None, 'function'),
    ('<arg_key>location</arg_key>', False, None, False, None, None, None),
    ('<arg_value>Beijing</arg_value>', False, None, False, None, None, None),
    ('</tool_call>', True, None, True, None, '{"location": "Beijing"}', None),
]


class TestGlm47ResponseParserStreaming:
    """Integration tests for ResponseParser.stream_chunk with glm47 tool
    parser."""

    def test_stream_chunk_matches_reference(self, response_parser):
        for (delta_text, exp_delta_msg, exp_content, exp_tool_emitted,
             exp_function_name, exp_function_arguments, exp_type) in REFERENCE_CHUNKS:
            delta_msg, tool_emitted = response_parser.stream_chunk(delta_text=delta_text, delta_token_ids=[])
            if not exp_delta_msg:
                assert delta_msg is None
                continue
            assert delta_msg is not None
            assert delta_msg.content == exp_content
            assert tool_emitted == exp_tool_emitted
            if tool_emitted:
                assert delta_msg.tool_calls is not None
                assert len(delta_msg.tool_calls) == 1
                call = delta_msg.tool_calls[0]
                assert call.type == exp_type
                assert call.function is not None
                assert call.function.name == exp_function_name
                assert call.function.arguments == exp_function_arguments

    def test_stream_chunk_handles_split_open_tag_and_zero_args(self, response_parser):
        chunks = ['prefix ', '<tool_', 'call>', 'get_weather', '</tool_call>']
        seen_name = False
        seen_args = False
        leaked_tag_text = []
        for chunk in chunks:
            delta, tool_emitted = response_parser.stream_chunk(delta_text=chunk, delta_token_ids=[])
            if delta is None:
                continue
            if delta.content:
                leaked_tag_text.append(delta.content)
            if tool_emitted and delta.tool_calls:
                for call in delta.tool_calls:
                    if call.function and call.function.name == 'get_weather':
                        seen_name = True
                    if call.function and call.function.arguments == '{}':
                        seen_args = True
        assert seen_name
        assert seen_args
        assert '<tool_call>' not in ''.join(leaked_tag_text)

    def test_stream_chunk_mixed_default_reasoning_and_glm47_tool(self, response_parser_with_reasoning):
        chunks = [
            '<think>',
            'first reason',
            '</think>\nAnswer: ',
            '<tool_call>get_weather',
            '<arg_key>location</arg_key><arg_value>Beijing</arg_value>',
            '</tool_call>',
        ]
        reasoning_seen = []
        content_seen = []
        emitted_name = None
        emitted_args = None

        for chunk in chunks:
            delta, tool_emitted = response_parser_with_reasoning.stream_chunk(delta_text=chunk, delta_token_ids=[])
            if delta is not None:
                if delta.reasoning_content:
                    reasoning_seen.append(delta.reasoning_content)
                if delta.content:
                    content_seen.append(delta.content)
            if tool_emitted and delta and delta.tool_calls:
                for call in delta.tool_calls:
                    if call.function and call.function.name:
                        emitted_name = call.function.name
                    if call.function and call.function.arguments:
                        emitted_args = call.function.arguments

        for _ in range(3):
            delta, tool_emitted = response_parser_with_reasoning.stream_chunk(delta_text='', delta_token_ids=[])
            if delta is not None:
                if delta.reasoning_content:
                    reasoning_seen.append(delta.reasoning_content)
                if delta.content:
                    content_seen.append(delta.content)
            if tool_emitted and delta and delta.tool_calls:
                for call in delta.tool_calls:
                    if call.function and call.function.name:
                        emitted_name = call.function.name
                    if call.function and call.function.arguments:
                        emitted_args = call.function.arguments

        assert ''.join(reasoning_seen) == 'first reason'
        assert ''.join(content_seen) == '\nAnswer: '
        assert emitted_name == 'get_weather'
        assert emitted_args == '{"location": "Beijing"}'

    def test_stream_chunk_keeps_string_without_schema(self, response_parser):
        chunks = [
            '<tool_call>',
            'no_schema_tool',
            '<arg_key>zip</arg_key><arg_value>77004</arg_value>',
            '<arg_key>active</arg_key><arg_value>true</arg_value>',
            '</tool_call>',
        ]
        emitted_name = None
        emitted_args = None
        for chunk in chunks:
            delta, tool_emitted = response_parser.stream_chunk(delta_text=chunk, delta_token_ids=[])
            if not tool_emitted or delta is None or not delta.tool_calls:
                continue
            for call in delta.tool_calls:
                if call.function and call.function.name:
                    emitted_name = call.function.name
                if call.function and call.function.arguments:
                    emitted_args = call.function.arguments
        assert emitted_name == 'no_schema_tool'
        assert emitted_args == '{"zip": "77004", "active": "true"}'


class TestGlm47ToolParserComplete:
    """Complete-parse tests for glm47 tool payloads."""

    def test_parse_tool_call_complete_with_arguments(self):
        parser = Glm47ToolParser(tokenizer=object())
        payload = (
            'get_weather'
            '<arg_key>location</arg_key><arg_value>Beijing</arg_value>'
            '<arg_key>unit</arg_key><arg_value>celsius</arg_value>'
        )
        tool_call = parser.parse_tool_call_complete(payload)
        assert tool_call is not None
        assert tool_call.function.name == 'get_weather'
        assert json.loads(tool_call.function.arguments) == {
            'location': 'Beijing',
            'unit': 'celsius',
        }

    def test_parse_tool_call_complete_without_arguments(self):
        parser = Glm47ToolParser(tokenizer=object())
        tool_call = parser.parse_tool_call_complete('get_time')
        assert tool_call is not None
        assert tool_call.function.name == 'get_time'
        assert json.loads(tool_call.function.arguments) == {}

    def test_parse_tool_call_complete_coerces_types_by_schema(self):
        parser = Glm47ToolParser(tokenizer=object())
        request = ChatCompletionRequest(
            model=MODEL_ID,
            messages=[],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'typed_tool',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'name': {
                                'type': 'string'
                            },
                            'age': {
                                'type': 'integer'
                            },
                            'height': {
                                'type': 'number'
                            },
                            'active': {
                                'type': 'boolean'
                            },
                            'meta': {
                                'type': 'object'
                            },
                            'scores': {
                                'type': 'array'
                            },
                            'misc': {
                                'type': 'null'
                            },
                        },
                    },
                },
            }],
            tool_choice='auto',
        )
        parser.adjust_request(request)
        payload = (
            'typed_tool'
            '<arg_key>name</arg_key><arg_value>Chen</arg_value>'
            '<arg_key>age</arg_key><arg_value>29</arg_value>'
            '<arg_key>height</arg_key><arg_value>1.75</arg_value>'
            '<arg_key>active</arg_key><arg_value>true</arg_value>'
            '<arg_key>meta</arg_key><arg_value>{"city":"Houston"}</arg_value>'
            '<arg_key>scores</arg_key><arg_value>[98,87]</arg_value>'
            '<arg_key>misc</arg_key><arg_value>null</arg_value>'
        )
        tool_call = parser.parse_tool_call_complete(payload)
        assert tool_call is not None
        assert tool_call.function.name == 'typed_tool'
        assert json.loads(tool_call.function.arguments) == {
            'name': 'Chen',
            'age': 29,
            'height': 1.75,
            'active': True,
            'meta': {
                'city': 'Houston'
            },
            'scores': [98, 87],
            'misc': None,
        }

    def test_parse_tool_call_complete_keeps_string_without_schema(self):
        parser = Glm47ToolParser(tokenizer=object())
        payload = (
            'no_schema_tool'
            '<arg_key>zip</arg_key><arg_value>77004</arg_value>'
            '<arg_key>active</arg_key><arg_value>true</arg_value>'
            '<arg_key>meta</arg_key><arg_value>{"city":"Houston"}</arg_value>'
        )
        tool_call = parser.parse_tool_call_complete(payload)
        assert tool_call is not None
        assert tool_call.function.name == 'no_schema_tool'
        assert json.loads(tool_call.function.arguments) == {
            'zip': '77004',
            'active': 'true',
            'meta': '{"city":"Houston"}',
        }
