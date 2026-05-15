from dataclasses import dataclass

import pytest

pytest.importorskip('openai_harmony')

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import _openai_harmony as openai_harmony_mod
from lmdeploy.serve.parsers import gpt_oss_response_parser as gpt_oss_mod


@dataclass
class _FakeMsg:
    channel: str
    recipient: str | None


class _FakeStreamableParser:
    """Scripted stand-in for openai_harmony.StreamableParser."""

    def __init__(self, script: dict[int, dict]):
        self._script = script
        self.current_channel = 'final'
        self.current_recipient = None
        self.last_content_delta = ''
        self.messages: list[_FakeMsg] = []

    def process(self, token: int):
        event = self._script[token]
        next_channel = event['channel']
        next_recipient = event.get('recipient')

        if (self.current_channel == 'commentary' and self.current_recipient
                and self.current_recipient.startswith('functions.') and next_recipient != self.current_recipient):
            self.messages.append(_FakeMsg(channel='commentary', recipient=self.current_recipient))

        self.current_channel = next_channel
        self.current_recipient = next_recipient
        self.last_content_delta = event.get('delta', '')


def _scripted_events() -> dict[int, dict]:
    return {
        1: {
            'channel': 'analysis',
            'recipient': None,
            'delta': 'Need tool. ',
        },
        2: {
            'channel': 'commentary',
            'recipient': 'functions.get_weather',
            'delta': '',
        },
        3: {
            'channel': 'commentary',
            'recipient': 'functions.get_weather',
            'delta': '{"location":"',
        },
        4: {
            'channel': 'commentary',
            'recipient': 'functions.get_weather',
            'delta': 'Beijing"}',
        },
        5: {
            'channel': 'commentary',
            'recipient': 'functions.get_time',
            'delta': '',
        },
        6: {
            'channel': 'commentary',
            'recipient': 'functions.get_time<|channel|>commentary',
            'delta': '{"tz":"UTC"}',
        },
        7: {
            'channel': 'final',
            'recipient': None,
            'delta': 'Result: ',
        },
        8: {
            'channel': 'final',
            'recipient': None,
            'delta': 'sunny',
        },
    }


class TestGptOssResponseParser:
    """Unit tests for :class:`GptOssResponseParser` (Harmony token
    streaming)."""

    def test_stream_chunk_full_sequence(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser(_scripted_events()),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object(), tokenizer=object())

        delta, tool_emitted = parser.stream_chunk(delta_text='ignored', delta_token_ids=[1, 2, 3, 4, 5, 6, 7, 8])
        assert delta is not None
        assert delta.content == 'Result: sunny'
        assert delta.reasoning_content == 'Need tool. '
        assert tool_emitted is True
        assert delta.tool_calls is not None
        assert len(delta.tool_calls) == 5

        # name delta + args delta for get_weather
        assert delta.tool_calls[0].function is not None
        assert delta.tool_calls[0].function.name == 'get_weather'
        assert delta.tool_calls[1].function is not None
        assert delta.tool_calls[1].function.arguments == '{"location":"'
        assert delta.tool_calls[2].function is not None
        assert delta.tool_calls[2].function.arguments == 'Beijing"}'

        # second tool: name delta + sanitized malformed recipient arguments delta.
        assert delta.tool_calls[3].function is not None
        assert delta.tool_calls[3].function.name == 'get_time'
        assert delta.tool_calls[4].function is not None
        assert delta.tool_calls[4].function.arguments == '{"tz":"UTC"}'

    def test_adjust_request_converts_tools_to_wrapper_dicts(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[],
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'city': {
                                    'type': 'string'
                                }
                            }
                        },
                    },
                },
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_time',
                        'parameters': {
                            'type': 'object',
                            'properties': {
                                'tz': {
                                    'type': 'string'
                                }
                            }
                        },
                    },
                },
            ],
            tool_choice={
                'type': 'function',
                'function': {
                    'name': 'get_time'
                },
            },
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request, tokenizer=object())

        assert parser.request.tools == [{
            'type': 'function',
            'function': {
                'name': 'get_time',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'tz': {
                            'type': 'string'
                        }
                    },
                },
                'description': None,
            },
        }]

    def test_parse_complete_full_sequence(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser(_scripted_events()),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object(), tokenizer=object())

        content, tool_calls, reasoning = parser.parse_complete(text='', token_ids=[1, 2, 3, 4, 5, 6, 7, 8])
        assert content == 'Result: sunny'
        assert reasoning == 'Need tool. '
        assert tool_calls is not None
        assert [call.function.name for call in tool_calls] == ['get_weather', 'get_time']
        assert [call.function.arguments for call in tool_calls] == ['{"location":"Beijing"}', '{"tz":"UTC"}']

    def test_stream_chunk_bootstrap_empty_before_any_content(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object(), tokenizer=object())

        delta, tool_emitted = parser.stream_chunk('', [])
        assert delta is not None
        assert delta.role == 'assistant'
        assert delta.content == ''
        assert tool_emitted is False

    def test_stream_chunk_empty_after_content_started_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object(), tokenizer=object())

        parser.stream_chunk('warmup', [])
        delta, tool_emitted = parser.stream_chunk('', [])
        assert delta is None
        assert tool_emitted is False

    def test_stream_chunk_text_only_without_token_ids(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object(), tokenizer=object())

        delta, tool_emitted = parser.stream_chunk('plain text', [])
        assert delta is not None
        assert delta.content == 'plain text'
        assert delta.reasoning_content is None
        assert delta.tool_calls is None
        assert tool_emitted is False

    def test_stream_chunk_token_ids_all_empty_delta_returns_none(self, monkeypatch):
        script = {
            10: {'channel': 'final', 'recipient': None, 'delta': ''},
        }
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser(script),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object(), tokenizer=object())

        delta, tool_emitted = parser.stream_chunk('', [10])
        assert delta is None
        assert tool_emitted is False

    def test_stream_chunk_analysis_without_tool_accumulates_reasoning(self, monkeypatch):
        script = {
            1: {'channel': 'analysis', 'recipient': None, 'delta': 'think '},
            2: {'channel': 'analysis', 'recipient': None, 'delta': 'more'},
        }
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser(script),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object(), tokenizer=object())

        delta, tool_emitted = parser.stream_chunk('', [1, 2])
        assert delta is not None
        assert delta.content is None
        assert delta.reasoning_content == 'think more'
        assert delta.tool_calls is None
        assert tool_emitted is False

    def test_parse_complete_without_token_ids_returns_raw_text(self):
        parser = gpt_oss_mod.GptOssResponseParser(request=object(), tokenizer=object())

        content, tool_calls, reasoning = parser.parse_complete('hello', token_ids=[])
        assert content == 'hello'
        assert tool_calls is None
        assert reasoning is None

    def test_parse_complete_without_token_ids_empty_text(self):
        parser = gpt_oss_mod.GptOssResponseParser(request=object(), tokenizer=object())

        content, tool_calls, reasoning = parser.parse_complete('', token_ids=None)
        assert content is None
        assert tool_calls is None
        assert reasoning is None

    def test_parse_complete_appends_tool_call_still_open_at_eof(self, monkeypatch):
        """Final `active` tool dict is appended when the stream ends in a tool
        channel."""
        script = {
            1: {
                'channel': 'commentary',
                'recipient': 'functions.echo',
                'delta': '{"x":1}',
            },
        }
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser(script),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object(), tokenizer=object())

        content, tool_calls, reasoning = parser.parse_complete(text='', token_ids=[1])
        assert content is None
        assert reasoning is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == 'echo'
        assert tool_calls[0].function.arguments == '{"x":1}'

    @pytest.mark.parametrize(
        ('recipient', 'expected'),
        [
            (None, None),
            ('', None),
            ('not-a-tool', None),
            ('functions.', None),
            ('functions.foo', 'foo'),
            ('prefix functions.bar suffix', 'bar'),
            ('functions.bash<|channel|>commentary', 'bash'),
            ('functions.tool_name<|extra|', 'tool_name'),
        ],
    )
    def test_extract_tool_name(self, recipient, expected):
        assert gpt_oss_mod.GptOssResponseParser._extract_tool_name(recipient) == expected


class TestGptOssResponseFormatHarmonyConversion:
    """Tests for
    :meth:`GptOssResponseParser._convert_response_format_to_harmony`."""

    @pytest.fixture(autouse=True)
    def _patch_streamable_parser(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser({}),
        )

    def test_response_format_cleared_after_conversion(self):
        """response_format must be None after the parser processes it."""
        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(
                    name='test',
                    schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}},
                ),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request, tokenizer=object())
        assert parser.request.response_format is None

    def test_schema_appended_to_existing_system_message(self):
        """When a system message already exists the schema is appended to
        it."""
        import json as _json

        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        schema_dict = {'type': 'object', 'properties': {'x': {'type': 'integer'}}}
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[
                {'role': 'system', 'content': 'You are helpful.'},
                {'role': 'user', 'content': 'hi'},
            ],
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(name='test', schema=schema_dict),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request, tokenizer=object())

        msgs = parser.request.messages
        assert msgs[0]['role'] == 'system'
        assert parser.request.response_format is None
        # The schema body must appear in the system message
        assert '# Response Formats' in msgs[0]['content']
        assert _json.dumps(schema_dict) in msgs[0]['content']
        # The original content is preserved before the appended section
        assert msgs[0]['content'].startswith('You are helpful.')
        # No leading blank lines in the appended section
        assert '\n\n# Response Formats' in msgs[0]['content']

    def test_schema_inserted_as_new_system_message_when_none_exists(self):
        """When no system message exists a new one is inserted at position
        0."""
        import json as _json

        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        schema_dict = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(name='test', schema=schema_dict),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request, tokenizer=object())

        msgs = parser.request.messages
        assert msgs[0]['role'] == 'system'
        assert parser.request.response_format is None
        # New system message content must NOT start with blank lines
        assert not msgs[0]['content'].startswith('\n')
        assert msgs[0]['content'].startswith('# Response Formats')
        assert _json.dumps(schema_dict) in msgs[0]['content']
        # The user message is still present after the inserted system message
        assert msgs[1]['role'] == 'user'

    def test_text_response_format_is_cleared_by_normalize(self):
        from lmdeploy.serve.openai.protocol import ResponseFormat

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(type='text'),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request, tokenizer=object())
        assert parser.request.response_format is None

    def test_no_response_format_leaves_request_unchanged(self):
        """When response_format is None the request is not modified."""
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request, tokenizer=object())
        assert parser.request.response_format is None
        assert len(parser.request.messages) == 1

    def test_str_messages_gets_schema_appended(self):
        """When messages is a string, the schema section is appended to it."""
        import json as _json

        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        schema_dict = {'type': 'object', 'properties': {'x': {'type': 'integer'}}}
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages='Tell me a joke',
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(name='test', schema=schema_dict),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request, tokenizer=object())

        assert parser.request.response_format is None
        assert isinstance(parser.request.messages, str)
        assert parser.request.messages.startswith('Tell me a joke')
        assert '# Response Formats' in parser.request.messages
        assert _json.dumps(schema_dict) in parser.request.messages

    def test_non_pydantic_request_messages_updated(self):
        """Non-Pydantic sentinel requests also get messages updated."""
        import json as _json

        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        schema_dict = {'type': 'object', 'properties': {'y': {'type': 'number'}}}
        fmt = ResponseFormat(
            type='json_schema',
            json_schema=JsonSchema(name='test', schema=schema_dict),
        )

        # Sentinel must NOT have tools/tool_choice attrs so that __init__
        # skips the Pydantic-dependent tool-rendering branch.
        class _Sentinel:
            messages = [{'role': 'user', 'content': 'hi'}]
            response_format = fmt

        sentinel = _Sentinel()
        parser = gpt_oss_mod.GptOssResponseParser(request=sentinel, tokenizer=object())

        assert parser.request.response_format is None
        msgs = parser.request.messages
        assert isinstance(msgs, list)
        assert msgs[0]['role'] == 'system'
        assert '# Response Formats' in msgs[0]['content']
        assert _json.dumps(schema_dict) in msgs[0]['content']

    def test_list_content_system_message_gets_text_block_appended(self):
        """When system message content is a list (multimodal), append a text
        block."""
        import json as _json

        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        schema_dict = {'type': 'object', 'properties': {'z': {'type': 'boolean'}}}
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[
                {'role': 'system', 'content': [
                    {'type': 'text', 'text': 'You are helpful.'},
                    {'type': 'image_url', 'image_url': {'url': 'http://example.com/img.png'}},
                ]},
                {'role': 'user', 'content': 'hi'},
            ],
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(name='test', schema=schema_dict),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request, tokenizer=object())

        assert parser.request.response_format is None
        sys_msg = parser.request.messages[0]
        assert sys_msg['role'] == 'system'
        content = sys_msg['content']
        assert isinstance(content, list)
        assert len(content) == 3
        # Original two blocks preserved
        assert content[0]['type'] == 'text'
        assert content[0]['text'] == 'You are helpful.'
        assert content[1]['type'] == 'image_url'
        # Schema appended as a text block
        assert content[2]['type'] == 'text'
        assert '# Response Formats' in content[2]['text']
        assert _json.dumps(schema_dict) in content[2]['text']

    def test_none_content_system_message_inserts_separate_system(self):
        """When system message content is None, insert a new system message."""
        import json as _json

        from lmdeploy.serve.openai.protocol import JsonSchema, ResponseFormat

        schema_dict = {'type': 'object', 'properties': {'w': {'type': 'string'}}}
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[
                {'role': 'system', 'content': None},
                {'role': 'user', 'content': 'hi'},
            ],
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(name='test', schema=schema_dict),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request, tokenizer=object())

        assert parser.request.response_format is None
        msgs = parser.request.messages
        # A new system message with the schema is inserted at position 0
        assert msgs[0]['role'] == 'system'
        assert '# Response Formats' in msgs[0]['content']
        assert _json.dumps(schema_dict) in msgs[0]['content']
