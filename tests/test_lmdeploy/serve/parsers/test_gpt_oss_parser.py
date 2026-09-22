import json

import pytest

pytest.importorskip('openai_harmony')

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, JsonSchema, ResponseFormat
from lmdeploy.serve.parsers import _openai_harmony as openai_harmony_mod
from lmdeploy.serve.parsers import gpt_oss_response_parser as gpt_oss_mod

from .helpers import first_stream_delta


class _FakeStreamableParser:
    """Scripted stand-in for openai_harmony.StreamableParser."""

    def __init__(self, script: dict[int, dict]):
        self._script = script
        self.current_channel = 'final'
        self.current_recipient = None
        self.last_content_delta = ''

    def process(self, token: int):
        event = self._script[token]
        self.current_channel = event['channel']
        self.current_recipient = event.get('recipient')
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


@pytest.fixture(autouse=True)
def _patch_streamable_parser(monkeypatch):
    """Mock ``get_encoding`` and ``StreamableParser`` so tests don't need the
    real Harmony vocab."""
    monkeypatch.setattr(openai_harmony_mod, 'get_encoding', lambda: None)
    monkeypatch.setattr(
        openai_harmony_mod,
        'StreamableParser',
        lambda *args, **kwargs: _FakeStreamableParser({}),
    )


class TestGptOssResponseParser:
    """Unit tests for :class:`GptOssResponseParser` (Harmony token
    streaming)."""

    def test_stream_chunk_full_sequence(self, monkeypatch):
        monkeypatch.setattr(
            openai_harmony_mod,
            'StreamableParser',
            lambda *args, **kwargs: _FakeStreamableParser(_scripted_events()),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        delta, tool_emitted = first_stream_delta(parser.stream_chunk(delta_text='ignored',
                                                                     delta_token_ids=[1, 2, 3, 4, 5, 6, 7, 8]))
        assert delta is not None
        assert delta.content == 'Result: sunny'
        assert delta.reasoning_content == 'Need tool. '
        assert parser.reasoning_tokens == 1
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

    def test_adjust_request_converts_tools_to_wrapper_dicts(self):
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
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

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
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        content, tool_calls, reasoning = parser.parse_complete(text='', token_ids=[1, 2, 3, 4, 5, 6, 7, 8])
        assert content == 'Result: sunny'
        assert reasoning == 'Need tool. '
        assert parser.reasoning_tokens == 1
        assert tool_calls is not None
        assert [call.function.name for call in tool_calls] == ['get_weather', 'get_time']
        assert [call.function.arguments for call in tool_calls] == ['{"location":"Beijing"}', '{"tz":"UTC"}']

    @pytest.mark.parametrize('delta_text', ['', 'plain text'])
    def test_stream_chunk_text_only(self, delta_text):
        """First call with no token_ids and empty script: bootstrap / plain text
        passthrough."""
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        delta, tool_emitted = first_stream_delta(parser.stream_chunk(delta_text, []))
        assert delta is not None
        assert delta.role == 'assistant'
        assert delta.content == delta_text
        assert delta.reasoning_content is None
        assert delta.tool_calls is None
        assert tool_emitted is False

    def test_stream_chunk_empty_after_content_started_returns_none(self):
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        parser.stream_chunk('warmup', [])
        delta, tool_emitted = first_stream_delta(parser.stream_chunk('', []))
        assert delta is None
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
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        delta, tool_emitted = first_stream_delta(parser.stream_chunk('', [10]))
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
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

        delta, tool_emitted = first_stream_delta(parser.stream_chunk('', [1, 2]))
        assert delta is not None
        assert delta.content is None
        assert delta.reasoning_content == 'think more'
        assert parser.reasoning_tokens == 2
        assert delta.tool_calls is None
        assert tool_emitted is False


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
        parser = gpt_oss_mod.GptOssResponseParser(request=object())

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


class TestGptOssResponseFormatGrammarConversion:
    """Tests for GptOssResponseParser response_format → structural_tag
    conversion (replaces the old Harmony-native prompt injection)."""

    @pytest.mark.parametrize('schema_dict', [
        {'type': 'object', 'properties': {'x': {'type': 'integer'}}},
        None,
    ])
    def test_json_schema_converted_to_structural_tag(self, schema_dict):
        """json_schema is converted to a structural_tag wrapping the schema in
        the Harmony final channel; messages are not modified.

        When no inner schema is provided it defaults to
        ``{'type': 'object'}`` instead of leaking the deprecated
        ``BaseModel.schema`` method.
        """
        json_schema = JsonSchema(name='test', schema=schema_dict)
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(type='json_schema', json_schema=json_schema),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        assert rf['structural_tag'] is not None
        st_json = json.dumps(rf['structural_tag'])
        # Harmony channel markers must be present
        assert '<|channel|>final<|message|>' in st_json
        assert '<|end|>' in st_json
        assert '<|channel|>analysis<|message|>' in st_json
        # Messages must NOT be modified (no prompt injection)
        assert len(parser.request.messages) == 1
        assert parser.request.messages[0]['role'] == 'user'
        if schema_dict is not None:
            assert json.dumps(schema_dict) in st_json
        else:
            assert 'bound method' not in st_json

    @pytest.mark.parametrize('fmt_type,kwargs', [
        ('regex_schema', {'regex_schema': '[0-9]+'}),
        ('json_object', {}),
    ])
    def test_simple_format_converted_to_structural_tag(self, fmt_type, kwargs):
        """regex_schema and json_object are converted to a structural_tag."""
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(type=fmt_type, **kwargs),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        assert rf['structural_tag'] is not None


    def test_grammar_failure_falls_back_to_prompt_injection(self, monkeypatch):
        """When xgrammar is unavailable, response_format is injected into the
        system prompt and cleared (legacy Harmony-native fallback)."""
        monkeypatch.setattr(
            gpt_oss_mod.GptOssResponseParser,
            '_build_response_format_grammar',
            staticmethod(lambda fmt: None),
        )

        schema_dict = {'type': 'object', 'properties': {'x': {'type': 'integer'}}}
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(name='test', schema=schema_dict),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        assert parser.request.response_format is None
        msgs = parser.request.messages
        assert msgs[0]['role'] == 'system'
        assert '# Response Formats' in msgs[0]['content']
        assert json.dumps(schema_dict) in msgs[0]['content']


class TestGptOssToolGrammarInjection:
    """Tests for GptOssResponseParser tool-calling structural_tag injection."""

    @pytest.mark.parametrize('tool_choice,extra_markers', [
        ('required', ['<|call|>', '<|constrain|>json']),
        ('auto', ['<|channel|>final']),
    ])
    def test_tool_choice_injects_structural_tag(self, tool_choice, extra_markers):
        """Required/auto tool_choice inject a structural_tag with the tool call
        grammar."""
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'What is the weather?'}],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'parameters': {
                        'type': 'object',
                        'properties': {'location': {'type': 'string'}},
                    },
                },
            }],
            tool_choice=tool_choice,
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        assert rf['structural_tag'] is not None
        st_json = json.dumps(rf['structural_tag'])
        assert 'functions.get_weather' in st_json
        for marker in extra_markers:
            assert marker in st_json

    def test_specific_tool_choice_injects_structural_tag(self):
        """tool_choice={"type":"function","function":{"name":"X"}} injects
        grammar for only that function."""
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'What is the weather?'}],
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                        'parameters': {'type': 'object', 'properties': {}},
                    },
                },
                {
                    'type': 'function',
                    'function': {
                        'name': 'get_time',
                        'parameters': {'type': 'object', 'properties': {}},
                    },
                },
            ],
            tool_choice={
                'type': 'function',
                'function': {'name': 'get_weather'},
            },
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        st_json = json.dumps(rf['structural_tag'])
        assert 'functions.get_weather' in st_json
        # The non-selected tool should NOT appear
        assert 'functions.get_time' not in st_json

    def test_none_tool_choice_does_not_inject_grammar(self):
        """tool_choice=none does not inject tool grammar."""
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'parameters': {'type': 'object', 'properties': {}},
                },
            }],
            tool_choice='none',
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is None or rf['type'] != 'structural_tag'

    def test_allowed_tool_choice_keeps_all_tools(self):
        """AllowedToolChoice (type='allowed_tools') must not crash and keeps
        all tools instead of filtering to a single function."""
        from lmdeploy.serve.openai.protocol import AllowedToolChoice

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'parameters': {'type': 'object', 'properties': {}},
                },
            }],
            tool_choice=AllowedToolChoice(allowed_tools={'mode': 'auto', 'tools': []}),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        assert parser.request.tools is not None
        assert len(parser.request.tools) == 1

    def test_tools_priority_over_response_format(self):
        """When both tools and response_format are present, tool grammar takes
        priority."""
        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'hi'}],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'parameters': {'type': 'object', 'properties': {}},
                },
            }],
            tool_choice='required',
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(
                    name='test',
                    schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}},
                ),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        rf = parser.request.response_format
        assert rf is not None
        assert rf['type'] == 'structural_tag'
        st_json = json.dumps(rf['structural_tag'])
        # Tool grammar wins — must contain tool call, not plain json_schema
        assert 'functions.get_weather' in st_json

    def test_tool_grammar_failure_clears_response_format(self, monkeypatch):
        """Tool grammar failure must clear a non-text response_format so it
        cannot conflict with Harmony tool-call constraints downstream.

        The monkeypatch is required because ``__init__`` filters tools to the
        selected function before grammar construction, so no real input can
        keep ``has_tools`` true while making ``_build_tool_grammar`` fail.
        """
        monkeypatch.setattr(
            gpt_oss_mod.GptOssResponseParser,
            '_build_tool_grammar',
            staticmethod(lambda tools, tool_choice: None),
        )

        request = ChatCompletionRequest(
            model='openai/gpt-oss-20b',
            messages=[{'role': 'user', 'content': 'What is the weather?'}],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'get_weather',
                    'parameters': {'type': 'object', 'properties': {}},
                },
            }],
            tool_choice='required',
            response_format=ResponseFormat(
                type='json_schema',
                json_schema=JsonSchema(name='test', schema={'type': 'object'}),
            ),
        )
        parser = gpt_oss_mod.GptOssResponseParser(request=request)

        assert parser.request.response_format is None
