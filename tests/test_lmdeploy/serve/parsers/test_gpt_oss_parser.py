from dataclasses import dataclass

import pytest

pytest.importorskip('openai_harmony')

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
