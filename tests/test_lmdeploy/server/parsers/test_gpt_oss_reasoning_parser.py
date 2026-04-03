from dataclasses import dataclass

from lmdeploy.serve.openai.reasoning_parser import gpt_oss_reasoning_parser as gpt_oss_mod


@dataclass
class _FakeMsg:
    channel: str
    recipient: str | None


class _FakeStreamableParser:
    """A tiny scripted parser to emulate openai_harmony.StreamableParser."""

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

        # Mirror completed function-call message accounting used by the parser
        # to compute tool call index.
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
            'recipient': 'functions.get_time',
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


def test_gpt_oss_chat_parser_routes_channels(monkeypatch):
    monkeypatch.setattr(
        gpt_oss_mod,
        'get_streamable_parser_for_assistant',
        lambda: _FakeStreamableParser(_scripted_events()),
    )

    parser = gpt_oss_mod.GptOssChatParser()
    delta = parser.parse_streaming([1, 2, 3, 4, 5, 6, 7, 8])

    assert delta.content == 'Result: sunny'
    assert delta.reasoning_content == 'Need tool. '
    assert delta.tool_calls is not None
    assert len(delta.tool_calls) == 2

    first, second = delta.tool_calls
    assert first.function is not None
    assert first.function.name == 'get_weather'
    assert first.function.arguments == '{"location":"Beijing"}'
    assert first.index == 0

    assert second.function is not None
    assert second.function.name == 'get_time'
    assert second.function.arguments == '{"tz":"UTC"}'
    assert second.index == 1


def test_gpt_oss_reasoning_parser_parse_full(monkeypatch):
    monkeypatch.setattr(
        gpt_oss_mod,
        'get_streamable_parser_for_assistant',
        lambda: _FakeStreamableParser(_scripted_events()),
    )

    parser = gpt_oss_mod.GptOssReasoningParser(tokenizer=object())
    message = parser.parse_full([1, 2, 3, 4, 5, 6, 7, 8])

    assert message.content == 'Result: sunny'
    assert message.reasoning_content == 'Need tool. '
    assert message.tool_calls is not None
    assert [call.function.name for call in message.tool_calls] == ['get_weather', 'get_time']
    assert [call.function.arguments for call in message.tool_calls] == ['{"location":"Beijing"}', '{"tz":"UTC"}']


def test_gpt_oss_reasoning_parser_tags():
    parser = gpt_oss_mod.GptOssReasoningParser(tokenizer=object())
    assert parser.get_reasoning_open_tag() is None
    assert parser.get_reasoning_close_tag() is None
    assert parser.starts_in_reasoning_mode() is False
