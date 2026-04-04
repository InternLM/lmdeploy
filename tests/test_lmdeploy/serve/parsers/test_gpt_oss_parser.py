from dataclasses import dataclass

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


def test_gpt_oss_response_parser_stream_chunk(monkeypatch):
    monkeypatch.setattr(
        gpt_oss_mod,
        'get_streamable_parser_for_assistant',
        lambda: _FakeStreamableParser(_scripted_events()),
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


def test_gpt_oss_response_parser_parse_complete(monkeypatch):
    monkeypatch.setattr(
        gpt_oss_mod,
        'get_streamable_parser_for_assistant',
        lambda: _FakeStreamableParser(_scripted_events()),
    )
    parser = gpt_oss_mod.GptOssResponseParser(request=object(), tokenizer=object())

    content, tool_calls, reasoning = parser.parse_complete(text='', token_ids=[1, 2, 3, 4, 5, 6, 7, 8])
    assert content == 'Result: sunny'
    assert reasoning == 'Need tool. '
    assert tool_calls is not None
    assert [call.function.name for call in tool_calls] == ['get_weather', 'get_time']
    assert [call.function.arguments for call in tool_calls] == ['{"location":"Beijing"}', '{"tz":"UTC"}']
