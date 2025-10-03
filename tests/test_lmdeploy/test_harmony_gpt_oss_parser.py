import collections
import json
import os
import sys
import time
import types
from typing import Generator, List

import pytest
import shortuuid

# Ensure local package is imported (not any site-packages installation)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _install_openai_harmony_stub():
    """Install a minimal stub for `openai_harmony` so the module imports
    without the real dependency.

    The GptOssChatParser test injects its own dummy parser, so the stub is sufficient.
    """
    if 'openai_harmony' in sys.modules:
        return
    m = types.ModuleType('openai_harmony')

    class HarmonyEncodingName:
        HARMONY_GPT_OSS = 'HARMONY_GPT_OSS'

    class Role:
        ASSISTANT = 'assistant'

    class StreamableParser:  # pragma: no cover - constructor only used

        def __init__(self, encoding, role=None):
            self.encoding = encoding
            self.role = role

    def load_harmony_encoding(name):  # pragma: no cover - not used in test
        return object()

    m.HarmonyEncodingName = HarmonyEncodingName
    m.Role = Role
    m.StreamableParser = StreamableParser
    m.load_harmony_encoding = load_harmony_encoding
    sys.modules['openai_harmony'] = m


TestExpects = collections.namedtuple('TestExpects', 'func_name location')


class DummyParser:
    """A minimal stand-in for Harmony's StreamableParser with channels.

    Control tokens:
      -1: start functions.get_weather (commentary)
      -4: start functions.get_time (commentary)
      -6: start functions.get_weather (again)
      -9: end current tool call, append to `messages`
      -2: switch to final (visible) content
      -3: switch to analysis (reasoning)
    Other tokens are interpreted as chr(token).
    """

    class _Msg:

        def __init__(self, channel, recipient):
            self.channel = channel
            self.recipient = recipient

    def __init__(self):
        self.current_channel = None
        self.current_recipient = None
        self.last_content_delta = ''
        self.messages = []

    def process(self, token):
        if token == -1:
            self.current_channel = 'commentary'
            self.current_recipient = 'functions.get_weather'
            self.last_content_delta = ''
            return
        if token == -4:
            self.current_channel = 'commentary'
            self.current_recipient = 'functions.get_time'
            self.last_content_delta = ''
            return
        if token == -6:
            self.current_channel = 'commentary'
            self.current_recipient = 'functions.get_weather'
            self.last_content_delta = ''
            return
        if token == -9:
            if self.current_channel == 'commentary' and self.current_recipient and self.current_recipient.startswith(
                    'functions.'):
                self.messages.append(self._Msg(self.current_channel, self.current_recipient))
            # reset recipient to signal end of current tool call
            self.current_recipient = None
            self.current_channel = None
            self.last_content_delta = ''
            return
        if token == -2:
            self.current_channel = 'final'
            self.current_recipient = None
            self.last_content_delta = ''
            return
        if token == -3:
            self.current_channel = 'analysis'
            self.current_recipient = None
            self.last_content_delta = ''
            return
        # regular character token
        self.last_content_delta = chr(token)


def _chat_completion_v1(request, token_chunks: List[List[int]]):
    from lmdeploy.serve.openai.harmony_utils import GptOssChatParser
    from lmdeploy.serve.openai.protocol import (ChatCompletionResponse, ChatCompletionResponseChoice,
                                                ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
                                                UsageInfo)

    request_id = f'chat-{shortuuid.random()}'
    created_time = int(time.time())
    model_name = request.model

    parser = GptOssChatParser()
    parser.parser = DummyParser()

    if request.stream:

        def completion_stream_generator() -> Generator['ChatCompletionStreamResponse', None, None]:
            finish_reason = 'stop'
            for chunk in token_chunks:
                delta_message = parser.parse_streaming(chunk)
                choice_data = ChatCompletionResponseStreamChoice(index=0,
                                                                 delta=delta_message,
                                                                 finish_reason=finish_reason,
                                                                 logprobs=None)
                response = ChatCompletionStreamResponse(id=request_id,
                                                        created=created_time,
                                                        model=model_name,
                                                        choices=[choice_data],
                                                        usage=None)
                yield response

        return completion_stream_generator()

    # Non-stream path: parse all tokens at once using parse_full
    tokens: List[int] = []
    for c in token_chunks:
        tokens.extend(c)
    message = parser.parse_full(tokens)
    finish_reason = 'tool_calls' if message.tool_calls else 'stop'
    choice_data = ChatCompletionResponseChoice(index=0, message=message, finish_reason=finish_reason)
    return ChatCompletionResponse(id=request_id,
                                  created=created_time,
                                  model=model_name,
                                  choices=[choice_data],
                                  usage=UsageInfo())


def _stream_parse(request, token_chunks: List[List[int]]):
    from lmdeploy.serve.openai.protocol import DeltaMessage

    content = ''
    reasoning_content = ''
    tool_calls_by_index = {}

    for i, stream_resp in enumerate(_chat_completion_v1(request, token_chunks)):
        delta_message: DeltaMessage = stream_resp.choices[0].delta
        if delta_message.content:
            content += delta_message.content
        if delta_message.reasoning_content:
            reasoning_content += delta_message.reasoning_content
        if delta_message.tool_calls:
            for c in delta_message.tool_calls:
                idx = c.index
                existing_call = tool_calls_by_index.get(idx, None)
                if not existing_call:
                    tool_calls_by_index[idx] = c
                    continue
                if c.function.name:
                    existing_call.function.name = c.function.name
                if c.function.arguments:
                    existing_call.function.arguments = existing_call.function.arguments or ''
                    existing_call.function.arguments += c.function.arguments
    # sorted list for stable order
    tool_calls = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index.keys())]
    return content, reasoning_content, tool_calls


def _t(s: str) -> List[int]:
    return [ord(c) for c in s]


# Basic: single function call split across two chunks (bug repro scenario)
TOKENS_SINGLE_CALL_TWO_CHUNKS = [
    [-1] + _t('{"location": "Paris'),
    _t(', France"}'),
]

# Multiple calls with indices and different function names
TOKENS_TWO_CALLS_DIFFERENT_FUNCS = [
    [-1] + _t('{"location": "Berlin"}') + [-9] + [-4] + _t('{"city": "New'),
    _t(' York"}') + [-9],
]

# Interleaved channels: analysis, tool call, final content
TOKENS_INTERLEAVED = [
    [-3] + _t('Thinking about the weather. ') + [-1] + _t('{"location": "Par'),
    _t('is, France"}') + [-9] + [-2] + _t('Fetching the weather now.'),
]

# Two calls, same function name, indices increment
TOKENS_TWO_CALLS_SAME_FUNC = [
    [-1] + _t('{"location": "Tokyo"}') + [-9],
    [-6] + _t('{"location": "Ky'),
    _t('oto"}') + [-9],
]


@pytest.mark.parametrize(('token_chunks', 'expects'), [
    (TOKENS_SINGLE_CALL_TWO_CHUNKS, [TestExpects('get_weather', 'Paris, France')]),
])
def test_parser_stream_basic(token_chunks: List[List[int]], expects: List[TestExpects]):
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

    _install_openai_harmony_stub()
    request = ChatCompletionRequest(model='gpt-oss', messages=[], stream=True)
    content, reasoning_content, tool_calls = _stream_parse(request, token_chunks)

    assert len(tool_calls) == len(expects)
    for parsed_call, expected_call in zip(tool_calls, expects):
        assert parsed_call.function.name == expected_call.func_name
        args = json.loads(parsed_call.function.arguments)
        assert args['location'] == expected_call.location
    assert content.strip() == ''
    assert (reasoning_content or '').strip() == ''


def test_parser_stream_multiple_calls_indices():
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

    _install_openai_harmony_stub()
    request = ChatCompletionRequest(model='gpt-oss', messages=[], stream=True)
    content, reasoning_content, tool_calls = _stream_parse(request, TOKENS_TWO_CALLS_DIFFERENT_FUNCS)

    assert len(tool_calls) == 2
    # tool_calls sorted by index ensures stable order
    tc0, tc1 = tool_calls
    assert tc0.index == 0 and tc1.index == 1
    assert tc0.function.name == 'get_weather'
    assert json.loads(tc0.function.arguments)['location'] == 'Berlin'
    assert tc1.function.name == 'get_time'
    assert json.loads(tc1.function.arguments)['city'] == 'New York'
    assert (content or '').strip() == ''
    assert (reasoning_content or '').strip() == ''


def test_parser_stream_interleaved_channels():
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

    _install_openai_harmony_stub()
    request = ChatCompletionRequest(model='gpt-oss', messages=[], stream=True)
    content, reasoning_content, tool_calls = _stream_parse(request, TOKENS_INTERLEAVED)

    assert json.loads(tool_calls[0].function.arguments)['location'] == 'Paris, France'
    assert reasoning_content == 'Thinking about the weather. '
    assert content == 'Fetching the weather now.'


@pytest.mark.parametrize(('token_chunks', 'expects'), [
    (TOKENS_TWO_CALLS_SAME_FUNC, [TestExpects('get_weather', 'Tokyo'),
                                  TestExpects('get_weather', 'Kyoto')]),
])
def test_parser_stream_two_calls_same_func(token_chunks: List[List[int]], expects: List[TestExpects]):
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

    _install_openai_harmony_stub()
    request = ChatCompletionRequest(model='gpt-oss', messages=[], stream=True)
    _, _, tool_calls = _stream_parse(request, token_chunks)

    assert len(tool_calls) == len(expects)
    for parsed_call, expected_call in zip(tool_calls, expects):
        assert parsed_call.function.name == expected_call.func_name
        args = json.loads(parsed_call.function.arguments)
        assert args['location'] == expected_call.location


def test_open_tool_call_no_args():
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

    _install_openai_harmony_stub()
    request = ChatCompletionRequest(model='gpt-oss', messages=[], stream=True)
    content, reasoning_content, tool_calls = _stream_parse(request, [[-1]])

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == 'get_weather'
    assert (tool_calls[0].function.arguments or '') == ''
    assert (content or '') == ''
    assert (reasoning_content or '') == ''


@pytest.mark.parametrize(('token_chunks', 'expects'), [
    (TOKENS_SINGLE_CALL_TWO_CHUNKS, [TestExpects('get_weather', 'Paris, France')]),
    (TOKENS_TWO_CALLS_SAME_FUNC, [TestExpects('get_weather', 'Tokyo'),
                                  TestExpects('get_weather', 'Kyoto')]),
])
def test_parser_nonstream(token_chunks: List[List[int]], expects: List[TestExpects]):
    from lmdeploy.serve.openai.protocol import ChatCompletionRequest

    _install_openai_harmony_stub()
    resp = _chat_completion_v1(ChatCompletionRequest(model='gpt-oss', messages=[], stream=False), token_chunks)

    assert len(resp.choices) == 1
    first_message = resp.choices[0].message
    assert first_message.content is None
    assert (first_message.reasoning_content or '') == ''
    assert len(first_message.tool_calls) == len(expects)
    for parsed_call, expected_call in zip(first_message.tool_calls, expects):
        assert parsed_call.function.name == expected_call.func_name
        args = json.loads(parsed_call.function.arguments)
        assert args['location'] == expected_call.location
