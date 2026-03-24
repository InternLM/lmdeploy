import collections
import json
import time
from collections.abc import Generator

import pytest
import shortuuid

from lmdeploy.serve.openai.api_server import VariableInterface
from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    DeltaToolCall,
    UsageInfo,
)
from lmdeploy.serve.openai.tool_parser.qwen3coder_parser import Qwen3CoderToolParser

TestExpects = collections.namedtuple('TestExpects', 'func_name kwargs')


class DummyTokenizer:

    def decode(self, token_ids: list[int]) -> str:
        return ' '.join(map(str, token_ids))

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


DELTA_TEXT_SEQUENCE = [
    '好的，我现在帮你调用工具。\n',
    '<tool_call>',
    '\n',
    '<function=get_wea',
    'ther>\n',
    '<parameter=loca',
    'tion>',
    '北京</par',
    'ameter>\n',
    '<parameter=uni',
    't>celsius</parameter>\n',
    '</function>\n',
    '</tool_call>',
]

DELTA_TEXT_SEQUENCE_MULTIPLE_CALLS = DELTA_TEXT_SEQUENCE + [
    '\n\n',
    '<tool_call>',
    '\n<function=get_weather',
    '>\n',
    '<parameter=location>上海</parameter>\n',
    '</function>\n',
    '</tool_call>',
]

EXPECTED_CONTENT = '好的，我现在帮你调用工具。'


def _chat_completion_v1(
        request: ChatCompletionRequest,
        text_sequence: list[str]) -> ChatCompletionResponse | Generator[ChatCompletionStreamResponse, None, None]:
    request_id = f'chat-{shortuuid.random()}'
    created_time = int(time.time())
    model_name = request.model
    if request.stream:

        def completion_stream_generator() -> Generator[ChatCompletionStreamResponse, None, None]:
            previous_text = ''
            current_text = ''
            finish_reason = 'stop'
            has_parser = (VariableInterface.tool_parser is not None or VariableInterface.reasoning_parser is not None)
            for text in text_sequence:
                logprobs, usage = None, None
                delta_message = DeltaMessage(role='assistant', content=text)
                if has_parser:
                    current_text = current_text + text
                has_tool = VariableInterface.tool_parser is not None
                if request.tool_choice != 'none' and has_tool:
                    tool_delta = VariableInterface.tool_parser.extract_tool_calls_streaming(
                        previous_text=previous_text,
                        current_text=current_text,
                        delta_text=delta_message.content,
                        previous_token_ids=[],
                        current_token_ids=[],
                        delta_token_ids=[],
                        request=request)
                    if tool_delta is not None:
                        delta_message.tool_calls = tool_delta.tool_calls
                        delta_message.content = tool_delta.content or ''
                if VariableInterface.reasoning_parser is not None:
                    parser = VariableInterface.reasoning_parser
                    reasoning_delta = parser.extract_reasoning_content_streaming(previous_text=previous_text,
                                                                                 current_text=current_text,
                                                                                 delta_text=delta_message.content,
                                                                                 previous_token_ids=[],
                                                                                 current_token_ids=[],
                                                                                 delta_token_ids=[])
                    if reasoning_delta is not None:
                        delta_message.reasoning_content = (reasoning_delta.reasoning_content)
                        delta_message.content = reasoning_delta.content or ''
                if has_parser:
                    previous_text = current_text

                choice_data = ChatCompletionResponseStreamChoice(index=0,
                                                                 delta=delta_message,
                                                                 finish_reason=finish_reason,
                                                                 logprobs=logprobs)
                response = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[choice_data],
                    usage=usage,
                )
                yield response

        return completion_stream_generator()

    text = ''.join(text_sequence)
    tool_calls = None
    reasoning_content = None
    finish_reason = 'stop'
    has_tool = VariableInterface.tool_parser is not None
    if request.tool_choice != 'none' and has_tool:
        tool_call_info = VariableInterface.tool_parser.extract_tool_calls(text, request=request)
        text, tool_calls = tool_call_info.content, tool_call_info.tool_calls
        if isinstance(tool_calls, list) and len(tool_calls):
            if finish_reason == 'stop':
                finish_reason = 'tool_calls'

    if VariableInterface.reasoning_parser is not None:
        parser = VariableInterface.reasoning_parser
        reasoning_content, text = parser.extract_reasoning_content(text, request)

    choices = []
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role='assistant', content=text, tool_calls=tool_calls, reasoning_content=reasoning_content),
        finish_reason=finish_reason,
    )
    choices.append(choice_data)

    return ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=UsageInfo(),
    )


def _stream_parse(request: ChatCompletionRequest, text_sequence: list[str]) -> tuple[str, str, list[DeltaToolCall]]:
    content = ''
    reasoning_content = ''
    tool_calls = {}

    for stream_resp in _chat_completion_v1(request, text_sequence):
        delta_message: DeltaMessage = stream_resp.choices[0].delta
        if delta_message.content:
            content += delta_message.content
        if delta_message.reasoning_content:
            reasoning_content += delta_message.reasoning_content
        if delta_message.tool_calls:
            for c in delta_message.tool_calls:
                existing_call = tool_calls.get(c.id, None)
                if not existing_call:
                    tool_calls[c.id] = c
                    continue
                # merge with existing
                if c.function.name:
                    existing_call.function.name = c.function.name
                if c.function.arguments:
                    existing_call.function.arguments = (existing_call.function.arguments or '')
                    existing_call.function.arguments += c.function.arguments
    return content, reasoning_content, list(sorted(tool_calls.values(), key=lambda x: x.index))


@pytest.mark.parametrize(('text_sequence', 'expects'), [
    (DELTA_TEXT_SEQUENCE, [TestExpects('get_weather', {
        'location': '北京',
        'unit': 'celsius'
    })]),
    (DELTA_TEXT_SEQUENCE_MULTIPLE_CALLS, [
        TestExpects('get_weather', {
            'location': '北京',
            'unit': 'celsius'
        }),
        TestExpects('get_weather', {'location': '上海'})
    ]),
])
def test_parser_stream(text_sequence: list[str], expects: list[TestExpects]):
    tokenizer = DummyTokenizer()
    VariableInterface.tool_parser = Qwen3CoderToolParser(tokenizer=tokenizer)
    VariableInterface.reasoning_parser = None
    request = ChatCompletionRequest(model='qwen3coder', messages=[], stream=True)
    content, reasoning_content, tool_calls = _stream_parse(request, text_sequence)
    assert len(tool_calls) == len(expects)
    for parsed_call, expected_call in zip(tool_calls, expects):
        assert parsed_call.function.name == expected_call.func_name
        args = json.loads(parsed_call.function.arguments)
        assert args == expected_call.kwargs
        assert content.strip() == EXPECTED_CONTENT


@pytest.mark.parametrize(('text_sequence', 'expects'), [
    (DELTA_TEXT_SEQUENCE, [TestExpects('get_weather', {
        'location': '北京',
        'unit': 'celsius'
    })]),
    (DELTA_TEXT_SEQUENCE_MULTIPLE_CALLS, [
        TestExpects('get_weather', {
            'location': '北京',
            'unit': 'celsius'
        }),
        TestExpects('get_weather', {'location': '上海'})
    ]),
])
def test_parser_nonstream(text_sequence: list[str], expects: list[TestExpects]):
    tokenizer = DummyTokenizer()
    VariableInterface.tool_parser = Qwen3CoderToolParser(tokenizer=tokenizer)
    VariableInterface.reasoning_parser = None
    resp: ChatCompletionResponse = _chat_completion_v1(
        ChatCompletionRequest(model='qwen3coder', messages=[], stream=False), text_sequence)

    assert len(resp.choices) == 1
    first_message = resp.choices[0].message
    assert first_message.content.strip() == EXPECTED_CONTENT
    assert first_message.reasoning_content is None
    assert len(first_message.tool_calls) == len(expects)
    for parsed_call, expected_call in zip(first_message.tool_calls, expects):
        assert parsed_call.function.name == expected_call.func_name
        args = json.loads(parsed_call.function.arguments)
        assert args == expected_call.kwargs


def test_no_think_nonstream():
    text_sequence = [
        '你好',
        '呀',
        '！',
        '✨',
        '',
        ' 很',
        '高兴',
        '见到',
        '你',
        '！',
    ]
    tokenizer = DummyTokenizer()
    VariableInterface.tool_parser = Qwen3CoderToolParser(tokenizer=tokenizer)
    VariableInterface.reasoning_parser = None
    resp: ChatCompletionResponse = _chat_completion_v1(
        ChatCompletionRequest(model='qwen3coder', messages=[], stream=False), text_sequence)

    assert len(resp.choices) == 1
    first_message = resp.choices[0].message
    assert first_message.content == '你好呀！✨ 很高兴见到你！'
    assert first_message.reasoning_content is None
