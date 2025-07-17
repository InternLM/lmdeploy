import collections
import json
import time
from typing import Generator, List, Tuple, Union

import pytest
import shortuuid

from lmdeploy.serve.openai.api_server import VariableInterface
from lmdeploy.serve.openai.protocol import (ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice,
                                            ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
                                            ChatMessage, DeltaMessage, DeltaToolCall, UsageInfo)
from lmdeploy.serve.openai.reasoning_parser.qwen_qwq_reasoning_parser import QwenQwQReasoningParser
from lmdeploy.serve.openai.tool_parser.qwen3_parser import Qwen3ToolParser

TestExpects = collections.namedtuple('TestExpects', 'func_name location')


class DummyTokenizer:

    def decode(self, token_ids: List[int]) -> str:
        return ' '.join(map(str, token_ids))

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]


DELTA_TEXT_SEQUENCE = [
    '<think>',
    '\n',
    '好的',
    '，',
    '用户',
    '问',
    '的是',
    '北京',
    '的',
    '天气',
    '怎么样',
    '。',
    '我',
    '需要',
    '调',
    '用',
    'get',
    '_weather',
    '这个',
    '工具',
    '来',
    '获取',
    '信息',
    '。',
    '首先',
    '，',
    '确认',
    '用户',
    '提供的',
    '地点',
    '是',
    '北京',
    '，',
    '参数',
    '正确',
    '。',
    '然后',
    '检查',
    '工具',
    '的',
    '参数',
    '要求',
    '，',
    '只需要',
    'location',
    '，',
    '类型',
    '是',
    '字符串',
    '。',
    '于是',
    '构造',
    '参数',
    '对象',
    '，',
    '调',
    '用',
    '函数',
    '，',
    '返回',
    '结果',
    '。',
    '确保',
    '没有',
    '遗漏',
    '必要',
    '参数',
    '，',
    '比如',
    'location',
    '是',
    '必须',
    '的',
    '，',
    '这里',
    '已经',
    '提供',
    '，',
    '所以',
    '没问题',
    '。',
    '最后',
    '将',
    '结果',
    '以',
    '自然',
    '语言',
    '回复',
    '用户',
    '。\n',
    '</think>',
    '\n\n',
    '<tool_call>',
    '\n',
    '{"',
    'name',
    '":',
    ' "',
    'get',
    '_weather',
    '",',
    ' "',
    'arguments',
    '":',
    ' {"',
    'location',
    '":',
    ' "',
    '北京',
    '"}}\n',
    '</tool_call>',
]

DELTA_TEXT_SEQUENCE_MULTIPLE_CALLS = DELTA_TEXT_SEQUENCE + [
    '\n\n',
    '<tool_call>',
    '\n',
    '{"',
    'name',
    '":',
    ' "',
    'get',
    '_weather',
    '",',
    ' "',
    'arguments',
    '":',
    ' {"',
    'location',
    '":',
    ' "',
    '上海',
    '"}}\n',
    '</tool_call>',
]

EXPECTED_CONTENT = ''
EXPECTED_REASONING_CONTENT = ''.join((
    '好的，用户问的是北京的天气怎么样。我需要调用get_weather这个工具来获取信息。',
    '首先，确认用户提供的地点是北京，参数正确。然后检查工具的参数要求，',
    '只需要location，类型是字符串。于是构造参数对象，调用函数，返回结果。',
    '确保没有遗漏必要参数，比如location是必须的，这里已经提供，所以没问题。',
    '最后将结果以自然语言回复用户。',
))


def _chat_completion_v1(
        request: ChatCompletionRequest,
        text_sequence: List[str]) -> Union[ChatCompletionResponse, Generator[ChatCompletionStreamResponse, None, None]]:
    request_id = f'chat-{shortuuid.random()}'
    created_time = int(time.time())
    model_name = request.model
    if request.stream:

        def completion_stream_generator() -> Generator[ChatCompletionStreamResponse, None, None]:
            previous_text = ''
            current_text = ''
            finish_reason = 'stop'
            has_parser = VariableInterface.tool_parser is not None or VariableInterface.reasoning_parser is not None
            for text in text_sequence:
                logprobs, usage = None, None
                delta_message = DeltaMessage(role='assistant', content=text)
                if has_parser:
                    current_text = current_text + text
                if request.tool_choice != 'none' and VariableInterface.tool_parser is not None:
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
                    reasoning_delta = VariableInterface.reasoning_parser.extract_reasoning_content_streaming(
                        previous_text=previous_text,
                        current_text=current_text,
                        delta_text=delta_message.content,
                        previous_token_ids=[],
                        current_token_ids=[],
                        delta_token_ids=[])
                    if reasoning_delta is not None:
                        delta_message.reasoning_content = reasoning_delta.reasoning_content
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

    # copied and simplified from api_server.py:chat_completions_v1
    text = ''.join(text_sequence)
    tool_calls = None
    reasoning_content = None
    finish_reason = 'stop'
    if request.tool_choice != 'none' and VariableInterface.tool_parser is not None:
        tool_call_info = VariableInterface.tool_parser.extract_tool_calls(text, request=request)
        text, tool_calls = tool_call_info.content, tool_call_info.tool_calls
        if isinstance(tool_calls, List) and len(tool_calls):
            if finish_reason == 'stop':
                finish_reason = 'tool_calls'

    if VariableInterface.reasoning_parser is not None:
        reasoning_content, text = VariableInterface.reasoning_parser.extract_reasoning_content(text, request)

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


def _stream_parse(request: ChatCompletionRequest, text_sequence: List[str]) -> Tuple[str, str, List[DeltaToolCall]]:
    # Call parser.extract_tool_calls_streaming with delta_text specified in `DELTA_TEXT_SEQUENCE`.
    # `current_text` and `previous_text` init values and update logic
    # can be found in lmdeploy/serve/openai/api_server.py:455-523.
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
                    existing_call.function.arguments = existing_call.function.arguments or ''
                    existing_call.function.arguments += c.function.arguments
    return content, reasoning_content, list(sorted(tool_calls.values(), key=lambda x: x.index))


@pytest.mark.parametrize(('text_sequence', 'expects'), [
    (DELTA_TEXT_SEQUENCE, [TestExpects('get_weather', '北京')]),
    (DELTA_TEXT_SEQUENCE_MULTIPLE_CALLS, [TestExpects('get_weather', '北京'),
                                          TestExpects('get_weather', '上海')]),
])
def test_parser_stream(text_sequence: List[str], expects: List[TestExpects]):
    tokenizer = DummyTokenizer()
    VariableInterface.tool_parser = Qwen3ToolParser(tokenizer=tokenizer)
    VariableInterface.reasoning_parser = QwenQwQReasoningParser(tokenizer=tokenizer)
    request = ChatCompletionRequest(model='qwen', messages=[], stream=True)
    content, reasoning_content, tool_calls = _stream_parse(request, text_sequence)
    assert len(tool_calls) == len(expects)
    for parsed_call, expected_call in zip(tool_calls, expects):
        assert parsed_call.function.name == expected_call.func_name
        args = json.loads(parsed_call.function.arguments)
        assert args['location'] == expected_call.location
        assert content.strip() == EXPECTED_CONTENT
        assert reasoning_content.strip() == EXPECTED_REASONING_CONTENT


@pytest.mark.parametrize(('text_sequence', 'expects'), [
    (DELTA_TEXT_SEQUENCE, [TestExpects('get_weather', '北京')]),
    (DELTA_TEXT_SEQUENCE_MULTIPLE_CALLS, [TestExpects('get_weather', '北京'),
                                          TestExpects('get_weather', '上海')]),
])
def test_parser_nonstream(text_sequence: List[str], expects: List[TestExpects]):
    tokenizer = DummyTokenizer()
    VariableInterface.tool_parser = Qwen3ToolParser(tokenizer=tokenizer)
    VariableInterface.reasoning_parser = QwenQwQReasoningParser(tokenizer=tokenizer)
    resp: ChatCompletionResponse = _chat_completion_v1(ChatCompletionRequest(model='qwen', messages=[], stream=False),
                                                       text_sequence)

    assert len(resp.choices) == 1
    first_message = resp.choices[0].message
    assert first_message.content is None
    assert first_message.reasoning_content == EXPECTED_REASONING_CONTENT
    assert len(first_message.tool_calls) == len(expects)
    for parsed_call, expected_call in zip(first_message.tool_calls, expects):
        assert parsed_call.function.name == expected_call.func_name
        args = json.loads(parsed_call.function.arguments)
        assert args['location'] == expected_call.location


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
    VariableInterface.tool_parser = Qwen3ToolParser(tokenizer=tokenizer)
    VariableInterface.reasoning_parser = QwenQwQReasoningParser(tokenizer=tokenizer)
    resp: ChatCompletionResponse = _chat_completion_v1(ChatCompletionRequest(model='qwen', messages=[], stream=False),
                                                       text_sequence)

    assert len(resp.choices) == 1
    first_message = resp.choices[0].message
    assert first_message.content == '你好呀！✨ 很高兴见到你！'
    assert first_message.reasoning_content is None
