import json
from typing import List, Tuple

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaToolCall
from lmdeploy.serve.openai.tool_parser.qwen3_parser import Qwen3ToolParser
from lmdeploy.serve.openai.tool_parser.tool_parser import ToolParser


class DummyTokenizer:

    def decode(self, token_ids: List[int]) -> str:
        return ' '.join(map(str, token_ids))

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]


delta_text_sequence = [
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

EXPECTED_CONTENT = ''
EXPECTED_REASONING_CONTENT = ''.join((
    '好的，用户问的是北京的天气怎么样。我需要调用get_weather这个工具来获取信息。',
    '首先，确认用户提供的地点是北京，参数正确。然后检查工具的参数要求，',
    '只需要location，类型是字符串。于是构造参数对象，调用函数，返回结果。',
    '确保没有遗漏必要参数，比如location是必须的，这里已经提供，所以没问题。',
    '最后将结果以自然语言回复用户。',
))


@pytest.mark.parametrize(
    "parser",
    [
        # Qwen2d5ToolParser(tokenizer=DummyTokenizer()), # not pass
        Qwen3ToolParser(tokenizer=DummyTokenizer()),
    ])
def test_parser_stream(parser: ToolParser):
    request = ChatCompletionRequest(model='qwen', messages=[])
    content, reasoning_content, tool_calls = _stream_parse(parser, request)
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == 'get_weather'
    args = json.loads(tool_calls[0].function.arguments)
    assert args['location'] == '北京'
    assert content.strip() == EXPECTED_CONTENT
    assert reasoning_content.strip() == EXPECTED_REASONING_CONTENT


def _stream_parse(parser: ToolParser, request: ChatCompletionRequest) -> Tuple[str, str, List[DeltaToolCall]]:
    # Call parser.extract_tool_calls_streaming with delta_text specified in `delta_text_sequence`.
    # `current_text` and `previous_text` init values and update logic
    # can be found in lmdeploy/serve/openai/api_server.py:455-523.
    content = ''
    reasoning_content = ''
    tool_calls = {}

    previous_text = ''
    current_text = ''
    for delta_text in delta_text_sequence:
        current_text += delta_text
        tool_delta = parser.extract_tool_calls_streaming(previous_text=previous_text,
                                                         current_text=current_text,
                                                         delta_text=delta_text,
                                                         previous_token_ids=[],
                                                         current_token_ids=[],
                                                         delta_token_ids=[],
                                                         request=request)
        previous_text = current_text
        if not tool_delta:
            continue
        if tool_delta.content:
            content += tool_delta.content
        if tool_delta.reasoning_content:
            reasoning_content += tool_delta.reasoning_content
        if tool_delta.tool_calls:
            for c in tool_delta.tool_calls:
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


@pytest.mark.parametrize("parser", [
    Qwen3ToolParser(tokenizer=DummyTokenizer()),
])
def test_parser_nonstream(parser: ToolParser):
    request = ChatCompletionRequest(model='qwen', messages=[])
    full_text = ''
    for delta_text in delta_text_sequence:
        full_text += delta_text

    extracted_info = parser.extract_tool_calls(full_text, request)
    content = extracted_info.content
    reasoning_content = extracted_info.reasoning_content
    tool_calls = extracted_info.tool_calls

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == 'get_weather'
    args = json.loads(tool_calls[0].function.arguments)
    assert args['location'] == '北京'
    assert content.strip() == EXPECTED_CONTENT
    assert reasoning_content.strip() == EXPECTED_REASONING_CONTENT
