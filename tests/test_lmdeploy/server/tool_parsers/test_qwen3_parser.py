import json
import time
from collections.abc import Generator

import pytest
import shortuuid

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo,
)
from lmdeploy.serve.openai.reasoning_parser import QwenReasoningParser
from lmdeploy.serve.openai.response_parser import StreamBuffer
from lmdeploy.serve.openai.tool_parser import Qwen3ToolParser
from lmdeploy.tokenizer import Tokenizer


@pytest.fixture(scope='module')
def tokenizer():
    from lmdeploy.tokenizer import HuggingFaceTokenizer
    return HuggingFaceTokenizer('Qwen/Qwen3-8B')

@pytest.fixture()
def reasoning_parser(tokenizer):
    return QwenReasoningParser(tokenizer)

@pytest.fixture()
def tool_parser(tokenizer):
    return Qwen3ToolParser(tokenizer)

DELTA_TEXT_SEQUENCE = [
    # (delta_text, reasoning_content, content, tool_calls)
    ('<think>', None, None, []),
    ('\n', '\n', None, []),
    ('好的', '好的', None, []),
    ('，', '，', None, []),
    ('用户', '用户', None, []),
    ('问', '问', None, []),
    ('的是', '的是', None, []),
    ('北京', '北京', None, []),
    ('的', '的', None, []),
    ('天气', '天气', None, []),
    ('怎么样', '怎么样', None, []),
    ('。', '。', None, []),
    ('我', '我', None, []),
    ('需要', '需要', None, []),
    ('调', '调', None, []),
    ('用', '用', None, []),
    ('get', 'get', None, []),
    ('_weather', '_weather', None, []),
    ('这个', '这个', None, []),
    ('工具', '工具', None, []),
    ('来', '来', None, []),
    ('获取', '获取', None, []),
    ('信息', '信息', None, []),
    ('。', '。', None, []),
    ('首先', '首先', None, []),
    ('，', '，', None, []),
    ('确认', '确认', None, []),
    ('用户', '用户', None, []),
    ('提供的', '提供的', None, []),
    ('地点', '地点', None, []),
    ('是', '是', None, []),
    ('北京', '北京', None, []),
    ('，', '，', None, []),
    ('参数', '参数', None, []),
    ('正确', '正确', None, []),
    ('。', '。', None, []),
    ('然后', '然后', None, []),
    ('检查', '检查', None, []),
    ('工具', '工具', None, []),
    ('的', '的', None, []),
    ('参数', '参数', None, []),
    ('要求', '要求', None, []),
    ('，', '，', None, []),
    ('只需要', '只需要', None, []),
    ('location', 'location', None, []),
    ('，', '，', None, []),
    ('类型', '类型', None, []),
    ('是', '是', None, []),
    ('字符串', '字符串', None, []),
    ('。', '。', None, []),
    ('于是', '于是', None, []),
    ('构造', '构造', None, []),
    ('参数', '参数', None, []),
    ('对象', '对象', None, []),
    ('，', '，', None, []),
    ('调', '调', None, []),
    ('用', '用', None, []),
    ('函数', '函数', None, []),
    ('，', '，', None, []),
    ('返回', '返回', None, []),
    ('结果', '结果', None, []),
    ('。', '。', None, []),
    ('确保', '确保', None, []),
    ('没有', '没有', None, []),
    ('遗漏', '遗漏', None, []),
    ('必要', '必要', None, []),
    ('参数', '参数', None, []),
    ('，', '，', None, []),
    ('比如', '比如', None, []),
    ('location', 'location', None, []),
    ('是', '是', None, []),
    ('必须', '必须', None, []),
    ('的', '的', None, []),
    ('，', '，', None, []),
    ('这里', '这里', None, []),
    ('已经', '已经', None, []),
    ('提供', '提供', None, []),
    ('，', '，', None, []),
    ('所以', '所以', None, []),
    ('没问题', '没问题', None, []),
    ('。', '。', None, []),
    ('最后', '最后', None, []),
    ('将', '将', None, []),
    ('结果', '结果', None, []),
    ('以', '以', None, []),
    ('自然', '自然', None, []),
    ('语言', '语言', None, []),
    ('回复', '回复', None, []),
    ('用户', '用户', None, []),
    ('。\n', '。\n', None, []),
    ('</think>', None, None, []),
    ('\n\n', None, '\n\n', []),
    ('<tool_call>', None, None, []),
    ('\n', None, None, '\n'),
    ('{"', None, None, '{"'),
    ('name', None, None, 'name'),
    ('":', None, None, '":'),
    (' "', None, None, ' "'),
    ('get', None, None, 'get'),
    ('_weather', None, None, '_weather'),
    ('",', None, None, '",'),
    (' "', None, None, ' "'),
    ('arguments', None, None, 'arguments'),
    ('":', None, None, '":'),
    (' {"', None, None, ' {"'),
    ('location', None, None, 'location'),
    ('":', None, None, '":'),
    (' "', None, None, ' "'),
    ('北京', None, None, '北京'),
    ('"}}\n', None, None, '"}}\n'),
    ('</tool_call>', None, None, None)
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


def _normalize_delta_sequence(text_sequence: list) -> list[str]:
    """Flatten streaming fixtures that use (delta, ...) tuples (possibly mixed
    with str chunks)."""
    if not text_sequence:
        return []
    out = []
    for item in text_sequence:
        out.append(item[0] if isinstance(item, tuple) else item)
    return out


def _chat_completion_v1(
    tokenizer: Tokenizer,
    reasoning_parser: QwenReasoningParser,
    tool_parser: Qwen3ToolParser,
    request: ChatCompletionRequest,
    text_sequence: list[str]) -> ChatCompletionResponse | Generator[ChatCompletionStreamResponse, None, None]:
    request_id = f'chat-{shortuuid.random()}'
    created_time = int(time.time())
    model_name = request.model
    delta_chunks = _normalize_delta_sequence(text_sequence)
    if request.stream:
        parser_state = StreamBuffer()
        has_parser = tool_parser is not None or reasoning_parser is not None

        def completion_stream_generator() -> Generator[ChatCompletionStreamResponse, None, None]:
            finish_reason = 'stop'
            for text in delta_chunks:
                print(f'delta_text: {text}')
                # delta_message = DeltaMessage(role='assistant', content=None)
                delta_message = DeltaMessage(role='assistant', content=text) if not has_parser else None
                content = text
                delta_token_ids = tokenizer.encode(content, add_bos=False)
                parser_state.update(content, delta_token_ids)
                if request.tool_choice != 'none' and tool_parser is not None:
                    delta_message = DeltaMessage(role='assistant')
                    tool_delta = tool_parser.extract_tool_calls_streaming(
                        delta_text=content,
                        delta_token_ids=delta_token_ids,
                        request=request,
                        stream_buffer=parser_state,
                    )
                    print(f'tool_delta: {tool_delta}')
                    if tool_delta is not None:
                        delta_message.tool_calls = tool_delta.tool_calls
                        delta_message.content = tool_delta.content
                if reasoning_parser is not None:
                    if tool_parser is None or delta_message is None:
                        content = text
                    elif delta_message.content is not None:
                         # delta_message.content is `content` if there is no tool call information in it
                        content = delta_message.content
                        # There might be reasoning content in `delta_message.content`.
                        # So we set it to None and let reasoning parser to extract the reasoning and content.
                        delta_message.content = None
                    else:
                        # tool_parser is consuming tool call information. We set Nont content to jump
                        # parsing reasoning.
                        content = None
                    reasoning_delta = reasoning_parser.extract_reasoning_streaming(
                        delta_text=content,
                        delta_token_ids=delta_token_ids,
                        request=request,
                        stream_buffer=parser_state,
                    )
                    print(f'reasoning_delta: {reasoning_delta}')
                    if reasoning_delta is not None:
                        delta_message.reasoning_content = reasoning_delta.reasoning_content
                        delta_message.content = reasoning_delta.content
                parser_state.step()
                choice_data = ChatCompletionResponseStreamChoice(index=0,
                                                                 delta=delta_message,
                                                                 finish_reason=finish_reason)
                response = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[choice_data]
                )
                yield response

        return completion_stream_generator()

    # copied and simplified from api_server.py:chat_completions_v1
    text = ''.join(delta_chunks)
    tool_calls = None
    reasoning_content = None
    finish_reason = 'stop'
    if request.tool_choice != 'none' and tool_parser is not None:
        tool_call_info = tool_parser.extract_tool_calls(text, request=request)
        text, tool_calls = tool_call_info.content, tool_call_info.tool_calls
        if isinstance(tool_calls, list) and len(tool_calls):
            if finish_reason == 'stop':
                finish_reason = 'tool_calls'

    if reasoning_parser is not None:
        reasoning_content, text = reasoning_parser.extract_reasoning(text, request)

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


# def _stream_parse(
#     tokenizer: Tokenizer,
#     reasoning_parser: QwenReasoningParser,
#     tool_parser: Qwen3ToolParser,
#     request: ChatCompletionRequest,
#     text_sequence: list[str],
# ) -> tuple[str, str, list[DeltaToolCall]]:
#     # Call parser.extract_tool_calls_streaming with delta_text specified in `DELTA_TEXT_SEQUENCE`.
#     # `current_text` and `previous_text` init values and update logic
#     # can be found in lmdeploy/serve/openai/api_server.py:455-523.
#     content = ''
#     reasoning_content = ''
#     tool_calls = {}

#     for stream_resp in _chat_completion_v1(tokenizer, reasoning_parser, tool_parser, request, text_sequence):
#         delta_message: DeltaMessage = stream_resp.choices[0].delta
#         if delta_message.content:
#             content += delta_message.content
#         if delta_message.reasoning_content:
#             reasoning_content += delta_message.reasoning_content
#         if delta_message.tool_calls:
#             for c in delta_message.tool_calls:
#                 existing_call = tool_calls.get(c.id, None)
#                 if not existing_call:
#                     tool_calls[c.id] = c
#                     continue
#                 # merge with existing
#                 if c.function.name:
#                     existing_call.function.name = c.function.name
#                 if c.function.arguments:
#                     existing_call.function.arguments = existing_call.function.arguments or ''
#                     existing_call.function.arguments += c.function.arguments
#     return content, reasoning_content, list(sorted(tool_calls.values(), key=lambda x: x.index))



class TestQwen3ToolStreamingParser:
    """Tests for Qwen3ToolParser streaming mode."""

    @pytest.mark.parametrize('text_sequence', [DELTA_TEXT_SEQUENCE])
    def test_parser_stream(self, tokenizer, reasoning_parser, tool_parser,
                           text_sequence: list[tuple[str, str, str, str]]):
        """Test streaming parser with single and multiple tool calls."""
        request = ChatCompletionRequest(model='qwen', messages=[], stream=True)
        delta_texts = [t[0] for t in text_sequence]
        responses = _chat_completion_v1(tokenizer, reasoning_parser, tool_parser, request, delta_texts)
        for response, t in zip(responses, text_sequence):
            delta_message: DeltaMessage = response.choices[0].delta
            print(f'delta_message: {delta_message}')
            assert delta_message.reasoning_content == t[1]
            assert delta_message.content == t[2]
            # assert delta_message.tool_calls == t[3]


    def test_incomplete_tool_call_streaming(self, tokenizer, reasoning_parser, tool_parser):
        """Test streaming parser with incomplete tool call (missing end
        tag)."""
        request = ChatCompletionRequest(model='qwen', messages=[], stream=True)

        # Incomplete tool call without end tag
        text_sequence = ['好的', '，', '让我', '调用', '工具', '。', 'Вот', '\n', 'ذهب', '\n',
                         '{"name": "get_weather", "arguments": {"location": "北京"']
        responses = _chat_completion_v1(
            tokenizer, reasoning_parser, tool_parser, request, text_sequence)
        for response in responses:
            delta_message: DeltaMessage = response.choices[0].delta
            print(f'delta_message: {delta_message}')
            assert not delta_message.tool_calls
        # Should not parse tool call since it's incomplete


class TestQwen3ToolNonStreamingParser:
    """Tests for Qwen3ToolParser non-streaming mode."""

    @pytest.mark.parametrize('text_sequence', [DELTA_TEXT_SEQUENCE, DELTA_TEXT_SEQUENCE_MULTIPLE_CALLS])
    def test_parser_nonstream(self, tokenizer, reasoning_parser, tool_parser, text_sequence: list[str]):
        """Test non-streaming parser with single and multiple tool calls."""
        full = ''.join(_normalize_delta_sequence(text_sequence))
        req = ChatCompletionRequest(model='qwen', messages=[], stream=False)
        tool_ref = tool_parser.extract_tool_calls(full, request=req)

        resp: ChatCompletionResponse = _chat_completion_v1(
            tokenizer, reasoning_parser, tool_parser, req, text_sequence)

        assert len(resp.choices) == 1
        first_message = resp.choices[0].message
        assert (first_message.content or '').strip() == EXPECTED_CONTENT
        assert (first_message.reasoning_content or '').strip() == EXPECTED_REASONING_CONTENT
        assert len(first_message.tool_calls) == len(tool_ref.tool_calls)
        for parsed_call, ref_call in zip(first_message.tool_calls, tool_ref.tool_calls):
            assert parsed_call.function.name == ref_call.function.name
            assert json.loads(parsed_call.function.arguments) == json.loads(ref_call.function.arguments)

    def test_no_think_nonstream(self, tokenizer, reasoning_parser, tool_parser):
        """Test non-streaming parser with plain text (no thinking tags)."""
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
        resp: ChatCompletionResponse = _chat_completion_v1(
            tokenizer, reasoning_parser, tool_parser,
            ChatCompletionRequest(model='qwen', messages=[], stream=False),
            text_sequence)

        assert len(resp.choices) == 1
        first_message = resp.choices[0].message
        assert first_message.content == '你好呀！✨ 很高兴见到你！'
        assert first_message.reasoning_content is None

    def test_invalid_json_tool_call(self, tokenizer, reasoning_parser, tool_parser):
        """Test non-streaming parser with invalid JSON in tool call."""
        # Invalid JSON in tool call
        text_sequence = ['好的，让我调用工具。', 'Вот', '\n', 'ذهب', '\n',
                         '{"name": "get_weather", "arguments": {invalid json}}', '666', '\n']

        resp: ChatCompletionResponse = _chat_completion_v1(
            tokenizer, reasoning_parser, tool_parser,
            ChatCompletionRequest(model='qwen', messages=[], stream=False),
            text_sequence)

        # Should handle gracefully - tool call may not be parsed due to invalid JSON
        assert len(resp.choices) == 1

    def test_empty_tool_call_content(self, tokenizer, reasoning_parser, tool_parser):
        """Test non-streaming parser with empty tool call content."""
        # Empty tool call
        text_sequence = ['好的', '。', 'Вот', '\n', 'ذهب', '\n', '666', '\n']

        resp: ChatCompletionResponse = _chat_completion_v1(
            tokenizer, reasoning_parser, tool_parser,
            ChatCompletionRequest(model='qwen', messages=[], stream=False),
            text_sequence)

        assert len(resp.choices) == 1
