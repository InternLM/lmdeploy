# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from lmdeploy.serve.openai.utils import maybe_filter_parallel_tool_calls


def _request(parallel_tool_calls: bool | None = True):
    return ChatCompletionRequest(
        model='fake-model',
        messages=[{
            'role': 'user',
            'content': 'hi',
        }],
        parallel_tool_calls=parallel_tool_calls,
    )


def test_chat_completion_request_accepts_parallel_tool_calls():
    assert _request(False).parallel_tool_calls is False
    assert _request().parallel_tool_calls is True


def test_parallel_tool_calls_false_keeps_first_non_streaming_tool_call():
    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role='assistant',
            tool_calls=[
                ToolCall(id='call_1', function=FunctionCall(name='search', arguments='{}')),
                ToolCall(id='call_2', function=FunctionCall(name='lookup', arguments='{}')),
            ],
        ),
    )

    maybe_filter_parallel_tool_calls(choice, _request(False))

    assert [tool_call.id for tool_call in choice.message.tool_calls] == ['call_1']


def test_parallel_tool_calls_true_keeps_all_non_streaming_tool_calls():
    choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(
            role='assistant',
            tool_calls=[
                ToolCall(id='call_1', function=FunctionCall(name='search', arguments='{}')),
                ToolCall(id='call_2', function=FunctionCall(name='lookup', arguments='{}')),
            ],
        ),
    )

    maybe_filter_parallel_tool_calls(choice, _request(True))

    assert [tool_call.id for tool_call in choice.message.tool_calls] == ['call_1', 'call_2']


def test_parallel_tool_calls_false_keeps_index_zero_streaming_tool_call():
    choice = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    id='call_1',
                    function=DeltaFunctionCall(name='search', arguments='{}'),
                ),
                DeltaToolCall(
                    index=1,
                    id='call_2',
                    function=DeltaFunctionCall(name='lookup', arguments='{}'),
                ),
            ],
        ),
    )

    maybe_filter_parallel_tool_calls(choice, _request(False))

    assert [tool_call.index for tool_call in choice.delta.tool_calls] == [0]


def test_parallel_tool_calls_true_keeps_all_streaming_tool_calls():
    choice = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=0,
                    id='call_1',
                    function=DeltaFunctionCall(name='search', arguments='{}'),
                ),
                DeltaToolCall(
                    index=1,
                    id='call_2',
                    function=DeltaFunctionCall(name='lookup', arguments='{}'),
                ),
            ],
        ),
    )

    maybe_filter_parallel_tool_calls(choice, _request(True))

    assert [tool_call.index for tool_call in choice.delta.tool_calls] == [0, 1]
