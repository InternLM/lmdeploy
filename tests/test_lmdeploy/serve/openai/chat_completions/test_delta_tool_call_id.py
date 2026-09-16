# Copyright (c) OpenMMLab. All rights reserved.
import json

from lmdeploy.serve.parsers.tool_parser.json_tool_parser import JsonToolParser


class _TestToolParser(JsonToolParser):
    @classmethod
    def get_tool_open_tag(cls):
        return None

    @classmethod
    def get_tool_close_tag(cls):
        return None


def test_decode_tool_incremental_json_id_only_on_first_chunk():
    """When streaming a tool call, id should appear only on the name-delta
    chunk, not on subsequent argument chunks."""

    parser = _TestToolParser()
    parser.begin_tool_block()
    pending = ''

    # Step 1: feed partial JSON with name
    pending += '{"name": "get_weather", '
    deltas = []
    consumed = parser.feed_tool_block(pending, deltas, final=False)
    pending = pending[consumed:]
    assert len(deltas) == 1
    name_delta = deltas[0]
    assert name_delta.function.name == 'get_weather'
    assert name_delta.id is not None
    assert name_delta.id.startswith('chatcmpl-tool-')
    assert name_delta.type == 'function'

    pending += '"arguments": {"city": "NY'
    deltas = []
    consumed = parser.feed_tool_block(pending, deltas, final=False)
    pending = pending[consumed:]
    assert len(deltas) == 1
    args_delta = deltas[0]
    assert args_delta.id is None
    assert args_delta.type is None
    assert args_delta.function.arguments


def test_stream_delta_tool_call_omits_null_id_and_type_in_json():
    """Serialized stream chunks should omit null id/type, not emit them as JSON
    null."""
    from lmdeploy.serve.openai.protocol import (
        ChatCompletionResponseStreamChoice,
        ChatCompletionStreamResponse,
        DeltaFunctionCall,
        DeltaMessage,
        DeltaToolCall,
    )

    delta = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                id=None,
                index=0,
                type=None,
                function=DeltaFunctionCall(arguments='{"city": "NYC"}'),
            )
        ]
    )
    response = ChatCompletionStreamResponse(
        model='test',
        choices=[ChatCompletionResponseStreamChoice(index=0, delta=delta)],
    )
    payload = json.loads(response.model_dump_json(exclude_none=True))
    tool_call = payload['choices'][0]['delta']['tool_calls'][0]
    assert 'id' not in tool_call
    assert 'type' not in tool_call
    assert tool_call['function']['arguments'] == '{"city": "NYC"}'
