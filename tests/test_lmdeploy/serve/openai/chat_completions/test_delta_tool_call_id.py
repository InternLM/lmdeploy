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


def _stream_argument_fragments(chunks, *, final_on_last):
    parser = _TestToolParser()
    parser.start_tool_call()
    fragments = []
    for idx, chunk in enumerate(chunks):
        deltas = parser.decode_tool_incremental(chunk, final=final_on_last and idx == len(chunks) - 1)
        fragments.extend(delta.function.arguments for delta in deltas if delta.function and delta.function.arguments)
    return fragments


def _complete_arguments(payload):
    call = _TestToolParser().parse_tool_call_complete(payload)
    return json.loads(call.function.arguments)


def test_decode_tool_incremental_json_id_only_on_first_chunk():
    """When streaming a tool call, id should appear only on the name-delta
    chunk, not on subsequent argument chunks."""

    parser = _TestToolParser()
    parser.start_tool_call()

    # Step 1: feed partial JSON with name
    deltas = parser.decode_tool_incremental('{"name": "get_weather", ', final=False)
    assert len(deltas) == 1
    name_delta = deltas[0]
    assert name_delta.function.name == 'get_weather'
    assert name_delta.id is not None
    assert name_delta.id.startswith('chatcmpl-tool-')
    assert name_delta.type == 'function'

    deltas = parser.decode_tool_incremental('"arguments": {"city": "NY', final=False)
    assert len(deltas) == 1
    args_delta = deltas[0]
    assert args_delta.id is None
    assert args_delta.type is None
    assert args_delta.function.arguments


def test_decode_tool_incremental_json_streams_empty_arguments():
    for arguments in ('{}', '[]', 'null'):
        argument_fragments = _stream_argument_fragments(
            ['{"name":"f","arguments":' + arguments + '}'],
            final_on_last=True,
        )

        assert argument_fragments == [arguments]


def test_decode_tool_incremental_json_streams_arguments_before_payload_complete():
    payload = '{"name":"f","arguments":{"city":"New York","units":"c"}}'
    fragments = _stream_argument_fragments(
        [
            '{"name":"f","arguments":{"city":"Ne',
            'w York","units":"c"}',
        ],
        final_on_last=False,
    )

    assert fragments
    assert json.loads(''.join(fragments)) == _complete_arguments(payload)


def test_decode_tool_incremental_json_streams_nested_and_escaped_arguments():
    args = {'outer': {'items': [1, {'text': 'a"b'}], 'path': 'C:\\tmp'}}
    payload = '{"name":"f","arguments":' + json.dumps(args) + '}'
    body_without_outer_close = payload[:-1]
    split_at = body_without_outer_close.find('a\\"b')
    fragments = _stream_argument_fragments(
        [
            body_without_outer_close[:split_at + 2],
            body_without_outer_close[split_at + 2:],
        ],
        final_on_last=False,
    )

    assert fragments
    assert json.loads(''.join(fragments)) == _complete_arguments(payload)


def test_decode_tool_incremental_json_streams_parameters_fallback_before_payload_complete():
    payload = '{"name":"f","parameters":{"p":1}}'
    fragments = _stream_argument_fragments(
        [
            '{"name":"f","parameters":{"p":',
            '1}',
        ],
        final_on_last=False,
    )

    assert fragments
    assert json.loads(''.join(fragments)) == _complete_arguments(payload)


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
