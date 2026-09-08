import json

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager

from .helpers import first_stream_delta


def _build_parser():
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = None
    cls.tool_parser_cls = ToolParserManager.get('llama3')
    request = ChatCompletionRequest(
        model='meta-llama/Llama-3.1-8B-Instruct',
        messages=[],
        stream=True,
        tool_choice='auto',
        tools=[{
            'type': 'function',
            'function': {
                'name': 'find_user_id_by_name_zip',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'first_name': {
                            'type': 'string'
                        },
                        'last_name': {
                            'type': 'string'
                        },
                        'zip': {
                            'type': 'integer'
                        },
                    },
                },
            },
        }],
    )
    return cls(request=request)


def test_llama3_streaming_without_close_tag():
    parser = _build_parser()
    payload = ('{"name":"find_user_id_by_name_zip","parameters":{"first_name":"Chen",'
               '"last_name":"Johnson","zip":77004}}')

    delta_msg, tool_emitted = first_stream_delta(parser.stream_chunk('<|python_tag|>', []))
    assert delta_msg is None
    assert tool_emitted is False

    delta_msg, tool_emitted = first_stream_delta(parser.stream_chunk(payload, []))
    assert delta_msg is not None
    assert tool_emitted is True
    assert delta_msg.tool_calls is not None

    names = [c.function.name for c in delta_msg.tool_calls if c.function and c.function.name]
    args_text = ''.join(c.function.arguments or '' for c in delta_msg.tool_calls if c.function)
    assert names == ['find_user_id_by_name_zip']
    assert json.loads(args_text) == {
        'first_name': 'Chen',
        'last_name': 'Johnson',
        'zip': 77004,
    }


def test_llama3_streaming_emits_arguments_before_json_payload_complete():
    parser = _build_parser()

    parser.stream_chunk('<|python_tag|>', [])
    chunks = [
        '{"name":"find_user_id_by_name_zip","parameters":{"first_name":"Ch',
        'en","last_name":"Johnson","zip":77004',
    ]

    argument_fragments = []
    for chunk in chunks:
        for delta_msg, tool_emitted in parser.stream_chunk(chunk, []):
            if not tool_emitted or delta_msg is None or not delta_msg.tool_calls:
                continue
            for call in delta_msg.tool_calls:
                if call.function and call.function.arguments:
                    argument_fragments.append(call.function.arguments)

    assert argument_fragments
    joined = ''.join(argument_fragments)
    assert 'Chen' in joined
    assert joined != '{"first_name":"Chen","last_name":"Johnson","zip":77004}'


def test_llama3_parse_complete_without_close_tag():
    parser = _build_parser()
    text = ('<|python_tag|>{"name":"find_user_id_by_name_zip","parameters":{"first_name":"Chen",'
            '"last_name":"Johnson","zip":77004}}')

    content, tool_calls, reasoning = parser.parse_complete(text)

    assert content is None
    assert reasoning is None
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == 'find_user_id_by_name_zip'
    assert json.loads(tool_calls[0].function.arguments) == {
        'first_name': 'Chen',
        'last_name': 'Johnson',
        'zip': 77004,
    }


def test_llama3_streaming_emits_plain_text_after_tool_call_finishes():
    parser = _build_parser()
    payload = ('{"name":"find_user_id_by_name_zip","parameters":{"first_name":"Chen",'
               '"last_name":"Johnson","zip":77004}}')

    parser.stream_chunk('<|python_tag|>', [])
    delta_msg, tool_emitted = first_stream_delta(parser.stream_chunk(payload, []))
    assert delta_msg is not None
    assert tool_emitted is True

    delta_msg, tool_emitted = first_stream_delta(parser.stream_chunk(' Done.', []))
    assert delta_msg is not None
    assert tool_emitted is False
    assert delta_msg.content == ' Done.'
    assert delta_msg.tool_calls is None


def test_llama3_preserves_post_tool_whitespace_across_chunk_boundaries():
    payload = ('{"name":"find_user_id_by_name_zip","parameters":{"first_name":"Chen",'
               '"last_name":"Johnson","zip":77004}}')
    text = f'<|python_tag|>{payload} Done.'

    single_chunk = _build_parser().stream_chunk(text, [], final=True)
    split_parser = _build_parser()
    split_chunks = split_parser.stream_chunk(f'<|python_tag|>{payload}', [])
    split_chunks.extend(split_parser.stream_chunk(' Done.', [], final=True))

    single_content = ''.join(delta.content or '' for delta, _ in single_chunk)
    split_content = ''.join(delta.content or '' for delta, _ in split_chunks)
    assert single_content == split_content == ' Done.'
