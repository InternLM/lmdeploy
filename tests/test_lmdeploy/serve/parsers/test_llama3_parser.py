import json

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager


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
    return cls(request=request, tokenizer=object())


def test_llama3_streaming_without_close_tag():
    parser = _build_parser()
    payload = ('{"name":"find_user_id_by_name_zip","parameters":{"first_name":"Chen",'
               '"last_name":"Johnson","zip":77004}}')

    delta_msg, tool_emitted = parser.stream_chunk('<|python_tag|>', [])
    assert delta_msg is None
    assert tool_emitted is False

    delta_msg, tool_emitted = parser.stream_chunk(payload, [])
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
    delta_msg, tool_emitted = parser.stream_chunk(payload, [])
    assert delta_msg is not None
    assert tool_emitted is True

    delta_msg, tool_emitted = parser.stream_chunk(' Done.', [])
    assert delta_msg is not None
    assert tool_emitted is False
    assert delta_msg.content == ' Done.'
    assert delta_msg.tool_calls is None
