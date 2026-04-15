from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager


def _build_parser():
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = None
    cls.tool_parser_cls = ToolParserManager.get('intern-s1')
    request = ChatCompletionRequest(
        model='intern-s1',
        messages=[],
        stream=True,
        tool_choice='auto',
    )
    return cls(request=request, tokenizer=object())


def test_stream_chunk_handles_split_internlm_open_tag():
    parser = _build_parser()
    chunks = [
        '<|action_start|>',
        '<|plugin|>',
        '\n{\n    "name": "get_weather",\n    "parameters": {"city": "Berlin"}\n}',
        '<|action_end|>',
    ]

    seen_name = False
    seen_args = False
    leaked_tag_text = []

    for chunk in chunks:
        delta, _tool_emitted = parser.stream_chunk(delta_text=chunk, delta_token_ids=[])
        if delta is None:
            continue
        if delta.content:
            leaked_tag_text.append(delta.content)
        if delta.tool_calls:
            for call in delta.tool_calls:
                if call.function and call.function.name == 'get_weather':
                    seen_name = True
                if call.function and call.function.arguments == '{"city": "Berlin"}':
                    seen_args = True

    assert seen_name
    assert seen_args
    assert '<|action_start|>' not in ''.join(leaked_tag_text)
    assert '<|plugin|>' not in ''.join(leaked_tag_text)
