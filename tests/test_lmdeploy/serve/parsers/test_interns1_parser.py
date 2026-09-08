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
        tools=[{'type': 'function', 'function': {'name': 'get_weather'}}],
        tool_choice='auto',
    )
    return cls(request=request)


def test_stream_chunk_matches_split_internlm_sequence():
    parser = _build_parser()
    reference = [
        ('<|action_start|>', []),
        ('<|plugin|>', []),
        ('\n{"name":"get_weather","parameters":{"city":"Ber', [
            ('function', 'get_weather', None),
            (None, None, '{"city":"Ber'),
        ]),
        ('lin"', [(None, None, 'lin"')]),
        ('}}', [(None, None, '}')]),
        ('<|action_end|>', []),
    ]

    for chunk, expected_calls in reference:
        deltas = parser.stream_chunk(delta_text=chunk, delta_token_ids=[])
        assert all(tool_emitted for _, tool_emitted in deltas)
        assert all(delta.content is None for delta, _ in deltas)
        actual_calls = [
            (call.type, call.function.name, call.function.arguments)
            for delta, _ in deltas
            for call in (delta.tool_calls or [])
        ]
        assert actual_calls == expected_calls
