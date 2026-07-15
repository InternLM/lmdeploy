from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager

from .helpers import first_stream_delta


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
    return cls(request=request)


class TestResponseParserToolCloseTagAcrossChunks:
    """A tool close tag (e.g. ``<|action_end|>``) is not a single vocab
    token, so real token-by-token streaming can split it across multiple
    ``stream_chunk`` calls. _consume_tool must buffer a possible
    partial-tag suffix instead of leaking it into the tool payload,
    mirroring how _consume_reasoning already handles a split reasoning
    close tag.
    """

    def test_close_tag_split_across_two_chunks_is_not_leaked(self):
        parser = _build_parser()
        chunks = [
            '<|action_start|><|plugin|>',
            '\n{\n    "name": "get_weather",\n    "parameters": {"city": "Berlin"}\n}',
            '<|action_e',
            'nd|>',
        ]

        seen_name = False
        seen_args = False
        leaked_tag_text = []

        for chunk in chunks:
            delta, _tool_emitted = first_stream_delta(parser.stream_chunk(delta_text=chunk, delta_token_ids=[]))
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
        assert '<|action_end|>' not in ''.join(leaked_tag_text)

    def test_intact_close_tag_in_single_chunk_still_works(self):
        """Regression guard: the common case (tag arrives whole) must be
        unaffected."""
        parser = _build_parser()
        chunks = [
            '<|action_start|><|plugin|>',
            '\n{\n    "name": "get_weather",\n    "parameters": {"city": "Berlin"}\n}<|action_end|>',
        ]

        seen_name = False
        for chunk in chunks:
            delta, _tool_emitted = first_stream_delta(parser.stream_chunk(delta_text=chunk, delta_token_ids=[]))
            if delta and delta.tool_calls:
                for call in delta.tool_calls:
                    if call.function and call.function.name == 'get_weather':
                        seen_name = True

        assert seen_name
