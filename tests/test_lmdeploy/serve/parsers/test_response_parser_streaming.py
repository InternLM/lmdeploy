from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager

from .helpers import first_stream_delta

MODEL_ID = 'test/default-reasoning-model'


def _make_parser():
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('default')
    cls.tool_parser_cls = None
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=True,
        chat_template_kwargs={'enable_thinking': True},
    )
    return cls(request=request)


class TestResponseParserReasoningCloseTagAcrossChunks:
    """A reasoning close tag (e.g. ``</think>``) is not a single vocab token,
    so real token-by-token streaming can split it across multiple
    ``stream_chunk`` calls.

    _consume_reasoning must buffer a possible partial-tag suffix instead of flushing it as reasoning content, mirroring
    how _consume_plain already handles split open tags.
    """

    def test_close_tag_split_across_two_chunks_is_not_leaked(self):
        parser = _make_parser()

        reasoning_text = ''
        content_text = ''
        for chunk in ['Let me think about this', '</th', 'ink>', ' The answer is 42.']:
            for delta_msg, _ in parser.stream_chunk(delta_text=chunk, delta_token_ids=[]):
                if delta_msg.reasoning_content:
                    reasoning_text += delta_msg.reasoning_content
                if delta_msg.content:
                    content_text += delta_msg.content

        assert reasoning_text == 'Let me think about this'
        assert content_text == ' The answer is 42.'

    def test_close_tag_split_byte_by_byte_is_not_leaked(self):
        parser = _make_parser()
        close_tag = '</think>'

        reasoning_text = ''
        for chunk in list('reasoning ') + list(close_tag):
            for delta_msg, _ in parser.stream_chunk(delta_text=chunk, delta_token_ids=[]):
                if delta_msg.reasoning_content:
                    reasoning_text += delta_msg.reasoning_content

        assert reasoning_text == 'reasoning '
        assert close_tag not in reasoning_text

    def test_intact_close_tag_in_single_chunk_still_works(self):
        """Regression guard: the common case (tag arrives whole) must be
        unaffected."""
        parser = _make_parser()

        delta_msg, _ = first_stream_delta(
            parser.stream_chunk(delta_text='reasoning</think> content', delta_token_ids=[]))
        assert delta_msg.reasoning_content == 'reasoning'
        assert delta_msg.content is None
