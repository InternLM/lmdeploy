import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaToolCall
from lmdeploy.serve.openai.reasoning_parser.reasoning_parser import ReasoningParser
from lmdeploy.serve.openai.response_parser import ResponseParser
from lmdeploy.serve.openai.tool_parser.qwen3_tool_parser import Qwen3ToolParser
from lmdeploy.tokenizer import HuggingFaceTokenizer

MODEL_ID = 'Qwen/Qwen3-8B'


@pytest.fixture(scope='module')
def tokenizer():
    try:
        return HuggingFaceTokenizer(MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'Could not load tokenizer for {MODEL_ID}: {exc}')


@pytest.fixture()
def response_parser(tokenizer):
    # Configure ResponseParser to use unified reasoning parser and Qwen3 tool parser.
    ResponseParser.reasoning_parser_cls = ReasoningParser
    ResponseParser.tool_parser_cls = Qwen3ToolParser

    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=True,
        # Enable tool parsing (any value other than "none" works).
        tool_choice='auto',
        # Explicitly enable thinking mode to exercise reasoning parsing.
        chat_template_kwargs={'enable_thinking': True},
    )
    return ResponseParser(request=request, tokenizer=tokenizer)


# Reference streaming sequence
# reasoning part: <think> This is the mock user prompt </think>
REASONING_0 = [
    # (delta_text, emitted_delta_msg, reasoning_content, content,
    # tool_emitted, function_name, function_arguments, tool_call_type)
    # reasoning part
    ('<think>', False, None, None, False, None, None, None),
    ('This is the mock', True, 'This is the mock', None, False, None, None, None),
    (' user prompt', True, ' user prompt', None, False, None, None, None),
    ('</think>', False, None, None, False, None, None, None),
]
# reasoning part: This is the mock user prompt </think>
REASONING_1 = [
    # (delta_text, emitted_delta_msg, reasoning_content, content,
    # tool_emitted, function_name, function_arguments, tool_call_type)
    # reasoning part
    ('This is the mock', True, 'This is the mock', None, False, None, None, None),
    (' user prompt', True, ' user prompt', None, False, None, None, None),
    ('</think>', False, None, None, False, None, None, None),
]

# tool call part: <tool_call> {"name": "get_weather", "arguments": {"location": "北京", "unit": "celsius"}} </tool_call>
TOOL_CALL_0 = [
    # (delta_text, emitted_delta_msg, reasoning_content, content,
    # tool_emitted, function_name, function_arguments, tool_call_type)
    # tool call part
    ('<tool_call>', False, None, None, False, None, None, None),
    ('\n', False, None, None, False, None, None, None),
    ('{"', False, None, None, False, None, None, None),
    ('name', False, None, None, False, None, None, None),
    ('":', False, None, None, False, None, None, None),
    (' "', False, None, None, False, None, None, None),
    ('get', False, None, None, False, None, None, None),
    ('_weather', False, None, None, False, None, None, None),
    ('",', True, None, None, True, 'get_weather', None, 'function'),
    (' "', False, None, None, False, None, None, None),
    ('arguments', False, None, None, False, None, None, None),
    ('":', False, None, None, False, None, None, None),
    (' {"', False, None, None, False, None, None, None),
    ('location', False, None, None, False, None, None, None),
    ('":', False, None, None, False, None, None, None),
    (' "', False, None, None, False, None, None, None),
    ('北京', False, None, None, False, None, None, None),
    ('",', False, None, None, False, None, None, None),
    (' "', False, None, None, False, None, None, None),
    ('unit', False, None, None, False, None, None, None),
    ('":', False, None, None, False, None, None, None),
    (' "', False, None, None, False, None, None, None),
    ('celsius', False, None, None, False, None, None, None),
    ('"}}\n', False, None, None, False, None, None, None),
    ('</tool_call>', True, None, None, True, None, '{"location": "北京", "unit": "celsius"}', None),
]

REFERENCE_CHUNKS_0 = REASONING_0 + [
    ('\n\n', True, None, '\n\n', False, None, None, None)] + TOOL_CALL_0 + [
    ('', True, None, '', False, None, None, None),
]

REFERENCE_CHUNKS_1 = REASONING_1 + [
    ('\n\n', True, None, '\n\n', False, None, None, None)] + TOOL_CALL_0 + [
    ('', True, None, '', False, None, None, None),
]

REFERENCE_CHUNKS_2 = [
    # (delta_text, emitted_delta_msg, reasoning_content, content,
    # tool_emitted, function_name, function_arguments, tool_call_type)
    # reasoning part
    ('This is the mock', True, 'This is the mock', None, False, None, None, None),
    (' user prompt.', True, ' user prompt.', None, False, None, None, None),
    (' reasoning</think>\n\n<tool_call>\n', True, ' reasoning', None, False, None, None, None),
    ('{"', True, None, '\n\n', False, None, None, None),
    ('name', False, None, None, False, None, None, None),
    ('":', False, None, None, False, None, None, None),
    (' "', False, None, None, False, None, None, None),
    ('get', False, None, None, False, None, None, None),
    ('_weather', False, None, None, False, None, None, None),
    ('",', True, None, None, True, 'get_weather', None, 'function'),
    (' "', False, None, None, False, None, None, None),
    ('arguments', False, None, None, False, None, None, None),
    ('":', False, None, None, False, None, None, None),
    (' {"', False, None, None, False, None, None, None),
    ('location', False, None, None, False, None, None, None),
    ('":', False, None, None, False, None, None, None),
    (' "', False, None, None, False, None, None, None),
    ('北京', False, None, None, False, None, None, None),
    ('",', False, None, None, False, None, None, None),
    (' "', False, None, None, False, None, None, None),
    ('unit', False, None, None, False, None, None, None),
    ('":', False, None, None, False, None, None, None),
    (' "', False, None, None, False, None, None, None),
    ('celsius', False, None, None, False, None, None, None),
    ('"}}\n', False, None, None, False, None, None, None),
    ('</tool_call>', True, None, None, True, None, '{"location": "北京", "unit": "celsius"}', None),
    ('', True, None, '', False, None, None, None),
]


class TestQwenResponseParserStreaming:
    """Integration test for ResponseParser.stream_chunk with Qwen3 parsers."""

    @staticmethod
    def _encode_ids(tokenizer, text: str) -> list[int]:
        return tokenizer.encode(text, add_bos=False, add_special_tokens=False)

    @pytest.mark.parametrize('reference_chunks', [REFERENCE_CHUNKS_0, REFERENCE_CHUNKS_1, REFERENCE_CHUNKS_2])
    def test_stream_chunk_matches_reference(self, tokenizer, response_parser, reference_chunks):
        """Feed the real streaming sequence into ResponseParser.stream_chunk
        and verify each parsed chunk.

        Input:
        - Strictly use the reference token stream (including <tool_call>, \\n, <,
          function, =get, ...).

        Checks:
        - reasoning: whenever an expected reasoning chunk is provided, the
          parser must emit exactly that reasoning_content.
        - content: only after </think>, we expect a single \\n\\n.
        - tool_calls:
          - for each step, tool_emitted must match expected_tool_emitted;
          - whenever ResponseParser actually emits DeltaToolCall, we check:
            - the first time a function.name appears, it must equal
              get_current_temperature;
            - any function.arguments increments are concatenated and validated
              after streaming completes.
        """

        for (delta_text, exp_delta_msg, exp_reasoning, exp_content, exp_tool_emitted,
             exp_function_name, exp_function_arguments,
             exp_type) in reference_chunks:
            delta_ids = self._encode_ids(tokenizer, delta_text)
            delta_msg, tool_emitted = response_parser.stream_chunk(
                delta_text=delta_text,
                delta_token_ids=delta_ids,
            )
            print(f'delta_text: {delta_text!r}, delta_msg: {delta_msg}')
            if not exp_delta_msg:
                assert delta_msg is None
                continue
            # reasoning: when an expected reasoning chunk is provided, it must match exactly.
            assert delta_msg.reasoning_content == exp_reasoning
            assert delta_msg.content == exp_content
            assert tool_emitted == exp_tool_emitted
            if tool_emitted:
                assert delta_msg.tool_calls is not None
                assert len(delta_msg.tool_calls) == 1
                call = delta_msg.tool_calls[0]
                assert isinstance(call, DeltaToolCall)
                assert call.type == exp_type
                assert call.function is not None
                assert call.function.name == exp_function_name
                assert call.function.arguments == exp_function_arguments

    def test_stream_chunk_handles_mixed_reasoning_content_tool(self, tokenizer, response_parser):
        """A single delta may contain reasoning/content/tool segments together.

        This test covers chunk shapes:
        1) ``<think>``
        2) ``<think> Let me think ``
        3) ``The answer is 9 </think> OK. The``
        4) ``fine. </think> \\n\\n <tool_call> ``
        """

        def _call(delta_text: str):
            ids = self._encode_ids(tokenizer, delta_text)
            return response_parser.stream_chunk(delta_text=delta_text, delta_token_ids=ids)

        # 1) tag-only chunk should be swallowed
        delta_msg, tool_emitted = _call('<think>')
        assert delta_msg is None
        assert tool_emitted is False

        # 2) open-think plus reasoning text should emit only reasoning
        delta_msg, tool_emitted = _call('<think> Let me think ')
        assert delta_msg is not None
        assert delta_msg.reasoning_content == ' Let me think '
        assert delta_msg.content is None
        assert tool_emitted is False

        # 3) chunk carries reasoning end + normal content.
        # New parser emits ordered events, so this call emits reasoning first.
        delta_msg, tool_emitted = _call('The answer is 9 </think> OK. The')
        assert delta_msg is not None
        assert delta_msg.reasoning_content == 'The answer is 9 '
        assert delta_msg.content is None
        assert tool_emitted is False

        # Next call flushes queued plain content from previous chunk first.
        delta_msg, tool_emitted = _call('fine. </think> \n\n <tool_call> ')
        assert delta_msg is not None
        assert delta_msg.reasoning_content is None
        assert delta_msg.content == ' OK. The'
        assert tool_emitted is False

        # Flush the next queued plain segment from chunk-4.
        delta_msg, tool_emitted = _call('')
        assert delta_msg is not None
        # Stray closing tag after reasoning has ended is treated as plain content.
        assert delta_msg.reasoning_content is None
        assert delta_msg.content == 'fine. </think> \n\n '
        assert tool_emitted is False

    def test_stream_chunk_tool_enabled_without_reasoning_parser(self, tokenizer):
        """When reasoning parser is disabled, tool parsing still works.

        This proves the tool branch is reachable from plain mode after seeing the tool open tag, even with no reasoning
        parser configured.
        """
        old_reasoning_cls = ResponseParser.reasoning_parser_cls
        old_tool_cls = ResponseParser.tool_parser_cls
        try:
            ResponseParser.reasoning_parser_cls = None
            ResponseParser.tool_parser_cls = Qwen3ToolParser

            request = ChatCompletionRequest(
                model=MODEL_ID,
                messages=[],
                stream=True,
                tool_choice='auto',
                chat_template_kwargs={'enable_thinking': False},
            )
            parser = ResponseParser(request=request, tokenizer=tokenizer)

            chunks = [
                'prefix ',
                '<tool_call>',
                '\n',
                '{"',
                'name',
                '":',
                ' "',
                'get',
                '_weather',
                '",',
            ]
            tool_seen = False
            for chunk in chunks:
                delta_ids = self._encode_ids(tokenizer, chunk)
                delta_msg, tool_emitted = parser.stream_chunk(delta_text=chunk, delta_token_ids=delta_ids)
                if delta_msg is not None:
                    assert delta_msg.reasoning_content is None
                if tool_emitted:
                    tool_seen = True
                    assert delta_msg is not None
                    assert delta_msg.tool_calls is not None
                    assert delta_msg.tool_calls[0].function is not None
                    assert delta_msg.tool_calls[0].function.name == 'get_weather'
            assert tool_seen is True
        finally:
            ResponseParser.reasoning_parser_cls = old_reasoning_cls
            ResponseParser.tool_parser_cls = old_tool_cls

    def test_stream_chunk_reasoning_without_open_tag(self, tokenizer, response_parser):
        """Qwen thinking mode may omit ``<think>`` and start directly with
        reasoning.

        In this case, chunks before ``</think>`` must be emitted as
        ``reasoning_content``.
        """

        def _call(delta_text: str):
            delta_ids = self._encode_ids(tokenizer, delta_text)
            return response_parser.stream_chunk(delta_text=delta_text, delta_token_ids=delta_ids)

        # No opening <think> tag, but still in reasoning mode initially.
        delta_msg, tool_emitted = _call('Let me reason ')
        assert delta_msg is not None
        assert delta_msg.reasoning_content == 'Let me reason '
        assert delta_msg.content is None
        assert tool_emitted is False

        delta_msg, tool_emitted = _call('step by step')
        assert delta_msg is not None
        assert delta_msg.reasoning_content == 'step by step'
        assert delta_msg.content is None
        assert tool_emitted is False

        # Closing tag chunk itself is swallowed.
        delta_msg, tool_emitted = _call('</think>')
        assert delta_msg is None
        assert tool_emitted is False

        # After close tag, emit normal content.
        delta_msg, tool_emitted = _call(' final answer')
        assert delta_msg is not None
        assert delta_msg.reasoning_content is None
        assert delta_msg.content == ' final answer'
        assert tool_emitted is False

    def test_stream_chunk_preserves_content_reasoning_content_order(self, tokenizer, response_parser):
        """Mixed single chunk should preserve event order without content
        merge."""
        class PlainStartQwenReasoningParser(ReasoningParser):

            def starts_in_reasoning_mode(self) -> bool:
                return False

        old_reasoning_cls = ResponseParser.reasoning_parser_cls
        old_tool_cls = ResponseParser.tool_parser_cls
        try:
            ResponseParser.reasoning_parser_cls = PlainStartQwenReasoningParser
            ResponseParser.tool_parser_cls = Qwen3ToolParser
            request = ChatCompletionRequest(
                model=MODEL_ID,
                messages=[],
                stream=True,
                tool_choice='auto',
                chat_template_kwargs={'enable_thinking': True},
            )
            parser = ResponseParser(request=request, tokenizer=tokenizer)

            delta_text = 'content-xxx <think> reasoning-yyy </think> content-zzz <tool_call> '
            delta_ids = self._encode_ids(tokenizer, delta_text)

            # 1st event: plain content before <think>
            delta_msg, tool_emitted = parser.stream_chunk(delta_text=delta_text, delta_token_ids=delta_ids)
            assert delta_msg is not None
            assert delta_msg.content == 'content-xxx '
            assert delta_msg.reasoning_content is None
            assert tool_emitted is False

            # 2nd event: reasoning segment
            delta_msg, tool_emitted = parser.stream_chunk(delta_text='', delta_token_ids=[])
            assert delta_msg is not None
            assert delta_msg.content is None
            assert delta_msg.reasoning_content == ' reasoning-yyy '
            assert tool_emitted is False

            # 3rd event: trailing content segment before <tool_call>
            delta_msg, tool_emitted = parser.stream_chunk(delta_text='', delta_token_ids=[])
            assert delta_msg is not None
            assert delta_msg.content == ' content-zzz '
            assert delta_msg.reasoning_content is None
            assert tool_emitted is False
        finally:
            ResponseParser.reasoning_parser_cls = old_reasoning_cls
            ResponseParser.tool_parser_cls = old_tool_cls
