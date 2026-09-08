import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager

from .helpers import first_stream_delta

MODEL_ID = 'Qwen/Qwen3-8B'
PARSER_TOOLS = [{'type': 'function', 'function': {'name': 'get_weather'}}]


class _ReasoningTokenizer:

    def get_vocab(self):
        return {
            '<think>': 1,
            '</think>': 2,
            '<tool_call>': 3,
            '</tool_call>': 5,
        }


def _collect_stream_events(parser, chunks):
    """Collect ordered response-channel events across streamed chunks."""
    events = []
    for index, chunk in enumerate(chunks):
        deltas = parser.stream_chunk(
            delta_text=chunk,
            delta_token_ids=[],
            final=index == len(chunks) - 1,
        )
        for message, _ in deltas:
            if message.reasoning_content is not None:
                events.append(('reasoning', message.reasoning_content))
            if message.tool_calls:
                events.append(('tool', message.tool_calls))
            if message.content is not None:
                events.append(('content', message.content))
    return events


def _flatten_delta_events(deltas):
    events = []
    for message, tool_emitted in deltas:
        if message is None:
            continue
        if message.tool_calls:
            for call in message.tool_calls:
                events.append((
                    message.reasoning_content,
                    message.content,
                    tool_emitted,
                    call.function.name,
                    call.function.arguments,
                    call.type,
                ))
        else:
            events.append((
                message.reasoning_content,
                message.content,
                tool_emitted,
                None,
                None,
                None,
            ))
    return events


@pytest.fixture()
def response_parser():
    # Configure ResponseParser to use unified reasoning parser and Qwen3 tool parser.
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('default')
    cls.tool_parser_cls = ToolParserManager.get('qwen3')

    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=True,
        tools=PARSER_TOOLS,
        # Enable tool parsing (any value other than "none" works).
        tool_choice='auto',
        # Explicitly enable thinking mode to exercise reasoning parsing.
        chat_template_kwargs={'enable_thinking': True},
    )
    return cls(request=request)


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
    (' {"', True, None, None, True, None, '{"', None),
    ('location', True, None, None, True, None, 'location', None),
    ('":', True, None, None, True, None, '":', None),
    (' "', True, None, None, True, None, ' "', None),
    ('北京', True, None, None, True, None, '北京', None),
    ('",', True, None, None, True, None, '",', None),
    (' "', True, None, None, True, None, ' "', None),
    ('unit', True, None, None, True, None, 'unit', None),
    ('":', True, None, None, True, None, '":', None),
    (' "', True, None, None, True, None, ' "', None),
    ('celsius', True, None, None, True, None, 'celsius', None),
    ('"}}\n', True, None, None, True, None, '"}', None),
    ('</tool_call>', False, None, None, False, None, None, None),
]

REFERENCE_CHUNKS_0 = REASONING_0 + [
    ('\n\n', True, None, '\n\n', False, None, None, None)] + TOOL_CALL_0 + [
    ('', False, None, None, False, None, None, None),
]

REFERENCE_CHUNKS_1 = REASONING_1 + [
    ('\n\n', True, None, '\n\n', False, None, None, None)] + TOOL_CALL_0 + [
    ('', False, None, None, False, None, None, None),
]

REFERENCE_CHUNKS_2 = [
    # (delta_text, emitted_delta_msg, reasoning_content, content,
    # tool_emitted, function_name, function_arguments, tool_call_type)
    # reasoning part
    ('This is the mock', True, 'This is the mock', None, False, None, None, None),
    (' user prompt.', True, ' user prompt.', None, False, None, None, None),
    (' reasoning</think>\n\n<tool_call>\n', True, ' reasoning', None, False, None, None, None),
    (None, True, None, '\n\n', False, None, None, None),
] + TOOL_CALL_0[2:] + [
    ('', False, None, None, False, None, None, None),
]


class TestQwenResponseParserStreaming:
    """Integration test for ResponseParser.stream_chunk with Qwen3 parsers."""

    @pytest.mark.parametrize('reference_chunks', [REFERENCE_CHUNKS_0, REFERENCE_CHUNKS_1, REFERENCE_CHUNKS_2])
    def test_stream_chunk_matches_reference(self, response_parser, reference_chunks):
        """Feed the real streaming sequence into ResponseParser.stream_chunk
        and verify each parsed chunk.

        Input:
        - Strictly use the reference token stream (including <tool_call>, \\n, <,
          function, =get, ...).

        Each input chunk is checked independently so the reference also
        enforces when a stable argument fragment becomes visible.
        """

        pos = 0
        while pos < len(reference_chunks):
            row = reference_chunks[pos]
            delta_text = row[0]
            assert delta_text is not None

            expected_rows = [row]
            pos += 1
            while pos < len(reference_chunks) and reference_chunks[pos][0] is None:
                expected_rows.append(reference_chunks[pos])
                pos += 1

            expected_events = [tuple(expected[2:]) for expected in expected_rows if expected[1]]
            actual_events = _flatten_delta_events(
                response_parser.stream_chunk(
                    delta_text=delta_text,
                    delta_token_ids=[],
                ))
            assert actual_events == expected_events

    def test_stream_chunk_handles_mixed_reasoning_content_tool(self, response_parser):
        """A single delta may contain reasoning/content/tool segments together.

        This test covers chunk shapes:
        1) ``<think>``
        2) ``<think> Let me think ``
        3) ``The answer is 9 </think> OK. The``
        4) ``fine. </think> \\n\\n <tool_call> ``
        """

        def _call(delta_text: str):
            return response_parser.stream_chunk(delta_text=delta_text, delta_token_ids=[])

        # 1) tag-only chunk should be swallowed
        delta_msg, tool_emitted = first_stream_delta(_call('<think>'))
        assert delta_msg is None
        assert tool_emitted is False

        # 2) open-think plus reasoning text should emit only reasoning
        delta_msg, tool_emitted = first_stream_delta(_call('<think> Let me think '))
        assert delta_msg is not None
        assert delta_msg.reasoning_content == ' Let me think '
        assert delta_msg.content is None
        assert tool_emitted is False

        # 3) chunk carries reasoning end + normal content in one step.
        deltas = _call('The answer is 9 </think> OK. The')
        assert len(deltas) >= 2
        assert deltas[0][0].reasoning_content == 'The answer is 9 '
        assert deltas[1][0].content == ' OK. The'

        # 4) chunk carries stray close tag + plain content before tool open tag.
        deltas = _call('fine. </think> \n\n <tool_call> ')
        assert len(deltas) == 1
        assert deltas[0][0].content == 'fine. </think> \n\n '
        assert deltas[0][1] is False

    def test_stream_chunk_tool_enabled_without_reasoning_parser(self):
        """When reasoning parser is disabled, tool parsing still works.

        This proves the tool branch is reachable from plain mode after seeing the tool open tag, even with no reasoning
        parser configured.
        """
        cls = ResponseParserManager.get('default')
        old_reasoning_cls = cls.reasoning_parser_cls
        old_tool_cls = cls.tool_parser_cls
        try:
            cls.reasoning_parser_cls = None
            cls.tool_parser_cls = ToolParserManager.get('qwen3')

            request = ChatCompletionRequest(
                model=MODEL_ID,
                messages=[],
                stream=True,
                tools=PARSER_TOOLS,
                tool_choice='auto',
                chat_template_kwargs={'enable_thinking': False},
            )
            parser = cls(request=request)

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
                delta_msg, tool_emitted = first_stream_delta(parser.stream_chunk(delta_text=chunk,
                                                                                 delta_token_ids=[]))
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
            cls.reasoning_parser_cls = old_reasoning_cls
            cls.tool_parser_cls = old_tool_cls

    def test_decode_incremental_keeps_json_argument_buffer_bounded(self):
        parser = ToolParserManager.get('qwen3')()
        parser.begin_tool_block()
        pending = '{"name":"write_file","arguments":{"content":"'
        calls = []
        consumed = parser.feed_tool_block(pending, calls, final=False)
        pending = pending[consumed:]
        for _ in range(200):
            pending += 'x' * 32
            consumed = parser.feed_tool_block(pending, calls, final=False)
            pending = pending[consumed:]

        assert pending == ''

    def test_stream_chunk_reasoning_without_open_tag(self, response_parser):
        """Qwen thinking mode may omit ``<think>`` and start directly with
        reasoning.

        In this case, chunks before ``</think>`` must be emitted as
        ``reasoning_content``.
        """

        def _call(delta_text: str):
            return response_parser.stream_chunk(delta_text=delta_text, delta_token_ids=[])

        # No opening <think> tag, but still in reasoning mode initially.
        delta_msg, tool_emitted = first_stream_delta(_call('Let me reason '))
        assert delta_msg is not None
        assert delta_msg.reasoning_content == 'Let me reason '
        assert delta_msg.content is None
        assert tool_emitted is False

        delta_msg, tool_emitted = first_stream_delta(_call('step by step'))
        assert delta_msg is not None
        assert delta_msg.reasoning_content == 'step by step'
        assert delta_msg.content is None
        assert tool_emitted is False

        # Closing tag chunk itself is swallowed.
        delta_msg, tool_emitted = first_stream_delta(_call('</think>'))
        assert delta_msg is None
        assert tool_emitted is False

        # After close tag, emit normal content.
        delta_msg, tool_emitted = first_stream_delta(_call(' final answer'))
        assert delta_msg is not None
        assert delta_msg.reasoning_content is None
        assert delta_msg.content == ' final answer'
        assert tool_emitted is False

    def test_stream_chunk_resumes_reasoning_after_nested_tool(self, response_parser):
        text = (
            '<think>before'
            '<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}</tool_call>'
            'after</think>answer'
        )

        events = _collect_stream_events(response_parser, [text])

        assert [kind for kind, _ in events] == ['reasoning', 'tool', 'reasoning', 'content']
        assert events[0] == ('reasoning', 'before')
        assert events[2:] == [('reasoning', 'after'), ('content', 'answer')]
        tool_deltas = events[1][1]
        assert tool_deltas[0].function.name == 'get_weather'
        assert ''.join(call.function.arguments or '' for call in tool_deltas) == '{"city":"Beijing"}'

    @pytest.mark.parametrize('separator', ['\n', '\n\n'])
    def test_stream_chunk_drops_separator_between_nested_tools(self, response_parser, separator):
        chunks = [
            (
                '<think>before'
                '<tool_call>{"name":"get_weather","arguments":{"n":1}}</tool_call>'
            ),
            separator,
            '<tool_',
            (
                'call>{"name":"get_weather","arguments":{"n":2}}</tool_call>'
                'after</think>answer'
            ),
        ]

        events = _collect_stream_events(response_parser, chunks)

        assert [kind for kind, _ in events] == ['reasoning', 'tool', 'tool', 'reasoning', 'content']
        assert ''.join(value for kind, value in events if kind == 'reasoning') == 'beforeafter'
        assert ''.join(value for kind, value in events if kind == 'content') == 'answer'
        tool_deltas = [call for kind, calls in events if kind == 'tool' for call in calls]
        assert [call.index for call in tool_deltas if call.function.name] == [0, 1]
        assert [
            ''.join(call.function.arguments or '' for call in tool_deltas if call.index == index)
            for index in (0, 1)
        ] == ['{"n":1}', '{"n":2}']

    def test_stream_chunk_filtered_nested_tool_still_resumes_reasoning(self, response_parser):
        chunks = [
            (
                '<think>before'
                '<tool_call>{"name":"not_available","arguments":{"x":1}}</tool_call>'
            ),
            '\n',
            'after</think>answer',
        ]

        events = _collect_stream_events(response_parser, chunks)

        assert [kind for kind, _ in events] == ['reasoning', 'reasoning', 'content']
        assert ''.join(value for kind, value in events if kind == 'reasoning') == 'before\nafter'
        assert events[-1] == ('content', 'answer')

    def test_stream_chunk_preserves_order(self):
        """Mixed single chunk should preserve event order without content
        merge."""
        class PlainStartQwenReasoningParser(ReasoningParserManager.get('default')):

            def starts_in_reasoning_mode(self) -> bool:
                return False

        cls = ResponseParserManager.get('default')
        old_reasoning_cls = cls.reasoning_parser_cls
        old_tool_cls = cls.tool_parser_cls
        try:
            cls.reasoning_parser_cls = PlainStartQwenReasoningParser
            cls.tool_parser_cls = ToolParserManager.get('qwen3')
            request = ChatCompletionRequest(
                model=MODEL_ID,
                messages=[],
                stream=True,
                tool_choice='auto',
                chat_template_kwargs={'enable_thinking': True},
            )
            parser = cls(request=request)

            delta_text = 'content-xxx <think> reasoning-yyy </think> content-zzz <tool_call> '

            deltas = parser.stream_chunk(delta_text=delta_text, delta_token_ids=[])
            assert len(deltas) >= 3
            assert deltas[0][0].content == 'content-xxx '
            assert deltas[0][0].reasoning_content is None
            assert deltas[0][1] is False
            assert deltas[1][0].reasoning_content == ' reasoning-yyy '
            assert deltas[1][0].content is None
            assert deltas[1][1] is False
            assert deltas[2][0].content == ' content-zzz '
            assert deltas[2][0].reasoning_content is None
            assert deltas[2][1] is False
        finally:
            cls.reasoning_parser_cls = old_reasoning_cls
            cls.tool_parser_cls = old_tool_cls

    def test_stream_chunk_counts_tool_tokens_until_reasoning_close(self):
        cls = ResponseParserManager.get('default')
        old_reasoning_cls = cls.reasoning_parser_cls
        old_tool_cls = cls.tool_parser_cls
        old_tokenizer = cls.tokenizer
        try:
            cls.reasoning_parser_cls = ReasoningParserManager.get('default')
            cls.tool_parser_cls = ToolParserManager.get('qwen3')
            cls.tokenizer = _ReasoningTokenizer()
            request = ChatCompletionRequest(
                model=MODEL_ID,
                messages=[],
                stream=True,
                tool_choice='auto',
                chat_template_kwargs={'enable_thinking': True},
            )
            parser = cls(request=request)

            deltas = parser.stream_chunk('<think>reason', [1, 10, 11])
            deltas += parser.stream_chunk('<tool_call>{}</tool_call>more', [3, 20, 5, 30])
            deltas += parser.stream_chunk('</think>answer', [2, 40], final=True)

            # Every token after <think> and before </think> counts, including
            # tool protocol, payload, and post-tool tokens.
            assert parser.reasoning_tokens == 6
            assert ''.join(delta.reasoning_content or '' for delta, _ in deltas) == 'reasonmore'
            assert ''.join(delta.content or '' for delta, _ in deltas) == 'answer'
        finally:
            cls.reasoning_parser_cls = old_reasoning_cls
            cls.tool_parser_cls = old_tool_cls
            cls.tokenizer = old_tokenizer


class TestQwenResponseParserComplete:

    def test_parse_complete_aggregates_reasoning_around_nested_tool(self, response_parser):
        text = (
            '<think>before'
            '<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}</tool_call>'
            'after</think>answer'
        )

        content, tool_calls, reasoning = response_parser.parse_complete(text)

        assert reasoning == 'beforeafter'
        assert content == 'answer'
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == 'get_weather'
        assert tool_calls[0].function.arguments == '{"city":"Beijing"}'

    def test_parse_complete_counts_only_reasoning_content_tokens(self):
        cls = ResponseParserManager.get('default')
        old_reasoning_cls = cls.reasoning_parser_cls
        old_tool_cls = cls.tool_parser_cls
        old_tokenizer = cls.tokenizer
        try:
            cls.reasoning_parser_cls = ReasoningParserManager.get('default')
            cls.tool_parser_cls = None
            cls.tokenizer = _ReasoningTokenizer()
            request = ChatCompletionRequest(
                model=MODEL_ID,
                messages=[],
                stream=False,
                tool_choice='none',
                chat_template_kwargs={'enable_thinking': True},
            )
            parser = cls(request=request)

            parser.parse_complete(
                '<think>reasoning</think>answer',
                token_ids=[1, 10, 11, 2, 12],
            )

            assert parser.reasoning_tokens == 2
        finally:
            cls.reasoning_parser_cls = old_reasoning_cls
            cls.tool_parser_cls = old_tool_cls
            cls.tokenizer = old_tokenizer

    def test_parse_complete_strips_reasoning_open_tag(self):
        cls = ResponseParserManager.get('default')
        old_reasoning_cls = cls.reasoning_parser_cls
        old_tool_cls = cls.tool_parser_cls
        try:
            cls.reasoning_parser_cls = ReasoningParserManager.get('default')
            cls.tool_parser_cls = None
            request = ChatCompletionRequest(
                model=MODEL_ID,
                messages=[],
                stream=False,
                tool_choice='none',
                chat_template_kwargs={'enable_thinking': True},
            )
            parser = cls(request=request)
            content, tool_calls, reasoning = parser.parse_complete('<think>\nabc\n</think>\n\nHello')
            assert reasoning == '\nabc\n'
            assert content == '\n\nHello'
            assert tool_calls is None
        finally:
            cls.reasoning_parser_cls = old_reasoning_cls
            cls.tool_parser_cls = old_tool_cls
