import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaToolCall
from lmdeploy.serve.openai.reasoning_parser.qwen_reasoning_parser import QwenReasoningParser
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
    # Configure ResponseParser to use Qwen3 reasoning and tool parsers.
    ResponseParser.reasoning_parser_cls = QwenReasoningParser
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


# Reference streaming sequence based on the attached example:
# - First: reasoning tokens (Chinese text explaining the need to call get_current_temperature).
# - Then: </think> and plain content (\n\n).
# - Finally: the <tool_call> section is streamed token-by-token, following the real model output:
#   <tool_call>, \n, <, function, =get, _current, _temperature, ... </tool_call>.
#
# For tool_call, we feed the raw token stream into ResponseParser.stream_chunk
# and rely on the ground-truth deltas to specify exactly which chunks should
# emit tool_calls and what those deltas should look like.
REFERENCE_CHUNKS = [
    # (delta_text, expected_delta_msg, expected_reasoning, expected_content,
    #  expected_tool_emitted, expected_function_name,
    #  expected_function_arguments, expected_type)
    ('', True, None, '', False, None, None, None),
    ('用户', True, '用户', None, False, None, None, None),
    ('询问', True, '询问', None, False, None, None, None),
    ('北京', True, '北京', None, False, None, None, None),
    ('今天的', True, '今天的', None, False, None, None, None),
    ('天气', True, '天气', None, False, None, None, None),
    ('情况', True, '情况', None, False, None, None, None),
    ('。', True, '。', None, False, None, None, None),
    ('我', True, '我', None, False, None, None, None),
    ('需要使用', True, '需要使用', None, False, None, None, None),
    ('get', True, 'get', None, False, None, None, None),
    ('_weather', True, '_weather', None, False, None, None, None),
    ('工具', True, '工具', None, False, None, None, None),
    ('来获取', True, '来获取', None, False, None, None, None),
    ('北京的', True, '北京的', None, False, None, None, None),
    ('天气', True, '天气', None, False, None, None, None),
    ('信息', True, '信息', None, False, None, None, None),
    ('。', True, '。', None, False, None, None, None),
    ('\n\n', True, '\n\n', None, False, None, None, None),
    ('参数', True, '参数', None, False, None, None, None),
    ('要求', True, '要求', None, False, None, None, None),
    ('：', True, '：', None, False, None, None, None),
    ('\n', True, '\n', None, False, None, None, None),
    ('-', True, '-', None, False, None, None, None),
    (' location', True, ' location', None, False, None, None, None),
    (':', True, ':', None, False, None, None, None),
    (' ', True, ' ', None, False, None, None, None),
    ('必需', True, '必需', None, False, None, None, None),
    ('参数', True, '参数', None, False, None, None, None),
    ('，', True, '，', None, False, None, None, None),
    ('用户', True, '用户', None, False, None, None, None),
    ('问', True, '问', None, False, None, None, None),
    ('的是', True, '的是', None, False, None, None, None),
    ('"', True, '"', None, False, None, None, None),
    ('北京', True, '北京', None, False, None, None, None),
    ('"', True, '"', None, False, None, None, None),
    ('，', True, '，', None, False, None, None, None),
    ('所以', True, '所以', None, False, None, None, None),
    ('location', True, 'location', None, False, None, None, None),
    ('应该是', True, '应该是', None, False, None, None, None),
    ('"', True, '"', None, False, None, None, None),
    ('北京', True, '北京', None, False, None, None, None),
    ('"', True, '"', None, False, None, None, None),
    ('\n', True, '\n', None, False, None, None, None),
    ('-', True, '-', None, False, None, None, None),
    (' unit', True, ' unit', None, False, None, None, None),
    (':', True, ':', None, False, None, None, None),
    (' ', True, ' ', None, False, None, None, None),
    ('可选', True, '可选', None, False, None, None, None),
    ('参数', True, '参数', None, False, None, None, None),
    ('，', True, '，', None, False, None, None, None),
    ('用户', True, '用户', None, False, None, None, None),
    ('没有', True, '没有', None, False, None, None, None),
    ('特别', True, '特别', None, False, None, None, None),
    ('指定', True, '指定', None, False, None, None, None),
    ('，', True, '，', None, False, None, None, None),
    ('我可以', True, '我可以', None, False, None, None, None),
    ('不', True, '不', None, False, None, None, None),
    ('填', True, '填', None, False, None, None, None),
    ('或者', True, '或者', None, False, None, None, None),
    ('用', True, '用', None, False, None, None, None),
    ('默认', True, '默认', None, False, None, None, None),
    ('值', True, '值', None, False, None, None, None),
    ('\n\n', True, '\n\n', None, False, None, None, None),
    ('我只', True, '我只', None, False, None, None, None),
    ('需要提供', True, '需要提供', None, False, None, None, None),
    ('location', True, 'location', None, False, None, None, None),
    ('参数', True, '参数', None, False, None, None, None),
    ('即可', True, '即可', None, False, None, None, None),
    ('。', True, '。', None, False, None, None, None),
    ('\n', True, '\n', None, False, None, None, None),
    ('</think>', False, None, None, False, None, None, None),
    ('\n\n', True, None, '\n\n', False, None, None, None),
    # (delta_text, expected_delta_msg,expected_reasoning, expected_content,
    #  expected_tool_emitted, expected_function_name,
    #  expected_function_arguments, expected_type)
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
    (' "', True, None, None, True, None, '{"location": "', None),
    ('北京', True, None, None, True, None, '北京', None),
    ('",', False, None, None, True, None, '",', None),
    (' "', False, None, None, False, None, None, None),
    ('unit', False, None, None, False, None, None, None),
    ('":', False, None, None, False, None, None, None),
    (' "', False, None, None, False, None, None, None),
    ('celsius', True, None, None, True, None, 'celsius', None),
    ('"}}\n', True, None, None, True, None, '"}', None),
    ('</tool_call>', False, None, None, False, None, None, None),
    ('', True, None, '', False, None, None, None),
]


class TestQwenResponseParserStreaming:
    """Integration test for ResponseParser.stream_chunk with Qwen3 parsers."""

    @staticmethod
    def _encode_ids(tokenizer, text: str) -> list[int]:
        return tokenizer.encode(text, add_bos=False, add_special_tokens=False)

    def test_stream_chunk_matches_reference(self, tokenizer, response_parser):
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
             exp_type) in REFERENCE_CHUNKS:
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

        # 3) chunk carries reasoning end + normal content
        delta_msg, tool_emitted = _call('The answer is 9 </think> OK. The')
        assert delta_msg is not None
        assert delta_msg.reasoning_content == 'The answer is 9 '
        assert delta_msg.content == ' OK. The'
        assert tool_emitted is False

        # 4) chunk carries stray think-close + content + tool-open
        delta_msg, tool_emitted = _call('fine. </think> \n\n <tool_call> ')
        assert delta_msg is not None
        # Stray closing tag after reasoning has ended is treated as plain content.
        assert delta_msg.reasoning_content is None
        assert delta_msg.content == 'fine. </think> \n\n '
        assert tool_emitted is False
