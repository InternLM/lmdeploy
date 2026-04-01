import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaToolCall
from lmdeploy.serve.openai.reasoning_parser.qwen_reasoning_parser import QwenReasoningParser
from lmdeploy.serve.openai.response_parser import ResponseParser
from lmdeploy.serve.openai.tool_parser.qwen3coder_tool_parser import Qwen3CoderToolParser
from lmdeploy.tokenizer import HuggingFaceTokenizer

MODEL_ID = 'Qwen/Qwen3.5-35B-A3B'


@pytest.fixture(scope='module')
def tokenizer():
    try:
        return HuggingFaceTokenizer(MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'Could not load tokenizer for {MODEL_ID}: {exc}')


@pytest.fixture()
def response_parser(tokenizer):
    # Configure ResponseParser to use Qwen3 reasoning parser and Qwen3.5 Coder tool parser.
    ResponseParser.reasoning_parser_cls = QwenReasoningParser
    ResponseParser.tool_parser_cls = Qwen3CoderToolParser

    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=True,
        tool_choice='auto',
        chat_template_kwargs={'enable_thinking': True},
    )
    return ResponseParser(request=request, tokenizer=tokenizer)


# NOTE: This REFERENCE_CHUNKS is currently a direct copy of the Qwen3 test.
# The user will later adjust it to match the actual Qwen3.5 XML-style ground
# truth stream. The structure is kept identical so the same assertions apply.
REFERENCE_CHUNKS = [
    # (delta_text, expected_reasoning, expected_content,
    #  expected_tool_emitted, expected_function_name,
    #  expected_function_arguments, expected_type)
    ('用户', '用户', None, False, None, None, None),
    ('询问', '询问', None, False, None, None, None),
    ('北京的', '北京的', None, False, None, None, None),
    ('天气', '天气', None, False, None, None, None),
    ('情况', '情况', None, False, None, None, None),
    ('。', '。', None, False, None, None, None),
    ('我', '我', None, False, None, None, None),
    ('需要使用', '需要使用', None, False, None, None, None),
    ('get', 'get', None, False, None, None, None),
    ('_current', '_current', None, False, None, None, None),
    ('_temperature', '_temperature', None, False, None, None, None),
    ('函数', '函数', None, False, None, None, None),
    ('来获取', '来获取', None, False, None, None, None),
    ('北京的', '北京的', None, False, None, None, None),
    ('当前', '当前', None, False, None, None, None),
    ('温度', '温度', None, False, None, None, None),
    ('。', '。', None, False, None, None, None),
    ('根据', '根据', None, False, None, None, None),
    ('函数', '函数', None, False, None, None, None),
    ('要求', '要求', None, False, None, None, None),
    ('，', '，', None, False, None, None, None),
    ('location', 'location', None, False, None, None, None),
    ('参数', '参数', None, False, None, None, None),
    ('需要', '需要', None, False, None, None, None),
    ('是', '是', None, False, None, None, None),
    ('"', '"', None, False, None, None, None),
    ('City', 'City', None, False, None, None, None),
    (',', ',', None, False, None, None, None),
    (' State', ' State', None, False, None, None, None),
    (',', ',', None, False, None, None, None),
    (' Country', ' Country', None, False, None, None, None),
    ('"', '"', None, False, None, None, None),
    ('的', '的', None, False, None, None, None),
    ('格式', '格式', None, False, None, None, None),
    ('，', '，', None, False, None, None, None),
    ('所以', '所以', None, False, None, None, None),
    ('北京', '北京', None, False, None, None, None),
    ('应该', '应该', None, False, None, None, None),
    ('写成', '写成', None, False, None, None, None),
    ('"', '"', None, False, None, None, None),
    ('Be', 'Be', None, False, None, None, None),
    ('ijing', 'ijing', None, False, None, None, None),
    (',', ',', None, False, None, None, None),
    (' China', ' China', None, False, None, None, None),
    ('"', '"', None, False, None, None, None),
    ('。', '。', None, False, None, None, None),
    ('unit', 'unit', None, False, None, None, None),
    ('参数', '参数', None, False, None, None, None),
    ('是', '是', None, False, None, None, None),
    ('可选', '可选', None, False, None, None, None),
    ('的', '的', None, False, None, None, None),
    ('，', '，', None, False, None, None, None),
    ('默认', '默认', None, False, None, None, None),
    ('是', '是', None, False, None, None, None),
    ('c', 'c', None, False, None, None, None),
    ('elsius', 'elsius', None, False, None, None, None),
    ('，', '，', None, False, None, None, None),
    ('我不', '我不', None, False, None, None, None),
    ('需要', '需要', None, False, None, None, None),
    ('特别', '特别', None, False, None, None, None),
    ('指定', '指定', None, False, None, None, None),
    ('。', '。', None, False, None, None, None),
    ('\n', '\n', None, False, None, None, None),
    ('</think>', None, None, False, None, None, None),
    ('\n\n', None, '\n\n', False, None, None, None),
    # Tool call section: placeholder; will be updated to match Qwen3.5 XML-style.
    ('<tool_call>', None, None, False, None, None, None),
    ('\n', None, None, False, None, None, None),
    ('<', None, None, False, None, None, None),
    ('function', None, None, False, None, None, None),
    ('=get', None, None, False, None, None, None),
    ('_current', None, None, False, None, None, None),
    ('_temperature', None, None, False, None, None, None),
    ('>', None, None, False, None, None, None),
    ('\n', None, None, False, None, None, None),
    ('<', None, None, False, None, None, None),
    ('parameter', None, None, False, None, None, None),
    ('=location', None, None, False, None, None, None),
    ('>', None, None, False, None, None, None),
    ('\n', None, None, False, None, None, None),
    ('Be', None, None, False, None, None, None),
    ('ijing', None, None, False, None, None, None),
    (',', None, None, False, None, None, None),
    (' China', None, None, False, None, None, None),
    ('\n', None, None, False, None, None, None),
    ('</', None, None, False, None, None, None),
    ('parameter', None, None, False, None, None, None),
    ('>', None, None, False, None, None, None),
    ('\n', None, None, False, None, None, None),
    ('</', None, None, False, None, None, None),
    ('function', None, None, False, None, None, None),
    ('>', None, None, False, None, None, None),
    ('\n', None, None, False, None, None, None),
    ('</tool_call>', None, None, False, None, None, None),
    ('', None, None, False, None, None, None),
]


class TestQwen3_5ResponseParserStreaming:
    """Integration test for ResponseParser.stream_chunk with Qwen3.5 Coder
    parsers."""

    @staticmethod
    def _encode_ids(tokenizer, text: str) -> list[int]:
        return tokenizer.encode(text, add_bos=False, add_special_tokens=False)

    def test_stream_chunk_matches_reference(self, tokenizer, response_parser):
        """Feed the real streaming sequence into ResponseParser.stream_chunk
        and verify each parsed chunk.

        Expectations for tool_calls will be refined once the Qwen3.5 ground-truth stream is finalized.
        """

        for (delta_text, exp_reasoning, exp_content, exp_tool_emitted,
             exp_function_name, exp_function_arguments,
             exp_type) in REFERENCE_CHUNKS:
            delta_ids = self._encode_ids(tokenizer, delta_text)
            delta_msg, tool_emitted = response_parser.stream_chunk(
                delta_text=delta_text,
                delta_token_ids=delta_ids,
            )

            assert delta_msg.reasoning_content == exp_reasoning
            if exp_content is not None:
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
