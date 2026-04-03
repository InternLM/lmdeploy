from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.openai.response_parser import ResponseParser


def _make_parser(enable_thinking):
    from lmdeploy.serve.openai.reasoning_parser.deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser

    old_reasoning_cls = ResponseParser.reasoning_parser_cls
    old_tool_cls = ResponseParser.tool_parser_cls
    ResponseParser.reasoning_parser_cls = DeepSeekV3ReasoningParser
    ResponseParser.tool_parser_cls = None
    request = ChatCompletionRequest(
        model='deepseek-v3',
        messages=[],
        stream=True,
        chat_template_kwargs={'enable_thinking': enable_thinking},
    )
    parser = ResponseParser(request=request, tokenizer=object())
    return parser, old_reasoning_cls, old_tool_cls


def test_deepseek_v3_starts_plain_when_enable_thinking_none():
    parser, old_reasoning_cls, old_tool_cls = _make_parser(enable_thinking=None)
    try:
        delta_msg, tool_emitted = parser.stream_chunk(delta_text='hello', delta_token_ids=[])
        assert tool_emitted is False
        assert delta_msg is not None
        assert delta_msg.content == 'hello'
        assert delta_msg.reasoning_content is None
    finally:
        ResponseParser.reasoning_parser_cls = old_reasoning_cls
        ResponseParser.tool_parser_cls = old_tool_cls


def test_deepseek_v3_starts_reasoning_when_enable_thinking_true():
    parser, old_reasoning_cls, old_tool_cls = _make_parser(enable_thinking=True)
    try:
        delta_msg, tool_emitted = parser.stream_chunk(delta_text='hello', delta_token_ids=[])
        assert tool_emitted is False
        assert delta_msg is not None
        assert delta_msg.content is None
        assert delta_msg.reasoning_content == 'hello'
    finally:
        ResponseParser.reasoning_parser_cls = old_reasoning_cls
        ResponseParser.tool_parser_cls = old_tool_cls
