import pytest
from transformers import AutoTokenizer

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers import ResponseParserManager
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager

MODEL_ID = 'deepseek-ai/DeepSeek-V3.1'

@pytest.fixture(scope='module')
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'Could not load tokenizer for {MODEL_ID}: {exc}')


def _make_parser(enable_thinking, tokenizer):
    cls = ResponseParserManager.get('default')
    cls.reasoning_parser_cls = ReasoningParserManager.get('deepseek-v3')
    cls.tool_parser_cls = None
    request = ChatCompletionRequest(
        model=MODEL_ID,
        messages=[],
        stream=True,
        chat_template_kwargs={'enable_thinking': enable_thinking},
    )
    return cls(request=request, tokenizer=tokenizer)


class TestDeepSeekV3ReasoningParser:

    def test_enable_thinking_none(self, tokenizer):
        parser = _make_parser(enable_thinking=None, tokenizer=tokenizer)
        delta_msg, tool_emitted = parser.stream_chunk(delta_text='hello', delta_token_ids=[])
        assert tool_emitted is False
        assert delta_msg is not None
        assert delta_msg.content == 'hello'
        assert delta_msg.reasoning_content is None

    def test_enable_thinking_true(self, tokenizer):
        parser = _make_parser(enable_thinking=True, tokenizer=tokenizer)
        delta_msg, tool_emitted = parser.stream_chunk(delta_text='hello', delta_token_ids=[])
        assert tool_emitted is False
        assert delta_msg is not None
        assert delta_msg.content is None
        assert delta_msg.reasoning_content == 'hello'
