# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import pytest
import transformers
from packaging.version import Version

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.openai.reasoning_parser.deepseek_v3_reasoning_parser import DeepSeekV3ReasoningParser
from lmdeploy.serve.openai.reasoning_parser.reasoning_parser import get_streaming_state
from lmdeploy.tokenizer import DetokenizeState, HuggingFaceTokenizer

TRANSFORMERS_LT_5 = Version(transformers.__version__) < Version('5.0.0')
REQUIRES_TRANSFORMERS_LT_5 = pytest.mark.skipif(
    not TRANSFORMERS_LT_5,
    reason=f'requires transformers < 5.0, got {transformers.__version__}',
)
pytestmark = REQUIRES_TRANSFORMERS_LT_5


MODEL_ID = 'deepseek-ai/DeepSeek-V3.1'

@pytest.fixture(scope='module')
def tokenizer():
    try:
        return HuggingFaceTokenizer(MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'Could not load tokenizer for {MODEL_ID}: {exc}')


def _make_request(stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=MODEL_ID, messages=[], stream=stream)


def _build_parser(tokenizer: HuggingFaceTokenizer, *, enable_thinking: bool | None) -> DeepSeekV3ReasoningParser:
    return DeepSeekV3ReasoningParser(tokenizer, enable_thinking=enable_thinking)


def simulate_pipeline_chunks(
    tokenizer: HuggingFaceTokenizer,
    full_text: str,
    *,
    chunk_size: int = 1,
    skip_special_tokens: bool = True,
    spaces_between_special_tokens: bool = True,
) -> list[tuple[str, list[int]]]:
    all_ids = tokenizer.encode(full_text, add_bos=False, add_special_tokens=False)
    state = DetokenizeState(0)
    accumulated: list[int] = []
    chunks: list[tuple[str, list[int]]] = []
    offset = 0
    while offset < len(all_ids):
        accumulated.extend(all_ids[offset:offset + chunk_size])
        offset += chunk_size
        ids_offset_before = state.ids_offset
        delta_text, state = tokenizer.detokenize_incrementally(
            accumulated,
            state,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        delta_ids = accumulated[ids_offset_before:len(accumulated)]
        chunks.append((delta_text, delta_ids))
    return chunks


def run_reasoning_stream(
    parser: DeepSeekV3ReasoningParser,
    request: object,
    chunks: list[tuple[str, list[int]]],
) -> tuple[str, str]:
    state = get_streaming_state(request)
    reasoning_acc = ''
    content_acc = ''
    for delta_text, delta_ids in chunks:
        state.update(delta_text, delta_ids)
        delta_msg = parser.extract_reasoning_streaming(
            delta_text=delta_text or '',
            delta_token_ids=delta_ids,
            request=request,
        )
        if delta_msg is not None:
            if delta_msg.reasoning_content:
                reasoning_acc += delta_msg.reasoning_content
            if delta_msg.content is not None:
                content_acc += delta_msg.content
        state.step()
    return reasoning_acc, content_acc


class TestExtractReasoning:

    def test_enable_thinking_true(self, tokenizer):
        parser = _build_parser(tokenizer, enable_thinking=True)
        full = '\nBrief chain of thought.\n</think>\n\nThe answer is 42.'
        reasoning, content = parser.extract_reasoning(full, _make_request())
        assert reasoning == '\nBrief chain of thought.\n'
        assert content == '\n\nThe answer is 42.'

    def test_enable_thinking_none(self, tokenizer):
        parser = _build_parser(tokenizer, enable_thinking=None)
        full = 'The answer is 42.'
        reasoning, content = parser.extract_reasoning(full, _make_request())
        assert reasoning is None
        assert content == full


class TestExtractReasoningStreaming:

    @pytest.mark.parametrize('chunk_size', [1, 3])
    def test_enable_thinking_true(self, tokenizer, chunk_size):
        parser = _build_parser(tokenizer, enable_thinking=True)
        full = '\nBrief chain of thought.\n</think>\n\nThe answer is 42.'
        chunks = simulate_pipeline_chunks(tokenizer, full, chunk_size=chunk_size)
        r_stream, c_stream = run_reasoning_stream(parser, _make_request(stream=True), chunks)
        r_ns, c_ns = parser.extract_reasoning(full, _make_request())
        assert r_stream == r_ns
        assert c_stream == c_ns

    @pytest.mark.parametrize('chunk_size', [1, 3])
    def test_enable_thinking_none(self, tokenizer, chunk_size):
        parser = _build_parser(tokenizer, enable_thinking=False)
        full = 'The answer is 42.'
        chunks = simulate_pipeline_chunks(tokenizer, full, chunk_size=chunk_size)
        r_stream, c_stream = run_reasoning_stream(parser, _make_request(stream=True), chunks)
        assert r_stream == ''
        assert c_stream == full
