# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for QwenReasoningParser covering three model behavior modes.

Scenario A – Thinking mode (Qwen3-8B, enable_thinking=True):
    Model generates ``<think>reasoning</think>\\n\\nAnswer``.

Scenario B – Non-thinking mode (Qwen3-8B, enable_thinking=False):
    Model generates plain content with no ``<think>`` tags at all.

Scenario C – Forceful Thinking (Qwen3-4B-Thinking-2507):
    ``<think>`` is injected into the prompt by the chat template, so the
    model's output starts directly with reasoning, then ``</think>``, then
    the answer.  No ``<think>`` appears in the generated output.
"""

from __future__ import annotations

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.openai.reasoning_parser.qwen_reasoning_parser import QwenReasoningParser
from lmdeploy.serve.openai.reasoning_parser.reasoning_parser import ReasoningParserManager, get_streaming_state
from lmdeploy.tokenizer import DetokenizeState, HuggingFaceTokenizer

# We use Qwen3-8B's tokenizer to simulate all the test cases.
MODEL_ID = 'Qwen/Qwen3-8B'

@pytest.fixture(scope='module')
def tokenizer():
    try:
        return HuggingFaceTokenizer(MODEL_ID)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f'Could not load tokenizer for {MODEL_ID}: {exc}')


@pytest.fixture()
def parser(tokenizer):
    return QwenReasoningParser(tokenizer)


def simulate_pipeline_chunks(
    tokenizer: HuggingFaceTokenizer,
    full_text: str,
    *,
    chunk_size: int = 1,
    skip_special_tokens: bool = True,
    spaces_between_special_tokens: bool = True,
) -> list[tuple[str, list[int]]]:
    """Split *full_text* into (delta_text, delta_token_ids) like
    ``AsyncEngine.generate``."""
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
    parser: QwenReasoningParser,
    request: object,
    chunks: list[tuple[str, list[int]]],
) -> tuple[str, str]:
    """Mirror ``api_server`` ``completion_stream_generator`` parser loop.

    Returns (accumulated_reasoning, accumulated_content).
    """
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


def _make_request(stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(model=MODEL_ID, messages=[], stream=stream)


class TestExtractReasoning:
    """Non-streaming ``extract_reasoning`` tests."""

    def test_thinking_mode(self, parser):
        """Qwen3-8B enable_thinking=True:

        <think>..reasoning..</think>answer.
        """
        full = '<think>\nBrief chain of thought.\n</think>\n\nThe answer is 42.'
        reasoning, content = parser.extract_reasoning(full, _make_request())
        assert reasoning == '\nBrief chain of thought.\n'
        assert content == '\n\nThe answer is 42.'

    def test_non_thinking_mode(self, parser):
        """Qwen3-8B enable_thinking=False: plain content, no tags."""
        full = 'The answer is 42.'
        reasoning, content = parser.extract_reasoning(full, _make_request())
        assert reasoning is None
        assert content == 'The answer is 42.'

    def test_forceful_thinking(self, parser):
        """Qwen3-4B-Thinking-2507: no <think> in output, model starts with reasoning."""
        full = '\nBrief chain of thought.\n</think>\n\nThe answer is 42.'
        reasoning, content = parser.extract_reasoning(full, _make_request())
        assert reasoning == '\nBrief chain of thought.\n'
        assert content == '\n\nThe answer is 42.'

    def test_empty_reasoning(self, parser):
        """Edge case: <think></think> with empty reasoning body."""
        full = '<think></think>\n\nThe answer is 42.'
        reasoning, content = parser.extract_reasoning(full, _make_request())
        assert reasoning is None
        assert content == '\n\nThe answer is 42.'

    def test_only_reasoning_no_answer(self, parser):
        """Edge case: reasoning present but no content after </think>."""
        full = '<think>reasoning only</think>'
        reasoning, content = parser.extract_reasoning(full, _make_request())
        assert reasoning == 'reasoning only'
        assert content is None

    def test_multiline_reasoning(self, parser):
        """Longer, multi-line reasoning body."""
        reasoning_text = (
            '\nStep 1: identify the problem.\n'
            'Step 2: solve it.\n'
            'Step 3: verify.\n'
        )
        full = f'<think>{reasoning_text}</think>\n\nFinal answer.'
        reasoning, content = parser.extract_reasoning(full, _make_request())
        assert reasoning == reasoning_text
        assert content == '\n\nFinal answer.'


class TestExtractReasoningStreaming:
    """Streaming ``extract_reasoning_streaming`` tests.

    Each test is parametrized over chunk_size to exercise both fine-grained (token-by-token) and coarse (multi-token)
    chunk boundaries.
    """

    @pytest.mark.parametrize('chunk_size', [1, 3])
    def test_thinking_mode(self, tokenizer, parser, chunk_size):
        """Qwen3-8B enable_thinking=True: streaming output matches non-
        streaming."""
        reasoning_body = '\nBrief chain of thought.\n'
        answer = 'The answer is 42.'
        full = f'<think>{reasoning_body}</think>\n\n{answer}'

        chunks = simulate_pipeline_chunks(tokenizer, full, chunk_size=chunk_size)
        request = _make_request(stream=True)
        r_stream, c_stream = run_reasoning_stream(parser, request, chunks)

        r_ns, c_ns = parser.extract_reasoning(full, _make_request())
        assert r_stream == r_ns
        assert c_stream == c_ns
        assert answer in c_stream

    @pytest.mark.parametrize('chunk_size', [1, 3])
    def test_forceful_thinking(self, tokenizer, parser, chunk_size):
        """Qwen3-4B-Thinking-2507: no <think>, streaming matches non-streaming."""
        reasoning_body = '\nBrief chain of thought.\n'
        answer = 'The answer is 42.'
        full = f'{reasoning_body}</think>\n\n{answer}'

        chunks = simulate_pipeline_chunks(tokenizer, full, chunk_size=chunk_size)
        request = _make_request(stream=True)
        r_stream, c_stream = run_reasoning_stream(parser, request, chunks)

        r_ns, c_ns = parser.extract_reasoning(full, _make_request())
        assert r_stream == r_ns
        assert c_stream == c_ns
        assert answer in c_stream

    @pytest.mark.parametrize('chunk_size', [1, 3])
    def test_non_thinking_mode(self, tokenizer, parser, chunk_size):
        """Qwen3-8B enable_thinking=False: no tags at all.

        The streaming parser has no way to know that </think> will never arrive, so it treats all text as
        reasoning_content.  The non-streaming path correctly returns it as content because it can inspect the full
        output.  This test documents the streaming behavior.
        """
        full = 'The answer is 42.'
        chunks = simulate_pipeline_chunks(tokenizer, full, chunk_size=chunk_size)
        request = _make_request(stream=True)
        r_stream, c_stream = run_reasoning_stream(parser, request, chunks)

        assert r_stream == full
        assert c_stream == ''

    @pytest.mark.parametrize('chunk_size', [1, 3])
    def test_empty_reasoning(self, tokenizer, parser, chunk_size):
        """Edge case: <think></think> with empty reasoning body."""
        answer = 'The answer is 42.'
        full = f'<think></think>\n\n{answer}'
        chunks = simulate_pipeline_chunks(tokenizer, full, chunk_size=chunk_size)
        request = _make_request(stream=True)
        r_stream, c_stream = run_reasoning_stream(parser, request, chunks)

        assert r_stream == ''
        assert answer in c_stream

    @pytest.mark.parametrize('chunk_size', [1, 3])
    def test_multiline_reasoning(self, tokenizer, parser, chunk_size):
        """Longer reasoning body, streaming matches non-streaming."""
        reasoning_text = (
            '\nStep 1: identify the problem.\n'
            'Step 2: solve it.\n'
            'Step 3: verify.\n'
        )
        answer = 'Final answer.'
        full = f'<think>{reasoning_text}</think>\n\n{answer}'
        chunks = simulate_pipeline_chunks(tokenizer, full, chunk_size=chunk_size)
        request = _make_request(stream=True)
        r_stream, c_stream = run_reasoning_stream(parser, request, chunks)

        r_ns, c_ns = parser.extract_reasoning(full, _make_request())
        assert r_stream == r_ns
        assert c_stream == c_ns
        assert answer in c_stream


class TestRegistry:

    @pytest.mark.parametrize('name', ['qwen-qwq', 'qwen3', 'intern-s1', 'deepseeek-r1'])
    def test_registered_names(self, tokenizer, name):
        """All registered aliases resolve to QwenReasoningParser."""
        cls = ReasoningParserManager.get(name)
        parser = cls(tokenizer)
        assert isinstance(parser, QwenReasoningParser)

    def test_basic_stream_round_trip(self, tokenizer):
        """Sanity check: registry-created parser works end-to-end."""
        cls = ReasoningParserManager.get('qwen3')
        parser = cls(tokenizer)
        full = f'{QwenReasoningParser.start_token}x{QwenReasoningParser.end_token}y'
        chunks = simulate_pipeline_chunks(tokenizer, full, chunk_size=2)
        request = _make_request(stream=True)
        r_stream, c_stream = run_reasoning_stream(parser, request, chunks)
        r_ns, c_ns = parser.extract_reasoning(full, _make_request())
        assert r_stream == r_ns
        assert c_stream == c_ns
