import pytest
from utils.tool_reasoning_definitions import THINK_START_TOKEN

from .conftest import (
    _DEFAULT_VOCAB,
    _apply_parser_marks,
    _get_qwen_parser_cls,
    _make_mock_tokenizer,
    _run_streaming_extraction,
)

# ===================================================================
# Qwen-specific init tests
# ===================================================================


@_apply_parser_marks
@pytest.mark.qwenqwq_parser
class TestQwenQwQParserInit:
    """Qwen-specific initialisation tests."""

    def test_missing_vocab_raises(self):
        """Qwen must raise RuntimeError when vocab lacks think tokens."""
        tok = _make_mock_tokenizer(vocab={'unrelated': 999})
        with pytest.raises(RuntimeError):
            _get_qwen_parser_cls()(tok)

    def test_token_ids_accessible(self):
        """Qwen must expose think-token IDs."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        has_start = (getattr(parser, 'think_start_token_id', None) is not None
                     or getattr(parser, 'start_token_id', None) is not None)
        has_end = (getattr(parser, 'think_end_token_id', None) is not None
                   or getattr(parser, 'end_token_id', None) is not None)
        assert has_start, 'parser should expose start token ID'
        assert has_end, 'parser should expose end token ID'


# ===================================================================
# Qwen-specific streaming: <think> is stripped from delta
#
# Qwen strips <think> from delta_text before processing, so the tag
# never leaks into reasoning.  DeepSeek does *not* strip — see the
# DeepSeek-specific tests.
# ===================================================================


@_apply_parser_marks
@pytest.mark.qwenqwq_parser
class TestQwenQwQParserStreaming:
    """Qwen-specific streaming tests."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tok = _make_mock_tokenizer(_DEFAULT_VOCAB)
        self.parser = _get_qwen_parser_cls()(self.tok)

    def test_embedded_think_in_delta_stripped(self):
        """``<think>`` embedded in a delta with other text must be stripped."""
        deltas = ['<think>I am reasoning', ' step by step', '</think>', 'The answer.']
        reasoning, content = _run_streaming_extraction(self.parser, deltas, _DEFAULT_VOCAB)
        assert reasoning is not None
        assert THINK_START_TOKEN not in reasoning
        assert 'I am reasoning step by step' in reasoning
        assert content is not None
        assert 'The answer.' in content

    def test_embedded_think_no_end_stripped(self):
        """``<think>`` in delta, output truncated (no ``</think>``) — Qwen
        strips the tag from reasoning."""
        deltas = ['<think>I am reasoning about', ' a complex problem...']
        reasoning, content = _run_streaming_extraction(self.parser, deltas, _DEFAULT_VOCAB)
        assert reasoning is not None
        assert THINK_START_TOKEN not in reasoning
        assert 'I am reasoning about a complex problem...' in reasoning
        assert content is None


# ===================================================================
# Qwen-specific multi-token delta tests
#
# When <think> is grouped with text in one delta, Qwen strips the
# tag.  DeepSeek does not — so this case is Qwen-specific.
# ===================================================================


QWEN_MULTI_TOKEN_DELTA_CASES = [
    pytest.param(
        ['<think>This is a reasoning section', '</think>', 'This is the rest'],
        'This is a reasoning section',
        'This is the rest',
        id='start_token_grouped_with_text',
    ),
]


@_apply_parser_marks
@pytest.mark.qwenqwq_parser
class TestQwenQwQMultiTokenDeltas:
    """Multi-token streaming delta tests specific to Qwen."""

    @pytest.mark.parametrize(
        'deltas, expected_reasoning, expected_content',
        QWEN_MULTI_TOKEN_DELTA_CASES,
    )
    def test_multi_token_deltas(self, deltas, expected_reasoning,
                                expected_content):
        tok = _make_mock_tokenizer(_DEFAULT_VOCAB)
        parser = _get_qwen_parser_cls()(tok)
        reasoning, content = _run_streaming_extraction(
            parser, deltas, _DEFAULT_VOCAB,
        )
        assert reasoning == expected_reasoning
        assert (content or None) == expected_content
