import pytest
from utils.tool_reasoning_definitions import THINK_START_TOKEN

from .conftest import (
    _DEFAULT_VOCAB,
    _apply_parser_marks,
    _get_deepseek_parser_cls,
    _make_mock_tokenizer,
    _run_streaming_extraction,
)

# ===================================================================
# DeepSeek-specific init tests
# ===================================================================


@_apply_parser_marks
@pytest.mark.deepseek_r1_parser
class TestDeepSeekR1ParserInit:
    """DeepSeek-specific initialisation tests."""

    def test_missing_vocab_tokens(self):
        """DeepSeek raises RuntimeError when vocab lacks think tokens."""
        tok = _make_mock_tokenizer(vocab={'unrelated': 999})
        with pytest.raises(RuntimeError, match='(?i)think.*token'):
            _get_deepseek_parser_cls()(tok)

    def test_token_ids_not_none(self):
        """DeepSeek exposes think_start_token_id / think_end_token_id."""
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        assert parser.think_start_token_id is not None
        assert parser.think_end_token_id is not None
        assert parser.think_start_token_id == 100
        assert parser.think_end_token_id == 101


# ===================================================================
# DeepSeek-specific streaming tests
#
# DeepSeek does **not** strip <think> from the delta text.  When
# <think> is embedded in a delta together with other text, the tag
# leaks into reasoning.  Qwen strips it — see Qwen-specific tests.
# ===================================================================


@_apply_parser_marks
@pytest.mark.deepseek_r1_parser
class TestDeepSeekR1ParserStreaming:
    """DeepSeek-specific streaming tests."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tok = _make_mock_tokenizer(_DEFAULT_VOCAB)
        self.parser = _get_deepseek_parser_cls()(self.tok)

    def test_streaming_multiple_end_tokens(self):
        """Multiple ``</think>`` in streaming."""
        deltas = ['<think>', 'First', '</think>', 'Middle', '</think>', 'Last']
        reasoning, content = _run_streaming_extraction(self.parser, deltas, _DEFAULT_VOCAB)
        assert reasoning is not None and 'First' in reasoning
        assert content is not None and 'Middle' in content

    def test_embedded_think_in_delta_leaks(self):
        """``<think>`` embedded in delta is NOT stripped by DeepSeek."""
        deltas = ['<think>I am reasoning', ' step by step', '</think>', 'The answer.']
        reasoning, content = _run_streaming_extraction(self.parser, deltas, _DEFAULT_VOCAB)
        assert reasoning is not None
        # DeepSeek does NOT strip <think> from embedded delta
        assert THINK_START_TOKEN in reasoning
        assert content is not None
        assert 'The answer.' in content

    def test_embedded_think_no_end_leaks(self):
        """``<think>`` embedded in delta without ``</think>`` — tag leaks."""
        deltas = ['<think>I am reasoning about', ' a complex problem...']
        reasoning, content = _run_streaming_extraction(self.parser, deltas, _DEFAULT_VOCAB)
        assert reasoning is not None
        assert THINK_START_TOKEN in reasoning
        assert content is None


# ===================================================================
# DeepSeek-specific multi-token delta tests
#
# When <think> is grouped with text in one delta, DeepSeek does NOT
# strip the tag.  This contrasts with Qwen.
# ===================================================================


DEEPSEEK_MULTI_TOKEN_DELTA_CASES = [
    pytest.param(
        ['<think>This is a reasoning section', '</think>', 'This is the rest'],
        '<think>This is a reasoning section',
        'This is the rest',
        id='start_token_grouped_with_text_leaks',
    ),
]


@_apply_parser_marks
@pytest.mark.deepseek_r1_parser
class TestDeepSeekR1MultiTokenDeltas:
    """Multi-token streaming delta tests specific to DeepSeek."""

    @pytest.mark.parametrize(
        'deltas, expected_reasoning, expected_content',
        DEEPSEEK_MULTI_TOKEN_DELTA_CASES,
    )
    def test_multi_token_deltas(self, deltas, expected_reasoning,
                                expected_content):
        tok = _make_mock_tokenizer(_DEFAULT_VOCAB)
        parser = _get_deepseek_parser_cls()(tok)
        reasoning, content = _run_streaming_extraction(
            parser, deltas, _DEFAULT_VOCAB,
        )
        assert reasoning == expected_reasoning
        assert (content or None) == expected_content
