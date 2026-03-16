import pytest
from utils.tool_reasoning_definitions import (
    REASONING_PARSER_NAMES,
    THINK_END_TOKEN,
    THINK_START_TOKEN,
)

from .conftest import (
    _apply_parser_marks,
    _BOTH_PARSERS,
    _DEFAULT_VOCAB,
    _get_deepseek_parser_cls,
    _get_parser_manager,
    _get_qwen_parser_cls,
    _make_mock_request,
    _make_mock_tokenizer,
    _run_extraction,
    _run_streaming_extraction,
)


# ===================================================================
# Test case data — each dict has {output, reasoning, content}
#
# Mirrors the data-driven style of vLLM's reasoning parser tests.
# ===================================================================

# --- Both <think> and </think> present ---

REASONING_WITH_THINK = {
    'output': '<think>This is a reasoning section</think>This is the rest',
    'reasoning': 'This is a reasoning section',
    'content': 'This is the rest',
}

# --- Only </think> present (Qwen3.5-style: <think> is in the prompt) ---

SIMPLE_REASONING = {
    'output': 'This is a reasoning section</think>This is the rest',
    'reasoning': 'This is a reasoning section',
    'content': 'This is the rest',
}

# --- Complete reasoning, nothing after </think> ---

COMPLETE_REASONING = {
    'output': '<think>This is a reasoning section</think>',
    'reasoning': 'This is a reasoning section',
    'content': None,
}

COMPLETE_REASONING_NO_START = {
    'output': 'This is a reasoning section</think>',
    'reasoning': 'This is a reasoning section',
    'content': None,
}

# --- Multiple lines ---

MULTIPLE_LINES = {
    'output': 'This\nThat</think>This is the rest\nThat',
    'reasoning': 'This\nThat',
    'content': 'This is the rest\nThat',
}

MULTIPLE_LINES_WITH_THINK = {
    'output': '<think>This\nThat</think>This is the rest\nThat',
    'reasoning': 'This\nThat',
    'content': 'This is the rest\nThat',
}

# --- Truncated: <think> present but no </think> ---

THINK_NO_END = {
    'output': '<think>This is a reasoning section',
    'reasoning': 'This is a reasoning section',
    'content': None,
}

# --- No think tokens at all (thinking enabled → all is reasoning) ---

NO_THINK_TOKENS = {
    'output': 'This is content',
    'reasoning': 'This is content',
    'content': None,
}

# --- Malformed similar tokens (<thinking> is NOT <think>) ---

MALFORMED_SIMILAR_TOKENS = {
    'output': '<thinking>Not real tags</thinking>Content',
    'reasoning': '<thinking>Not real tags</thinking>Content',
    'content': None,
}

# --- Newlines inside reasoning preserved ---

NEWLINES_IN_REASONING = {
    'output': '<think>\nStep 1: think\nStep 2: conclude\n</think>Done.',
    'reasoning': '\nStep 1: think\nStep 2: conclude\n',
    'content': 'Done.',
}

# --- Newlines between </think> and content ---

NEWLINES_AFTER_THINK = {
    'output': '<think>Step 1: 37*43=1591</think>\n\nThe answer is 1591.',
    'reasoning': 'Step 1: 37*43=1591',
    'content': '\n\nThe answer is 1591.',
}

SINGLE_NEWLINE_AFTER_THINK = {
    'output': '<think>reasoning</think>\nThe answer.',
    'reasoning': 'reasoning',
    'content': '\nThe answer.',
}

# --- Shortest reasoning: empty string before </think> ---

SHORTEST_REASONING_NS = {
    'output': '</think>This is the rest',
    'reasoning': '',
    'content': 'This is the rest',
}

SHORTEST_REASONING_STREAMING = {
    'output': '</think>This is the rest',
    'reasoning': None,
    'content': 'This is the rest',
}

# --- Empty output ---

EMPTY = {
    'output': '',
    'reasoning': '',
    'content': None,
}

EMPTY_STREAMING = {
    'output': '',
    'reasoning': None,
    'content': None,
}

# --- Empty reasoning: <think></think>content ---

EMPTY_REASONING_TAGS = {
    'output': '<think></think>Direct answer.',
    'reasoning': '',
    'content': 'Direct answer.',
}

EMPTY_REASONING_TAGS_STREAMING = {
    'output': '<think></think>Direct answer.',
    'reasoning': None,
    'content': 'Direct answer.',
}

# --- Newline prefix before <think> ---

NEW_LINE = {
    'output': '\n<think>This is a reasoning section</think>\nThis is the rest',
    'reasoning': 'This is a reasoning section',
    'content': '\nThis is the rest',
}

NEW_LINE_STREAMING = {
    'output': '\n<think>This is a reasoning section</think>\nThis is the rest',
    'reasoning': '\nThis is a reasoning section',
    'content': '\nThis is the rest',
}


# ===================================================================
# Parametrized TEST_CASES — (streaming, param_dict)
# ===================================================================

TEST_CASES = [
    # --- Streaming == non-streaming ---
    pytest.param(False, REASONING_WITH_THINK, id='reasoning_with_think'),
    pytest.param(True, REASONING_WITH_THINK, id='reasoning_with_think_streaming'),
    pytest.param(False, SIMPLE_REASONING, id='simple_reasoning'),
    pytest.param(True, SIMPLE_REASONING, id='simple_reasoning_streaming'),
    pytest.param(False, COMPLETE_REASONING, id='complete_reasoning'),
    pytest.param(True, COMPLETE_REASONING, id='complete_reasoning_streaming'),
    pytest.param(False, COMPLETE_REASONING_NO_START, id='complete_reasoning_no_start'),
    pytest.param(True, COMPLETE_REASONING_NO_START, id='complete_reasoning_no_start_streaming'),
    pytest.param(False, MULTIPLE_LINES, id='multiple_lines'),
    pytest.param(True, MULTIPLE_LINES, id='multiple_lines_streaming'),
    pytest.param(False, MULTIPLE_LINES_WITH_THINK, id='multiple_lines_with_think'),
    pytest.param(True, MULTIPLE_LINES_WITH_THINK, id='multiple_lines_with_think_streaming'),
    pytest.param(False, THINK_NO_END, id='think_no_end'),
    pytest.param(True, THINK_NO_END, id='think_no_end_streaming'),
    pytest.param(False, NO_THINK_TOKENS, id='no_think_tokens'),
    pytest.param(True, NO_THINK_TOKENS, id='no_think_tokens_streaming'),
    pytest.param(False, MALFORMED_SIMILAR_TOKENS, id='malformed_similar_tokens'),
    pytest.param(True, MALFORMED_SIMILAR_TOKENS, id='malformed_similar_tokens_streaming'),
    pytest.param(False, NEWLINES_IN_REASONING, id='newlines_in_reasoning'),
    pytest.param(True, NEWLINES_IN_REASONING, id='newlines_in_reasoning_streaming'),
    pytest.param(False, NEWLINES_AFTER_THINK, id='newlines_after_think'),
    pytest.param(True, NEWLINES_AFTER_THINK, id='newlines_after_think_streaming'),
    pytest.param(False, SINGLE_NEWLINE_AFTER_THINK, id='single_newline_after_think'),
    pytest.param(True, SINGLE_NEWLINE_AFTER_THINK, id='single_newline_after_think_streaming'),
    # --- Streaming != non-streaming ---
    pytest.param(False, SHORTEST_REASONING_NS, id='shortest_reasoning'),
    pytest.param(True, SHORTEST_REASONING_STREAMING, id='shortest_reasoning_streaming'),
    pytest.param(False, EMPTY, id='empty'),
    pytest.param(True, EMPTY_STREAMING, id='empty_streaming'),
    pytest.param(False, EMPTY_REASONING_TAGS, id='empty_reasoning_tags'),
    pytest.param(True, EMPTY_REASONING_TAGS_STREAMING, id='empty_reasoning_tags_streaming'),
    pytest.param(False, NEW_LINE, id='new_line'),
    pytest.param(True, NEW_LINE_STREAMING, id='new_line_streaming'),
]


# ===================================================================
# Multi-token delta test cases (streaming only, common to both parsers)
# ===================================================================

MULTI_TOKEN_DELTA_CASES = [
    pytest.param(
        ['reasoning section', '</think>This is the rest'],
        'reasoning section',
        'This is the rest',
        id='end_token_grouped_with_content',
    ),
    pytest.param(
        ['<think>reasoning</think>'],
        'reasoning',
        None,
        id='start_and_end_in_one_delta_no_content',
    ),
    pytest.param(
        ['<think>short reasoning</think>answer'],
        'short reasoning',
        'answer',
        id='start_and_end_in_one_delta_with_content',
    ),
    pytest.param(
        ['reasoning section', '</think>content'],
        'reasoning section',
        'content',
        id='no_start_end_grouped_with_content',
    ),
]


# ===================================================================
# Core data-driven extraction tests
# ===================================================================


@_apply_parser_marks
class TestReasoningExtraction:
    """Core extraction tests — data-driven, parametrized over both parsers
    and streaming/non-streaming modes."""

    @_BOTH_PARSERS
    @pytest.mark.parametrize('streaming, param_dict', TEST_CASES)
    def test_reasoning(self, parser_factory, streaming, param_dict):
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        reasoning, content = _run_extraction(
            parser, param_dict['output'], streaming=streaming,
        )
        assert reasoning == param_dict['reasoning']
        assert content == param_dict['content']


@_apply_parser_marks
class TestMultiTokenDeltas:
    """Multi-token streaming delta tests — common to both parsers."""

    @_BOTH_PARSERS
    @pytest.mark.parametrize(
        'deltas, expected_reasoning, expected_content',
        MULTI_TOKEN_DELTA_CASES,
    )
    def test_multi_token_deltas(self, parser_factory, deltas,
                                expected_reasoning, expected_content):
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        reasoning, content = _run_streaming_extraction(
            parser, deltas, _DEFAULT_VOCAB,
        )
        assert reasoning == expected_reasoning
        assert (content or None) == expected_content


# ===================================================================
# Parser manager
# ===================================================================


@_apply_parser_marks
class TestReasoningParserManager:
    """Verify that all parsers are correctly registered."""

    def test_all_parser_names_registered(self):
        mgr = _get_parser_manager()
        for name in REASONING_PARSER_NAMES:
            cls = mgr.get(name)
            assert cls is not None, f'Parser "{name}" not found'

    @pytest.mark.parametrize('name,expected_cls_name', [
        ('deepseek-r1', 'DeepSeekR1ReasoningParser'),
        ('qwen-qwq', 'QwenQwQReasoningParser'),
        ('intern-s1', 'QwenQwQReasoningParser'),
    ])
    def test_specific_parser_class(self, name, expected_cls_name):
        mgr = _get_parser_manager()
        cls = mgr.get(name)
        assert cls is not None
        assert cls.__name__ == expected_cls_name

    def test_unknown_parser_returns_none(self):
        mgr = _get_parser_manager()
        result = mgr.get('unknown-parser-xyz')
        assert result is None


# ===================================================================
# Parser init / robustness
# ===================================================================


@_apply_parser_marks
class TestReasoningParserInitErrors:
    """Initialization error handling common to both parsers."""

    @_BOTH_PARSERS
    def test_none_tokenizer(self, parser_factory):
        """Both parsers must raise ValueError when tokenizer is None."""
        with pytest.raises(ValueError, match='(?i)tokenizer'):
            parser_factory()(None)

    @_BOTH_PARSERS
    def test_missing_vocab_raises_runtime_error(self, parser_factory):
        """Both parsers must raise RuntimeError when vocab lacks think
        tokens."""
        tok = _make_mock_tokenizer(vocab={'unrelated': 999})
        with pytest.raises(RuntimeError):
            parser_factory()(tok)

    @_BOTH_PARSERS
    def test_vocab_property_accessible(self, parser_factory):
        """Both parsers expose a vocab property via the base class."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        assert '<think>' in parser.vocab
        assert '</think>' in parser.vocab
        assert parser.vocab['<think>'] == 100


# ===================================================================
# Complex edge cases (relaxed assertions)
# ===================================================================


@_apply_parser_marks
class TestReasoningParserComplexEdgeCases:
    """Edge cases requiring relaxed assertions or parser-specific
    behaviour that cannot be expressed as simple data dicts."""

    @_BOTH_PARSERS
    def test_multiple_end_tokens(self, parser_factory):
        """Multiple ``</think>`` — extraction stops at the first one."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = '<think>First</think>Middle</think>Last'
        reasoning, content = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None and 'First' in reasoning
        assert content is not None and 'Middle' in content

    @_BOTH_PARSERS
    def test_nested_think_tokens(self, parser_factory):
        """Nested ``<think>`` inside ``<think>``."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = '<think>Outer<think>Inner</think>Content'
        reasoning, content = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert 'Outer' in reasoning and 'Inner' in reasoning
        assert content is not None and 'Content' in content

    @_BOTH_PARSERS
    def test_multiple_think_blocks(self, parser_factory):
        """Multiple ``<think>…</think>`` blocks — stops at first ``</think>``."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = ('<think>first reasoning</think>middle text'
                        '<think>second reasoning</think>final text')
        reasoning, content = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None and 'first reasoning' in reasoning
        assert content is not None and 'middle text' in content

    @_BOTH_PARSERS
    def test_prefix_before_think_tags(self, parser_factory):
        """Text before ``<think>`` must be discarded."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = 'Some prefix<think>reasoning here</think>The answer.'
        reasoning, content = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None and 'reasoning here' in reasoning
        assert content is not None and content == 'The answer.'

    @_BOTH_PARSERS
    def test_whitespace_prefix_before_think(self, parser_factory):
        """Whitespace-only prefix before ``<think>`` must be discarded."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = ' <think>reasoning</think>answer'
        reasoning, content = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None and 'reasoning' in reasoning
        assert content is not None and content == 'answer'

    @_BOTH_PARSERS
    def test_unicode_in_reasoning(self, parser_factory):
        """Chinese / Unicode content inside think tags."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = '<think>让我想想... 37 × 43 = 1591</think>答案是1591。'
        reasoning, content = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None and '37' in reasoning
        assert content is not None and '1591' in content

    @_BOTH_PARSERS
    def test_special_chars_in_reasoning(self, parser_factory):
        """Special characters inside think tags."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = ('<think>Let\'s check: 2 > 1 & 3 < 5</think>'
                        'All good.')
        reasoning, content = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert content is not None and content == 'All good.'

    @_BOTH_PARSERS
    def test_very_long_reasoning(self, parser_factory):
        """Long reasoning content should be fully extracted."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        long_reasoning = 'Step ' + '. Step '.join(str(i) for i in range(100))
        model_output = f'<think>{long_reasoning}</think>Final.'
        reasoning, content = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None and 'Step 99' in reasoning
        assert content == 'Final.'

    @pytest.mark.parametrize('parser_name', REASONING_PARSER_NAMES)
    def test_parser_instantiation(self, parser_name):
        """Every registered parser should be instantiable with a mock
        tokenizer."""
        mgr = _get_parser_manager()
        cls = mgr.get(parser_name)
        assert cls is not None
        tok = _make_mock_tokenizer()
        parser = cls(tok)
        assert parser is not None
        assert parser.think_start_token == THINK_START_TOKEN
        assert parser.think_end_token == THINK_END_TOKEN

    @_BOTH_PARSERS
    def test_prefix_before_think_streaming(self, parser_factory):
        """Text prefix before ``<think>`` in streaming — content correct."""
        tok = _make_mock_tokenizer(_DEFAULT_VOCAB)
        parser = parser_factory()(tok)
        deltas = ['Some prefix', '<think>', 'reasoning here', '</think>', 'The answer.']
        reasoning, content = _run_streaming_extraction(parser, deltas, _DEFAULT_VOCAB)
        assert reasoning is not None and 'reasoning here' in reasoning
        assert content is not None and 'The answer.' in content
        assert '<' not in content and '>' not in content


# ===================================================================
# Parser independence
# ===================================================================


@_apply_parser_marks
class TestReasoningParserIndependence:
    """Verify that different parser instances don't share state."""

    def test_parsers_do_not_share_state(self):
        tok = _make_mock_tokenizer()
        req = _make_mock_request()
        ds = _get_deepseek_parser_cls()(tok)
        qw = _get_qwen_parser_cls()(tok)
        output = '<think>Shared reasoning</think>Shared content'
        ds_r, ds_c = ds.extract_reasoning_content(output, req)
        qw_r, qw_c = qw.extract_reasoning_content(output, req)
        assert 'Shared reasoning' in ds_r and 'Shared reasoning' in qw_r
        assert 'Shared content' in ds_c and 'Shared content' in qw_c

    def test_parsers_independent_streaming(self):
        tok = _make_mock_tokenizer()
        ds = _get_deepseek_parser_cls()(tok)
        qw = _get_qwen_parser_cls()(tok)
        deltas = ['<think>', 'Step 1 ', 'Step 2', '</think>', 'Answer']
        ds_r, ds_c = _run_streaming_extraction(ds, deltas, _DEFAULT_VOCAB)
        qw_r, qw_c = _run_streaming_extraction(qw, deltas, _DEFAULT_VOCAB)
        assert ds_r is not None and qw_r is not None
        assert ds_c is not None and qw_c is not None

    def test_multiple_instances_same_parser(self):
        tok1 = _make_mock_tokenizer()
        tok2 = _make_mock_tokenizer()
        req = _make_mock_request()
        p1 = _get_deepseek_parser_cls()(tok1)
        p2 = _get_deepseek_parser_cls()(tok2)
        r1, c1 = p1.extract_reasoning_content('<think>A</think>B', req)
        r2, c2 = p2.extract_reasoning_content('<think>C</think>D', req)
        assert 'A' in r1 and 'C' in r2
        assert 'B' in c1 and 'D' in c2


# ===================================================================
# Detection mechanism — token-ID–based
# ===================================================================


@_apply_parser_marks
class TestReasoningParserDetectionMechanism:
    """Both parsers use token-ID–based detection.

    When token IDs are missing the special tokens (e.g. the tokenizer
    splits ``<think>`` into sub-tokens), neither parser detects the
    boundary and both fall back to treating text as reasoning.
    """

    @_BOTH_PARSERS
    def test_missing_token_ids_fallback_to_reasoning(self, parser_factory):
        """Token IDs lack special-token IDs → all text treated as
        reasoning regardless of parser."""
        tok = _make_mock_tokenizer(_DEFAULT_VOCAB)
        parser = parser_factory()(tok)

        # previous_text contains <think> but previous_token_ids lacks 100
        # delta_text contains </think> but delta_token_ids lacks 101
        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Step 1',
            current_text='<think>Step 1. Done</think>Answer',
            delta_text='. Done</think>Answer',
            previous_token_ids=[300, 301],           # No 100 (think_start)
            current_token_ids=[300, 301, 302, 303, 304],
            delta_token_ids=[302, 303, 304],          # No 101 (think_end)
        )

        assert result is not None
        # No token IDs → parser cannot split; everything is reasoning
        assert result.reasoning_content == '. Done</think>Answer'
        assert result.content is None
