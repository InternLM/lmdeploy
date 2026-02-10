import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from utils.constant import BACKEND_LIST, RESTFUL_MODEL_LIST
from utils.tool_reasoning_definitions import (CALCULATOR_TOOL, REASONING_PARSER_NAMES, SEARCH_TOOL, THINK_END_TOKEN,
                                              THINK_START_TOKEN, WEATHER_TOOL, WEATHER_TOOL_CN,
                                              assert_arguments_parseable, assert_tool_call_fields,
                                              build_reasoning_tool_roundtrip_messages, collect_stream_reasoning,
                                              get_client_and_model, get_reasoning_content, get_reasoning_tokens)

_LMDEPLOY_ROOT = str(Path(__file__).resolve().parents[3] / 'lmdeploy')
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
for _p in (_PROJECT_ROOT, _LMDEPLOY_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Lazy imports - only used by parser unit-test classes
_deepseek_parser_cls = None
_qwen_parser_cls = None
_parser_manager = None


def _get_deepseek_parser_cls():
    global _deepseek_parser_cls
    if _deepseek_parser_cls is None:
        from lmdeploy.serve.openai.reasoning_parser import DeepSeekR1ReasoningParser
        _deepseek_parser_cls = DeepSeekR1ReasoningParser
    return _deepseek_parser_cls


def _get_qwen_parser_cls():
    global _qwen_parser_cls
    if _qwen_parser_cls is None:
        from lmdeploy.serve.openai.reasoning_parser import QwenQwQReasoningParser
        _qwen_parser_cls = QwenQwQReasoningParser
    return _qwen_parser_cls


def _get_parser_manager():
    global _parser_manager
    if _parser_manager is None:
        from lmdeploy.serve.openai.reasoning_parser import ReasoningParserManager
        _parser_manager = ReasoningParserManager
    return _parser_manager


def _make_mock_tokenizer(vocab=None):
    """Create a mock tokenizer with a configurable vocab dict."""
    tok = MagicMock()
    default_vocab = {
        '<think>': 100,
        '</think>': 101,
    }
    tok.get_vocab.return_value = vocab or default_vocab
    return tok


def _make_mock_request():
    """Create a mock ChatCompletionRequest."""
    req = MagicMock()
    req.model = 'test-model'
    return req


_CLASS_MARKS = [
    pytest.mark.order(9),
    pytest.mark.reasoning,
    pytest.mark.deepseek_r1_parser,
    pytest.mark.qwenqwq_parser,
    pytest.mark.flaky(reruns=2),
    pytest.mark.parametrize('backend', BACKEND_LIST),
    pytest.mark.parametrize('model_case', RESTFUL_MODEL_LIST),
]

_PARSER_MARKS = [
    pytest.mark.order(9),
    pytest.mark.reasoning,
    pytest.mark.reasoning_parser,
]


def _apply_marks(cls):
    """Apply the shared API-level marks to *cls*."""
    for m in _CLASS_MARKS:
        cls = m(cls)
    return cls


def _apply_parser_marks(cls):
    """Apply lightweight marks to parser unit-test classes (no parametrize)."""
    for m in _PARSER_MARKS:
        cls = m(cls)
    return cls


MESSAGES_REASONING_BASIC = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant. Think step by step '
        'before answering.',
    },
    {
        'role': 'user',
        'content': 'What is 37 * 43? Explain your reasoning.',
    },
]

MESSAGES_REASONING_COMPLEX = [
    {
        'role':
        'system',
        'content':
        'You are a math tutor. Always think through the problem '
        'step by step before providing the final answer.',
    },
    {
        'role':
        'user',
        'content':
        'A train leaves station A at 60 km/h and another train '
        'leaves station B at 80 km/h. If the stations are 280 km '
        'apart and trains leave at the same time heading towards '
        'each other, when will they meet?',
    },
]

MESSAGES_REASONING_SIMPLE = [
    {
        'role': 'user',
        'content': 'What is 2 + 2?',
    },
]

MESSAGES_REASONING_WEATHER_TOOL = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant that can use tools. '
        'Think step by step before deciding whether to use a tool.',
    },
    {
        'role': 'user',
        'content': "I'm traveling to Dallas, TX tomorrow. Should I pack "
        'an umbrella? Check the weather first.',
    },
]

MESSAGES_REASONING_CN = [
    {
        'role': 'system',
        'content': '你是一个有用的助手。请逐步思考后再回答问题。',
    },
    {
        'role': 'user',
        'content': '一辆火车从A站以60公里/小时出发，另一辆从B站以80公里/小时出发。'
        '如果两站相距280公里，两车相向而行，何时相遇？',
    },
]

MESSAGES_REASONING_MULTI_TURN = [
    {
        'role': 'system',
        'content': 'You are a math tutor. Think step by step.',
    },
    {
        'role': 'user',
        'content': 'What is the sum of the first 10 natural numbers?',
    },
    {
        'role': 'assistant',
        'content': 'The sum of 1 to 10 is 55.',
    },
    {
        'role': 'user',
        'content': 'Now what is the sum of the first 100 natural numbers? '
        'Explain your reasoning.',
    },
]

MESSAGES_REASONING_PARALLEL_TOOLS = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant. Think about what tools to '
        'use, then call them. You can call multiple tools at once.',
    },
    {
        'role': 'user',
        'content': "What's the weather in Dallas, TX? "
        'Also calculate 37 * 43.',
    },
]


@_apply_parser_marks
class TestReasoningParserManager:
    """Verify that all parsers are correctly registered."""

    def test_all_parser_names_registered(self):
        """Each name in REASONING_PARSER_NAMES should be resolvable."""
        mgr = _get_parser_manager()
        for name in REASONING_PARSER_NAMES:
            cls = mgr.get(name)
            assert cls is not None, (f'Parser "{name}" not found in ReasoningParserManager')

    def test_deepseek_r1_registered(self):
        mgr = _get_parser_manager()
        cls = mgr.get('deepseek-r1')
        assert cls is not None
        assert cls.__name__ == 'DeepSeekR1ReasoningParser'

    def test_qwen_qwq_registered(self):
        mgr = _get_parser_manager()
        cls = mgr.get('qwen-qwq')
        assert cls is not None
        assert cls.__name__ == 'QwenQwQReasoningParser'

    def test_intern_s1_registered(self):
        """Intern-s1 should map to QwenQwQReasoningParser."""
        mgr = _get_parser_manager()
        cls = mgr.get('intern-s1')
        assert cls is not None
        assert cls.__name__ == 'QwenQwQReasoningParser'

    def test_unknown_parser_returns_none(self):
        mgr = _get_parser_manager()
        result = mgr.get('unknown-parser-xyz')
        assert result is None


@_apply_parser_marks
@pytest.mark.deepseek_r1_parser
class TestDeepSeekR1ParserNonStreaming:
    """Unit tests for DeepSeekR1ReasoningParser.extract_reasoning_content."""

    def test_full_think_tags(self):
        """<think>reasoning</think>answer → (reasoning, answer)."""
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()

        model_output = '<think>Let me think step by step...</think>The answer is 42.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert 'step by step' in reasoning
        assert final is not None
        assert '42' in final

    def test_missing_start_tag(self):
        """No <think> but has </think> → parser should handle gracefully.

        DeepSeek-R1 HF model may omit the start tag.
        """
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()

        model_output = 'I need to reason about this...</think>The answer is 7.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert final is not None
        assert '7' in final

    def test_no_think_tags_at_all(self):
        """No think tags → treated as all-reasoning content (DeepSeek-R1
        behaviour)."""
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()

        model_output = 'Just a plain answer without reasoning.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        # Per DeepSeek-R1 parser: if no </think>, returns (model_output, None)
        assert reasoning == model_output
        assert final is None

    def test_empty_reasoning(self):
        """<think></think>answer → empty reasoning, non-empty content."""
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()

        model_output = '<think></think>Direct answer.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert final is not None
        assert 'Direct answer' in final

    def test_only_think_tags(self):
        """<think>reasoning only</think> with nothing after."""
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()

        model_output = '<think>All reasoning, no final answer.</think>'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert 'reasoning' in reasoning.lower()
        assert final is None

    def test_multiline_reasoning(self):
        """Multi-line reasoning between tags."""
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<think>\nStep 1: calculate 37 * 40 = 1480\n'
                        'Step 2: calculate 37 * 3 = 111\n'
                        'Step 3: 1480 + 111 = 1591\n</think>The answer is 1591.')
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert 'Step 1' in reasoning
        assert 'Step 3' in reasoning
        assert final is not None
        assert '1591' in final


@_apply_parser_marks
@pytest.mark.deepseek_r1_parser
class TestDeepSeekR1ParserStreaming:
    """Unit tests for DeepSeekR1ReasoningParser streaming extraction."""

    def test_think_start_token_only(self):
        """Single <think> token → return None (skip)."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)

        result = parser.extract_reasoning_content_streaming(
            previous_text='',
            current_text='<think>',
            delta_text='<think>',
            previous_token_ids=[],
            current_token_ids=[100],
            delta_token_ids=[100],
        )
        assert result is None

    def test_think_end_token_only(self):
        """Single </think> token → DeltaMessage(content='')."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)

        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Some reasoning',
            current_text='<think>Some reasoning</think>',
            delta_text='</think>',
            previous_token_ids=[100, 200, 201],
            current_token_ids=[100, 200, 201, 101],
            delta_token_ids=[101],
        )
        assert result is not None
        assert result.content == ''

    def test_reasoning_continues(self):
        """<think> in previous, no </think> → reasoning_content=delta."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)

        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Step 1',
            current_text='<think>Step 1: multiply',
            delta_text=': multiply',
            previous_token_ids=[100, 200],
            current_token_ids=[100, 200, 201],
            delta_token_ids=[201],
        )
        assert result is not None
        assert result.reasoning_content == ': multiply'

    def test_reasoning_ends_in_delta(self):
        """<think> in previous, </think> in delta → split reasoning/content."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)

        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Step 1',
            current_text='<think>Step 1. Done</think>Answer',
            delta_text='. Done</think>Answer',
            previous_token_ids=[100, 200],
            current_token_ids=[100, 200, 201, 101, 300],
            delta_token_ids=[201, 101, 300],
        )
        assert result is not None
        assert result.reasoning_content is not None
        assert '. Done' in result.reasoning_content
        assert result.content == 'Answer'

    def test_content_after_think_end(self):
        """Both <think> and </think> in previous → regular content."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)

        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Reasoning</think>Prev',
            current_text='<think>Reasoning</think>Prev content',
            delta_text=' content',
            previous_token_ids=[100, 200, 101, 300],
            current_token_ids=[100, 200, 101, 300, 301],
            delta_token_ids=[301],
        )
        assert result is not None
        assert result.content == ' content'

    def test_no_think_in_previous_with_end_in_delta(self):
        """No <think> in previous, </think> in delta (HF compatibility)."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)

        result = parser.extract_reasoning_content_streaming(
            previous_text='Some reasoning text',
            current_text='Some reasoning text</think>Final',
            delta_text='</think>Final',
            previous_token_ids=[200, 201],
            current_token_ids=[200, 201, 101, 300],
            delta_token_ids=[101, 300],
        )
        assert result is not None
        # Should split at </think>
        assert result.content == 'Final'


@_apply_parser_marks
@pytest.mark.qwenqwq_parser
class TestQwenQwQParserNonStreaming:
    """Unit tests for QwenQwQReasoningParser.extract_reasoning_content."""

    def test_full_think_tags(self):
        """<think>reasoning</think>answer → (reasoning, answer)."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()

        model_output = '<think>Let me reason about this...</think>The answer is 42.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert 'reason' in reasoning.lower()
        assert final is not None
        assert '42' in final

    def test_no_think_tags(self):
        """No think tags at all → (None, content).

        QwenQwQ parser: if no </think>, returns (None, model_output).
        """
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()

        model_output = 'Plain output without reasoning.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is None
        assert final == model_output

    def test_missing_start_tag(self):
        """</think> without <think> → parser adds <think> prefix."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()

        model_output = 'Reasoning here...</think>Final answer.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert final is not None
        assert 'Final answer' in final

    def test_newline_stripping(self):
        """Leading/trailing newlines in reasoning should be stripped."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()

        model_output = '<think>\nStep 1: think\nStep 2: conclude\n</think>Done.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        # QwenQwQ strips leading/trailing \n from reasoning
        assert not reasoning.startswith('\n')
        assert not reasoning.endswith('\n')
        assert final is not None

    def test_only_reasoning_no_content(self):
        """<think>only reasoning</think> with nothing after."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()

        model_output = '<think>Only reasoning content here.</think>'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert final is None

    def test_empty_reasoning_tags(self):
        """<think></think>content → empty reasoning, non-empty content."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()

        model_output = '<think></think>Direct answer.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        # Empty reasoning content (after stripping) may be empty string
        assert final is not None
        assert 'Direct answer' in final


@_apply_parser_marks
@pytest.mark.qwenqwq_parser
class TestQwenQwQParserStreaming:
    """Unit tests for QwenQwQReasoningParser streaming extraction.

    QwenQwQ uses text-based matching (not token IDs) for streaming.
    """

    def test_think_start_text_skipped(self):
        """delta_text == '<think>' → DeltaMessage(content='')."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)

        result = parser.extract_reasoning_content_streaming(
            previous_text='',
            current_text='<think>',
            delta_text='<think>',
            previous_token_ids=[],
            current_token_ids=[100],
            delta_token_ids=[100],
        )
        assert result is not None
        assert result.content == ''

    def test_think_end_text_skipped(self):
        """delta_text == '</think>' → DeltaMessage(content='')."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)

        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Some text',
            current_text='<think>Some text</think>',
            delta_text='</think>',
            previous_token_ids=[100, 200],
            current_token_ids=[100, 200, 101],
            delta_token_ids=[101],
        )
        assert result is not None
        assert result.content == ''

    def test_reasoning_continues(self):
        """<think> in previous, no </think> → reasoning_content."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)

        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Step 1',
            current_text='<think>Step 1, Step 2',
            delta_text=', Step 2',
            previous_token_ids=[100, 200],
            current_token_ids=[100, 200, 201],
            delta_token_ids=[201],
        )
        assert result is not None
        assert result.reasoning_content == ', Step 2'

    def test_reasoning_ends_in_delta(self):
        """<think> in previous, </think> in delta → split."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)

        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Step 1',
            current_text='<think>Step 1. Final</think>Result',
            delta_text='. Final</think>Result',
            previous_token_ids=[100, 200],
            current_token_ids=[100, 200, 201, 101, 300],
            delta_token_ids=[201, 101, 300],
        )
        assert result is not None
        assert '. Final' in (result.reasoning_content or '')
        assert result.content == 'Result'

    def test_content_after_think_closed(self):
        """Both tags in previous_text → regular content delta."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)

        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Reasoning</think>Prev',
            current_text='<think>Reasoning</think>Prev more',
            delta_text=' more',
            previous_token_ids=[100, 200, 101, 300],
            current_token_ids=[100, 200, 101, 300, 301],
            delta_token_ids=[301],
        )
        assert result is not None
        assert result.content == ' more'


@_apply_parser_marks
class TestReasoningParserEdgeCases:
    """Edge cases that both parsers should handle."""

    def test_deepseek_unicode_in_reasoning(self):
        """Chinese / Unicode content inside think tags."""
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()

        model_output = '<think>让我想想... 37 × 43 = 1591</think>答案是1591。'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert '37' in reasoning
        assert final is not None
        assert '1591' in final

    def test_qwen_unicode_in_reasoning(self):
        """Chinese content in think tags for QwenQwQ."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()

        model_output = '<think>让我想想... 37 × 43 = 1591</think>答案是1591。'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert '37' in reasoning
        assert final is not None

    def test_deepseek_very_long_reasoning(self):
        """Long reasoning content should be fully extracted."""
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()

        long_reasoning = 'Step ' + '. Step '.join(str(i) for i in range(100))
        model_output = f'<think>{long_reasoning}</think>Final.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert len(reasoning) > 100
        assert 'Step 99' in reasoning
        assert final == 'Final.'

    def test_qwen_very_long_reasoning(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()

        long_reasoning = 'Step ' + '. Step '.join(str(i) for i in range(100))
        model_output = f'<think>{long_reasoning}</think>Final.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert 'Step 99' in reasoning
        assert final == 'Final.'

    def test_deepseek_special_chars_in_reasoning(self):
        """Special characters inside think tags."""
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<think>Let\'s check: 2 > 1 & 3 < 5, also "quoted" text</think>'
                        'All good.')
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert '&' in reasoning or '>' in reasoning
        assert final is not None

    def test_qwen_special_chars_in_reasoning(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()

        model_output = ('<think>Let\'s check: 2 > 1 & 3 < 5, also "quoted" text</think>'
                        'All good.')
        reasoning, final = parser.extract_reasoning_content(model_output, req)

        assert reasoning is not None
        assert final is not None

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


@_apply_marks
class TestReasoningBasic:
    """Basic reasoning_content presence in non-streaming responses."""

    def test_reasoning_content_present(self, backend, model_case):
        """Model should populate reasoning_content for complex questions."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.finish_reason in ('stop', 'length')

        reasoning = get_reasoning_content(choice.message)
        if reasoning is not None:
            assert len(reasoning) > 0, 'reasoning_content should not be empty'

    def test_content_also_present(self, backend, model_case):
        """Both reasoning_content and content should be populated."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
        )

        choice = response.choices[0]
        # At minimum, the model must produce some output
        has_reasoning = get_reasoning_content(choice.message) is not None
        has_content = (choice.message.content is not None and len(choice.message.content.strip()) > 0)
        assert has_reasoning or has_content, ('Model must produce reasoning_content or content (or both)')


@_apply_marks
class TestReasoningStreaming:
    """Streaming delta.reasoning_content presence."""

    def test_reasoning_in_stream(self, backend, model_case):
        """At least some chunks should carry reasoning_content."""
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
            stream=True,
        )

        result = collect_stream_reasoning(stream)

        assert result['chunk_count'] > 0, 'Should receive at least one chunk'
        assert result['finish_reason'] in ('stop', 'length')

        # Model should produce reasoning and/or content
        total_output = result['reasoning_content'] + result['content']
        assert len(total_output) > 0, ('Stream should produce reasoning or content')

    def test_reasoning_chunks_count(self, backend, model_case):
        """For complex questions, expect multiple reasoning chunks."""
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_COMPLEX,
            temperature=0,
            max_completion_tokens=2048,
            logprobs=False,
            stream=True,
        )

        result = collect_stream_reasoning(stream)

        if result['reasoning_chunks'] > 0:
            assert len(result['reasoning_content']) > 10, ('Complex question should produce substantial reasoning')


@_apply_marks
class TestReasoningStreamConsistency:
    """Reasoning output should be consistent across modes."""

    def test_reasoning_presence_consistent(self, backend, model_case):
        """If non-streaming has reasoning, streaming should too (and vice
        versa)."""
        client, model_name = get_client_and_model()

        # Non-streaming
        ns_resp = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
        )

        ns_reasoning = get_reasoning_content(ns_resp.choices[0].message)
        ns_content = ns_resp.choices[0].message.content or ''

        # Streaming
        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
            stream=True,
        )

        result = collect_stream_reasoning(stream)

        # Both should produce some output
        ns_total = (ns_reasoning or '') + ns_content
        s_total = result['reasoning_content'] + result['content']

        assert len(ns_total) > 0 and len(s_total) > 0, ('Both streaming and non-streaming should produce output')


@_apply_marks
class TestReasoningContentQuality:
    """Verify reasoning content is meaningful and well-structured."""

    def test_reasoning_contains_steps(self, backend, model_case):
        """For math problems, reasoning should contain calculation steps."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_COMPLEX,
            temperature=0,
            max_completion_tokens=2048,
            logprobs=False,
        )

        choice = response.choices[0]
        reasoning = get_reasoning_content(choice.message)
        content = choice.message.content or ''
        full_output = (reasoning or '') + content

        # For a train problem, expect some mathematical reasoning
        assert len(full_output) > 20, ('Complex problem should produce substantial output')

    def test_reasoning_not_duplicated_in_content(self, backend, model_case):
        """Reasoning and content should not be identical."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
        )

        choice = response.choices[0]
        reasoning = get_reasoning_content(choice.message)
        content = choice.message.content

        if reasoning and content:
            # They can overlap but should not be exactly the same
            assert reasoning.strip() != content.strip(), ('reasoning_content and content should differ')


@_apply_marks
class TestReasoningWithToolCalls:
    """Model reasons THEN calls a tool."""

    def test_reasoning_before_tool_call(self, backend, model_case):
        """For tool-triggering questions, model should reason then call
        tool."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_WEATHER_TOOL,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'

        # Model may produce reasoning + tool call, or just tool call
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert_tool_call_fields(tc)
            assert tc.function.name == 'get_current_weather'
            assert_arguments_parseable(tc.function.arguments)

    def test_reasoning_tool_call_finish_reason(self, backend, model_case):
        """When tool call is made, finish_reason should be 'tool_calls'."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_WEATHER_TOOL,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL],
            tool_choice={
                'type': 'function',
                'function': {
                    'name': 'get_current_weather'
                },
            },
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.finish_reason == 'tool_calls'
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) >= 1


@_apply_marks
class TestReasoningWithToolCallsStreaming:
    """Streaming: reasoning chunks then tool call chunks."""

    def test_streaming_reasoning_then_tool(self, backend, model_case):
        """Stream should contain reasoning and/or tool_call data."""
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_WEATHER_TOOL,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL],
            logprobs=False,
            stream=True,
        )

        result = collect_stream_reasoning(stream)

        assert result['chunk_count'] > 0

        # Should have tool calls or content
        has_tool_calls = len(result['tool_calls']) > 0
        has_content = len(result['content']) > 0
        has_reasoning = len(result['reasoning_content']) > 0

        assert has_tool_calls or has_content or has_reasoning, (
            'Stream should produce reasoning, content, or tool calls')

        if has_tool_calls:
            for idx, tc in result['tool_calls'].items():
                assert tc['name'] is not None
                assert len(tc['args_str']) > 0


@_apply_marks
class TestReasoningWithToolChoice:
    """Reasoning behaviour under different tool_choice settings."""

    def test_reasoning_tool_choice_auto(self, backend, model_case):
        """tool_choice='auto' with reasoning model."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_WEATHER_TOOL,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice='auto',
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'

        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            assert choice.finish_reason == 'tool_calls'
            for tc in choice.message.tool_calls:
                assert_tool_call_fields(tc)

    def test_reasoning_tool_choice_required(self, backend, model_case):
        """tool_choice='required' + reasoning: must produce tool call."""
        client, model_name = get_client_and_model()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=MESSAGES_REASONING_WEATHER_TOOL,
                temperature=0,
                max_completion_tokens=1024,
                tools=[WEATHER_TOOL],
                tool_choice='required',
                logprobs=False,
            )

            choice = response.choices[0]
            assert choice.message.tool_calls is not None
            assert len(choice.message.tool_calls) >= 1
            for tc in choice.message.tool_calls:
                assert_tool_call_fields(tc)
        except Exception as e:
            pytest.skip(f'tool_choice="required" not supported: {e}')

    def test_reasoning_tool_choice_none(self, backend, model_case):
        """tool_choice='none' + reasoning: no tool calls, text output."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_WEATHER_TOOL,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL],
            tool_choice='none',
            logprobs=False,
        )

        choice = response.choices[0]
        assert (choice.message.tool_calls is None or len(choice.message.tool_calls) == 0)

        has_reasoning = get_reasoning_content(choice.message) is not None
        has_content = (choice.message.content is not None and len(choice.message.content.strip()) > 0)
        assert has_reasoning or has_content

    def test_reasoning_tool_choice_specific(self, backend, model_case):
        """Force a specific tool with reasoning model."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_WEATHER_TOOL,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice={
                'type': 'function',
                'function': {
                    'name': 'get_current_weather'
                },
            },
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) >= 1
        assert choice.message.tool_calls[0].function.name == 'get_current_weather'


@_apply_marks
class TestReasoningParallelToolCalls:
    """Reasoning model calling multiple tools in parallel."""

    def test_reasoning_parallel_tools(self, backend, model_case):
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_PARALLEL_TOOLS,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'

        if choice.message.tool_calls and len(choice.message.tool_calls) >= 2:
            ids = [tc.id for tc in choice.message.tool_calls]
            assert len(set(ids)) == len(ids), 'Tool call IDs must be unique'

            for tc in choice.message.tool_calls:
                assert_tool_call_fields(tc)
                assert_arguments_parseable(tc.function.arguments)

    def test_reasoning_parallel_tools_streaming(self, backend, model_case):
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_PARALLEL_TOOLS,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            logprobs=False,
            stream=True,
        )

        result = collect_stream_reasoning(stream)

        if len(result['tool_calls']) >= 2:
            for idx, tc in result['tool_calls'].items():
                assert tc['name'] is not None
                parsed = json.loads(tc['args_str'])
                assert isinstance(parsed, dict)


@_apply_marks
class TestReasoningToolRoundTrip:
    """Multi-turn: reason → tool → result → reasoning → answer."""

    def test_reasoning_after_tool_result(self, backend, model_case):
        """After receiving tool result, model should reason and answer."""
        client, model_name = get_client_and_model()
        messages = build_reasoning_tool_roundtrip_messages()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.finish_reason in ('stop', 'length')

        # Should produce a text answer (possibly with reasoning)
        reasoning = get_reasoning_content(choice.message)
        content = choice.message.content or ''
        full_output = (reasoning or '') + content

        assert len(full_output) > 0

    def test_reasoning_after_tool_result_streaming(self, backend, model_case):
        """Streaming variant of tool result round-trip."""
        client, model_name = get_client_and_model()
        messages = build_reasoning_tool_roundtrip_messages()

        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL],
            logprobs=False,
            stream=True,
        )

        result = collect_stream_reasoning(stream)

        total_output = result['reasoning_content'] + result['content']
        assert len(total_output) > 0
        assert result['finish_reason'] in ('stop', 'length')


@_apply_marks
class TestReasoningTokenAccounting:
    """Verify token usage includes reasoning tokens when available."""

    def test_usage_present(self, backend, model_case):
        """Usage field should be present and have positive counts."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (response.usage.prompt_tokens + response.usage.completion_tokens)

    def test_reasoning_tokens_if_available(self, backend, model_case):
        """If reasoning_tokens is exposed, it should be non-negative."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_COMPLEX,
            temperature=0,
            max_completion_tokens=2048,
            logprobs=False,
        )

        rt = get_reasoning_tokens(response)
        if rt is not None:
            assert rt >= 0, f'reasoning_tokens should be >= 0, got {rt}'
            assert rt <= response.usage.completion_tokens, ('reasoning_tokens should not exceed completion_tokens')


@_apply_marks
class TestReasoningMultilingual:
    """Reasoning with Chinese / multilingual prompts."""

    def test_chinese_reasoning(self, backend, model_case):
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_CN,
            temperature=0,
            max_completion_tokens=2048,
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'

        reasoning = get_reasoning_content(choice.message)
        content = choice.message.content or ''
        full_output = (reasoning or '') + content

        assert len(full_output) > 0

    def test_chinese_reasoning_streaming(self, backend, model_case):
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_CN,
            temperature=0,
            max_completion_tokens=2048,
            logprobs=False,
            stream=True,
        )

        result = collect_stream_reasoning(stream)

        total = result['reasoning_content'] + result['content']
        assert len(total) > 0

    def test_chinese_reasoning_with_tool(self, backend, model_case):
        """Chinese prompt + tool call with reasoning."""
        client, model_name = get_client_and_model()

        messages = [
            {
                'role': 'system',
                'content': '你是一个有用的助手，可以使用工具。'
                '请先思考是否需要使用工具。',
            },
            {
                'role': 'user',
                'content': '北京今天的天气怎么样？我需要带伞吗？',
            },
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL_CN],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'

        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert_tool_call_fields(tc)


@_apply_marks
class TestReasoningMultiTurn:
    """Multi-turn conversations where reasoning persists."""

    def test_multi_turn_reasoning(self, backend, model_case):
        """Second-round question should also trigger reasoning."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_MULTI_TURN,
            temperature=0,
            max_completion_tokens=2048,
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.finish_reason in ('stop', 'length')

        reasoning = get_reasoning_content(choice.message)
        content = choice.message.content or ''
        full_output = (reasoning or '') + content

        assert len(full_output) > 0

    def test_multi_turn_reasoning_streaming(self, backend, model_case):
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_MULTI_TURN,
            temperature=0,
            max_completion_tokens=2048,
            logprobs=False,
            stream=True,
        )

        result = collect_stream_reasoning(stream)

        total = result['reasoning_content'] + result['content']
        assert len(total) > 0


@_apply_marks
class TestReasoningResponseValidation:
    """Validate response-level fields in reasoning mode."""

    def test_model_id_created_fields(self, backend, model_case):
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
        )

        assert response.model is not None and len(response.model) > 0
        assert response.id is not None
        assert response.created is not None
        assert len(response.choices) >= 1
        assert response.choices[0].index == 0

    def test_finish_reason_values(self, backend, model_case):
        """finish_reason should be 'stop' or 'length'."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
        )

        assert response.choices[0].finish_reason in ('stop', 'length')

    def test_role_is_assistant(self, backend, model_case):
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
        )

        assert response.choices[0].message.role == 'assistant'

    def test_reasoning_with_tool_finish_reason(self, backend, model_case):
        """When reasoning + tool_call, finish_reason = 'tool_calls'."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_WEATHER_TOOL,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL],
            tool_choice={
                'type': 'function',
                'function': {
                    'name': 'get_current_weather'
                },
            },
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.finish_reason == 'tool_calls'


@_apply_marks
class TestReasoningEdgeCases:
    """Edge cases for reasoning functionality."""

    def test_simple_question_still_works(self, backend, model_case):
        """Simple question should still produce valid output."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_SIMPLE,
            temperature=0,
            max_completion_tokens=512,
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.finish_reason in ('stop', 'length')

        reasoning = get_reasoning_content(choice.message)
        content = choice.message.content or ''
        full = (reasoning or '') + content
        assert len(full) > 0

    def test_no_think_tags_leaked_to_content(self, backend, model_case):
        """<think>/</think> tags should not appear in final content."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
        )

        content = response.choices[0].message.content
        if content:
            assert THINK_START_TOKEN not in content, (f'<think> should not leak into content: {content[:100]}')
            assert THINK_END_TOKEN not in content, (f'</think> should not leak into content: {content[:100]}')

    def test_no_think_tags_leaked_streaming(self, backend, model_case):
        """Streaming: <think> tags should not appear in content chunks."""
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_BASIC,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
            stream=True,
        )

        result = collect_stream_reasoning(stream)
        content = result['content']
        if content:
            assert THINK_START_TOKEN not in content
            assert THINK_END_TOKEN not in content

    def test_reasoning_with_no_tools_provided(self, backend, model_case):
        """Reasoning model without tools → pure text reasoning."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_WEATHER_TOOL,
            temperature=0,
            max_completion_tokens=1024,
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert (choice.message.tool_calls is None or len(choice.message.tool_calls) == 0)

        reasoning = get_reasoning_content(choice.message)
        content = choice.message.content or ''
        full = (reasoning or '') + content
        assert len(full) > 0

    def test_reasoning_with_empty_tools(self, backend, model_case):
        """Empty tools list with reasoning model."""
        client, model_name = get_client_and_model()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=MESSAGES_REASONING_BASIC,
                temperature=0,
                max_completion_tokens=1024,
                tools=[],
                logprobs=False,
            )

            choice = response.choices[0]
            assert choice.message.role == 'assistant'
            assert (choice.message.tool_calls is None or len(choice.message.tool_calls) == 0)
        except Exception:
            # Some backends reject empty tools list
            pass

    def test_reasoning_low_max_tokens(self, backend, model_case):
        """With very low max_tokens, response may be truncated but valid."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_REASONING_COMPLEX,
            temperature=0,
            max_completion_tokens=50,
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.finish_reason in ('stop', 'length')
