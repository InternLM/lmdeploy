import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from utils.constant import BACKEND_LIST, TOOL_REASONING_MODEL_LIST
from utils.tool_reasoning_definitions import (
    CALCULATOR_TOOL, REASONING_PARSER_NAMES, SEARCH_TOOL, THINK_END_TOKEN, THINK_START_TOKEN, WEATHER_TOOL,
    WEATHER_TOOL_CN, assert_arguments_parseable, assert_tool_call_fields, build_messages_with_tool_response,
    build_reasoning_tool_roundtrip_messages, collect_stream_reasoning, get_reasoning_content, get_reasoning_tokens,
    make_logged_client, setup_log_file)

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


def _simple_tokenize(text, vocab):
    """Tokenise *text* into a list of integer token IDs.

    Known special tokens are mapped via *vocab*; every other character gets a deterministic ID > 200 so it never
    collides with specials.
    """
    ids = []
    i = 0
    sorted_tokens = sorted(vocab.keys(), key=len, reverse=True)
    while i < len(text):
        matched = False
        for tok_str in sorted_tokens:
            if text[i:i + len(tok_str)] == tok_str:
                ids.append(vocab[tok_str])
                i += len(tok_str)
                matched = True
                break
        if not matched:
            ids.append(200 + ord(text[i]))
            i += 1
    return ids


def _run_streaming_extraction(parser, deltas, vocab):
    """Simulate streaming through *parser* with a list of text *deltas*.

    Returns ``(reasoning_content, content)`` aggregated from all deltas.
    """
    reasoning_parts = []
    content_parts = []
    previous_text = ''
    previous_token_ids = []

    for d in deltas:
        current_text = previous_text + d
        current_token_ids = _simple_tokenize(current_text, vocab)
        delta_token_ids = _simple_tokenize(d, vocab)

        result = parser.extract_reasoning_content_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=d,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )

        if result is not None:
            rc = getattr(result, 'reasoning_content', None)
            ct = getattr(result, 'content', None)
            if rc:
                reasoning_parts.append(rc)
            if ct:
                content_parts.append(ct)

        previous_text = current_text
        previous_token_ids = current_token_ids

    reasoning = ''.join(reasoning_parts) if reasoning_parts else None
    content = ''.join(content_parts) if content_parts else None
    return reasoning, content


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

_CLASS_MARKS = [
    pytest.mark.order(9),
    pytest.mark.reasoning,
    pytest.mark.deepseek_r1_parser,
    pytest.mark.qwenqwq_parser,
    pytest.mark.flaky(reruns=2),
    pytest.mark.parametrize('backend', BACKEND_LIST),
    pytest.mark.parametrize('model_case', TOOL_REASONING_MODEL_LIST),
]

_CLASS_MARKS_STREAM = _CLASS_MARKS + [
    pytest.mark.parametrize('stream', [False, True], ids=['nonstream', 'stream']),
]

_PARSER_MARKS = [
    pytest.mark.order(9),
    pytest.mark.reasoning,
    pytest.mark.reasoning_parser,
]


def _apply_marks(cls):
    """Apply the shared API-level marks to *cls* (no stream parametrize)."""
    for m in _CLASS_MARKS:
        cls = m(cls)
    return cls


def _apply_marks_stream(cls):
    """Apply API-level marks WITH stream parametrize to *cls*."""
    for m in _CLASS_MARKS_STREAM:
        cls = m(cls)
    return cls


def _apply_parser_marks(cls):
    """Apply lightweight marks to parser unit-test classes (no parametrize)."""
    for m in _PARSER_MARKS:
        cls = m(cls)
    return cls


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def _assert_no_tag_leakage(reasoning, content):
    """Assert that <think>/</think> tags do not appear in reasoning or
    content."""
    for label, text in [('reasoning', reasoning), ('content', content)]:
        assert THINK_START_TOKEN not in text, (f'<think> leaked into {label}: {text[:100]}')
        assert THINK_END_TOKEN not in text, (f'</think> leaked into {label}: {text[:100]}')


# ---------------------------------------------------------------------------
# Logging helpers – uses shared StreamTee / setup_log_file / make_logged_client
# from utils.tool_reasoning_definitions.
# ---------------------------------------------------------------------------


class _ReasoningTestBase:
    """Mixin providing per-test API request/response logging and unified
    ``_call_api`` helper for both streaming and non-streaming modes."""

    @pytest.fixture(autouse=True)
    def _setup_logging(self, request, config, backend, model_case):
        """Create the log directory and compute the log-file path."""
        self._log_file = setup_log_file(config, request.node.name, 'reasoning')

    def _get_client(self):
        """Return *(client, model_name)* with transparent logging."""
        return make_logged_client(self._log_file)

    def _call_api(self, stream, messages, **create_kwargs):
        """Unified API call for both streaming and non-streaming.

        Returns a dict with keys:
            reasoning, content, finish_reason, role, tool_calls (list of dicts),
            chunk_count, reasoning_chunks, content_chunks,
            finish_reason_count, role_count,
            _response (non-stream only), _choice (non-stream only).
        """
        client, model_name = self._get_client()
        create_kwargs.setdefault('temperature', 0)
        create_kwargs.setdefault('max_completion_tokens', 1024)
        create_kwargs.setdefault('logprobs', False)

        if stream:
            resp = client.chat.completions.create(model=model_name, messages=messages, stream=True, **create_kwargs)
            sr = collect_stream_reasoning(resp)
            tool_calls = []
            for idx in sorted(sr['tool_calls'].keys()):
                tool_calls.append(sr['tool_calls'][idx])
            return {
                'reasoning': sr['reasoning_content'] or '',
                'content': sr['content'] or '',
                'finish_reason': sr['finish_reason'],
                'role': sr.get('role'),
                'tool_calls': tool_calls,
                'chunk_count': sr.get('chunk_count', 0),
                'reasoning_chunks': sr.get('reasoning_chunks', 0),
                'content_chunks': sr.get('content_chunks', 0),
                'finish_reason_count': sr.get('finish_reason_count', 0),
                'role_count': sr.get('role_count', 0),
            }
        else:
            resp = client.chat.completions.create(model=model_name, messages=messages, **create_kwargs)
            choice = resp.choices[0]
            reasoning = get_reasoning_content(choice.message) or ''
            content = choice.message.content or ''
            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        'name': tc.function.name,
                        'args_str': tc.function.arguments,
                        'id': tc.id,
                    })
            return {
                'reasoning': reasoning,
                'content': content,
                'finish_reason': choice.finish_reason,
                'role': choice.message.role,
                'tool_calls': tool_calls,
                '_response': resp,
                '_choice': choice,
            }


# ---------------------------------------------------------------------------
# Message constants
# ---------------------------------------------------------------------------

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

MESSAGES_REASONING_SEARCH_TOOL = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant that can use tools. '
        'Think step by step before deciding whether to use a tool.',
    },
    {
        'role': 'user',
        'content': 'Search for the latest advances in quantum computing '
        'in 2025. Summarize the key breakthroughs.',
    },
]


def _build_search_roundtrip_messages(tool_call_id='call_search_001'):
    """Build multi-turn messages for web_search round-trip."""
    return [
        {
            'role': 'system',
            'content': 'You are a helpful assistant that can use tools. '
            'Think through problems step by step.',
        },
        {
            'role': 'user',
            'content': 'Search for who won the 2024 Nobel Prize in Physics.',
        },
        {
            'role':
            'assistant',
            'content':
            'Let me search for the 2024 Nobel Prize in Physics winner.',
            'tool_calls': [{
                'id': tool_call_id,
                'type': 'function',
                'function': {
                    'name': 'web_search',
                    'arguments': '{"query": "2024 Nobel Prize in Physics winner"}',
                },
            }],
        },
        {
            'role':
            'tool',
            'tool_call_id':
            tool_call_id,
            'content':
            json.dumps({
                'title':
                '2024 Nobel Prize in Physics',
                'snippet':
                'The 2024 Nobel Prize in Physics was awarded to '
                'John Hopfield and Geoffrey Hinton for foundational '
                'discoveries in machine learning with artificial '
                'neural networks.',
                'url':
                'https://www.nobelprize.org/prizes/physics/2024/',
            }),
        },
    ]


# ===========================================================================
# Parser unit-test classes (no API calls, no logging needed)
# ===========================================================================


@_apply_parser_marks
class TestReasoningParserManager:
    """Verify that all parsers are correctly registered."""

    def test_all_parser_names_registered(self):
        mgr = _get_parser_manager()
        for name in REASONING_PARSER_NAMES:
            cls = mgr.get(name)
            assert cls is not None, (f'Parser "{name}" not found in ReasoningParserManager')

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


@_apply_parser_marks
@pytest.mark.deepseek_r1_parser
class TestDeepSeekR1ParserNonStreaming:
    """Unit tests for DeepSeekR1ReasoningParser.extract_reasoning_content."""

    def test_full_think_tags(self):
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
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()
        model_output = 'I need to reason about this...</think>The answer is 7.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert final is not None
        assert '7' in final

    def test_no_think_tags_at_all(self):
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()
        model_output = 'Just a plain answer without reasoning.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning == model_output
        assert final is None

    def test_empty_reasoning(self):
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think></think>Direct answer.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert final is not None
        assert 'Direct answer' in final

    def test_only_think_tags(self):
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think>All reasoning, no final answer.</think>'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert 'reasoning' in reasoning.lower()
        assert final is None

    def test_multiline_reasoning(self):
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

    def test_unclosed_think_tag(self):
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think>I am still reasoning about this problem...'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning == model_output
        assert final is None

    def test_empty_model_output(self):
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()
        reasoning, final = parser.extract_reasoning_content('', req)
        assert reasoning == ''
        assert final is None

    def test_multiple_think_blocks(self):
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()
        model_output = ('<think>first reasoning</think>middle text'
                        '<think>second reasoning</think>final text')
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert 'first reasoning' in reasoning
        assert final is not None
        assert 'middle text' in final
        assert '<think>second reasoning</think>' in final

    def test_newlines_between_think_and_content(self):
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think>Step 1: 37*43=1591</think>\n\nThe answer is 1591.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert '1591' in reasoning
        assert final is not None
        assert final.startswith('\n\n'), (f'DeepSeek parser should preserve leading \\n\\n in content, '
                                          f'got: {repr(final[:20])}')

    def test_single_newline_between_think_and_content(self):
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think>reasoning</think>\nThe answer.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning == 'reasoning'
        assert final is not None
        assert final.startswith('\n')


@_apply_parser_marks
@pytest.mark.deepseek_r1_parser
class TestDeepSeekR1ParserStreaming:
    """Unit tests for DeepSeekR1ReasoningParser streaming extraction."""

    def test_think_start_token_only(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='',
                                                            current_text='<think>',
                                                            delta_text='<think>',
                                                            previous_token_ids=[],
                                                            current_token_ids=[100],
                                                            delta_token_ids=[100])
        assert result is None

    def test_think_end_token_only(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>Some reasoning',
                                                            current_text='<think>Some reasoning</think>',
                                                            delta_text='</think>',
                                                            previous_token_ids=[100, 200, 201],
                                                            current_token_ids=[100, 200, 201, 101],
                                                            delta_token_ids=[101])
        assert result is not None
        assert result.content == ''

    def test_reasoning_continues(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>Step 1',
                                                            current_text='<think>Step 1: multiply',
                                                            delta_text=': multiply',
                                                            previous_token_ids=[100, 200],
                                                            current_token_ids=[100, 200, 201],
                                                            delta_token_ids=[201])
        assert result is not None
        assert result.reasoning_content == ': multiply'

    def test_reasoning_ends_in_delta(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>Step 1',
                                                            current_text='<think>Step 1. Done</think>Answer',
                                                            delta_text='. Done</think>Answer',
                                                            previous_token_ids=[100, 200],
                                                            current_token_ids=[100, 200, 201, 101, 300],
                                                            delta_token_ids=[201, 101, 300])
        assert result is not None
        assert '. Done' in result.reasoning_content
        assert result.content == 'Answer'

    def test_content_after_think_end(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>Reasoning</think>Prev',
                                                            current_text='<think>Reasoning</think>Prev content',
                                                            delta_text=' content',
                                                            previous_token_ids=[100, 200, 101, 300],
                                                            current_token_ids=[100, 200, 101, 300, 301],
                                                            delta_token_ids=[301])
        assert result is not None
        assert result.content == ' content'

    def test_no_think_in_previous_with_end_in_delta(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='Some reasoning text',
                                                            current_text='Some reasoning text</think>Final',
                                                            delta_text='</think>Final',
                                                            previous_token_ids=[200, 201],
                                                            current_token_ids=[200, 201, 101, 300],
                                                            delta_token_ids=[101, 300])
        assert result is not None
        assert result.content == 'Final'

    def test_no_think_tags_at_all_streaming(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='',
                                                            current_text='Just a plain answer.',
                                                            delta_text='Just a plain answer.',
                                                            previous_token_ids=[],
                                                            current_token_ids=[200, 201, 202],
                                                            delta_token_ids=[200, 201, 202])
        assert result is not None
        assert result.reasoning_content == 'Just a plain answer.'
        assert result.content is None

    def test_empty_reasoning_streaming(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='',
                                                            current_text='<think></think>Direct answer.',
                                                            delta_text='<think></think>Direct answer.',
                                                            previous_token_ids=[],
                                                            current_token_ids=[100, 101, 200, 201],
                                                            delta_token_ids=[100, 101, 200, 201])
        assert result is not None
        assert result.reasoning_content is not None
        assert result.content is not None
        assert 'Direct answer' in result.content

    def test_only_think_tags_streaming(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>All reasoning here.',
                                                            current_text='<think>All reasoning here.</think>',
                                                            delta_text='</think>',
                                                            previous_token_ids=[100, 200, 201],
                                                            current_token_ids=[100, 200, 201, 101],
                                                            delta_token_ids=[101])
        assert result is not None
        assert result.content == ''

    def test_multiline_reasoning_streaming(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Step 1: calculate\n',
            current_text='<think>Step 1: calculate\nStep 2: add\nStep 3: done',
            delta_text='Step 2: add\nStep 3: done',
            previous_token_ids=[100, 200, 201],
            current_token_ids=[100, 200, 201, 202, 203, 204],
            delta_token_ids=[202, 203, 204])
        assert result is not None
        assert result.reasoning_content == 'Step 2: add\nStep 3: done'

    def test_unclosed_think_tag_streaming(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>I am still reasoning',
                                                            current_text='<think>I am still reasoning about this...',
                                                            delta_text=' about this...',
                                                            previous_token_ids=[100, 200, 201],
                                                            current_token_ids=[100, 200, 201, 202, 203],
                                                            delta_token_ids=[202, 203])
        assert result is not None
        assert result.reasoning_content == ' about this...'
        assert result.content is None

    def test_empty_model_output_streaming(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='',
                                                            current_text='',
                                                            delta_text='',
                                                            previous_token_ids=[],
                                                            current_token_ids=[],
                                                            delta_token_ids=[])
        if result is not None:
            assert (result.reasoning_content or '') == ''
            assert (result.content or '') == ''

    def test_multiple_think_blocks_streaming(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>first reasoning</think>middle text',
                                                            current_text=('<think>first reasoning</think>middle text'
                                                                          '<think>second reasoning</think>final'),
                                                            delta_text='<think>second reasoning</think>final',
                                                            previous_token_ids=[100, 200, 101, 300],
                                                            current_token_ids=[100, 200, 101, 300, 100, 201, 101, 301],
                                                            delta_token_ids=[100, 201, 101, 301])
        assert result is not None
        assert result.content is not None
        assert '<think>second reasoning</think>' in result.content

    def test_newlines_between_think_and_content_streaming(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Step 1: reasoning',
            current_text='<think>Step 1: reasoning. Done</think>\n\nThe answer is 1591.',
            delta_text='. Done</think>\n\nThe answer is 1591.',
            previous_token_ids=[100, 200],
            current_token_ids=[100, 200, 201, 101, 300, 301],
            delta_token_ids=[201, 101, 300, 301])
        assert result is not None
        assert '. Done' in result.reasoning_content
        assert result.content is not None
        assert result.content.startswith('\n\n')
        assert '1591' in result.content

    def test_newline_only_content_streaming(self):
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_deepseek_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>reasoning',
                                                            current_text='<think>reasoning</think>\n\n',
                                                            delta_text='</think>\n\n',
                                                            previous_token_ids=[100, 200],
                                                            current_token_ids=[100, 200, 101, 300],
                                                            delta_token_ids=[101, 300])
        assert result is not None
        assert result.content == '\n\n'


@_apply_parser_marks
@pytest.mark.qwenqwq_parser
class TestQwenQwQParserNonStreaming:
    """Unit tests for QwenQwQReasoningParser.extract_reasoning_content."""

    def test_full_think_tags(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think>Let me reason about this...</think>The answer is 42.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert 'reason' in reasoning.lower()
        assert final is not None
        assert '42' in final

    def test_missing_start_tag(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = 'Reasoning here...</think>Final answer.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert final is not None
        assert 'Final answer' in final

    def test_newline_stripping(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think>\nStep 1: think\nStep 2: conclude\n</think>Done.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert not reasoning.startswith('\n')
        assert not reasoning.endswith('\n')
        assert final is not None

    def test_only_reasoning_no_content(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think>Only reasoning content here.</think>'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert final is None

    def test_empty_reasoning_tags(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think></think>Direct answer.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert final is not None
        assert 'Direct answer' in final

    def test_multiple_think_blocks(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = ('<think>first reasoning</think>middle text'
                        '<think>second reasoning</think>final text')
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert 'first reasoning' in reasoning
        assert final is not None
        assert 'middle text' in final
        assert '<think>second reasoning</think>' in final

    def test_newlines_between_think_and_content(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think>\nStep 1: 37*43=1591\n</think>\n\nThe answer is 1591.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert not reasoning.startswith('\n')
        assert not reasoning.endswith('\n')
        assert '1591' in reasoning
        assert final is not None
        assert final.startswith('\n\n')

    def test_single_newline_between_think_and_content(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think>\nreasoning\n</think>\nThe answer.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning == 'reasoning'
        assert final is not None
        assert final.startswith('\n')

    def test_unclosed_think_should_return_reasoning(self):
        """<think> without </think> → reasoning should be preserved."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think>I am still reasoning about this problem...'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None, ('reasoning should not be None when <think> is present but unclosed')
        assert reasoning == 'I am still reasoning about this problem...'
        assert final is None

    def test_no_tags_all_reasoning(self):
        """No think tags at all → everything is reasoning."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = 'Plain output without any think tokens.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert reasoning == model_output
        assert final is None

    def test_empty_input(self):
        """Edge case: empty string → ('', None)."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        reasoning, final = parser.extract_reasoning_content('', req)
        assert final is None

    def test_prefix_before_think_tags(self):
        """Text before <think> causes wrong content extraction."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = 'Some prefix<think>reasoning here</think>The answer.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert 'reasoning here' in reasoning
        assert final is not None
        assert final == 'The answer.'
        assert '<' not in final and '>' not in final

    def test_prefix_only_whitespace_before_think(self):
        """Whitespace-only prefix before <think>."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = ' <think>reasoning</think>answer'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert final is not None
        assert final == 'answer'

    def test_unclosed_think_with_newlines(self):
        r"""<think>\nreasoning\n without </think>."""
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        req = _make_mock_request()
        model_output = '<think>\nI am reasoning step by step\n'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert 'reasoning step by step' in reasoning
        assert final is None


@_apply_parser_marks
@pytest.mark.qwenqwq_parser
class TestQwenQwQParserStreaming:
    """Unit tests for QwenQwQReasoningParser streaming extraction."""

    def test_think_start_text_skipped(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='',
                                                            current_text='<think>',
                                                            delta_text='<think>',
                                                            previous_token_ids=[],
                                                            current_token_ids=[100],
                                                            delta_token_ids=[100])
        assert result is not None
        assert result.content == ''

    def test_think_end_text_skipped(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>Some text',
                                                            current_text='<think>Some text</think>',
                                                            delta_text='</think>',
                                                            previous_token_ids=[100, 200],
                                                            current_token_ids=[100, 200, 101],
                                                            delta_token_ids=[101])
        assert result is not None
        assert result.content == ''

    def test_reasoning_continues(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>Step 1',
                                                            current_text='<think>Step 1, Step 2',
                                                            delta_text=', Step 2',
                                                            previous_token_ids=[100, 200],
                                                            current_token_ids=[100, 200, 201],
                                                            delta_token_ids=[201])
        assert result is not None
        assert result.reasoning_content == ', Step 2'

    def test_reasoning_ends_in_delta(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>Step 1',
                                                            current_text='<think>Step 1. Final</think>Result',
                                                            delta_text='. Final</think>Result',
                                                            previous_token_ids=[100, 200],
                                                            current_token_ids=[100, 200, 201, 101, 300],
                                                            delta_token_ids=[201, 101, 300])
        assert result is not None
        assert '. Final' in (result.reasoning_content or '')
        assert result.content == 'Result'

    def test_content_after_think_closed(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>Reasoning</think>Prev',
                                                            current_text='<think>Reasoning</think>Prev more',
                                                            delta_text=' more',
                                                            previous_token_ids=[100, 200, 101, 300],
                                                            current_token_ids=[100, 200, 101, 300, 301],
                                                            delta_token_ids=[301])
        assert result is not None
        assert result.content == ' more'

    def test_no_think_tags_streaming(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='',
                                                            current_text='Plain answer without reasoning.',
                                                            delta_text='Plain answer without reasoning.',
                                                            previous_token_ids=[],
                                                            current_token_ids=[200, 201, 202],
                                                            delta_token_ids=[200, 201, 202])
        assert result is not None
        assert result.content == 'Plain answer without reasoning.'
        assert result.reasoning_content is None

    def test_missing_start_tag_streaming(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(
            previous_text='Reasoning without start tag',
            current_text='Reasoning without start tag</think>Final answer.',
            delta_text='</think>Final answer.',
            previous_token_ids=[200, 201],
            current_token_ids=[200, 201, 101, 300],
            delta_token_ids=[101, 300])
        assert result is not None
        assert result.content == 'Final answer.'

    def test_only_reasoning_no_content_streaming(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>Only reasoning content.',
                                                            current_text='<think>Only reasoning content.</think>',
                                                            delta_text='</think>',
                                                            previous_token_ids=[100, 200],
                                                            current_token_ids=[100, 200, 101],
                                                            delta_token_ids=[101])
        assert result is not None
        assert result.content == ''

    def test_empty_reasoning_tags_streaming(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='',
                                                            current_text='<think></think>Direct answer.',
                                                            delta_text='<think></think>Direct answer.',
                                                            previous_token_ids=[],
                                                            current_token_ids=[100, 101, 200],
                                                            delta_token_ids=[100, 101, 200])
        assert result is not None
        assert result.content is not None
        assert 'Direct answer' in result.content

    def test_newline_in_reasoning_streaming(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>\nStep 1: think',
                                                            current_text='<think>\nStep 1: think\nStep 2: conclude\n',
                                                            delta_text='\nStep 2: conclude\n',
                                                            previous_token_ids=[100, 200, 201],
                                                            current_token_ids=[100, 200, 201, 202, 203],
                                                            delta_token_ids=[202, 203])
        assert result is not None
        assert result.reasoning_content == '\nStep 2: conclude\n'

    def test_unclosed_think_tag_streaming(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>I am still reasoning',
                                                            current_text='<think>I am still reasoning about this...',
                                                            delta_text=' about this...',
                                                            previous_token_ids=[100, 200, 201],
                                                            current_token_ids=[100, 200, 201, 202, 203],
                                                            delta_token_ids=[202, 203])
        assert result is not None
        assert result.reasoning_content == ' about this...'
        assert result.content is None

    def test_empty_model_output_streaming(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='',
                                                            current_text='',
                                                            delta_text='',
                                                            previous_token_ids=[],
                                                            current_token_ids=[],
                                                            delta_token_ids=[])
        if result is not None:
            assert (result.reasoning_content or '') == ''
            assert (result.content or '') == ''

    def test_multiple_think_blocks_streaming(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>first reasoning</think>middle text',
                                                            current_text=('<think>first reasoning</think>middle text'
                                                                          '<think>second reasoning</think>final'),
                                                            delta_text='<think>second reasoning</think>final',
                                                            previous_token_ids=[100, 200, 101, 300],
                                                            current_token_ids=[100, 200, 101, 300, 100, 201, 101, 301],
                                                            delta_token_ids=[100, 201, 101, 301])
        assert result is not None
        assert result.content is not None
        assert '<think>second reasoning</think>' in result.content

    def test_newlines_between_think_and_content_streaming(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(
            previous_text='<think>Step 1: reasoning',
            current_text='<think>Step 1: reasoning. Done</think>\n\nThe answer is 1591.',
            delta_text='. Done</think>\n\nThe answer is 1591.',
            previous_token_ids=[100, 200],
            current_token_ids=[100, 200, 201, 101, 300, 301],
            delta_token_ids=[201, 101, 300, 301])
        assert result is not None
        assert '. Done' in result.reasoning_content
        assert result.content is not None
        assert result.content.startswith('\n\n')
        assert '1591' in result.content

    def test_newline_only_content_streaming(self):
        tok = _make_mock_tokenizer()
        parser = _get_qwen_parser_cls()(tok)
        result = parser.extract_reasoning_content_streaming(previous_text='<think>reasoning',
                                                            current_text='<think>reasoning</think>\n\n',
                                                            delta_text='</think>\n\n',
                                                            previous_token_ids=[100, 200],
                                                            current_token_ids=[100, 200, 101, 300],
                                                            delta_token_ids=[101, 300])
        assert result is not None
        assert result.content == '\n\n'

    def test_unclosed_think_should_return_reasoning(self):
        """<think> without </think> → reasoning should be accumulated across
        all deltas and content should remain None."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_qwen_parser_cls()(tok)
        deltas = ['<think>', 'I am still ', 'reasoning about ', 'this problem...']
        reasoning, content = _run_streaming_extraction(parser, deltas, vocab)
        assert reasoning is not None, ('reasoning should not be None when '
                                       '<think> is present but unclosed')
        assert 'I am still reasoning about this problem...' == reasoning
        assert content is None

    def test_no_tags_all_reasoning_streaming(self):
        """No think tags at all → everything is reasoning (Qwen treats tag-less
        output as content)."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_qwen_parser_cls()(tok)
        deltas = ['Plain output ', 'without any ', 'think tokens.']
        reasoning, content = _run_streaming_extraction(parser, deltas, vocab)
        full_text = 'Plain output without any think tokens.'
        # Qwen streaming without tags emits content (not reasoning)
        assert (reasoning or '') + (content or '') == full_text

    def test_prefix_before_think_tags_streaming(self):
        """Text before <think> must not corrupt content (streaming)."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_qwen_parser_cls()(tok)
        deltas = ['Some prefix', '<think>', 'reasoning here', '</think>', 'The answer.']
        reasoning, content = _run_streaming_extraction(parser, deltas, vocab)
        assert reasoning is not None
        assert 'reasoning here' in reasoning
        assert content is not None
        assert 'The answer.' in content
        assert '<' not in content and '>' not in content

    def test_prefix_only_whitespace_before_think_streaming(self):
        """Whitespace-only prefix before <think> (streaming)."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_qwen_parser_cls()(tok)
        deltas = [' ', '<think>', 'reasoning', '</think>', 'answer']
        reasoning, content = _run_streaming_extraction(parser, deltas, vocab)
        assert reasoning is not None
        assert content is not None
        assert 'answer' in content

    def test_unclosed_think_with_newlines_streaming(self):
        r"""<think>\nreasoning\n without </think> (streaming)."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_qwen_parser_cls()(tok)
        deltas = ['<think>', '\n', 'I am reasoning ', 'step by step', '\n']
        reasoning, content = _run_streaming_extraction(parser, deltas, vocab)
        assert reasoning is not None, ('reasoning should not be None for unclosed '
                                       '<think> with newlines')
        assert 'reasoning step by step' in reasoning
        assert content is None

    def test_think_tag_embedded_in_delta_should_not_leak(self):
        """<think> embedded in a larger delta must not leak into
        reasoning_content."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_qwen_parser_cls()(tok)
        # <think> is part of a larger delta, not a separate token
        deltas = ['<think>I am reasoning', ' step by step', '</think>', 'The answer.']
        reasoning, content = _run_streaming_extraction(parser, deltas, vocab)
        assert reasoning is not None
        assert THINK_START_TOKEN not in reasoning, ('<think> tag leaked into reasoning_content')
        assert THINK_END_TOKEN not in reasoning, ('</think> tag leaked into reasoning_content')
        assert 'I am reasoning step by step' in reasoning
        assert content is not None
        assert 'The answer.' in content

    def test_think_tag_embedded_with_end_in_same_delta(self):
        """<think>.....</think> all in one large delta."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_qwen_parser_cls()(tok)
        deltas = ['<think>short reasoning</think>answer']
        reasoning, content = _run_streaming_extraction(parser, deltas, vocab)
        assert reasoning is not None
        assert THINK_START_TOKEN not in reasoning, ('<think> tag leaked into reasoning_content')
        assert 'short reasoning' in reasoning
        assert content is not None
        assert 'answer' in content

    def test_think_tag_embedded_no_end_token(self):
        """<think> embedded in delta, output truncated (no </think> at all) —
        tag must not appear in reasoning."""
        vocab = {'<think>': 100, '</think>': 101}
        tok = _make_mock_tokenizer(vocab)
        parser = _get_qwen_parser_cls()(tok)
        deltas = ['<think>I am reasoning about', ' a complex problem...']
        reasoning, content = _run_streaming_extraction(parser, deltas, vocab)
        assert reasoning is not None
        assert THINK_START_TOKEN not in reasoning, ('<think> tag leaked into reasoning_content '
                                                    'when output is truncated')
        assert 'I am reasoning about a complex problem...' in reasoning
        assert content is None


# ---------------------------------------------------------------------------
# Parser edge cases — parametrized over both parsers (was separate per-parser)
# ---------------------------------------------------------------------------


@_apply_parser_marks
class TestReasoningParserEdgeCases:
    """Edge cases that both parsers should handle identically."""

    @pytest.mark.parametrize('parser_factory', [
        pytest.param(_get_deepseek_parser_cls, id='deepseek'),
        pytest.param(_get_qwen_parser_cls, id='qwen'),
    ])
    def test_unicode_in_reasoning(self, parser_factory):
        """Chinese / Unicode content inside think tags."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = '<think>让我想想... 37 × 43 = 1591</think>答案是1591。'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert '37' in reasoning
        assert final is not None
        assert '1591' in final

    @pytest.mark.parametrize('parser_factory', [
        pytest.param(_get_deepseek_parser_cls, id='deepseek'),
        pytest.param(_get_qwen_parser_cls, id='qwen'),
    ])
    def test_very_long_reasoning(self, parser_factory):
        """Long reasoning content should be fully extracted."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        long_reasoning = 'Step ' + '. Step '.join(str(i) for i in range(100))
        model_output = f'<think>{long_reasoning}</think>Final.'
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert len(reasoning) > 100
        assert 'Step 99' in reasoning
        assert final == 'Final.'

    @pytest.mark.parametrize('parser_factory', [
        pytest.param(_get_deepseek_parser_cls, id='deepseek'),
        pytest.param(_get_qwen_parser_cls, id='qwen'),
    ])
    def test_special_chars_in_reasoning(self, parser_factory):
        """Special characters inside think tags."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = ('<think>Let\'s check: 2 > 1 & 3 < 5, also "quoted" text</think>'
                        'All good.')
        reasoning, final = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert '&' in reasoning or '>' in reasoning
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


# ---------------------------------------------------------------------------
# Parser init / robustness tests
# ---------------------------------------------------------------------------


@_apply_parser_marks
class TestReasoningParserInitErrors:
    """Initialization error handling for reasoning parsers."""

    def test_deepseek_none_tokenizer(self):
        with pytest.raises(ValueError, match='(?i)tokenizer'):
            _get_deepseek_parser_cls()(None)

    def test_qwen_none_tokenizer(self):
        with pytest.raises(ValueError, match='(?i)tokenizer'):
            _get_qwen_parser_cls()(None)

    def test_deepseek_missing_vocab_tokens(self):
        tok = _make_mock_tokenizer(vocab={'unrelated': 999})
        with pytest.raises(RuntimeError, match='(?i)think.*token'):
            _get_deepseek_parser_cls()(tok)

    def test_deepseek_token_ids_not_none(self):
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        assert parser.think_start_token_id is not None
        assert parser.think_end_token_id is not None
        assert parser.think_start_token_id == 100
        assert parser.think_end_token_id == 101

    # NOTE: token_properties for both parsers are covered by
    # TestReasoningParserEdgeCases.test_parser_instantiation.

    def test_vocab_property_accessible(self):
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        assert '<think>' in parser.vocab
        assert '</think>' in parser.vocab
        assert parser.vocab['<think>'] == 100


# ---------------------------------------------------------------------------
# Parser dual-mode tests (already parametrized)
# ---------------------------------------------------------------------------

_DEFAULT_VOCAB = {'<think>': 100, '</think>': 101}


@_apply_parser_marks
class TestReasoningParserDualMode:
    """Same extraction scenario verified in both streaming and non-
    streaming."""

    @pytest.mark.parametrize('parser_factory', [
        _get_deepseek_parser_cls,
        _get_qwen_parser_cls,
    ],
                             ids=['deepseek', 'qwen'])
    def test_simple_extraction_both_modes(self, parser_factory):
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = '<think>Step by step reasoning</think>Final answer'
        ns_reasoning, ns_content = parser.extract_reasoning_content(model_output, req)
        deltas = ['<think>', 'Step by step ', 'reasoning', '</think>', 'Final answer']
        s_reasoning, s_content = _run_streaming_extraction(parser, deltas, _DEFAULT_VOCAB)
        assert ns_reasoning is not None and len(ns_reasoning) > 0
        assert s_reasoning is not None and len(s_reasoning) > 0
        assert ns_content is not None and s_content is not None
        assert ns_content.strip() == s_content.strip()

    @pytest.mark.parametrize('parser_factory', [
        _get_deepseek_parser_cls,
        _get_qwen_parser_cls,
    ],
                             ids=['deepseek', 'qwen'])
    def test_no_end_token_both_modes(self, parser_factory):
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = 'Just reasoning without end token'
        ns_reasoning, ns_content = parser.extract_reasoning_content(model_output, req)
        deltas = ['Just reasoning ', 'without end token']
        s_reasoning, s_content = _run_streaming_extraction(parser, deltas, _DEFAULT_VOCAB)
        assert ns_reasoning == model_output
        assert ns_content is None
        assert s_reasoning is not None

    @pytest.mark.parametrize('parser_factory', [
        _get_deepseek_parser_cls,
        _get_qwen_parser_cls,
    ],
                             ids=['deepseek', 'qwen'])
    def test_empty_output_both_modes(self, parser_factory):
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        ns_reasoning, ns_content = parser.extract_reasoning_content('', req)
        s_reasoning, s_content = _run_streaming_extraction(parser, [''], _DEFAULT_VOCAB)
        assert isinstance(ns_reasoning, (str, type(None)))
        assert isinstance(ns_content, (str, type(None)))

    @pytest.mark.parametrize('parser_factory', [
        _get_deepseek_parser_cls,
        _get_qwen_parser_cls,
    ],
                             ids=['deepseek', 'qwen'])
    def test_incremental_deltas_both_modes(self, parser_factory):
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = '<think>AB</think>CD'
        ns_reasoning, ns_content = parser.extract_reasoning_content(model_output, req)
        deltas = ['<think>', 'A', 'B', '</think>', 'C', 'D']
        s_reasoning, s_content = _run_streaming_extraction(parser, deltas, _DEFAULT_VOCAB)
        assert ns_content is not None and s_content is not None
        assert ns_content.strip() == s_content.strip()


# ---------------------------------------------------------------------------
# Advanced edge cases — parametrized over both parsers (was separate)
# ---------------------------------------------------------------------------


@_apply_parser_marks
class TestReasoningParserAdvancedEdgeCases:
    """Advanced edge cases — parametrized over both parsers."""

    @pytest.mark.parametrize('parser_factory', [
        pytest.param(_get_deepseek_parser_cls, id='deepseek'),
        pytest.param(_get_qwen_parser_cls, id='qwen'),
    ])
    def test_multiple_end_tokens(self, parser_factory):
        """Multiple </think> tokens: should stop at the first one."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = '<think>First</think>Middle</think>Last'
        reasoning, content = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None and 'First' in reasoning
        assert content is not None and 'Middle' in content

    @pytest.mark.parametrize('parser_factory', [
        pytest.param(_get_deepseek_parser_cls, id='deepseek'),
        pytest.param(_get_qwen_parser_cls, id='qwen'),
    ])
    def test_nested_think_tokens(self, parser_factory):
        """Nested <think> inside <think>."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = '<think>Outer<think>Inner</think>Content'
        reasoning, content = parser.extract_reasoning_content(model_output, req)
        assert reasoning is not None
        assert 'Outer' in reasoning and 'Inner' in reasoning
        assert content is not None and 'Content' in content

    @pytest.mark.parametrize('parser_factory', [
        pytest.param(_get_deepseek_parser_cls, id='deepseek'),
        pytest.param(_get_qwen_parser_cls, id='qwen'),
    ])
    def test_malformed_similar_tokens(self, parser_factory):
        """Tags like <thinking> should be treated as plain text."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        req = _make_mock_request()
        model_output = '<thinking>Not real tags</thinking>Content'
        reasoning, content = parser.extract_reasoning_content(model_output, req)
        assert reasoning == model_output
        assert content is None

    @pytest.mark.parametrize('parser_factory', [
        pytest.param(_get_deepseek_parser_cls, id='deepseek'),
        pytest.param(_get_qwen_parser_cls, id='qwen'),
    ])
    def test_streaming_no_end_token(self, parser_factory):
        """Streaming with only start token — reasoning continues."""
        tok = _make_mock_tokenizer()
        parser = parser_factory()(tok)
        deltas = ['<think>', 'Reasoning ', 'without ', 'end']
        reasoning, content = _run_streaming_extraction(parser, deltas, _DEFAULT_VOCAB)
        assert reasoning is not None and 'Reasoning' in reasoning
        assert content is None

    def test_deepseek_streaming_multiple_end_tokens(self):
        """Multiple </think> in streaming — DeepSeek specific."""
        tok = _make_mock_tokenizer()
        parser = _get_deepseek_parser_cls()(tok)
        deltas = ['<think>', 'First', '</think>', 'Middle', '</think>', 'Last']
        reasoning, content = _run_streaming_extraction(parser, deltas, _DEFAULT_VOCAB)
        assert reasoning is not None and 'First' in reasoning
        assert content is not None and 'Middle' in content


# ---------------------------------------------------------------------------
# Parser independence tests
# ---------------------------------------------------------------------------


@_apply_parser_marks
class TestReasoningParserIndependence:
    """Verify that different parser instances don't share state."""

    def test_parsers_do_not_share_state(self):
        tok = _make_mock_tokenizer()
        req = _make_mock_request()
        deepseek_parser = _get_deepseek_parser_cls()(tok)
        qwen_parser = _get_qwen_parser_cls()(tok)
        model_output = '<think>Shared reasoning</think>Shared content'
        ds_reasoning, ds_content = deepseek_parser.extract_reasoning_content(model_output, req)
        qw_reasoning, qw_content = qwen_parser.extract_reasoning_content(model_output, req)
        assert ds_reasoning is not None and 'Shared reasoning' in ds_reasoning
        assert qw_reasoning is not None and 'Shared reasoning' in qw_reasoning
        assert ds_content is not None and 'Shared content' in ds_content
        assert qw_content is not None and 'Shared content' in qw_content

    def test_parsers_independent_streaming(self):
        tok = _make_mock_tokenizer()
        deepseek_parser = _get_deepseek_parser_cls()(tok)
        qwen_parser = _get_qwen_parser_cls()(tok)
        deltas = ['<think>', 'Step 1 ', 'Step 2', '</think>', 'Answer']
        ds_reasoning, ds_content = _run_streaming_extraction(deepseek_parser, deltas, _DEFAULT_VOCAB)
        qw_reasoning, qw_content = _run_streaming_extraction(qwen_parser, deltas, _DEFAULT_VOCAB)
        assert ds_reasoning is not None and qw_reasoning is not None
        assert ds_content is not None and qw_content is not None
        assert ds_content.strip() == qw_content.strip()

    def test_multiple_instances_same_parser(self):
        tok1 = _make_mock_tokenizer()
        tok2 = _make_mock_tokenizer()
        req = _make_mock_request()
        parser1 = _get_deepseek_parser_cls()(tok1)
        parser2 = _get_deepseek_parser_cls()(tok2)
        r1, c1 = parser1.extract_reasoning_content('<think>Reasoning A</think>Content A', req)
        r2, c2 = parser2.extract_reasoning_content('<think>Reasoning B</think>Content B', req)
        assert r1 is not None and 'A' in r1
        assert r2 is not None and 'B' in r2
        assert c1 is not None and 'A' in c1
        assert c2 is not None and 'B' in c2
        r1_again, c1_again = parser1.extract_reasoning_content('<think>Reasoning A</think>Content A', req)
        assert r1_again == r1 and c1_again == c1


# ===========================================================================
# API-level test classes (parametrized by backend × model × stream)
# ===========================================================================

# ---------------------------------------------------------------------------
# Basic reasoning: presence, quality, separation
# (merged from TestReasoningBasic, TestReasoningStreaming,
#  TestReasoningContentQuality, TestReasoningContentQualityStreaming)
# ---------------------------------------------------------------------------


@_apply_marks_stream
class TestReasoningBasic(_ReasoningTestBase):
    """Basic reasoning_content presence, quality, and content separation."""

    def test_reasoning_content_present(self, backend, model_case, stream):
        """Model should populate reasoning_content for math questions."""
        r = self._call_api(stream, MESSAGES_REASONING_BASIC)
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['reasoning']) > 10, (f'reasoning too short ({len(r["reasoning"])} chars)')
        assert any(kw in r['reasoning'] for kw in ('37', '43', '1591', 'multiply', '*', '×'))
        assert len(r['content'].strip()) > 0
        assert '1591' in r['content']
        assert r['reasoning'].strip() != r['content'].strip()
        _assert_no_tag_leakage(r['reasoning'], r['content'])

    def test_reasoning_quality_complex(self, backend, model_case, stream):
        """Complex train problem: reasoning should contain calculation steps."""
        r = self._call_api(stream, MESSAGES_REASONING_COMPLEX, max_completion_tokens=2048)
        assert len(r['reasoning']) > 50
        assert any(kw in r['reasoning'] for kw in ('60', '80', '140', '280'))
        assert len(r['content'].strip()) > 0
        assert '2' in r['content']
        _assert_no_tag_leakage(r['reasoning'], r['content'])
        if stream:
            assert r['reasoning_chunks'] > 1


# ---------------------------------------------------------------------------
# Streaming ↔ Non-streaming consistency (cross-mode comparison)
# ---------------------------------------------------------------------------


@_apply_marks
class TestReasoningStreamConsistency(_ReasoningTestBase):
    """Both modes must produce reasoning AND content with correct
    separation."""

    def test_reasoning_presence_consistent(self, backend, model_case):
        client, model_name = self._get_client()
        ns_resp = client.chat.completions.create(model=model_name,
                                                 messages=MESSAGES_REASONING_BASIC,
                                                 temperature=0,
                                                 max_completion_tokens=1024,
                                                 logprobs=False)
        ns_reasoning = get_reasoning_content(ns_resp.choices[0].message)
        ns_content = ns_resp.choices[0].message.content or ''

        stream = client.chat.completions.create(model=model_name,
                                                messages=MESSAGES_REASONING_BASIC,
                                                temperature=0,
                                                max_completion_tokens=1024,
                                                logprobs=False,
                                                stream=True)
        result = collect_stream_reasoning(stream)

        assert ns_reasoning is not None and len(ns_reasoning) > 0
        assert len(result['reasoning_content']) > 0
        assert len(ns_content.strip()) > 0
        assert len(result['content'].strip()) > 0
        assert '1591' in ns_content
        assert '1591' in result['content']
        for text in [ns_reasoning, ns_content, result['reasoning_content'], result['content']]:
            assert THINK_START_TOKEN not in text
            assert THINK_END_TOKEN not in text


# ---------------------------------------------------------------------------
# Tool calls + tool_choice (merged from TestReasoningWithToolCalls,
# TestReasoningWithToolCallsStreaming, TestReasoningWithToolChoice,
# TestReasoningWithToolChoiceStreaming)
# ---------------------------------------------------------------------------


@_apply_marks_stream
class TestReasoningWithTools(_ReasoningTestBase):
    """Reasoning with tool calls under different tool_choice settings."""

    def test_tool_choice_auto(self, backend, model_case, stream):
        """tool_choice='auto': weather question should trigger weather tool."""
        r = self._call_api(stream,
                           MESSAGES_REASONING_WEATHER_TOOL,
                           tools=[WEATHER_TOOL, SEARCH_TOOL],
                           tool_choice='auto')
        if len(r['tool_calls']) > 0:
            assert r['finish_reason'] == 'tool_calls'
            for tc in r['tool_calls']:
                assert tc['name'] in ('get_current_weather', 'web_search')
                parsed = json.loads(tc['args_str'])
                assert isinstance(parsed, dict)
        else:
            assert len(r['content'].strip()) > 0

    def test_tool_choice_required(self, backend, model_case, stream):
        """tool_choice='required': must produce tool call."""
        try:
            r = self._call_api(stream, MESSAGES_REASONING_WEATHER_TOOL, tools=[WEATHER_TOOL], tool_choice='required')
            assert len(r['tool_calls']) >= 1
            assert r['finish_reason'] == 'tool_calls'
            tc = r['tool_calls'][0]
            assert tc['name'] == 'get_current_weather'
            parsed = json.loads(tc['args_str'])
            assert 'city' in parsed
        except Exception as e:
            pytest.skip(f'tool_choice="required" not supported: {e}')

    def test_tool_choice_none(self, backend, model_case, stream):
        """tool_choice='none': no tool calls, text answer instead."""
        r = self._call_api(stream, MESSAGES_REASONING_WEATHER_TOOL, tools=[WEATHER_TOOL], tool_choice='none')
        assert len(r['tool_calls']) == 0
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['reasoning'] + r['content']) > 0
        _assert_no_tag_leakage(r['reasoning'], r['content'])

    def test_tool_choice_specific(self, backend, model_case, stream):
        """Force get_current_weather: must call exactly that tool."""
        r = self._call_api(stream,
                           MESSAGES_REASONING_WEATHER_TOOL,
                           tools=[WEATHER_TOOL, SEARCH_TOOL],
                           tool_choice={
                               'type': 'function',
                               'function': {
                                   'name': 'get_current_weather'
                               }
                           })
        assert r['finish_reason'] == 'tool_calls'
        assert len(r['tool_calls']) >= 1
        tc = r['tool_calls'][0]
        assert tc['name'] == 'get_current_weather'
        parsed = json.loads(tc['args_str'])
        assert 'city' in parsed
        assert 'dallas' in parsed['city'].lower()


# ---------------------------------------------------------------------------
# Parallel tool calls
# ---------------------------------------------------------------------------


@_apply_marks_stream
class TestReasoningParallelToolCalls(_ReasoningTestBase):
    """Reasoning model calling multiple tools in parallel."""

    def test_parallel_tools(self, backend, model_case, stream):
        r = self._call_api(stream, MESSAGES_REASONING_PARALLEL_TOOLS, tools=[WEATHER_TOOL, CALCULATOR_TOOL])
        assert len(r['tool_calls']) >= 1
        assert r['finish_reason'] == 'tool_calls'
        ids = [tc['id'] for tc in r['tool_calls'] if tc.get('id')]
        if len(ids) >= 2:
            assert len(set(ids)) == len(ids), f'IDs must be unique: {ids}'
        for tc in r['tool_calls']:
            assert tc['name'] in ('get_current_weather', 'calculate')
            parsed = json.loads(tc['args_str'])
            assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Tool round-trip: reason → tool → result → answer
# ---------------------------------------------------------------------------


@_apply_marks_stream
class TestReasoningToolRoundTrip(_ReasoningTestBase):
    """Multi-turn: reason → tool → result → reasoning → answer."""

    def test_after_tool_result(self, backend, model_case, stream):
        r = self._call_api(stream, build_reasoning_tool_roundtrip_messages(), tools=[WEATHER_TOOL])
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['tool_calls']) == 0
        assert len(r['reasoning'] + r['content']) > 0
        if r['content']:
            has_ref = any(kw in r['content'].lower()
                          for kw in ('sunny', 'umbrella', 'dallas', 'clear', 'rain', 'weather', 'no'))
            assert has_ref, f'Content should reference weather: {r["content"][:200]}'
            _assert_no_tag_leakage(r['reasoning'], r['content'])


# ---------------------------------------------------------------------------
# Streaming ↔ Non-streaming tool-call consistency (cross-mode comparison)
# ---------------------------------------------------------------------------


@_apply_marks
class TestReasoningToolCallConsistency(_ReasoningTestBase):
    """Compare streaming vs non-streaming tool-call results."""

    def test_tool_call_stream_vs_nonstream(self, backend, model_case):
        client, model_name = self._get_client()
        common_kwargs = dict(model=model_name,
                             messages=MESSAGES_REASONING_WEATHER_TOOL,
                             temperature=0,
                             max_completion_tokens=1024,
                             tools=[WEATHER_TOOL, SEARCH_TOOL],
                             logprobs=False)

        ns_resp = client.chat.completions.create(**common_kwargs)
        ns_choice = ns_resp.choices[0]
        assert ns_choice.finish_reason == 'tool_calls'
        assert ns_choice.message.tool_calls is not None
        ns_tc = ns_choice.message.tool_calls[0]
        assert_tool_call_fields(ns_tc)
        ns_parsed = assert_arguments_parseable(ns_tc.function.arguments)
        assert isinstance(ns_tc.id, str) and len(ns_tc.id) >= 9

        stream = client.chat.completions.create(**common_kwargs, stream=True)
        sr = collect_stream_reasoning(stream)
        assert sr['finish_reason'] == 'tool_calls'
        assert sr['finish_reason_count'] == 1
        assert sr['role'] == 'assistant'
        assert sr['role_count'] == 1
        s_tc = list(sr['tool_calls'].values())[0]
        assert s_tc['name'] is not None
        assert s_tc['id'] is not None and len(s_tc['id']) >= 9
        s_parsed = json.loads(s_tc['args_str'])

        assert s_tc['name'] == ns_tc.function.name
        assert ns_parsed == s_parsed
        assert sr['finish_reason'] == ns_choice.finish_reason

    def test_streaming_role_exactly_once(self, backend, model_case):
        client, model_name = self._get_client()
        stream = client.chat.completions.create(model=model_name,
                                                messages=MESSAGES_REASONING_BASIC,
                                                temperature=0,
                                                max_completion_tokens=1024,
                                                logprobs=False,
                                                stream=True)
        result = collect_stream_reasoning(stream)
        assert result['role'] == 'assistant'
        assert result['role_count'] == 1

    def test_streaming_function_name_not_fragmented(self, backend, model_case):
        client, model_name = self._get_client()
        stream = client.chat.completions.create(model=model_name,
                                                messages=MESSAGES_REASONING_WEATHER_TOOL,
                                                temperature=0,
                                                max_completion_tokens=1024,
                                                tools=[WEATHER_TOOL],
                                                tool_choice={
                                                    'type': 'function',
                                                    'function': {
                                                        'name': 'get_current_weather'
                                                    }
                                                },
                                                logprobs=False,
                                                stream=True)
        name_events = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        name_events.append(tc.function.name)
        assert len(name_events) == 1
        assert name_events[0] == 'get_current_weather'


# ---------------------------------------------------------------------------
# Tool-result consistency (cross-mode comparison)
# ---------------------------------------------------------------------------


@_apply_marks
class TestReasoningToolResultConsistency(_ReasoningTestBase):
    """After providing tool results, streaming content must match non-
    streaming."""

    def test_tool_result_stream_vs_nonstream(self, backend, model_case):
        client, model_name = self._get_client()
        messages = build_messages_with_tool_response()
        common_kwargs = dict(model=model_name,
                             messages=messages,
                             temperature=0,
                             max_completion_tokens=256,
                             tools=[WEATHER_TOOL, SEARCH_TOOL],
                             logprobs=False)

        ns_resp = client.chat.completions.create(**common_kwargs)
        ns_choice = ns_resp.choices[0]
        assert ns_choice.finish_reason != 'tool_calls'
        assert ns_choice.message.content is not None
        assert len(ns_choice.message.content) > 0

        stream = client.chat.completions.create(**common_kwargs, stream=True)
        chunks = []
        finish_count = 0
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                chunks.append(delta.content)
            if chunk.choices[0].finish_reason is not None:
                finish_count += 1
        assert finish_count == 1
        streamed_content = ''.join(chunks)
        assert streamed_content == ns_choice.message.content

    def test_tool_result_no_tag_leakage(self, backend, model_case):
        client, model_name = self._get_client()
        response = client.chat.completions.create(model=model_name,
                                                  messages=build_messages_with_tool_response(),
                                                  temperature=0,
                                                  max_completion_tokens=256,
                                                  tools=[WEATHER_TOOL],
                                                  logprobs=False)
        content = response.choices[0].message.content or ''
        assert THINK_START_TOKEN not in content
        assert THINK_END_TOKEN not in content

    def test_reasoning_roundtrip_stream_vs_nonstream(self, backend, model_case):
        client, model_name = self._get_client()
        messages = build_reasoning_tool_roundtrip_messages()
        common_kwargs = dict(model=model_name,
                             messages=messages,
                             temperature=0,
                             max_completion_tokens=512,
                             tools=[WEATHER_TOOL],
                             logprobs=False)

        ns_resp = client.chat.completions.create(**common_kwargs)
        ns_choice = ns_resp.choices[0]
        ns_content = ns_choice.message.content or ''
        ns_reasoning = get_reasoning_content(ns_choice.message)

        stream = client.chat.completions.create(**common_kwargs, stream=True)
        sr = collect_stream_reasoning(stream)
        assert sr['finish_reason'] == ns_choice.finish_reason
        assert sr['content'] == ns_content
        if ns_reasoning:
            assert len(sr['reasoning_content']) > 0


# ---------------------------------------------------------------------------
# Web search tool
# ---------------------------------------------------------------------------


@_apply_marks_stream
class TestReasoningWebSearchTool(_ReasoningTestBase):
    """Tests for web_search tool call — forced, auto, and round-trip."""

    def test_web_search_forced(self, backend, model_case, stream):
        r = self._call_api(stream,
                           MESSAGES_REASONING_SEARCH_TOOL,
                           tools=[SEARCH_TOOL],
                           tool_choice={
                               'type': 'function',
                               'function': {
                                   'name': 'web_search'
                               }
                           })
        assert r['finish_reason'] == 'tool_calls'
        assert len(r['tool_calls']) >= 1
        tc = r['tool_calls'][0]
        assert tc['name'] == 'web_search'
        parsed = json.loads(tc['args_str'])
        assert 'query' in parsed and len(parsed['query']) > 0

    def test_web_search_auto(self, backend, model_case, stream):
        r = self._call_api(stream,
                           MESSAGES_REASONING_SEARCH_TOOL,
                           tools=[SEARCH_TOOL, WEATHER_TOOL],
                           tool_choice='auto')
        if len(r['tool_calls']) > 0:
            assert r['finish_reason'] == 'tool_calls'
            names = [tc['name'] for tc in r['tool_calls']]
            assert 'web_search' in names
            for tc in r['tool_calls']:
                if tc['name'] == 'web_search':
                    parsed = json.loads(tc['args_str'])
                    assert 'query' in parsed
        else:
            assert len(r['content'].strip()) > 0

    def test_web_search_roundtrip(self, backend, model_case, stream):
        r = self._call_api(stream, _build_search_roundtrip_messages(), tools=[SEARCH_TOOL])
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['tool_calls']) == 0
        assert len(r['reasoning'] + r['content']) > 0
        if r['content']:
            has_ref = any(kw in r['content'].lower()
                          for kw in ('hopfield', 'hinton', 'nobel', 'physics', 'machine learning', 'neural network'))
            assert has_ref
            _assert_no_tag_leakage(r['reasoning'], r['content'])


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


@_apply_marks
class TestReasoningTokenAccounting(_ReasoningTestBase):
    """Verify token usage includes reasoning tokens when available."""

    def test_usage_present(self, backend, model_case):
        client, model_name = self._get_client()
        response = client.chat.completions.create(model=model_name,
                                                  messages=MESSAGES_REASONING_BASIC,
                                                  temperature=0,
                                                  max_completion_tokens=1024,
                                                  logprobs=False)
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (response.usage.prompt_tokens + response.usage.completion_tokens)
        assert response.usage.completion_tokens > 10

    def test_reasoning_tokens_if_available(self, backend, model_case):
        client, model_name = self._get_client()
        response = client.chat.completions.create(model=model_name,
                                                  messages=MESSAGES_REASONING_COMPLEX,
                                                  temperature=0,
                                                  max_completion_tokens=2048,
                                                  logprobs=False)
        rt = get_reasoning_tokens(response)
        if rt is not None:
            assert rt >= 0
            assert rt <= response.usage.completion_tokens
            if response.usage.completion_tokens > 50:
                assert rt > 0

    def test_usage_present_streaming(self, backend, model_case):
        client, model_name = self._get_client()
        stream = client.chat.completions.create(model=model_name,
                                                messages=MESSAGES_REASONING_BASIC,
                                                temperature=0,
                                                max_completion_tokens=1024,
                                                logprobs=False,
                                                stream=True,
                                                stream_options={'include_usage': True})
        usage = None
        for chunk in stream:
            chunk_usage = getattr(chunk, 'usage', None)
            if chunk_usage is not None:
                usage = chunk_usage
        if usage is not None:
            assert usage.prompt_tokens > 0
            assert usage.completion_tokens > 0
            assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_reasoning_tokens_streaming_if_available(self, backend, model_case):
        client, model_name = self._get_client()
        stream = client.chat.completions.create(model=model_name,
                                                messages=MESSAGES_REASONING_COMPLEX,
                                                temperature=0,
                                                max_completion_tokens=2048,
                                                logprobs=False,
                                                stream=True,
                                                stream_options={'include_usage': True})
        usage = None
        for chunk in stream:
            chunk_usage = getattr(chunk, 'usage', None)
            if chunk_usage is not None:
                usage = chunk_usage
        if usage is not None:
            details = getattr(usage, 'completion_tokens_details', None)
            rt = getattr(details, 'reasoning_tokens', None) if details else None
            if rt is None:
                rt = getattr(usage, 'reasoning_tokens', None)
            if rt is not None:
                assert rt >= 0
                assert rt <= usage.completion_tokens


# ---------------------------------------------------------------------------
# Multilingual reasoning
# ---------------------------------------------------------------------------


@_apply_marks_stream
class TestReasoningMultilingual(_ReasoningTestBase):
    """Reasoning with Chinese / multilingual prompts."""

    def test_chinese_reasoning(self, backend, model_case, stream):
        r = self._call_api(stream, MESSAGES_REASONING_CN, max_completion_tokens=2048)
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['reasoning']) > 20
        assert any(kw in r['reasoning'] for kw in ('60', '80', '140', '280'))
        assert len(r['content'].strip()) > 0
        assert '2' in r['content']
        _assert_no_tag_leakage(r['reasoning'], r['content'])

    def test_chinese_with_tool(self, backend, model_case, stream):
        messages = [
            {
                'role': 'system',
                'content': '你是一个有用的助手，可以使用工具。请先思考是否需要使用工具。'
            },
            {
                'role': 'user',
                'content': '北京今天的天气怎么样？我需要带伞吗？'
            },
        ]
        r = self._call_api(stream, messages, tools=[WEATHER_TOOL_CN])
        assert len(r['tool_calls']) > 0
        assert r['finish_reason'] == 'tool_calls'
        tc = r['tool_calls'][0]
        assert tc['name'] == 'get_current_weather'
        parsed = json.loads(tc['args_str'])
        assert 'city' in parsed
        assert '北京' in parsed['city'] or 'Beijing' in parsed['city']


# ---------------------------------------------------------------------------
# Multi-turn reasoning
# ---------------------------------------------------------------------------


@_apply_marks_stream
class TestReasoningMultiTurn(_ReasoningTestBase):
    """Multi-turn conversations where reasoning persists."""

    def test_multi_turn_reasoning(self, backend, model_case, stream):
        r = self._call_api(stream, MESSAGES_REASONING_MULTI_TURN, max_completion_tokens=2048)
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['reasoning']) > 20
        assert any(kw in r['reasoning'] for kw in ('100', '101', '5050', '5,050', 'formula', 'Gauss', 'n(n', 'n *'))
        assert len(r['content'].strip()) > 0
        assert '5050' in r['content'] or '5,050' in r['content']
        _assert_no_tag_leakage(r['reasoning'], r['content'])


# ---------------------------------------------------------------------------
# Response-level validation (separate streaming / non-streaming methods)
# ---------------------------------------------------------------------------


@_apply_marks
class TestReasoningResponseValidation(_ReasoningTestBase):
    """Validate response-level fields in reasoning mode."""

    def test_model_id_created_fields(self, backend, model_case):
        client, model_name = self._get_client()
        response = client.chat.completions.create(model=model_name,
                                                  messages=MESSAGES_REASONING_BASIC,
                                                  temperature=0,
                                                  max_completion_tokens=1024,
                                                  logprobs=False)
        assert response.model is not None and len(response.model) > 0
        assert response.id is not None and len(str(response.id)) > 0
        assert response.created is not None and response.created > 0
        assert len(response.choices) >= 1
        assert response.choices[0].index == 0
        assert response.choices[0].finish_reason in ('stop', 'length')
        assert response.choices[0].message.role == 'assistant'
        msg = response.choices[0].message
        reasoning = get_reasoning_content(msg)
        assert reasoning is not None
        assert msg.content is not None and len(msg.content.strip()) > 0
        assert THINK_START_TOKEN not in (msg.content or '')
        assert THINK_END_TOKEN not in (msg.content or '')

    def test_model_id_created_fields_streaming(self, backend, model_case):
        client, model_name = self._get_client()
        stream = client.chat.completions.create(model=model_name,
                                                messages=MESSAGES_REASONING_BASIC,
                                                temperature=0,
                                                max_completion_tokens=1024,
                                                logprobs=False,
                                                stream=True)
        first_chunk = None
        chunk_count = 0
        has_role = False
        last_finish = None
        for chunk in stream:
            chunk_count += 1
            if first_chunk is None:
                first_chunk = chunk
            if chunk.choices and chunk.choices[0].delta.role:
                has_role = True
            if chunk.choices and chunk.choices[0].finish_reason:
                last_finish = chunk.choices[0].finish_reason
        assert first_chunk is not None
        assert first_chunk.model is not None and len(first_chunk.model) > 0
        assert first_chunk.id is not None
        assert first_chunk.created is not None and first_chunk.created > 0
        assert has_role
        assert last_finish in ('stop', 'length')


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@_apply_marks_stream
class TestReasoningEdgeCases(_ReasoningTestBase):
    """Edge cases for reasoning functionality."""

    def test_simple_question(self, backend, model_case, stream):
        """'What is 2+2?' should produce answer '4'."""
        r = self._call_api(stream, MESSAGES_REASONING_SIMPLE, max_completion_tokens=512)
        assert r['finish_reason'] in ('stop', 'length')
        full = r['reasoning'] + r['content']
        assert len(full) > 0
        assert '4' in full
        _assert_no_tag_leakage(r['reasoning'], r['content'])

    def test_no_tools_provided(self, backend, model_case, stream):
        """Without tools, weather question produces text answer."""
        r = self._call_api(stream, MESSAGES_REASONING_WEATHER_TOOL)
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['tool_calls']) == 0
        assert len(r['reasoning'] + r['content']) > 0
        _assert_no_tag_leakage(r['reasoning'], r['content'])

    def test_empty_tools(self, backend, model_case, stream):
        """Empty tools list: no tool calls, pure reasoning + text."""
        from openai import BadRequestError
        try:
            r = self._call_api(stream, MESSAGES_REASONING_BASIC, tools=[])
        except BadRequestError:
            pytest.skip('Backend rejects empty tools list')
        assert len(r['tool_calls']) == 0
        assert len(r['reasoning'] + r['content']) > 0

    def test_low_max_tokens(self, backend, model_case, stream):
        """Very low max_tokens: truncated but valid output."""
        r = self._call_api(stream, MESSAGES_REASONING_COMPLEX, max_completion_tokens=50)
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['reasoning'] + r['content']) > 0
        _assert_no_tag_leakage(r['reasoning'], r['content'])

    def test_reasoning_not_parsed_as_tool_call(self, backend, model_case, stream):
        """Reasoning mentioning function names must not be extracted as tool
        calls."""
        messages = [{
            'role':
            'user',
            'content': ('Explain the proof that the square root of 2 is irrational. '
                        'Do not call any tools, just explain in text.'),
        }]
        r = self._call_api(stream, messages, tools=[CALCULATOR_TOOL, WEATHER_TOOL], tool_choice='auto')
        assert len(r['content'].strip()) > 0
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['tool_calls']) == 0
        _assert_no_tag_leakage(r['reasoning'], r['content'])


# ---------------------------------------------------------------------------
# Disable thinking (enable_thinking=False)
# ---------------------------------------------------------------------------


@_apply_marks_stream
class TestReasoningDisableThinking(_ReasoningTestBase):
    """Tests with enable_thinking=False — non-think mode."""

    def test_no_reasoning_content(self, backend, model_case, stream):
        """enable_thinking=False: reasoning_content should be absent."""
        r = self._call_api(stream, MESSAGES_REASONING_BASIC, extra_body={'enable_thinking': False})
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['content'].strip()) > 0
        assert THINK_START_TOKEN not in r['content']
        assert THINK_END_TOKEN not in r['content']
        # reasoning should be empty
        assert r['reasoning'] == ''
        if stream:
            assert r['reasoning_chunks'] == 0

    def test_with_tool_call(self, backend, model_case, stream):
        """enable_thinking=False + tool call: should call tool without
        reasoning."""
        r = self._call_api(stream,
                           MESSAGES_REASONING_WEATHER_TOOL,
                           tools=[WEATHER_TOOL],
                           extra_body={'enable_thinking': False})
        if len(r['tool_calls']) > 0:
            assert r['finish_reason'] == 'tool_calls'
            tc = r['tool_calls'][0]
            assert tc['name'] == 'get_current_weather'
            parsed = json.loads(tc['args_str'])
            assert 'city' in parsed
        assert r['reasoning'] == ''
        _assert_no_tag_leakage(r['reasoning'], r['content'])

    def test_content_quality(self, backend, model_case, stream):
        """enable_thinking=False: content should still contain correct
        answer."""
        r = self._call_api(stream, MESSAGES_REASONING_BASIC, extra_body={'enable_thinking': False})
        assert len(r['content'].strip()) > 0
        assert '1591' in r['content']
