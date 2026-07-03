import json

import pytest
from utils.constant import BACKEND_LIST, DEFAULT_MAX_COMPLETION_TOKENS, TOOL_REASONING_MODEL_LIST

from utils.tool_reasoning_definitions import (  # isort: skip
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    _stream_choice_extension,
    assert_tool_call_dict_fields,
    assert_tool_call_fields,
    attach_decoded_validation,
    collect_stream_reasoning,
    make_logged_client,
    resolve_tokenizer_model_path,
    resolve_tool_parser_name,
    setup_log_file,
)

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

_CLASS_MARKS = [
    pytest.mark.order(9),
    pytest.mark.reasoning,
    pytest.mark.deepseek_r1_parser,
    pytest.mark.deepseek_v3_parser,
    pytest.mark.gpt_oss_parser,
    pytest.mark.qwenqwq_parser,
    pytest.mark.flaky(reruns=2),
    pytest.mark.parametrize('backend', BACKEND_LIST),
    pytest.mark.parametrize('model_case', TOOL_REASONING_MODEL_LIST),
]

_CLASS_MARKS_STREAM = _CLASS_MARKS + [
    pytest.mark.parametrize('stream', [False, True], ids=['nonstream', 'stream']),
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


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def _require_str(value: str | None, field: str) -> str:
    """Fail fast when a required protocol string field is missing (no ``or ''``
    coercion)."""
    assert value is not None, f'{field} must be present'
    return value


def _assert_reasoning_absent(reasoning: str | None, *, stream: bool) -> None:
    """``enable_thinking=False``: non-stream omits field (None); stream
    aggregates to ''."""
    if stream:
        assert reasoning == '', f'expected empty reasoning stream aggregate, got {reasoning!r}'
    else:
        assert reasoning is None, f'expected absent reasoning_content, got {reasoning!r}'


def _assert_no_tag_leakage(reasoning: str | None, content: str | None) -> None:
    """Assert that <think>/</think> tags do not appear in reasoning or
    content."""
    for label, text in [('reasoning', reasoning), ('content', content)]:
        if text is None:
            continue
        assert THINK_START_TOKEN not in text, (f'<think> leaked into {label}: {text[:100]}')
        assert THINK_END_TOKEN not in text, (f'</think> leaked into {label}: {text[:100]}')


def _assert_after_tool_turn(r, content_keywords, *, hint):
    """Post-tool second turn: content must answer; reasoning_content is optional."""
    assert r['finish_reason'] in ('stop', 'length')
    assert len(r['tool_calls']) == 0
    content = _require_str(r['content'], 'content')
    assert len(content.strip()) > 0, f'Expected non-empty content after tool result, got: {content!r}'
    lower = content.lower()
    assert any(kw in lower for kw in content_keywords), (
        f'Content should reference {hint}: {content[:200]}')
    assert THINK_START_TOKEN not in content, (
        f'<think> leaked into content: {content[:100]}')
    assert THINK_END_TOKEN not in content, (
        f'</think> leaked into content: {content[:100]}')


_SUM_100_PATTERNS = ('5050', '5,050', '5{,}050')


def thinking_extra_body(enable_thinking: bool = True) -> dict:
    """``extra_body`` for reasoning API calls that need raw ``output_ids``."""
    return {
        'chat_template_kwargs': {'enable_thinking': enable_thinking},
        'return_token_ids': True,
    }


def _assert_content_has_sum_5050(content: str) -> None:
    """Accept plain or LaTeX-formatted 5050 in final answer text."""
    assert any(p in content for p in _SUM_100_PATTERNS), (
        f'Expected 5050 in content (plain, comma, or LaTeX 5{{,}}050); got: {content[:300]!r}...')


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
        self._model_case = model_case
        self._client, self._model_name = make_logged_client(self._log_file)
        self._tokenizer_path = resolve_tokenizer_model_path(config, model_case)

    def _parser_validation_kwargs(self, tools=None):
        """Kwargs for parser-backed validation helpers."""
        kwargs = {
            'tokenizer_path': self._tokenizer_path,
            'tool_parser_name': resolve_tool_parser_name(self._model_case),
        }
        if tools is not None:
            kwargs['tools'] = tools
        return kwargs

    def _collect_stream_reasoning_validated(self, stream, *, tools=None, enable_thinking=True, validate_decoded=True):
        """Collect stream chunks and validate decoded ``output_ids`` markup."""
        sr = collect_stream_reasoning(stream)
        attach_decoded_validation(
            sr,
            enable_thinking=enable_thinking,
            model_case=self._model_case,
            reasoning_parser_name='default',
            validate_decoded=validate_decoded,
            **self._parser_validation_kwargs(tools=tools),
        )
        return sr

    def _get_client(self):
        """Return *(client, model_name)* with transparent logging."""
        return self._client, self._model_name

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
        create_kwargs.setdefault('max_completion_tokens', DEFAULT_MAX_COMPLETION_TOKENS)
        create_kwargs.setdefault('logprobs', False)
        extra_body = create_kwargs.pop('extra_body', None)
        extra_body = {} if extra_body is None else dict(extra_body)
        legacy_et = extra_body.pop('enable_thinking', None)
        ctk = extra_body.pop('chat_template_kwargs', None)
        ctk = {} if ctk is None else dict(ctk)
        if legacy_et is not None and 'enable_thinking' not in ctk:
            ctk['enable_thinking'] = legacy_et
        if 'enable_thinking' not in ctk:
            ctk['enable_thinking'] = True
        extra_body['chat_template_kwargs'] = ctk
        enable_thinking = ctk['enable_thinking']
        validate_decoded = create_kwargs.pop('validate_decoded', True)
        if validate_decoded:
            extra_body.setdefault('return_token_ids', True)
        else:
            extra_body['return_token_ids'] = False
        create_kwargs['extra_body'] = extra_body
        tools = create_kwargs.get('tools')
        parser_kwargs = self._parser_validation_kwargs(tools=tools)

        if stream:
            resp = client.chat.completions.create(model=model_name, messages=messages, stream=True, **create_kwargs)
            sr = collect_stream_reasoning(resp)
            attach_decoded_validation(
                sr,
                enable_thinking=enable_thinking,
                model_case=self._model_case,
                reasoning_parser_name='default',
                validate_decoded=validate_decoded,
                **parser_kwargs,
            )
            tool_calls = []
            for idx in sorted(sr['tool_calls'].keys()):
                tc_entry = sr['tool_calls'][idx]
                if tc_entry['name']:
                    assert_tool_call_dict_fields(tc_entry)
                tool_calls.append(tc_entry)
            return {
                'reasoning': sr['reasoning_content'],
                'content': sr['content'],
                'finish_reason': sr['finish_reason'],
                'role': sr['role'],
                'tool_calls': tool_calls,
                'chunk_count': sr['chunk_count'],
                'reasoning_chunks': sr['reasoning_chunks'],
                'content_chunks': sr['content_chunks'],
                'finish_reason_count': sr['finish_reason_count'],
                'role_count': sr['role_count'],
                'role_inconsistent': sr['role_inconsistent'],
            }
        else:
            resp = client.chat.completions.create(model=model_name, messages=messages, **create_kwargs)
            choice = resp.choices[0]
            message = choice.message
            ns_result = {
                'output_ids': _stream_choice_extension(choice, 'output_ids') or [],
                'finish_reason': choice.finish_reason,
            }
            attach_decoded_validation(
                ns_result,
                enable_thinking=enable_thinking,
                model_case=self._model_case,
                reasoning_parser_name='default',
                validate_decoded=validate_decoded,
                **parser_kwargs,
            )
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    assert_tool_call_fields(tc)
                    tool_calls.append({
                        'name': tc.function.name,
                        'args_str': tc.function.arguments,
                        'id': tc.id,
                    })
            return {
                'reasoning': message.reasoning_content,
                'content': message.content,
                'finish_reason': choice.finish_reason,
                'role': message.role,
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
            'name':
            'web_search',
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
