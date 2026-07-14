import json

import pytest
from utils.constant import BACKEND_LIST, TOOL_REASONING_MODEL_LIST

from utils.tool_reasoning_definitions import (  # isort: skip
    THINK_END_TOKEN, THINK_START_TOKEN, collect_stream_reasoning, get_reasoning_content, make_logged_client,
    setup_log_file)

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
        extra_body = dict(create_kwargs.pop('extra_body', {}) or {})
        legacy_et = extra_body.pop('enable_thinking', None)
        ctk = dict(extra_body.pop('chat_template_kwargs', None) or {})
        if legacy_et is not None and 'enable_thinking' not in ctk:
            ctk['enable_thinking'] = legacy_et
        if 'enable_thinking' not in ctk:
            ctk['enable_thinking'] = True
        extra_body['chat_template_kwargs'] = ctk
        create_kwargs['extra_body'] = extra_body

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
