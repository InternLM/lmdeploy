import json

import pytest
from utils.tool_reasoning_definitions import (
    CALCULATOR_TOOL,
    SEARCH_TOOL,
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    WEATHER_TOOL,
    WEATHER_TOOL_CN,
    build_messages_with_tool_response,
    build_reasoning_tool_roundtrip_messages,
    collect_stream_reasoning,
    get_reasoning_content,
    get_reasoning_tokens,
)

from .conftest import (
    MESSAGES_REASONING_BASIC,
    MESSAGES_REASONING_CN,
    MESSAGES_REASONING_COMPLEX,
    MESSAGES_REASONING_MULTI_TURN,
    MESSAGES_REASONING_PARALLEL_TOOLS,
    MESSAGES_REASONING_SEARCH_TOOL,
    MESSAGES_REASONING_SIMPLE,
    MESSAGES_REASONING_WEATHER_TOOL,
    _apply_marks,
    _apply_marks_stream,
    _assert_no_tag_leakage,
    _build_search_roundtrip_messages,
    _ReasoningTestBase,
)


# ===========================================================================
# Basic reasoning: presence, quality, separation
# ===========================================================================


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
        assert '1591' in r['content'] or '1,591' in r['content'], \
            f"Expected '1591' or '1,591' in response, got: {r['content']!r}"
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


# ===========================================================================
# Streaming ↔ Non-streaming consistency (cross-mode comparison)
# ===========================================================================


@_apply_marks
class TestReasoningStreamConsistency(_ReasoningTestBase):
    """Both modes must produce reasoning AND content with correct
    separation."""

    def test_reasoning_presence_consistent(self, backend, model_case):
        client, model_name = self._get_client()
        common_kwargs = dict(model=model_name,
                             messages=MESSAGES_REASONING_BASIC,
                             temperature=0,
                             max_completion_tokens=1024,
                             logprobs=False,
                             extra_body={'enable_thinking': True})
        ns_resp = client.chat.completions.create(**common_kwargs)
        ns_reasoning = get_reasoning_content(ns_resp.choices[0].message)
        ns_content = ns_resp.choices[0].message.content or ''

        stream = client.chat.completions.create(**common_kwargs, stream=True)
        result = collect_stream_reasoning(stream)

        assert ns_reasoning is not None and len(ns_reasoning) > 0
        assert len(result['reasoning_content']) > 0
        assert len(ns_content.strip()) > 0
        assert len(result['content'].strip()) > 0
        assert '1591' in ns_content or '1,591' in ns_content, \
            f"Expected '1591' or '1,591' in response, got: {ns_content!r}"
        assert '1591' in result['content'] or '1,591' in result['content'], \
            f"Expected '1591' or '1,591' in response, got: {result['content']!r}"
        for text in [ns_reasoning, ns_content, result['reasoning_content'], result['content']]:
            assert THINK_START_TOKEN not in text
            assert THINK_END_TOKEN not in text


# ===========================================================================
# Tool calls + tool_choice
# ===========================================================================


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
        assert len(r['reasoning'].strip()) > 0, (
            f'Expected non-empty reasoning_content for reasoning model, '
            f'got reasoning={r["reasoning"]!r}, content={r["content"][:200]!r}')
        assert len(r['content'].strip()) > 0, (
            f'Expected non-empty content, got content={r["content"]!r}')
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


# ===========================================================================
# Parallel tool calls
# ===========================================================================


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


# ===========================================================================
# Tool round-trip: reason → tool → result → answer
# ===========================================================================


@_apply_marks_stream
class TestReasoningToolRoundTrip(_ReasoningTestBase):
    """Multi-turn: reason → tool → result → reasoning → answer."""

    def test_after_tool_result(self, backend, model_case, stream):
        r = self._call_api(stream, build_reasoning_tool_roundtrip_messages(), tools=[WEATHER_TOOL])
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['tool_calls']) == 0
        assert len(r['reasoning'].strip()) > 0, (
            f'Expected non-empty reasoning_content for reasoning model, '
            f'got reasoning={r["reasoning"]!r}, content={r["content"][:200]!r}')
        assert len(r['content'].strip()) > 0, (
            f'Expected non-empty content after tool result, '
            f'got content={r["content"]!r}')
        has_ref = any(kw in r['content'].lower()
                      for kw in ('sunny', 'umbrella', 'dallas', 'clear', 'rain', 'weather', 'no'))
        assert has_ref, f'Content should reference weather: {r["content"][:200]}'
        _assert_no_tag_leakage(r['reasoning'], r['content'])


# ===========================================================================
# Streaming ↔ Non-streaming tool-call consistency (cross-mode comparison)
# ===========================================================================


@_apply_marks
class TestReasoningToolCallConsistency(_ReasoningTestBase):
    """Compare streaming vs non-streaming tool-call results."""

    def test_tool_call_stream_vs_nonstream(self, backend, model_case):
        from utils.tool_reasoning_definitions import assert_arguments_parseable, assert_tool_call_fields
        client, model_name = self._get_client()
        common_kwargs = dict(model=model_name,
                             messages=MESSAGES_REASONING_WEATHER_TOOL,
                             temperature=0,
                             max_completion_tokens=1024,
                             tools=[WEATHER_TOOL, SEARCH_TOOL],
                             logprobs=False,
                             extra_body={'enable_thinking': True})

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
                                                extra_body={'enable_thinking': True},
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
                                                extra_body={'enable_thinking': True},
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


# ===========================================================================
# Tool-result consistency (cross-mode comparison)
# ===========================================================================


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
                             logprobs=False,
                             extra_body={'enable_thinking': True})

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
                                                  logprobs=False,
                                                  extra_body={'enable_thinking': True})
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
                             logprobs=False,
                             extra_body={'enable_thinking': True})

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


# ===========================================================================
# Web search tool
# ===========================================================================


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
        assert len(r['reasoning'].strip()) > 0, (
            f'Expected non-empty reasoning_content for reasoning model, '
            f'got reasoning={r["reasoning"]!r}, content={r["content"][:200]!r}')
        assert len(r['content'].strip()) > 0, (
            f'Expected non-empty content after search result, '
            f'got content={r["content"]!r}')
        has_ref = any(kw in r['content'].lower()
                      for kw in ('hopfield', 'hinton', 'nobel', 'physics', 'machine learning', 'neural network'))
        assert has_ref, f'Content should reference Nobel Prize: {r["content"][:200]}'
        _assert_no_tag_leakage(r['reasoning'], r['content'])


# ===========================================================================
# Token accounting
# ===========================================================================


@_apply_marks
class TestReasoningTokenAccounting(_ReasoningTestBase):
    """Verify token usage includes reasoning tokens when available."""

    def test_usage_present(self, backend, model_case):
        client, model_name = self._get_client()
        response = client.chat.completions.create(model=model_name,
                                                  messages=MESSAGES_REASONING_BASIC,
                                                  temperature=0,
                                                  max_completion_tokens=1024,
                                                  logprobs=False,
                                                  extra_body={'enable_thinking': True})
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
                                                  logprobs=False,
                                                  extra_body={'enable_thinking': True})
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
                                                extra_body={'enable_thinking': True},
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
                                                extra_body={'enable_thinking': True},
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


# ===========================================================================
# Multilingual reasoning
# ===========================================================================


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


# ===========================================================================
# Multi-turn reasoning
# ===========================================================================


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


# ===========================================================================
# Response-level validation
# ===========================================================================


@_apply_marks
class TestReasoningResponseValidation(_ReasoningTestBase):
    """Validate response-level fields in reasoning mode."""

    def test_model_id_created_fields(self, backend, model_case):
        client, model_name = self._get_client()
        response = client.chat.completions.create(model=model_name,
                                                  messages=MESSAGES_REASONING_BASIC,
                                                  temperature=0,
                                                  max_completion_tokens=1024,
                                                  logprobs=False,
                                                  extra_body={'enable_thinking': True})
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
                                                extra_body={'enable_thinking': True},
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


# ===========================================================================
# Edge cases
# ===========================================================================


@_apply_marks_stream
class TestReasoningEdgeCases(_ReasoningTestBase):
    """Edge cases for reasoning functionality."""

    def test_simple_question(self, backend, model_case, stream):
        """'What is 2+2?' should produce answer '4'."""
        r = self._call_api(stream, MESSAGES_REASONING_SIMPLE, max_completion_tokens=512)
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['reasoning'].strip()) > 0, (
            f'Expected non-empty reasoning_content for reasoning model, '
            f'got reasoning={r["reasoning"]!r}, content={r["content"][:200]!r}')
        assert len(r['content'].strip()) > 0, (
            f'Expected non-empty content, got content={r["content"]!r}')
        assert '4' in r['reasoning'] + r['content']
        _assert_no_tag_leakage(r['reasoning'], r['content'])

    def test_no_tools_provided(self, backend, model_case, stream):
        """Without tools, weather question produces text answer."""
        r = self._call_api(stream, MESSAGES_REASONING_WEATHER_TOOL)
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['tool_calls']) == 0
        assert len(r['reasoning'].strip()) > 0, (
            f'Expected non-empty reasoning_content for reasoning model, '
            f'got reasoning={r["reasoning"]!r}, content={r["content"][:200]!r}')
        assert len(r['content'].strip()) > 0, (
            f'Expected non-empty content, got content={r["content"]!r}')
        _assert_no_tag_leakage(r['reasoning'], r['content'])

    def test_empty_tools(self, backend, model_case, stream):
        """Empty tools list: no tool calls, pure reasoning + text."""
        from openai import BadRequestError
        try:
            r = self._call_api(stream, MESSAGES_REASONING_BASIC, tools=[])
        except BadRequestError:
            pytest.skip('Backend rejects empty tools list')
        assert len(r['tool_calls']) == 0
        assert len(r['reasoning'].strip()) > 0, (
            f'Expected non-empty reasoning_content for reasoning model, '
            f'got reasoning={r["reasoning"]!r}, content={r["content"][:200]!r}')
        assert len(r['content'].strip()) > 0, (
            f'Expected non-empty content, got content={r["content"]!r}')

    def test_low_max_tokens(self, backend, model_case, stream):
        """Very low max_tokens: truncated but valid output."""
        r = self._call_api(stream, MESSAGES_REASONING_COMPLEX, max_completion_tokens=50)
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['reasoning'].strip()) > 0, (
            f'Expected non-empty reasoning_content for reasoning model, '
            f'got reasoning={r["reasoning"]!r}, content={r["content"][:200]!r}')
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
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['tool_calls']) == 0
        assert len(r['reasoning'].strip()) > 0, (
            f'Expected non-empty reasoning_content for reasoning model, '
            f'got reasoning={r["reasoning"]!r}, content={r["content"][:200]!r}')
        assert len(r['content'].strip()) > 0, (
            f'Expected non-empty content, got content={r["content"]!r}')
        _assert_no_tag_leakage(r['reasoning'], r['content'])


# ===========================================================================
# Disable thinking (enable_thinking=False)
# ===========================================================================


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
        assert '1591' in r['content'] or '1,591' in r['content'], \
            f"Expected '1591' or '1,591' in response, got: {r['content']!r}"
