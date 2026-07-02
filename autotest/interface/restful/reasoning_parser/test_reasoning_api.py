import pytest
from openai import BadRequestError
from utils.constant import DEFAULT_MAX_COMPLETION_TOKENS
from utils.tool_reasoning_definitions import (
    CALCULATOR_TOOL,
    SEARCH_TOOL,
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    WEATHER_TOOL,
    _stream_choice_extension,
    _stream_delta_field,
    assert_arguments_parseable,
    assert_tool_call_dict_fields,
    assert_tool_call_fields,
    assert_tool_name_single_delta,
    attach_decoded_validation,
    build_messages_with_tool_response,
    build_reasoning_tool_roundtrip_messages,
    validate_stream_reasoning_result,
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
    _assert_after_tool_turn,
    _assert_content_has_sum_5050,
    _assert_no_tag_leakage,
    _assert_reasoning_absent,
    _build_search_roundtrip_messages,
    _ReasoningTestBase,
    _require_str,
    thinking_extra_body,
)

_EXTRA_BODY_THINKING_OFF = {
    'chat_template_kwargs': {'enable_thinking': False},
}

# ===========================================================================
# Basic reasoning: presence, quality, separation
# ===========================================================================


@_apply_marks_stream
class TestReasoningBasic(_ReasoningTestBase):
    """Basic reasoning_content presence, quality, and content separation."""

    def test_reasoning_content_present(self, backend, model_case, stream):
        """Model should populate reasoning_content for math questions."""
        r = self._call_api(stream, MESSAGES_REASONING_BASIC, max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS)
        assert r['finish_reason'] in ('stop', 'length')
        reasoning = _require_str(r['reasoning'], 'reasoning_content')
        content = _require_str(r['content'], 'content')
        assert len(reasoning) > 10, (f'reasoning too short ({len(reasoning)} chars)')
        assert any(kw in reasoning for kw in ('37', '43', '1591', 'multiply', '*', '×'))
        assert len(content.strip()) > 0
        assert '1591' in content or '1,591' in content, \
            f"Expected '1591' or '1,591' in response, got: {content!r}"
        assert reasoning.strip() != content.strip()
        _assert_no_tag_leakage(reasoning, content)

    def test_reasoning_quality_complex(self, backend, model_case, stream):
        """Complex train problem: reasoning should contain calculation steps."""
        r = self._call_api(stream, MESSAGES_REASONING_COMPLEX, max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS)
        reasoning = _require_str(r['reasoning'], 'reasoning_content')
        content = _require_str(r['content'], 'content')
        assert len(reasoning) > 50
        assert any(kw in reasoning for kw in ('60', '80', '140', '280'))
        assert len(content.strip()) > 0
        assert '2' in content
        _assert_no_tag_leakage(reasoning, content)
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
                             max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                             logprobs=False,
                             extra_body=thinking_extra_body())
        ns_resp = client.chat.completions.create(**common_kwargs)
        ns_choice = ns_resp.choices[0]
        ns_message = ns_choice.message
        ns_reasoning = _require_str(ns_message.reasoning_content, 'reasoning_content')
        ns_content = _require_str(ns_message.content, 'content')
        attach_decoded_validation(
            {'output_ids': _stream_choice_extension(ns_choice, 'output_ids') or []},
            enable_thinking=True,
            model_case=self._model_case,
            reasoning_parser_name='default',
            **self._parser_validation_kwargs(),
        )

        stream = client.chat.completions.create(**common_kwargs, stream=True)
        result = self._collect_stream_reasoning_validated(stream)

        assert len(ns_reasoning) > 0
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
                assert_tool_call_dict_fields(tc)
        else:
            content = _require_str(r['content'], 'content')
            assert len(content.strip()) > 0

    def test_tool_choice_required(self, backend, model_case, stream):
        """tool_choice='required': must produce tool call."""
        try:
            r = self._call_api(stream, MESSAGES_REASONING_WEATHER_TOOL, tools=[WEATHER_TOOL], tool_choice='required')
        except BadRequestError as e:
            pytest.skip(f'tool_choice="required" rejected by server (HTTP 400): {e}')
        assert len(r['tool_calls']) >= 1
        assert r['finish_reason'] == 'tool_calls'
        tc = r['tool_calls'][0]
        assert tc['name'] == 'get_current_weather'
        parsed = assert_tool_call_dict_fields(tc)
        assert 'city' in parsed

    def test_tool_choice_none(self, backend, model_case, stream):
        """tool_choice='none': no tool calls, text answer instead."""
        r = self._call_api(stream, MESSAGES_REASONING_WEATHER_TOOL, tools=[WEATHER_TOOL], tool_choice='none')
        assert len(r['tool_calls']) == 0
        assert r['finish_reason'] in ('stop', 'length')
        reasoning = _require_str(r['reasoning'], 'reasoning_content')
        content = _require_str(r['content'], 'content')
        assert len(reasoning.strip()) > 0, (f'Expected non-empty reasoning_content for reasoning model, '
                                            f'got reasoning={reasoning!r}, content={content[:200]!r}')
        assert len(content.strip()) > 0, (f'Expected non-empty content, got content={content!r}')
        _assert_no_tag_leakage(reasoning, content)

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
        parsed = assert_tool_call_dict_fields(tc)
        assert 'city' in parsed
        assert 'dallas' in parsed['city'].lower()


# ===========================================================================
# Parallel tool calls
# ===========================================================================


@_apply_marks_stream
class TestReasoningParallelToolCalls(_ReasoningTestBase):
    """Reasoning model calling multiple tools in parallel."""

    def test_parallel_tools(self, backend, model_case, stream):
        """User asks for Dallas weather and 37*43: expect both tools in one
        assistant turn."""
        r = self._call_api(stream, MESSAGES_REASONING_PARALLEL_TOOLS, tools=[WEATHER_TOOL, CALCULATOR_TOOL])
        assert r['finish_reason'] == 'tool_calls'
        tcs = r['tool_calls']
        assert len(tcs) >= 2, (f'Expected >=2 parallel tool calls, got {len(tcs)}: '
                               f'{[tc["name"] for tc in tcs]}')
        names = {tc['name'] for tc in tcs}
        assert 'get_current_weather' in names and 'calculate' in names, (
            f'Expected both get_current_weather and calculate, got {names}')
        ids = [tc['id'] for tc in tcs]
        assert len(set(ids)) == len(ids), f'IDs must be unique: {ids}'
        for tc in tcs:
            assert tc['name'] in ('get_current_weather', 'calculate')
            assert_tool_call_dict_fields(tc)


# ===========================================================================
# Tool round-trip: reason → tool → result → answer
# ===========================================================================


@_apply_marks_stream
class TestReasoningToolRoundTrip(_ReasoningTestBase):
    """Multi-turn: tool → result → answer (second turn may omit reasoning_content)."""

    def test_after_tool_result(self, backend, model_case, stream):
        r = self._call_api(stream, build_reasoning_tool_roundtrip_messages(), tools=[WEATHER_TOOL])
        _assert_after_tool_turn(
            r,
            ('sunny', 'umbrella', 'dallas', 'clear', 'rain', 'weather', 'no'),
            hint='weather',
        )


# ===========================================================================
# Streaming ↔ Non-streaming tool-call consistency (cross-mode comparison)
# ===========================================================================


@_apply_marks
class TestReasoningToolCallConsistency(_ReasoningTestBase):
    """Compare streaming vs non-streaming tool-call results."""

    def test_tool_call_stream_vs_nonstream(self, backend, model_case):
        client, model_name = self._get_client()
        common_kwargs = dict(model=model_name,
                             messages=MESSAGES_REASONING_WEATHER_TOOL,
                             temperature=0,
                             max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                             tools=[WEATHER_TOOL, SEARCH_TOOL],
                             logprobs=False,
                             extra_body=thinking_extra_body())

        ns_resp = client.chat.completions.create(**common_kwargs)
        ns_choice = ns_resp.choices[0]
        assert ns_choice.finish_reason == 'tool_calls'
        assert ns_choice.message.tool_calls is not None
        ns_tc = ns_choice.message.tool_calls[0]
        assert_tool_call_fields(ns_tc)
        ns_parsed = assert_arguments_parseable(ns_tc.function.arguments)
        assert isinstance(ns_tc.id, str) and len(ns_tc.id) >= 9
        attach_decoded_validation(
            {'output_ids': _stream_choice_extension(ns_choice, 'output_ids') or []},
            enable_thinking=True,
            model_case=self._model_case,
            reasoning_parser_name='default',
            **self._parser_validation_kwargs(tools=[WEATHER_TOOL, SEARCH_TOOL]),
        )

        stream = client.chat.completions.create(**common_kwargs, stream=True)
        sr = self._collect_stream_reasoning_validated(stream, tools=[WEATHER_TOOL, SEARCH_TOOL])
        validate_stream_reasoning_result(sr, expected_finish_reason='tool_calls', require_tool_calls=True)
        s_tc = list(sr['tool_calls'].values())[0]
        s_parsed = assert_tool_call_dict_fields(s_tc)

        assert s_tc['name'] == ns_tc.function.name
        assert ns_parsed == s_parsed
        assert sr['finish_reason'] == ns_choice.finish_reason

    def test_streaming_role_on_chunks(self, backend, model_case):
        """Each stream chunk may carry role; all roles must be assistant."""
        client, model_name = self._get_client()
        stream = client.chat.completions.create(model=model_name,
                                                messages=MESSAGES_REASONING_BASIC,
                                                temperature=0,
                                                max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                                                logprobs=False,
                                                extra_body=thinking_extra_body(),
                                                stream=True)
        result = self._collect_stream_reasoning_validated(stream)
        assert result['role'] == 'assistant'
        assert result['role_count'] >= 1
        assert not result['role_inconsistent']

    def test_streaming_function_name_not_fragmented(self, backend, model_case):
        client, model_name = self._get_client()
        stream = client.chat.completions.create(model=model_name,
                                                messages=MESSAGES_REASONING_WEATHER_TOOL,
                                                temperature=0,
                                                max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                                                tools=[WEATHER_TOOL],
                                                tool_choice={
                                                    'type': 'function',
                                                    'function': {
                                                        'name': 'get_current_weather'
                                                    }
                                                },
                                                logprobs=False,
                                                extra_body={'chat_template_kwargs': {'enable_thinking': True}},
                                                stream=True)
        assert_tool_name_single_delta(stream, 'get_current_weather')


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
                             max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                             tools=[WEATHER_TOOL, SEARCH_TOOL],
                             logprobs=False,
                             extra_body=thinking_extra_body())

        ns_resp = client.chat.completions.create(**common_kwargs)
        ns_choice = ns_resp.choices[0]
        assert ns_choice.finish_reason != 'tool_calls'
        ns_content = _require_str(ns_choice.message.content, 'content')

        stream = client.chat.completions.create(**common_kwargs, stream=True)
        sr = self._collect_stream_reasoning_validated(
            stream, tools=[WEATHER_TOOL, SEARCH_TOOL])
        assert sr['finish_reason_count'] == 1
        assert sr['content'] == ns_content

    def test_tool_result_no_tag_leakage(self, backend, model_case):
        client, model_name = self._get_client()
        response = client.chat.completions.create(model=model_name,
                                                  messages=build_messages_with_tool_response(),
                                                  temperature=0,
                                                  max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                                                  tools=[WEATHER_TOOL],
                                                  logprobs=False,
                                                  extra_body={'chat_template_kwargs': {'enable_thinking': True}})
        content = _require_str(response.choices[0].message.content, 'content')
        assert THINK_START_TOKEN not in content
        assert THINK_END_TOKEN not in content

    def test_reasoning_roundtrip_stream_vs_nonstream(self, backend, model_case):
        client, model_name = self._get_client()
        messages = build_reasoning_tool_roundtrip_messages()
        common_kwargs = dict(model=model_name,
                             messages=messages,
                             temperature=0,
                             max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                             tools=[WEATHER_TOOL],
                             logprobs=False,
                             extra_body=thinking_extra_body())

        ns_resp = client.chat.completions.create(**common_kwargs)
        ns_choice = ns_resp.choices[0]
        ns_message = ns_choice.message
        ns_content = _require_str(ns_message.content, 'content')
        ns_reasoning = ns_message.reasoning_content

        stream = client.chat.completions.create(**common_kwargs, stream=True)
        sr = self._collect_stream_reasoning_validated(stream, tools=[WEATHER_TOOL])
        assert sr['finish_reason'] == ns_choice.finish_reason
        assert sr['content'] == ns_content
        if ns_reasoning is not None:
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
        parsed = assert_tool_call_dict_fields(tc)
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
                    parsed = assert_tool_call_dict_fields(tc)
                    assert 'query' in parsed
        else:
            content = _require_str(r['content'], 'content')
            assert len(content.strip()) > 0

    def test_web_search_roundtrip(self, backend, model_case, stream):
        r = self._call_api(stream, _build_search_roundtrip_messages(), tools=[SEARCH_TOOL])
        _assert_after_tool_turn(
            r,
            ('hopfield', 'hinton', 'nobel', 'physics', 'machine learning', 'neural network'),
            hint='Nobel Prize',
        )


# ===========================================================================
# Token accounting
# ===========================================================================


@_apply_marks
class TestReasoningTokenAccounting(_ReasoningTestBase):
    """Verify ``usage`` (prompt / completion / total tokens) on reasoning
    requests."""

    def test_usage_present(self, backend, model_case):
        client, model_name = self._get_client()
        response = client.chat.completions.create(model=model_name,
                                                  messages=MESSAGES_REASONING_BASIC,
                                                  temperature=0,
                                                  max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                                                  logprobs=False,
                                                  extra_body={'chat_template_kwargs': {'enable_thinking': True}})
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (response.usage.prompt_tokens + response.usage.completion_tokens)
        assert response.usage.completion_tokens > 10

    def test_usage_present_streaming(self, backend, model_case):
        client, model_name = self._get_client()
        stream = client.chat.completions.create(model=model_name,
                                                messages=MESSAGES_REASONING_BASIC,
                                                temperature=0,
                                                max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                                                logprobs=False,
                                                extra_body={'chat_template_kwargs': {'enable_thinking': True}},
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


# ===========================================================================
# Multilingual reasoning
# ===========================================================================


@_apply_marks_stream
class TestReasoningMultilingual(_ReasoningTestBase):
    """Reasoning with Chinese prompts."""

    def test_chinese_reasoning(self, backend, model_case, stream):
        r = self._call_api(stream, MESSAGES_REASONING_CN, max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS)
        assert r['finish_reason'] in ('stop', 'length')
        reasoning = _require_str(r['reasoning'], 'reasoning_content')
        content = _require_str(r['content'], 'content')
        assert len(reasoning) > 20
        assert any(kw in reasoning for kw in ('60', '80', '140', '280'))
        assert len(content.strip()) > 0
        assert '2' in content
        _assert_no_tag_leakage(reasoning, content)

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
        r = self._call_api(stream, messages, tools=[WEATHER_TOOL])
        assert len(r['tool_calls']) > 0
        assert r['finish_reason'] == 'tool_calls'
        tc = r['tool_calls'][0]
        assert tc['name'] == 'get_current_weather'
        parsed = assert_tool_call_dict_fields(tc)
        assert 'city' in parsed
        assert '北京' in parsed['city'] or 'Beijing' in parsed['city']


# ===========================================================================
# Multi-turn reasoning
# ===========================================================================


@_apply_marks_stream
class TestReasoningMultiTurn(_ReasoningTestBase):
    """Multi-turn conversations where reasoning persists."""

    def test_multi_turn_reasoning(self, backend, model_case, stream):
        r = self._call_api(stream, MESSAGES_REASONING_MULTI_TURN, max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS)
        assert r['finish_reason'] in ('stop', 'length')
        reasoning = _require_str(r['reasoning'], 'reasoning_content')
        content = _require_str(r['content'], 'content')
        assert len(reasoning) > 20
        assert any(
            kw in reasoning
            for kw in ('100', '101', '5050', '5,050', '5{,}050', 'formula', 'Gauss', 'n(n', 'n *'))
        assert len(content.strip()) > 0
        _assert_content_has_sum_5050(content)
        _assert_no_tag_leakage(reasoning, content)


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
                                                  max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                                                  logprobs=False,
                                                  extra_body={'chat_template_kwargs': {'enable_thinking': True}})
        assert response.model is not None and len(response.model) > 0
        assert response.id is not None and len(str(response.id)) > 0
        assert response.created is not None and response.created > 0
        assert len(response.choices) >= 1
        assert response.choices[0].index == 0
        assert response.choices[0].finish_reason in ('stop', 'length')
        assert response.choices[0].message.role == 'assistant'
        msg = response.choices[0].message
        _require_str(msg.reasoning_content, 'reasoning_content')
        content = _require_str(msg.content, 'content')
        assert len(content.strip()) > 0
        assert THINK_START_TOKEN not in content
        assert THINK_END_TOKEN not in content

    def test_model_id_created_fields_streaming(self, backend, model_case):
        client, model_name = self._get_client()
        stream = client.chat.completions.create(model=model_name,
                                                messages=MESSAGES_REASONING_BASIC,
                                                temperature=0,
                                                max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
                                                logprobs=False,
                                                extra_body={'chat_template_kwargs': {'enable_thinking': True}},
                                                stream=True)
        first_chunk = None
        chunk_count = 0
        has_role = False
        last_finish = None
        for chunk in stream:
            chunk_count += 1
            if first_chunk is None:
                first_chunk = chunk
            if chunk.choices and _stream_delta_field(chunk.choices[0].delta, 'role'):
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
        r = self._call_api(stream, MESSAGES_REASONING_SIMPLE, max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS)
        assert r['finish_reason'] in ('stop', 'length')
        reasoning = _require_str(r['reasoning'], 'reasoning_content')
        content = _require_str(r['content'], 'content')
        assert len(reasoning.strip()) > 0, (f'Expected non-empty reasoning_content for reasoning model, '
                                            f'got reasoning={reasoning!r}, content={content[:200]!r}')
        assert len(content.strip()) > 0, (f'Expected non-empty content, got content={content!r}')
        assert '4' in reasoning + content
        _assert_no_tag_leakage(reasoning, content)

    def test_no_tools_provided(self, backend, model_case, stream):
        """Without tools, weather question produces text answer."""
        r = self._call_api(stream, MESSAGES_REASONING_WEATHER_TOOL)
        assert r['finish_reason'] in ('stop', 'length')
        assert len(r['tool_calls']) == 0
        reasoning = _require_str(r['reasoning'], 'reasoning_content')
        content = _require_str(r['content'], 'content')
        assert len(reasoning.strip()) > 0, (f'Expected non-empty reasoning_content for reasoning model, '
                                            f'got reasoning={reasoning!r}, content={content[:200]!r}')
        assert len(content.strip()) > 0, (f'Expected non-empty content, got content={content!r}')
        _assert_no_tag_leakage(reasoning, content)

    def test_empty_tools(self, backend, model_case, stream):
        """Empty tools list: no tool calls, pure reasoning + text."""
        try:
            r = self._call_api(stream, MESSAGES_REASONING_BASIC, tools=[])
        except BadRequestError:
            pytest.skip('Backend rejects empty tools list')
        assert len(r['tool_calls']) == 0
        reasoning = _require_str(r['reasoning'], 'reasoning_content')
        content = _require_str(r['content'], 'content')
        assert len(reasoning.strip()) > 0, (f'Expected non-empty reasoning_content for reasoning model, '
                                            f'got reasoning={reasoning!r}, content={content[:200]!r}')
        assert len(content.strip()) > 0, (f'Expected non-empty content, got content={content!r}')

    def test_low_max_tokens(self, backend, model_case, stream):
        """Very low max_tokens: truncated but valid output."""
        r = self._call_api(
            stream,
            MESSAGES_REASONING_COMPLEX,
            max_completion_tokens=50,
            validate_decoded=False,
        )
        assert r['finish_reason'] in ('stop', 'length')
        reasoning = _require_str(r['reasoning'], 'reasoning_content')
        assert len(reasoning.strip()) > 0, (f'Expected non-empty reasoning_content for reasoning model, '
                                            f'got reasoning={reasoning!r}')
        _assert_no_tag_leakage(reasoning, r['content'])

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
        reasoning = _require_str(r['reasoning'], 'reasoning_content')
        content = _require_str(r['content'], 'content')
        assert len(reasoning.strip()) > 0, (f'Expected non-empty reasoning_content for reasoning model, '
                                            f'got reasoning={reasoning!r}, content={content[:200]!r}')
        assert len(content.strip()) > 0, (f'Expected non-empty content, got content={content!r}')
        _assert_no_tag_leakage(reasoning, content)


# ===========================================================================
# Disable thinking (enable_thinking=False)
# ===========================================================================


@_apply_marks_stream
class TestReasoningDisableThinking(_ReasoningTestBase):
    """Tests with enable_thinking=False — non-think mode."""

    def test_no_reasoning_content(self, backend, model_case, stream):
        """enable_thinking=False: reasoning_content should be absent."""
        r = self._call_api(stream, MESSAGES_REASONING_BASIC, extra_body=_EXTRA_BODY_THINKING_OFF)
        assert r['finish_reason'] in ('stop', 'length')
        content = _require_str(r['content'], 'content')
        assert len(content.strip()) > 0
        assert THINK_START_TOKEN not in content
        assert THINK_END_TOKEN not in content
        _assert_reasoning_absent(r['reasoning'], stream=stream)
        if stream:
            assert r['reasoning_chunks'] == 0

    def test_with_tool_call(self, backend, model_case, stream):
        """enable_thinking=False + tool call: should call tool without
        reasoning."""
        r = self._call_api(stream,
                           MESSAGES_REASONING_WEATHER_TOOL,
                           tools=[WEATHER_TOOL],
                           extra_body=_EXTRA_BODY_THINKING_OFF)
        if len(r['tool_calls']) > 0:
            assert r['finish_reason'] == 'tool_calls'
            tc = r['tool_calls'][0]
            assert tc['name'] == 'get_current_weather'
            parsed = assert_tool_call_dict_fields(tc)
            assert 'city' in parsed
        _assert_reasoning_absent(r['reasoning'], stream=stream)
        _assert_no_tag_leakage(r['reasoning'], r['content'])

    def test_content_quality(self, backend, model_case, stream):
        """enable_thinking=False: content should still contain correct
        answer."""
        r = self._call_api(stream, MESSAGES_REASONING_BASIC, extra_body=_EXTRA_BODY_THINKING_OFF)
        content = _require_str(r['content'], 'content')
        assert len(content.strip()) > 0
        assert '1591' in content or '1,591' in content, \
            f"Expected '1591' or '1,591' in response, got: {content!r}"
