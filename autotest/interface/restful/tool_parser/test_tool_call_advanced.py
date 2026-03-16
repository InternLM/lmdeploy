import json

import pytest
from utils.tool_reasoning_definitions import (ALL_OPTIONAL_TOOL, CALCULATOR_TOOL, NESTED_PARAM_TOOL, SEARCH_TOOL,
                                              WEATHER_TOOL, WEATHER_TOOL_CN, assert_arguments_parseable,
                                              assert_tool_call_fields, build_messages_with_parallel_tool_responses,
                                              build_messages_with_tool_response, collect_stream_parallel_tool_calls,
                                              collect_stream_tool_call)

from .conftest import (MESSAGES_ASKING_FOR_CALCULATION, MESSAGES_ASKING_FOR_WEATHER, MESSAGES_ASKING_FOR_WEATHER_CN,
                       MESSAGES_NO_TOOL_NEEDED, MESSAGES_PARALLEL_MIXED, MESSAGES_PARALLEL_WEATHER, _apply_marks,
                       _ToolCallTestBase)

# ===========================================================================
# Model should pick the right tool from a multi-tool list
# ===========================================================================


@_apply_marks
class TestToolCallMultipleTools(_ToolCallTestBase):
    """Model should pick the right tool from a multi-tool list."""

    def test_selects_weather_tool(self, backend, model_case):
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL, CALCULATOR_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            assert choice.message.tool_calls[0].function.name == ('get_current_weather')

    def test_selects_calculator_tool(self, backend, model_case):
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_CALCULATION,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL, CALCULATOR_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert tc.function.name == 'calculate'
            parsed = json.loads(tc.function.arguments)
            assert 'expression' in parsed

    def test_no_tool_when_not_needed(self, backend, model_case):
        """Unrelated question + tool_choice=auto → prefer text."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_NO_TOOL_NEEDED,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice='auto',
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            for tc in choice.message.tool_calls:
                assert_tool_call_fields(tc)
        else:
            assert choice.message.content is not None
            assert len(choice.message.content) > 0

    def test_large_number_of_tools(self, backend, model_case):
        """Model should still pick the right tool among 10+ definitions."""
        client, model_name = self._get_client()

        tools = [WEATHER_TOOL]
        for i in range(10):
            tools.append({
                'type': 'function',
                'function': {
                    'name': f'dummy_tool_{i}',
                    'description': f'A dummy tool number {i} (does nothing).',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'input': {
                                'type': 'string',
                                'description': f'Input for dummy tool {i}',
                            },
                        },
                        'required': ['input'],
                    },
                },
            })

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=tools,
            logprobs=False,
        )

        choice = response.choices[0]
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert_tool_call_fields(tc)
            assert_arguments_parseable(tc.function.arguments)
            assert tc.function.name == 'get_current_weather', (f'Expected weather tool, got "{tc.function.name}"')


# ===========================================================================
# Parallel tool calls in a single response
# ===========================================================================


@_apply_marks
class TestToolCallParallel(_ToolCallTestBase):
    """Parallel tool calls in a single response."""

    def test_parallel_same_tool(self, backend, model_case):
        """Two cities → two weather tool calls."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_PARALLEL_WEATHER,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        tool_calls = response.choices[0].message.tool_calls

        # Hard assertion: two cities asked → must get ≥ 2 tool calls
        assert tool_calls is not None and len(tool_calls) >= 2, (f'Expected ≥2 parallel tool calls for two cities, '
                                                                 f'got {len(tool_calls) if tool_calls else 0}')

        for tc in tool_calls:
            assert_tool_call_fields(tc)
            assert tc.function.name == 'get_current_weather'
            parsed = assert_arguments_parseable(tc.function.arguments)
            assert 'city' in parsed and 'state' in parsed

        ids = [tc.id for tc in tool_calls]
        assert len(set(ids)) == len(ids), (f'IDs should be unique, got {ids}')
        assert response.choices[0].finish_reason == 'tool_calls'

    def test_parallel_same_tool_streaming(self, backend, model_case):
        """Streaming: parallel tool calls indexed correctly."""
        client, model_name = self._get_client()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_PARALLEL_WEATHER,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL],
            logprobs=False,
            stream=True,
        )

        tc_data, fr_count = collect_stream_parallel_tool_calls(stream)
        assert fr_count == 1

        # Hard assertion: must receive ≥ 2 distinct tool call indices
        assert len(tc_data) >= 2, (f'Expected ≥2 parallel streaming tool calls, '
                                   f'got {len(tc_data)} indices: {list(tc_data.keys())}')

        for idx, data in tc_data.items():
            assert data['name'] is not None, (f'Index {idx}: missing function name')
            assert len(data['args_str']) > 0, (f'Index {idx}: missing arguments')
            parsed = json.loads(data['args_str'])
            assert isinstance(parsed, dict)

    def test_parallel_mixed_tools(self, backend, model_case):
        """Weather + calculator in one request."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_PARALLEL_MIXED,
            temperature=0,
            max_completion_tokens=400,
            tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            logprobs=False,
        )

        tool_calls = response.choices[0].message.tool_calls

        # Hard assertion: weather + calculation asked → ≥ 2 tool calls
        assert tool_calls is not None and len(tool_calls) >= 2, (f'Expected ≥2 parallel tool calls (weather+calc), '
                                                                 f'got {len(tool_calls) if tool_calls else 0}')

        for tc in tool_calls:
            assert_tool_call_fields(tc)
            assert_arguments_parseable(tc.function.arguments)

        ids = [tc.id for tc in tool_calls]
        assert len(set(ids)) == len(ids), (f'Tool call IDs should be unique, got {ids}')

        names = {tc.function.name for tc in tool_calls}
        assert len(names) >= 2, (f'Expected ≥2 distinct tool names, got {names}')
        assert 'get_current_weather' in names, (f'Expected get_current_weather in tool calls, got {names}')
        assert 'calculate' in names, (f'Expected calculate in tool calls, got {names}')


# ===========================================================================
# Feed tool results back; model should reply with text
# ===========================================================================


@_apply_marks
class TestToolCallWithResults(_ToolCallTestBase):
    """Feed tool results back; model should reply with text."""

    def test_single_result(self, backend, model_case):
        client, model_name = self._get_client()
        messages = build_messages_with_tool_response()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.finish_reason in ('stop', 'length')
        assert choice.message.role == 'assistant'
        assert (choice.message.tool_calls is None or len(choice.message.tool_calls) == 0)
        assert choice.message.content and len(choice.message.content) > 0
        assert '98' in choice.message.content or 'Dallas' in choice.message.content

    def test_multiple_results(self, backend, model_case):
        """Feed two parallel tool results back at once."""
        client, model_name = self._get_client()
        messages = build_messages_with_parallel_tool_responses()

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
        assert choice.message.content and len(choice.message.content) > 0

        content = choice.message.content
        has_dallas = 'Dallas' in content or '98' in content
        has_sf = 'San Francisco' in content or '65' in content
        assert has_dallas or has_sf


# ===========================================================================
# Multilingual tool calls
# ===========================================================================


@_apply_marks
class TestToolCallMultilingual(_ToolCallTestBase):

    def test_chinese_description(self, backend, model_case):
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER_CN,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL_CN],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'

        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert_tool_call_fields(tc)
            assert tc.function.name == 'get_current_weather'
            parsed = assert_arguments_parseable(tc.function.arguments)
            assert 'city' in parsed
            assert isinstance(parsed['city'], str) and len(parsed['city']) > 0

    def test_chinese_description_streaming(self, backend, model_case):
        client, model_name = self._get_client()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER_CN,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL_CN],
            logprobs=False,
            stream=True,
        )

        r = collect_stream_tool_call(stream)
        if r['function_name'] is not None:
            assert r['function_name'] == 'get_current_weather'
            parsed = json.loads(r['args_str'])
            assert isinstance(parsed, dict)
            assert 'city' in parsed

    def test_mixed_language_tools(self, backend, model_case):
        """Pass Chinese + English tool definitions together."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER_CN,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL_CN, SEARCH_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            for tc in choice.message.tool_calls:
                assert_tool_call_fields(tc)
                assert_arguments_parseable(tc.function.arguments)

    def test_unicode_arguments(self, backend, model_case):
        """Chinese query → tool arguments with Unicode chars."""
        client, model_name = self._get_client()

        messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant. Use the search tool.'
            },
            {
                'role': 'user',
                'content': '请搜索一下"人工智能最新进展"'
            },
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=200,
            tools=[SEARCH_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert_tool_call_fields(tc)
            assert_arguments_parseable(tc.function.arguments)


# ===========================================================================
# Nested objects, arrays, enum constraints, all-optional params
# ===========================================================================


@_apply_marks
class TestToolCallComplexParams(_ToolCallTestBase):
    """Nested objects, arrays, enum constraints, all-optional params."""

    def test_nested_object_parameters(self, backend, model_case):
        client, model_name = self._get_client()

        messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant. Use the create_event '
                'tool when asked to schedule events.'
            },
            {
                'role':
                'user',
                'content':
                'Schedule a team meeting titled "Sprint Review" at '
                'the Conference Room in New York with attendees '
                'alice@example.com and bob@example.com, high priority.'
            },
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=400,
            tools=[NESTED_PARAM_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert_tool_call_fields(tc)
            assert tc.function.name == 'create_event'

            parsed = json.loads(tc.function.arguments)
            assert 'title' in parsed

            if 'location' in parsed:
                assert isinstance(parsed['location'], dict)
            if 'attendees' in parsed:
                assert isinstance(parsed['attendees'], list)
            if 'priority' in parsed:
                assert parsed['priority'] in ('low', 'medium', 'high')

    def test_all_optional_parameters(self, backend, model_case):
        client, model_name = self._get_client()

        messages = [
            {
                'role': 'system',
                'content': 'You are a logging assistant. '
                'Use the log_message tool to log messages.'
            },
            {
                'role': 'user',
                'content': 'Log an info message saying '
                '"System started successfully".'
            },
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=200,
            tools=[ALL_OPTIONAL_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert_tool_call_fields(tc)
            parsed = assert_arguments_parseable(tc.function.arguments)

            if 'message' in parsed:
                assert isinstance(parsed['message'], str)
            if 'level' in parsed:
                assert parsed['level'] in ('debug', 'info', 'warning', 'error')


# ===========================================================================
# Validate response-level fields when tool calls are returned
# ===========================================================================


@_apply_marks
class TestToolCallResponseValidation(_ToolCallTestBase):
    """Validate response-level fields when tool calls are returned."""

    def test_content_null_when_tool_calls_present(self, backend, model_case):
        """Per OpenAI spec, content should be null or empty when tool_calls
        exist."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
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
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) >= 1

        # Per OpenAI spec: content should be null or empty when tool_calls
        # are present.
        if choice.message.content is not None:
            assert choice.message.content.strip() == '', (f'content should be null/empty when tool_calls are '
                                                          f'present, got: {choice.message.content!r}')

    def test_usage_field_present(self, backend, model_case):
        """usage.prompt_tokens / completion_tokens / total_tokens."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens > 0

        choice = response.choices[0]
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            assert response.usage.completion_tokens > 0
            assert response.usage.total_tokens == (response.usage.prompt_tokens + response.usage.completion_tokens)

    def test_model_and_metadata_fields(self, backend, model_case):
        """Response must contain model, id, and created."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        assert response.model is not None
        assert isinstance(response.model, str) and len(response.model) > 0
        assert response.id is not None
        assert response.created is not None

    def test_choices_structure(self, backend, model_case):
        """choices[0].index should be 0."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL],
            logprobs=False,
            n=1,
        )

        assert len(response.choices) >= 1
        assert response.choices[0].index == 0


# ===========================================================================
# Edge cases and robustness tests
# ===========================================================================


@_apply_marks
class TestToolCallEdgeCases(_ToolCallTestBase):
    """Edge cases and robustness tests."""

    def test_empty_tools_list(self, backend, model_case):
        """Empty tools list should behave like no tools."""
        from openai import BadRequestError
        client, model_name = self._get_client()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=MESSAGES_NO_TOOL_NEEDED,
                temperature=0,
                max_completion_tokens=100,
                tools=[],
                logprobs=False,
            )
        except BadRequestError:
            pytest.skip('Backend rejects empty tools list')

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.message.content is not None
        assert (choice.message.tool_calls is None or len(choice.message.tool_calls) == 0)

    def test_tool_call_with_max_tokens(self, backend, model_case):
        """With sufficient max_tokens, tool call structure must be valid."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=500,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert_tool_call_fields(tc)
            assert_arguments_parseable(tc.function.arguments)

    def test_tool_call_id_format(self, backend, model_case):
        """ID should be a non-empty string with no leading/trailing spaces."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert isinstance(tc.id, str)
            assert len(tc.id) >= 1
            assert tc.id.strip() == tc.id

    def test_multi_turn_conversation(self, backend, model_case):
        """Tool call → result → follow-up question → possible second call."""
        client, model_name = self._get_client()

        messages = build_messages_with_tool_response()
        messages.append({
            'role': 'user',
            'content': 'Now search the web for how to stay cool in hot weather.',
        })

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert_tool_call_fields(tc)
            if tc.function.name == 'web_search':
                parsed = json.loads(tc.function.arguments)
                assert 'query' in parsed
        else:
            assert choice.message.content and len(choice.message.content) > 0

    def test_special_characters_in_query(self, backend, model_case):
        """Quotes, angle brackets, Unicode → JSON args still parseable."""
        client, model_name = self._get_client()

        messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant that can use tools.'
            },
            {
                'role':
                'user',
                'content':
                'Search for "what\'s the latest on AI & ML?" '
                '(include results with special chars: <>, "quotes", '
                'and unicode: café, naïve, résumé)'
            },
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=200,
            tools=[SEARCH_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert_tool_call_fields(tc)
            parsed = assert_arguments_parseable(tc.function.arguments)
            if 'query' in parsed:
                assert isinstance(parsed['query'], str)
                assert len(parsed['query']) > 0
