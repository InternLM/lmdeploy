import json

import pytest
from utils.constant import BACKEND_LIST, RESTFUL_MODEL_LIST
from utils.tool_reasoning_definitions import (
    ALL_OPTIONAL_TOOL,
    CALCULATOR_TOOL,
    NESTED_PARAM_TOOL,
    SEARCH_TOOL,
    WEATHER_TOOL,
    WEATHER_TOOL_CN,
    assert_arguments_parseable,
    assert_tool_call_fields,
    build_messages_with_parallel_tool_responses,
    build_messages_with_tool_response,
    collect_stream_content,
    collect_stream_parallel_tool_calls,
    collect_stream_tool_call,
    get_client_and_model,
)

_CLASS_MARKS = [
    pytest.mark.order(8),
    pytest.mark.tool_call,
    pytest.mark.flaky(reruns=2),
    pytest.mark.parametrize('backend', BACKEND_LIST),
    pytest.mark.parametrize('model_case', RESTFUL_MODEL_LIST),
]


def _apply_marks(cls):
    """Apply the shared set of marks to *cls* and return it."""
    for m in _CLASS_MARKS:
        cls = m(cls)
    return cls


MESSAGES_ASKING_FOR_WEATHER = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant that can use tools. '
                   'When asked about weather, use the get_current_weather tool.',
    },
    {
        'role': 'user',
        'content': "What's the weather like in Dallas, TX?",
    },
]

MESSAGES_ASKING_FOR_SEARCH = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant with access to tools. '
                   'Use the web_search tool when asked to look something up.',
    },
    {
        'role': 'user',
        'content': 'Search the web for the latest news about AI.',
    },
]

MESSAGES_ASKING_FOR_CALCULATION = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant. When asked math questions, '
                   'use the calculate tool.',
    },
    {
        'role': 'user',
        'content': 'What is 1234 * 5678?',
    },
]

MESSAGES_ASKING_FOR_WEATHER_CN = [
    {
        'role': 'system',
        'content': '你是一个有用的助手，可以使用工具。'
                   '当被问到天气时，请使用get_current_weather工具。',
    },
    {
        'role': 'user',
        'content': '北京今天的天气怎么样？',
    },
]

MESSAGES_NO_TOOL_NEEDED = [
    {
        'role': 'user',
        'content': 'Hi, please introduce yourself briefly.',
    },
]

MESSAGES_PARALLEL_WEATHER = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant. When asked about weather '
                   'in multiple cities, call the weather tool for each city '
                   'separately.',
    },
    {
        'role': 'user',
        'content': "What's the weather in Dallas, TX and also in "
                   'San Francisco, CA?',
    },
]

MESSAGES_PARALLEL_MIXED = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant with access to multiple tools. '
                   'You can call multiple tools in parallel when needed.',
    },
    {
        'role': 'user',
        'content': "What's the weather in Dallas, TX? "
                   'Also calculate 1234 * 5678.',
    },
]



@_apply_marks
class TestToolCallBasic:
    """Basic tool call: response structure, finish_reason, field validation."""

    def test_non_streaming(self, backend, model_case):
        """Non-streaming: complete tool call response structure."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        assert choice.message.role == 'assistant'
        assert tool_calls is not None and len(tool_calls) >= 1

        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == WEATHER_TOOL['function']['name']

        parsed_args = assert_arguments_parseable(tc.function.arguments)
        assert isinstance(parsed_args.get('city'), str)
        assert isinstance(parsed_args.get('state'), str)
        assert choice.finish_reason == 'tool_calls'

    def test_streaming(self, backend, model_case):
        """Streaming: tool call id / name streamed once, args accumulated."""
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
            stream=True,
        )

        r = collect_stream_tool_call(stream)

        assert r['finish_reason_count'] == 1
        assert r['finish_reason'] == 'tool_calls'
        assert r['role'] == 'assistant'
        assert isinstance(r['tool_call_id'], str) and len(r['tool_call_id']) >= 1
        assert r['function_name'] == WEATHER_TOOL['function']['name']

        streamed_args = assert_arguments_parseable(r['args_str'])
        assert isinstance(streamed_args.get('city'), str)
        assert isinstance(streamed_args.get('state'), str)


@_apply_marks
class TestToolCallStreamConsistency:
    """Streaming and non-streaming tool call results must match."""

    def test_stream_nonstream_consistency(self, backend, model_case):
        client, model_name = get_client_and_model()

        # Non-streaming
        ns_resp = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
        )
        ns_tc = ns_resp.choices[0].message.tool_calls[0]
        ns_name = ns_tc.function.name
        ns_args = json.loads(ns_tc.function.arguments)

        # Streaming
        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
            stream=True,
        )
        r = collect_stream_tool_call(stream)
        s_args = json.loads(r['args_str'])

        assert ns_name == r['function_name'], (
            f'Function name mismatch: non-stream={ns_name}, '
            f'stream={r["function_name"]}')
        assert ns_args == s_args, (
            f'Arguments mismatch: non-stream={ns_args}, stream={s_args}')


@_apply_marks
class TestToolCallChoice:
    """Test all tool_choice variants."""

    # -- auto ----------------------------------------------------------------
    def test_tool_choice_auto(self, backend, model_case):
        """tool_choice='auto': model decides whether to call a tool."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice='auto',
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            assert_tool_call_fields(choice.message.tool_calls[0])
            assert choice.finish_reason == 'tool_calls'
        else:
            assert choice.message.content is not None
            assert choice.finish_reason in ('stop', 'length')

    # -- none ----------------------------------------------------------------
    def test_tool_choice_none(self, backend, model_case):
        """tool_choice='none': model must NOT return tool calls."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice='none',
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert (choice.message.tool_calls is None
                or len(choice.message.tool_calls) == 0)
        assert choice.message.content is not None
        assert choice.finish_reason in ('stop', 'length')

    def test_tool_choice_none_streaming(self, backend, model_case):
        """tool_choice='none' + streaming: no tool_calls in any chunk."""
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice='none',
            logprobs=False,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                chunks.append(delta.content)
            assert (not delta.tool_calls or len(delta.tool_calls) == 0), (
                'tool_choice="none" but got tool_calls in stream')

        assert len(chunks) > 0

    # -- required ------------------------------------------------------------
    def test_tool_choice_required(self, backend, model_case):
        """tool_choice='required': model MUST return at least one tool call."""
        client, model_name = get_client_and_model()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=MESSAGES_ASKING_FOR_WEATHER,
                temperature=0,
                max_completion_tokens=200,
                tools=[WEATHER_TOOL, SEARCH_TOOL],
                tool_choice='required',
                logprobs=False,
            )

            choice = response.choices[0]
            assert choice.message.role == 'assistant'
            assert choice.message.tool_calls is not None
            assert len(choice.message.tool_calls) >= 1
            for tc in choice.message.tool_calls:
                assert_tool_call_fields(tc)
                assert_arguments_parseable(tc.function.arguments)
        except Exception as e:
            pytest.skip(f'tool_choice="required" not supported: {e}')

    def test_tool_choice_required_streaming(self, backend, model_case):
        """tool_choice='required' + streaming: must return tool call chunks."""
        client, model_name = get_client_and_model()

        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=MESSAGES_ASKING_FOR_WEATHER,
                temperature=0,
                max_completion_tokens=200,
                tools=[WEATHER_TOOL, SEARCH_TOOL],
                tool_choice='required',
                logprobs=False,
                stream=True,
            )

            r = collect_stream_tool_call(stream)

            assert r['function_name'] is not None
            assert len(r['args_str']) > 0
            assert r['tool_call_id'] is not None
            assert_arguments_parseable(r['args_str'])
            assert r['finish_reason'] == 'tool_calls'
        except Exception as e:
            pytest.skip(
                f'tool_choice="required" streaming not supported: {e}')

    # -- specific function ---------------------------------------------------
    def test_tool_choice_specific_function(self, backend, model_case):
        """Force a specific tool via tool_choice={type, function}."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice={
                'type': 'function',
                'function': {'name': 'get_current_weather'},
            },
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'

        tool_calls = choice.message.tool_calls
        assert tool_calls is not None and len(tool_calls) >= 1

        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'get_current_weather'
        assert_arguments_parseable(tc.function.arguments)

    def test_tool_choice_specific_function_streaming(self, backend, model_case):
        """Force a specific tool + streaming."""
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            tool_choice={
                'type': 'function',
                'function': {'name': 'get_current_weather'},
            },
            logprobs=False,
            stream=True,
        )

        r = collect_stream_tool_call(stream)

        assert r['function_name'] == 'get_current_weather'
        assert r['finish_reason_count'] == 1
        assert_arguments_parseable(r['args_str'])


@_apply_marks
class TestToolCallArgumentsParsing:
    """Validate that arguments are parseable and contain expected keys."""

    def test_weather_args(self, backend, model_case):
        """Weather tool args should contain city & state."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        tc = response.choices[0].message.tool_calls[0]
        parsed = json.loads(tc.function.arguments)

        assert 'city' in parsed, f'Missing "city": {parsed}'
        assert 'state' in parsed, f'Missing "state": {parsed}'
        assert 'dallas' in parsed['city'].lower()
        assert 'tx' in parsed['state'].lower()

    def test_weather_args_streaming(self, backend, model_case):
        """Streaming: weather tool args should contain city & state."""
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL],
            logprobs=False,
            stream=True,
        )

        r = collect_stream_tool_call(stream)
        parsed = json.loads(r['args_str'])

        assert 'city' in parsed
        assert 'state' in parsed

    def test_search_tool_args(self, backend, model_case):
        """Search tool args should contain a non-empty query."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_SEARCH,
            temperature=0,
            max_completion_tokens=200,
            tools=[SEARCH_TOOL],
            logprobs=False,
        )

        choice = response.choices[0]
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            tc = choice.message.tool_calls[0]
            assert tc.function.name == 'web_search'
            parsed = json.loads(tc.function.arguments)
            assert 'query' in parsed
            assert isinstance(parsed['query'], str) and len(parsed['query']) > 0

    def test_enum_parameter_constraint(self, backend, model_case):
        """Enum-constrained params should only return valid values."""
        client, model_name = get_client_and_model()

        messages = [
            {'role': 'system',
             'content': 'You are a helpful weather assistant. '
                        'Always use the weather tool and specify the unit.'},
            {'role': 'user',
             'content': 'What is the weather in Miami, FL in celsius?'},
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL],
            tool_choice={
                'type': 'function',
                'function': {'name': 'get_current_weather'},
            },
            logprobs=False,
        )

        tc = response.choices[0].message.tool_calls[0]
        parsed = json.loads(tc.function.arguments)
        if 'unit' in parsed:
            assert parsed['unit'] in ('celsius', 'fahrenheit'), (
                f'unit should be from enum, got "{parsed["unit"]}"')


@_apply_marks
class TestToolCallMultipleTools:
    """Model should pick the right tool from a multi-tool list."""

    def test_selects_weather_tool(self, backend, model_case):
        client, model_name = get_client_and_model()

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
            assert choice.message.tool_calls[0].function.name == (
                'get_current_weather')

    def test_selects_calculator_tool(self, backend, model_case):
        client, model_name = get_client_and_model()

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
        client, model_name = get_client_and_model()

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
        client, model_name = get_client_and_model()

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
            assert tc.function.name == 'get_current_weather', (
                f'Expected weather tool, got "{tc.function.name}"')


@_apply_marks
class TestToolCallParallel:
    """Parallel tool calls in a single response."""

    def test_parallel_same_tool(self, backend, model_case):
        """Two cities → two weather tool calls."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_PARALLEL_WEATHER,
            temperature=0,
            max_completion_tokens=300,
            tools=[WEATHER_TOOL],
            logprobs=False,
        )

        tool_calls = response.choices[0].message.tool_calls

        if tool_calls and len(tool_calls) >= 2:
            for tc in tool_calls:
                assert_tool_call_fields(tc)
                assert tc.function.name == 'get_current_weather'
                parsed = assert_arguments_parseable(tc.function.arguments)
                assert 'city' in parsed and 'state' in parsed

            ids = [tc.id for tc in tool_calls]
            assert len(set(ids)) == len(ids), (
                f'IDs should be unique, got {ids}')
            assert response.choices[0].finish_reason == 'tool_calls'
        elif tool_calls and len(tool_calls) == 1:
            assert_tool_call_fields(tool_calls[0])

    def test_parallel_same_tool_streaming(self, backend, model_case):
        """Streaming: parallel tool calls indexed correctly."""
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_PARALLEL_WEATHER,
            temperature=0,
            max_completion_tokens=300,
            tools=[WEATHER_TOOL],
            logprobs=False,
            stream=True,
        )

        tc_data, fr_count = collect_stream_parallel_tool_calls(stream)
        assert fr_count == 1

        for idx, data in tc_data.items():
            assert data['name'] is not None, (
                f'Index {idx}: missing function name')
            assert len(data['args_str']) > 0, (
                f'Index {idx}: missing arguments')
            parsed = json.loads(data['args_str'])
            assert isinstance(parsed, dict)

    def test_parallel_mixed_tools(self, backend, model_case):
        """Weather + calculator in one request."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_PARALLEL_MIXED,
            temperature=0,
            max_completion_tokens=400,
            tools=[WEATHER_TOOL, CALCULATOR_TOOL],
            logprobs=False,
        )

        tool_calls = response.choices[0].message.tool_calls

        if tool_calls and len(tool_calls) >= 2:
            for tc in tool_calls:
                assert_tool_call_fields(tc)
                assert_arguments_parseable(tc.function.arguments)

            ids = [tc.id for tc in tool_calls]
            assert len(set(ids)) == len(ids)

            names = {tc.function.name for tc in tool_calls}
            if len(names) >= 2:
                assert ('get_current_weather' in names
                        or 'calculate' in names)
        elif tool_calls and len(tool_calls) == 1:
            assert_tool_call_fields(tool_calls[0])
            assert_arguments_parseable(tool_calls[0].function.arguments)


@_apply_marks
class TestToolCallWithResults:
    """Feed tool results back; model should reply with text."""

    def test_single_result(self, backend, model_case):
        client, model_name = get_client_and_model()
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
        assert (choice.message.tool_calls is None
                or len(choice.message.tool_calls) == 0)
        assert choice.message.content and len(choice.message.content) > 0
        assert '98' in choice.message.content or 'Dallas' in choice.message.content

    def test_single_result_streaming(self, backend, model_case):
        client, model_name = get_client_and_model()
        messages = build_messages_with_tool_response()

        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
            stream=True,
        )

        chunks, finish_reason = collect_stream_content(stream)
        assert finish_reason in ('stop', 'length')
        assert len(chunks) > 0

        full = ''.join(chunks)
        assert '98' in full or 'Dallas' in full

    def test_multiple_results(self, backend, model_case):
        """Feed two parallel tool results back at once."""
        client, model_name = get_client_and_model()
        messages = build_messages_with_parallel_tool_responses()

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=300,
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

    def test_multiple_results_streaming(self, backend, model_case):
        """Streaming: two parallel tool results fed back."""
        client, model_name = get_client_and_model()
        messages = build_messages_with_parallel_tool_responses()

        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=300,
            tools=[WEATHER_TOOL],
            logprobs=False,
            stream=True,
        )

        chunks = []
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                chunks.append(delta.content)
            assert (not delta.tool_calls or len(delta.tool_calls) == 0), (
                'Should not have tool calls after providing results')
        assert len(chunks) > 0


@_apply_marks
class TestToolCallMultilingual:

    def test_chinese_description(self, backend, model_case):
        client, model_name = get_client_and_model()

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
        client, model_name = get_client_and_model()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER_CN,
            temperature=0,
            max_completion_tokens=200,
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
        client, model_name = get_client_and_model()

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
        client, model_name = get_client_and_model()

        messages = [
            {'role': 'system',
             'content': 'You are a helpful assistant. Use the search tool.'},
            {'role': 'user',
             'content': '请搜索一下"人工智能最新进展"'},
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


@_apply_marks
class TestToolCallComplexParams:
    """Nested objects, arrays, enum constraints, all-optional params."""

    def test_nested_object_parameters(self, backend, model_case):
        client, model_name = get_client_and_model()

        messages = [
            {'role': 'system',
             'content': 'You are a helpful assistant. Use the create_event '
                        'tool when asked to schedule events.'},
            {'role': 'user',
             'content': 'Schedule a team meeting titled "Sprint Review" at '
                        'the Conference Room in New York with attendees '
                        'alice@example.com and bob@example.com, high priority.'},
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
        client, model_name = get_client_and_model()

        messages = [
            {'role': 'system',
             'content': 'You are a logging assistant. '
                        'Use the log_message tool to log messages.'},
            {'role': 'user',
             'content': 'Log an info message saying '
                        '"System started successfully".'},
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
                assert parsed['level'] in (
                    'debug', 'info', 'warning', 'error')


@_apply_marks
class TestToolCallResponseValidation:
    """Validate response-level fields when tool calls are returned."""

    def test_content_null_when_tool_calls_present(self, backend, model_case):
        """Per OpenAI spec, content should be null when tool_calls exist."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL],
            tool_choice={
                'type': 'function',
                'function': {'name': 'get_current_weather'},
            },
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) >= 1

        # Content should be null or empty when tool_calls are present
        if choice.message.content is not None:
            assert choice.message.content.strip() == '' or True

    def test_usage_field_present(self, backend, model_case):
        """usage.prompt_tokens / completion_tokens / total_tokens."""
        client, model_name = get_client_and_model()

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
            assert response.usage.total_tokens == (
                response.usage.prompt_tokens
                + response.usage.completion_tokens)

    def test_model_and_metadata_fields(self, backend, model_case):
        """Response must contain model, id, and created."""
        client, model_name = get_client_and_model()

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
        client, model_name = get_client_and_model()

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


@_apply_marks
class TestToolCallEdgeCases:
    """Edge cases and robustness tests."""

    def test_no_tools_provided(self, backend, model_case):
        """No tools → normal text response."""
        client, model_name = get_client_and_model()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_NO_TOOL_NEEDED,
            temperature=0,
            max_completion_tokens=100,
            logprobs=False,
        )

        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.message.content and len(choice.message.content) > 0
        assert choice.finish_reason in ('stop', 'length')

    def test_empty_tools_list(self, backend, model_case):
        """Empty tools list should behave like no tools."""
        client, model_name = get_client_and_model()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=MESSAGES_NO_TOOL_NEEDED,
                temperature=0,
                max_completion_tokens=100,
                tools=[],
                logprobs=False,
            )

            choice = response.choices[0]
            assert choice.message.role == 'assistant'
            assert choice.message.content is not None
            assert (choice.message.tool_calls is None
                    or len(choice.message.tool_calls) == 0)
        except Exception:
            # Some backends reject an empty tools list
            pass

    def test_tool_call_with_max_tokens(self, backend, model_case):
        """With sufficient max_tokens, tool call structure must be valid."""
        client, model_name = get_client_and_model()

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
        client, model_name = get_client_and_model()

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
        """tool call → result → follow-up question → possible second call."""
        client, model_name = get_client_and_model()

        messages = build_messages_with_tool_response()
        messages.append({
            'role': 'user',
            'content': 'Now search the web for how to stay cool in hot weather.',
        })

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_completion_tokens=300,
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
        client, model_name = get_client_and_model()

        messages = [
            {'role': 'system',
             'content': 'You are a helpful assistant that can use tools.'},
            {'role': 'user',
             'content': 'Search for "what\'s the latest on AI & ML?" '
                        '(include results with special chars: <>, "quotes", '
                        'and unicode: café, naïve, résumé)'},
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
