import json

import pytest
from utils.tool_reasoning_definitions import (SEARCH_TOOL, WEATHER_TOOL, assert_arguments_parseable,
                                              assert_tool_call_fields, collect_stream_tool_call)

from .conftest import MESSAGES_ASKING_FOR_SEARCH, MESSAGES_ASKING_FOR_WEATHER, _apply_marks, _ToolCallTestBase

# ===========================================================================
# Basic tool call: response structure, finish_reason, field validation
# ===========================================================================


@_apply_marks
class TestToolCallBasic(_ToolCallTestBase):
    """Basic tool call: response structure, finish_reason, field validation."""

    def test_non_streaming(self, backend, model_case):
        """Non-streaming: complete tool call response structure."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
        )

        # Response-level checks
        assert response.object == 'chat.completion'
        assert response.id is not None and len(response.id) > 0
        assert response.model is not None and len(response.model) > 0
        assert len(response.choices) == 1

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        assert choice.message.role == 'assistant'
        assert choice.finish_reason == 'tool_calls'
        assert tool_calls is not None and len(tool_calls) >= 1

        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.type == 'function'
        assert tc.function.name == WEATHER_TOOL['function']['name']

        parsed_args = assert_arguments_parseable(tc.function.arguments)
        assert isinstance(parsed_args.get('city'), str) and len(parsed_args['city']) > 0
        assert isinstance(parsed_args.get('state'), str) and len(parsed_args['state']) > 0

        # Token usage sanity
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    def test_streaming(self, backend, model_case):
        """Streaming: tool call id / name streamed once, args accumulated."""
        client, model_name = self._get_client()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
            stream=True,
        )

        r = collect_stream_tool_call(stream)

        assert r['finish_reason_count'] == 1, (f'Expected exactly 1 finish_reason, got {r["finish_reason_count"]}')
        assert r['finish_reason'] == 'tool_calls'
        assert r['role'] == 'assistant'
        assert isinstance(r['tool_call_id'], str) and len(r['tool_call_id']) >= 1
        assert r['tool_call_id'].strip() == r['tool_call_id'], 'tool_call_id has leading/trailing whitespace'
        assert r['function_name'] == WEATHER_TOOL['function']['name']

        streamed_args = assert_arguments_parseable(r['args_str'])
        assert isinstance(streamed_args.get('city'), str) and len(streamed_args['city']) > 0
        assert isinstance(streamed_args.get('state'), str) and len(streamed_args['state']) > 0


# ===========================================================================
# Streaming and non-streaming tool call results must match
# ===========================================================================


@_apply_marks
class TestToolCallStreamConsistency(_ToolCallTestBase):
    """Streaming and non-streaming tool call results must match."""

    def test_stream_nonstream_consistency(self, backend, model_case):
        client, model_name = self._get_client()

        # Use 1024 tokens to avoid truncation — reasoning models consume
        # thinking tokens before emitting the tool call JSON.
        common_kwargs = dict(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=1024,
            tools=[WEATHER_TOOL, SEARCH_TOOL],
            logprobs=False,
        )

        # Non-streaming
        ns_resp = client.chat.completions.create(**common_kwargs)
        ns_choice = ns_resp.choices[0]
        assert ns_choice.finish_reason == 'tool_calls'
        assert ns_choice.message.tool_calls is not None and len(ns_choice.message.tool_calls) >= 1
        ns_tc = ns_choice.message.tool_calls[0]
        ns_name = ns_tc.function.name
        ns_args = json.loads(ns_tc.function.arguments)

        # Streaming
        stream = client.chat.completions.create(**common_kwargs, stream=True)
        r = collect_stream_tool_call(stream)
        assert r['finish_reason'] == 'tool_calls'
        s_args = json.loads(r['args_str'])

        assert ns_name == r['function_name'], (f'Function name mismatch: non-stream={ns_name}, '
                                               f'stream={r["function_name"]}')
        assert ns_args == s_args, (f'Arguments mismatch: non-stream={ns_args}, stream={s_args}')
        # Verify both resolved to a valid tool name
        assert ns_name in ('get_current_weather', 'web_search'), (f'Unexpected function name: {ns_name}')


# ===========================================================================
# Test all tool_choice variants
# ===========================================================================


@_apply_marks
class TestToolCallChoice(_ToolCallTestBase):
    """Test all tool_choice variants."""

    # -- auto ----------------------------------------------------------------
    def test_tool_choice_auto(self, backend, model_case):
        """tool_choice='auto': model decides whether to call a tool."""
        client, model_name = self._get_client()

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
        assert choice.finish_reason in ('stop', 'length',
                                        'tool_calls'), (f'Unexpected finish_reason: {choice.finish_reason}')
        if choice.message.tool_calls and len(choice.message.tool_calls) > 0:
            for tc in choice.message.tool_calls:
                assert_tool_call_fields(tc)
                assert tc.type == 'function'
                assert tc.function.name in ('get_current_weather',
                                            'web_search'), (f'Unexpected tool: {tc.function.name}')
                assert_arguments_parseable(tc.function.arguments)
            assert choice.finish_reason == 'tool_calls'
        else:
            assert choice.message.content is not None and len(choice.message.content.strip()) > 0
            assert choice.finish_reason in ('stop', 'length')

    # -- none ----------------------------------------------------------------
    def test_tool_choice_none(self, backend, model_case):
        """tool_choice='none': model must NOT return tool calls."""
        client, model_name = self._get_client()

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
                or len(choice.message.tool_calls) == 0), ('tool_choice="none" but got tool_calls in response')
        assert choice.message.content is not None
        assert len(choice.message.content.strip()) > 0, ('tool_choice="none" should produce non-empty text content')
        assert choice.finish_reason in ('stop', 'length')

    # -- required ------------------------------------------------------------
    def test_tool_choice_required(self, backend, model_case):
        """tool_choice='required': model MUST return at least one tool call.

        Only skip when the *server* rejects the request (HTTP error).
        """
        client, model_name = self._get_client()

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
        except Exception as e:
            # Only skip if the server itself rejects the request
            pytest.skip(f'tool_choice="required" not supported by server: {e}')

        # Validation MUST fail loudly — never skip on assertion errors
        choice = response.choices[0]
        assert choice.message.role == 'assistant'
        assert choice.message.tool_calls is not None, ('tool_choice="required" but got no tool_calls')
        assert len(choice.message.tool_calls) >= 1
        for tc in choice.message.tool_calls:
            assert_tool_call_fields(tc)
            assert_arguments_parseable(tc.function.arguments)

    def test_tool_choice_required_streaming(self, backend, model_case):
        """tool_choice='required' + streaming: must return tool call chunks.

        Only skip if the server rejects the request, not on parse errors.
        """
        client, model_name = self._get_client()

        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=MESSAGES_ASKING_FOR_WEATHER,
                temperature=0,
                max_completion_tokens=1024,
                tools=[WEATHER_TOOL, SEARCH_TOOL],
                tool_choice='required',
                logprobs=False,
                stream=True,
            )
            r = collect_stream_tool_call(stream)
        except Exception as e:
            pytest.skip(f'tool_choice="required" streaming not supported by server: {e}')

        # Validation MUST fail loudly
        assert r['function_name'] is not None, ('tool_choice="required" streaming but no function name received')
        assert len(r['args_str']) > 0, ('tool_choice="required" streaming but no arguments received')
        assert r['tool_call_id'] is not None
        assert_arguments_parseable(r['args_str'])
        assert r['finish_reason'] == 'tool_calls'

    # -- specific function ---------------------------------------------------
    def test_tool_choice_specific_function(self, backend, model_case):
        """Force a specific tool via tool_choice={type, function}."""
        client, model_name = self._get_client()

        response = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=200,
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
        assert choice.message.role == 'assistant'

        tool_calls = choice.message.tool_calls
        assert tool_calls is not None and len(tool_calls) >= 1

        tc = tool_calls[0]
        assert_tool_call_fields(tc)
        assert tc.function.name == 'get_current_weather'
        assert_arguments_parseable(tc.function.arguments)


# ===========================================================================
# Validate that arguments are parseable and contain expected keys
# ===========================================================================


@_apply_marks
class TestToolCallArgumentsParsing(_ToolCallTestBase):
    """Validate that arguments are parseable and contain expected keys."""

    def test_weather_args(self, backend, model_case):
        """Weather tool args should contain city & state."""
        client, model_name = self._get_client()

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
        client, model_name = self._get_client()

        stream = client.chat.completions.create(
            model=model_name,
            messages=MESSAGES_ASKING_FOR_WEATHER,
            temperature=0,
            max_completion_tokens=1024,
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
        client, model_name = self._get_client()

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
        client, model_name = self._get_client()

        messages = [
            {
                'role': 'system',
                'content': 'You are a helpful weather assistant. '
                'Always use the weather tool and specify the unit.'
            },
            {
                'role': 'user',
                'content': 'What is the weather in Miami, FL in celsius?'
            },
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
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

        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls and len(tool_calls) > 0, (
            'Model did not return any tool calls despite tool_choice forcing '
            f'get_current_weather. finish_reason={response.choices[0].finish_reason}, '
            f'content={response.choices[0].message.content!r}')
        tc = tool_calls[0]
        parsed = json.loads(tc.function.arguments)
        if 'unit' in parsed:
            assert parsed['unit'] in ('celsius', 'fahrenheit'), (f'unit should be from enum, got "{parsed["unit"]}"')
