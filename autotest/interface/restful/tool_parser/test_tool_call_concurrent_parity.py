

import asyncio

import aiohttp
import pytest
from utils.constant import BACKEND_LIST, TOOL_REASONING_MODEL_LIST
from utils.tool_reasoning_definitions import (
    CONCURRENT_WEATHER_TOOL,
    DEFAULT_TOOL_CALL_CONCURRENCY,
    DEFAULT_TOOL_CALL_HTTP_ERROR_WORKERS,
    HttpToolCallError,
    RoutedExpertsNotSupported,
    collect_stream_tool_call_http_async,
    validate_reference_turn_result,
    validate_stream_tool_call_with_tokens,
)

from .conftest import (
    MULTI_TURN_WEATHER_CITIES,
    _apply_marks,
    _ToolCallTestBase,
)

_CLASS_MARKS_STRESS = [
    pytest.mark.order(8),
    pytest.mark.tool_call,
    pytest.mark.stress,
    pytest.mark.flaky(reruns=1),
    pytest.mark.parametrize('backend', BACKEND_LIST),
    pytest.mark.parametrize('model_case', TOOL_REASONING_MODEL_LIST),
]


def _apply_marks_stress(cls):
    for m in _CLASS_MARKS_STRESS:
        cls = m(cls)
    return cls


@_apply_marks
class TestToolCallConcurrentParity(_ToolCallTestBase):
    """Concurrent tool-call streaming parity (input_ids, multi-turn,
    errors)."""

    def test_http_error_raises_formatted_message(self, backend, model_case):
        """Non-200 responses raise ``HttpToolCallError`` with formatted
        text."""
        async def _run():
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                with pytest.raises(HttpToolCallError) as exc_info:
                    await collect_stream_tool_call_http_async(
                        session,
                        '__invalid_model_name_for_tool_call_test__',
                        messages=[{'role': 'user', 'content': 'hi'}],
                        use_input_ids=False,
                    )
                assert 'HTTP' in exc_info.value.message

        asyncio.run(_run())

    def test_input_ids_path_streaming(self, backend, model_case):
        """``input_ids`` + empty ``messages`` in request payload."""
        messages = []
        messages.append({'role': 'user', 'content': 'What is the weather in Tokyo?'})
        r = self._stream_tool_call_with_tokens(
            messages,
            tools=[CONCURRENT_WEATHER_TOOL],
            use_input_ids=True,
            reference_payload=True,
        )
        prompt_tokens = r['prompt_tokens_computed'] or r['prompt_tokens']
        assert prompt_tokens > 0
        validate_reference_turn_result(
            r,
            prompt_tokens,
            expected_function_name='get_weather',
            **self._parser_validation_kwargs([CONCURRENT_WEATHER_TOOL]),
        )

    def test_messages_path_streaming_with_done_and_tokens(self, backend, model_case):
        """Standard messages path still requires ``[DONE]`` + token fields."""
        messages = []
        messages.append({'role': 'user', 'content': 'What is the weather in London?'})
        r = self._stream_tool_call_with_tokens(
            messages,
            tools=[CONCURRENT_WEATHER_TOOL],
            use_input_ids=False,
        )
        prompt_tokens = r['prompt_tokens_computed'] or r['prompt_tokens']
        try:
            validate_stream_tool_call_with_tokens(
                r,
                prompt_tokens=prompt_tokens,
                expected_function_name='get_weather',
                **self._parser_validation_kwargs([CONCURRENT_WEATHER_TOOL]),
            )
        except RoutedExpertsNotSupported as exc:
            pytest.skip(str(exc))

    def test_parser_drop_via_decoded_output_ids(self, backend, model_case):
        """Per-turn validation including ``decoded_str`` parser-drop check."""
        messages = []
        messages.append({'role': 'user', 'content': 'What is the weather in Paris?'})
        r = self._stream_tool_call_with_tokens(
            messages,
            tools=[CONCURRENT_WEATHER_TOOL],
            use_input_ids=True,
            reference_payload=True,
        )
        prompt_tokens = r['prompt_tokens_computed'] or r['prompt_tokens']
        validate_reference_turn_result(
            r,
            prompt_tokens,
            expected_function_name='get_weather',
            **self._parser_validation_kwargs([CONCURRENT_WEATHER_TOOL]),
        )

    def test_multi_turn_input_ids_with_tool_name_in_history(self, backend, model_case):
        """3-turn loop via ``input_ids``; tool messages include ``name``
        field."""
        messages = []
        num_turns = 3

        for turn in range(num_turns):
            city = MULTI_TURN_WEATHER_CITIES[turn % len(MULTI_TURN_WEATHER_CITIES)]
            messages.append({'role': 'user', 'content': f'What is the weather in {city}?'})
            r = self._stream_tool_call_with_tokens(
                messages,
                tools=[CONCURRENT_WEATHER_TOOL],
                use_input_ids=True,
                reference_payload=True,
            )
            prompt_tokens = r['prompt_tokens_computed'] or r['prompt_tokens']
            validate_reference_turn_result(
                r,
                prompt_tokens,
                expected_function_name='get_weather',
                **self._parser_validation_kwargs([CONCURRENT_WEATHER_TOOL]),
            )
            self._append_assistant_and_tool_messages(messages, r)
            last_tool = messages[-1]
            assert last_tool['role'] == 'tool'
            assert 'name' in last_tool
            assert last_tool['name'] == 'get_weather'

        assert len(messages) == num_turns * 3


@_apply_marks_stress
class TestToolCallConcurrentStress(_ToolCallTestBase):
    """High concurrency stress (default 50 workers; override via
    TOOL_CALL_CONCURRENCY)."""

    def test_concurrent_multi_turn_workers(self, backend, model_case):
        """N workers × 3 turns over the ``input_ids`` streaming path."""
        num_workers = DEFAULT_TOOL_CALL_CONCURRENCY
        self._run_concurrent_workers(
            num_workers=num_workers,
            num_turns=3,
            use_input_ids=True,
            tools=[CONCURRENT_WEATHER_TOOL],
            reference_payload=True,
        )

    def test_concurrent_http_errors_formatted(self, backend, model_case):
        """Concurrent invalid-model requests each raise formatted
        ``HttpToolCallError``."""
        num_workers = DEFAULT_TOOL_CALL_HTTP_ERROR_WORKERS
        successes, total = self._run_concurrent_http_error_workers(num_workers=num_workers)
        assert successes == total == num_workers
