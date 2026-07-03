import pytest
from utils.constant import BACKEND_LIST, DEFAULT_MAX_COMPLETION_TOKENS, TOOL_REASONING_MODEL_LIST
from utils.tool_reasoning_definitions import (
    CONCURRENT_WEATHER_TOOL,
    DEFAULT_TOOL_CALL_CONCURRENCY,
    HttpToolCallError,
    RoutedExpertsNotSupported,
    append_concurrent_turn_to_messages,
    build_input_ids_and_prompt_tokens,
    collect_stream_tool_call,
    collect_stream_tool_call_http,
    make_logged_client,
    resolve_tokenizer_model_path,
    resolve_tool_parser_name,
    run_concurrent_http_error_workers,
    run_concurrent_tool_call_workers,
    setup_log_file,
)

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

_CLASS_MARKS = [
    pytest.mark.order(8),
    pytest.mark.tool_call,
    pytest.mark.flaky(reruns=2),
    pytest.mark.parametrize('backend', BACKEND_LIST),
    pytest.mark.parametrize('model_case', TOOL_REASONING_MODEL_LIST),
]


def _apply_marks(cls):
    """Apply the shared set of marks to *cls* and return it."""
    for m in _CLASS_MARKS:
        cls = m(cls)
    return cls


def llama31_single_tool_only(model_case: str) -> bool:
    """True when the model uses Meta-Llama-3.1 chat template (one tool call per
    turn)."""
    return 'llama-3.1' in model_case.lower().replace('_', '-')


LLAMA31_SKIP_PARALLEL_REASON = (
    'Meta-Llama 3.1 chat template allows only one tool call per turn '
    '(apply_chat_template: single tool-calls at once)')


def _llama31_parallel_skip_target(item) -> bool:
    """True for TestToolCallParallel and test_multiple_results (parametrize-
    safe)."""
    cls_name = item.cls.__name__ if item.cls is not None else ''
    if cls_name == 'TestToolCallParallel':
        return True
    test_name = getattr(item, 'originalname', None) or item.name.split('[')[0]
    return test_name == 'test_multiple_results'


def pytest_collection_modifyitems(config, items):
    """Skip parallel-tool tests on Llama 3.1 at collection time (reliable vs
    node.name)."""
    for item in items:
        if not _llama31_parallel_skip_target(item):
            continue
        callspec = getattr(item, 'callspec', None)
        if callspec is None:
            continue
        model_case = callspec.params.get('model_case')
        if model_case and llama31_single_tool_only(model_case):
            item.add_marker(pytest.mark.skip(reason=LLAMA31_SKIP_PARALLEL_REASON))


# ---------------------------------------------------------------------------
# Logging helpers – uses shared StreamTee / setup_log_file / make_logged_client
# from utils.tool_reasoning_definitions.
# ---------------------------------------------------------------------------


class _ToolCallTestBase:
    """Mixin providing per-test API request/response logging to *log_path*."""

    _DEFAULT_STREAM_KWARGS = {
        'temperature': 0,
        'max_completion_tokens': DEFAULT_MAX_COMPLETION_TOKENS,
        'logprobs': False,
    }

    @pytest.fixture(autouse=True)
    def _setup_logging(self, request, config, backend, model_case):
        """Create the log directory and compute the log-file path."""
        self._log_file = setup_log_file(config, request.node.name, 'tool_calls')
        self._model_case = model_case
        self._client, self._api_model_name = make_logged_client(self._log_file)
        self._model_name = self._api_model_name
        self._tokenizer_path = resolve_tokenizer_model_path(config, model_case)

    def _get_client(self):
        """Return *(client, api_model_name)* with transparent logging."""
        return self._client, self._api_model_name

    def _parser_validation_kwargs(self, tools=None):
        """Kwargs for ``validate_*`` helpers using
        ``ResponseParser.validate_complete``."""
        kwargs = {
            'tokenizer_path': self._tokenizer_path,
            'tool_parser_name': resolve_tool_parser_name(self._model_case),
        }
        if tools is not None:
            kwargs['tools'] = tools
        return kwargs

    def _stream_tool_call(self, messages, tools=None, **create_kwargs):
        """Run a streaming tool-call request and return aggregated result."""
        client, model_name = self._get_client()
        kwargs = {**self._DEFAULT_STREAM_KWARGS, **create_kwargs}
        if tools is not None:
            kwargs['tools'] = tools
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            **kwargs,
        )
        return collect_stream_tool_call(stream)

    def _stream_tool_call_with_tokens(
        self,
        messages,
        tools=None,
        use_input_ids=False,
        reference_payload=False,
        **payload_extra,
    ):
        """Stream via HTTP with return_token_ids + return_routed_experts."""
        if use_input_ids:
            try:
                build_input_ids_and_prompt_tokens(messages, self._tokenizer_path, tools)
            except Exception as exc:
                pytest.skip(f'input_ids path requires local tokenizer: {exc}')
        if not reference_payload:
            payload_extra = {
                **self._DEFAULT_STREAM_KWARGS,
                **payload_extra,
            }
        try:
            return collect_stream_tool_call_http(
                self._api_model_name,
                messages,
                tools=tools,
                log_file=self._log_file,
                use_input_ids=use_input_ids,
                tokenizer_path=self._tokenizer_path,
                reference_payload=reference_payload,
                **payload_extra,
            )
        except RoutedExpertsNotSupported as exc:
            pytest.skip(str(exc))
        except HttpToolCallError as exc:
            pytest.fail(exc.message)

    def _append_assistant_and_tool_messages(self, messages, stream_result):
        """Append assistant + tool turns (includes tool ``name``, like
        concurrent script)."""
        append_concurrent_turn_to_messages(messages, stream_result)

    def _run_concurrent_workers(
        self,
        num_workers=None,
        num_turns=3,
        use_input_ids=True,
        tools=None,
        reference_payload=True,
    ):
        """Run parallel multi-turn asyncio workers."""
        return run_concurrent_tool_call_workers(
            self._api_model_name,
            tokenizer_path=self._tokenizer_path,
            num_workers=num_workers or DEFAULT_TOOL_CALL_CONCURRENCY,
            num_turns=num_turns,
            tools=tools or [CONCURRENT_WEATHER_TOOL],
            use_input_ids=use_input_ids,
            log_file=self._log_file,
            reference_payload=reference_payload,
        )

    def _run_concurrent_http_error_workers(self, num_workers=None, invalid_model_name=None):
        """Concurrent invalid-model HTTP error probes."""
        return run_concurrent_http_error_workers(
            self._api_model_name,
            num_workers=num_workers,
            invalid_model_name=invalid_model_name,
        )


# ---------------------------------------------------------------------------
# Message constants
# ---------------------------------------------------------------------------

MESSAGES_ASKING_FOR_WEATHER = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant that can use tools. '
        'When asked about weather, use the get_current_weather tool.',
    },
    {
        'role': 'user',
        'content': "What's the weather like in Dallas, TX?",
    },
]

MESSAGES_ASKING_FOR_SEARCH = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant with access to tools. '
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
        'role':
        'system',
        'content':
        'You are a helpful assistant. When asked about weather '
        'in multiple cities, call the weather tool for each city '
        'separately.',
    },
    {
        'role': 'user',
        'content': "What's the weather in Dallas, TX and also in "
        'San Francisco, CA?',
    },
]

MULTI_TURN_WEATHER_CITIES = ['Tokyo', 'London', 'Paris', 'New York']

MESSAGES_CONCURRENT_WEATHER = [
    {
        'role': 'system',
        'content': 'You are a helpful assistant that can use tools. '
        'When asked about weather, use the get_weather tool.',
    },
]

MESSAGES_PARALLEL_MIXED = [
    {
        'role':
        'system',
        'content':
        'You are a helpful assistant with access to multiple tools. '
        'You can call multiple tools in parallel when needed.',
    },
    {
        'role': 'user',
        'content': "What's the weather in Dallas, TX? "
        'Also calculate 1234 * 5678.',
    },
]
