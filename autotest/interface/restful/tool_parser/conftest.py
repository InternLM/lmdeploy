import os

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

from lmdeploy.serve.processors.multimodal import MULTIMODAL_TYPES

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

_CLASS_MARKS_MM = [
    pytest.mark.order(8),
    pytest.mark.tool_call,
    pytest.mark.mm_tool_call,
    pytest.mark.flaky(reruns=2),
    pytest.mark.parametrize('backend', BACKEND_LIST),
    pytest.mark.parametrize('model_case', TOOL_REASONING_MODEL_LIST),
]

def _apply_marks(cls):
    """Apply the shared set of marks to *cls* and return it."""
    for m in _CLASS_MARKS:
        cls = m(cls)
    return cls


def _apply_marks_mm(cls):
    """Apply multimodal tool-call marks to *cls*."""
    for m in _CLASS_MARKS_MM:
        cls = m(cls)
    return cls


def llama31_single_tool_only(model_case: str) -> bool:
    """True when the model uses Meta-Llama-3.1 chat template (one tool call per
    turn)."""
    return 'llama-3.1' in model_case.lower().replace('_', '-')


LLAMA31_SKIP_PARALLEL_REASON = (
    'Meta-Llama 3.1 chat template allows only one tool call per turn '
    '(apply_chat_template: single tool-calls at once)')

MM_TOOL_CALL_SKIP_REASON = (
    'Multimodal tool-call tests require native VL models (Qwen3.5 / Intern-S2)')

# Test media filenames under config['resource_path'].
MM_TEST_IMAGE_TIGER = 'tiger.jpeg'
MM_TEST_IMAGE_BEIJING = 'Beijing_Small.jpeg'
MM_TEST_IMAGE_POSE = 'human-pose.jpg'
MM_TEST_VIDEO = 'red-panda.mp4'
MM_TEST_AUDIO = 'zh.wav'

# MULTIMODAL_TYPES from lmdeploy; local fixtures cover image + video + audio.
MM_IMAGE_MEDIA_TYPES = (
    'image_url',
    'image',
    'image_data',
)
MM_VIDEO_MEDIA_TYPES = (
    'video_url',
    'video',
)
MM_AUDIO_MEDIA_TYPES = (
    'audio_url',
    'audio',
)
MM_MEDIA_TYPES_WITH_FIXTURES = (
    MM_IMAGE_MEDIA_TYPES + MM_VIDEO_MEDIA_TYPES + MM_AUDIO_MEDIA_TYPES
)
MM_MEDIA_TYPES_WITHOUT_FIXTURES = tuple(
    t for t in MULTIMODAL_TYPES if t not in MM_MEDIA_TYPES_WITH_FIXTURES)

MM_VIDEO_EXTRA_BODY = {'media_io_kwargs': {'video': {'num_frames': 3}}}

# Substrings matched against model_case for multimodal tool-call capability.
_MM_TOOL_CALL_MODEL_MARKERS = (
    'Qwen3.5',
    'Intern-S2',
    'Qwen3-VL',
)
# Audio tool-call media types need native audio models (not VL-only).
_MM_AUDIO_TOOL_CALL_MODEL_MARKERS = (
    'Qwen2.5-Omni',
    'Qwen3-Omni',
)
MM_AUDIO_TOOL_CALL_SKIP_REASON = (
    'Audio tool-call media types require native audio models '
    '(e.g. Qwen-Omni); VL-only models are skipped')


def is_mm_tool_call_capable(model_case: str) -> bool:
    """True for model families that support image input and tool calling."""
    name = model_case.lower()
    return any(marker.lower() in name for marker in _MM_TOOL_CALL_MODEL_MARKERS)


def is_mm_audio_tool_call_capable(model_case: str) -> bool:
    """True for model families that support audio input and tool calling."""
    name = model_case.lower()
    return any(marker.lower() in name for marker in _MM_AUDIO_TOOL_CALL_MODEL_MARKERS)


def resolve_mm_resource_path(config, filename: str) -> str | None:
    """Return a local filesystem path for a test media file, or None if
    missing."""
    path = os.path.join(config['resource_path'], filename)
    if os.path.isfile(path):
        return path
    return None


def resolve_mm_image_url(config, filename: str) -> str | None:
    """Return a local filesystem URL for a test image, or None if missing."""
    return resolve_mm_resource_path(config, filename)


def mm_file_to_data_url(path: str, *, mime: str | None = None) -> str:
    """Encode a local media file as a ``data:<mime>;base64,...`` URL."""
    import base64
    import mimetypes

    if mime is None:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or 'application/octet-stream'
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'data:{mime};base64,{b64}'


def build_multimodal_user_message(text: str, image_url: str, *, image_first: bool = False) -> dict:
    """OpenAI-compatible user message with text + image_url parts."""
    text_part = {'type': 'text', 'text': text}
    image_part = {'type': 'image_url', 'image_url': {'url': image_url}}
    parts = [image_part, text_part] if image_first else [text_part, image_part]
    return {'role': 'user', 'content': parts}


def build_multimodal_user_message_multi(text: str, image_urls: list[str]) -> dict:
    """User message with one text part followed by multiple images."""
    parts: list[dict] = [{'type': 'text', 'text': text}]
    for url in image_urls:
        parts.append({'type': 'image_url', 'image_url': {'url': url}})
    return {'role': 'user', 'content': parts}


def build_multimodal_media_part(
        media_type: str,
        source,
        **fields) -> dict:
    """Build one OpenAI-style multimodal content part for *media_type*.

    ``image_data`` must be JSON-serializable for OpenAI REST
    (``chat.completions.create``); pass a local path / data-URL string, not
    a ``PIL.Image`` (Python pipeline API only).
    """
    if media_type == 'image_data':
        if hasattr(source, 'size'):
            import base64
            from io import BytesIO
            buf = BytesIO()
            source.save(buf, format='JPEG')
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            data = f'data:image/jpeg;base64,{b64}'
        else:
            data = source
        return {'type': 'image_data', 'image_data': {'data': data, **fields}}
    if media_type in ('image_url', 'image', 'video_url', 'video',
                       'audio_url', 'audio', 'time_series_url', 'time_series'):
        return {media_type: {'url': source, **fields}, 'type': media_type}
    raise ValueError(f'Unsupported multimodal media type: {media_type!r}')


def build_multimodal_user_message_media(
        text: str,
        media_type: str,
        source,
        *,
        media_first: bool = False,
        **media_fields) -> dict:
    """User message with text + one multimodal part (any MULTIMODAL_TYPES)."""
    text_part = {'type': 'text', 'text': text}
    media_part = build_multimodal_media_part(
        media_type, source, **media_fields)
    parts = [media_part, text_part] if media_first else [text_part, media_part]
    return {'role': 'user', 'content': parts}


def mm_media_fixture_filename(media_type: str) -> str:
    """Map a media type to a filename under resource_path."""
    if media_type in ('image_url', 'image', 'image_data'):
        return MM_TEST_IMAGE_TIGER
    if media_type in ('video_url', 'video'):
        return MM_TEST_VIDEO
    if media_type in ('audio_url', 'audio'):
        return MM_TEST_AUDIO
    raise ValueError(f'No local fixture for media type {media_type!r}')


def mm_create_extra_body_for_media_type(media_type: str) -> dict | None:
    """Optional extra_body for chat.completions.create per media type."""
    if media_type in ('video_url', 'video'):
        return dict(MM_VIDEO_EXTRA_BODY)
    return None


def _llama31_parallel_skip_target(item) -> bool:
    """True for TestToolCallParallel and test_multiple_results (parametrize-
    safe)."""
    cls_name = item.cls.__name__ if item.cls is not None else ''
    if cls_name in ('TestToolCallParallel', 'TestToolCallMultimodalParallel'):
        return True
    test_name = getattr(item, 'originalname', None) or item.name.split('[')[0]
    return test_name == 'test_multiple_results'


def _mm_tool_call_skip_target(item) -> bool:
    cls_name = item.cls.__name__ if item.cls is not None else ''
    return cls_name.startswith('TestToolCallMultimodal')


def _mm_audio_media_type_skip_target(item) -> bool:
    """True for TestToolCallMultimodalMediaTypes audio parametrize cases."""
    cls_name = item.cls.__name__ if item.cls is not None else ''
    if cls_name != 'TestToolCallMultimodalMediaTypes':
        return False
    callspec = getattr(item, 'callspec', None)
    if callspec is None:
        return False
    media_type = callspec.params.get('media_type')
    return media_type in MM_AUDIO_MEDIA_TYPES


def pytest_collection_modifyitems(config, items):
    """Skip parallel-tool tests on Llama 3.1; skip MM tool tests on text-only
    models; skip audio media types on VL-only models."""
    for item in items:
        callspec = getattr(item, 'callspec', None)
        if callspec is None:
            continue
        model_case = callspec.params['model_case']
        if _llama31_parallel_skip_target(item):
            if model_case and llama31_single_tool_only(model_case):
                item.add_marker(pytest.mark.skip(reason=LLAMA31_SKIP_PARALLEL_REASON))
        if _mm_tool_call_skip_target(item):
            if model_case and not is_mm_tool_call_capable(model_case):
                item.add_marker(pytest.mark.skip(reason=MM_TOOL_CALL_SKIP_REASON))
        if _mm_audio_media_type_skip_target(item):
            if model_case and not is_mm_audio_tool_call_capable(model_case):
                item.add_marker(pytest.mark.skip(reason=MM_AUDIO_TOOL_CALL_SKIP_REASON))


# ---------------------------------------------------------------------------
# Per-test API request/response logging fixtures.
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
        self._config = config
        self._model_case = model_case
        self._client, self._api_model_name = make_logged_client(self._log_file)
        self._model_name = self._api_model_name
        self._tokenizer_path = resolve_tokenizer_model_path(config, model_case)

    def _get_client(self):
        """Return *(client, api_model_name)* with transparent logging."""
        return self._client, self._api_model_name

    def _require_mm_image(self, filename: str) -> str:
        """Resolve a test image path or skip when ``resource_path`` is
        unset."""
        image_url = resolve_mm_image_url(self._config, filename)
        if image_url is None:
            pytest.skip(
                f'Missing multimodal test image {filename!r} under '
                f'resource_path={self._config["resource_path"]!r}')
        return image_url

    def _require_mm_resource(self, filename: str) -> str:
        """Resolve a test media path or skip when missing."""
        path = resolve_mm_resource_path(self._config, filename)
        if path is None:
            pytest.skip(
                f'Missing multimodal test media {filename!r} under '
                f'resource_path={self._config["resource_path"]!r}')
        return path

    def _require_mm_media_source(self, media_type: str):
        """Resolve local fixture path/object for a MULTIMODAL_TYPES entry."""
        filename = mm_media_fixture_filename(media_type)
        return self._require_mm_resource(filename)

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


MM_SCENE_DALLAS = 'Dallas Zoo photo.'
MM_SCENE_BEIJING = 'Beijing in photo.'
MM_SCENE_MIAMI = 'Outdoor workout in Miami (see photo).'
MM_SCENE_VIDEO = 'Red panda video from Sichuan.'
MM_SCENE_AUDIO = 'Chinese audio about running.'

MM_USER_BEIJING_WEATHER = 'Weather in Beijing, China (state: Beijing)?'
MM_USER_MIAMI_WEATHER = "What's the weather like in Miami, FL?"
MM_USER_SICHUAN_WEATHER = "What's the weather like in Chengdu, Sichuan?"
MM_USER_TIGER_SEARCH = 'Tiger in photo. Search recent tiger conservation news.'
MM_USER_AUDIO_SEARCH = (
    'Listen to the audio and search the web for how running '
    'benefits physical health as described.')


def _mm_user_text(user_prompt: str, *, scene: str | None = None) -> str:
    if scene:
        return f'{scene} {user_prompt}'
    return user_prompt


def _mm_scene_for_media_type(media_type: str) -> str:
    if media_type in ('video_url', 'video'):
        return MM_SCENE_VIDEO
    return MM_SCENE_DALLAS


def _mm_weather_user_prompt_for_media_type(media_type: str) -> str:
    if media_type in ('video_url', 'video'):
        return MM_USER_SICHUAN_WEATHER
    return MESSAGES_ASKING_FOR_WEATHER[1]['content']


def mm_weather_messages_for_media_type(
        media_type: str,
        source,
        *,
        media_first: bool = False) -> list[dict]:
    """Weather tool-call messages using any supported MULTIMODAL_TYPES part."""
    text = _mm_user_text(
        _mm_weather_user_prompt_for_media_type(media_type),
        scene=_mm_scene_for_media_type(media_type),
    )
    return [
        MESSAGES_ASKING_FOR_WEATHER[0],
        build_multimodal_user_message_media(
            text, media_type, source, media_first=media_first),
    ]


def mm_audio_search_messages_for_media_type(
        media_type: str,
        source,
        *,
        media_first: bool = False) -> list[dict]:
    """Search tool-call messages for audio MULTIMODAL_TYPES (zh.wav clip)."""
    text = _mm_user_text(MM_USER_AUDIO_SEARCH, scene=MM_SCENE_AUDIO)
    return [
        MESSAGES_ASKING_FOR_SEARCH[0],
        build_multimodal_user_message_media(
            text, media_type, source, media_first=media_first),
    ]


def build_mm_weather_messages(
        image_url: str,
        *,
        user_prompt: str,
        scene: str | None = None,
        image_first: bool = False) -> list[dict]:
    """System + multimodal user turn asking for weather via tool."""
    text = _mm_user_text(user_prompt, scene=scene)
    return [
        MESSAGES_ASKING_FOR_WEATHER[0],
        build_multimodal_user_message(
            text, image_url, image_first=image_first),
    ]


def build_mm_dallas_weather_user_message(
        image_url: str, *, image_first: bool = False) -> dict:
    text = _mm_user_text(
        MESSAGES_ASKING_FOR_WEATHER[1]['content'], scene=MM_SCENE_DALLAS)
    return build_multimodal_user_message(
        text, image_url, image_first=image_first)


def mm_dallas_weather_messages(
        image_url: str, *, image_first: bool = False) -> list[dict]:
    return build_mm_weather_messages(
        image_url,
        user_prompt=MESSAGES_ASKING_FOR_WEATHER[1]['content'],
        scene=MM_SCENE_DALLAS,
        image_first=image_first,
    )


def mm_beijing_weather_messages(image_url: str) -> list[dict]:
    return build_mm_weather_messages(
        image_url,
        user_prompt=MM_USER_BEIJING_WEATHER,
        scene=MM_SCENE_BEIJING,
    )


def build_mm_miami_weather_user_message(image_url: str) -> dict:
    text = _mm_user_text(MM_USER_MIAMI_WEATHER, scene=MM_SCENE_MIAMI)
    return build_multimodal_user_message(text, image_url)


def mm_miami_weather_messages(image_url: str) -> list[dict]:
    return build_mm_weather_messages(
        image_url,
        user_prompt=MM_USER_MIAMI_WEATHER,
        scene=MM_SCENE_MIAMI,
    )


def build_mm_tiger_search_messages(image_url: str) -> list[dict]:
    return [
        MESSAGES_ASKING_FOR_SEARCH[0],
        build_multimodal_user_message(MM_USER_TIGER_SEARCH, image_url),
    ]


def build_mm_parallel_weather_user_message(image_url: str) -> dict:
    text = _mm_user_text(
        MESSAGES_PARALLEL_WEATHER[1]['content'], scene=MM_SCENE_DALLAS)
    return build_multimodal_user_message_multi(text, [image_url])


def build_mm_parallel_weather_messages(image_url: str) -> list[dict]:
    return [
        MESSAGES_PARALLEL_WEATHER[0],
        build_mm_parallel_weather_user_message(image_url),
    ]


def build_mm_dual_image_dallas_messages(
        tiger_url: str, pose_url: str) -> list[dict]:
    text = _mm_user_text(
        MESSAGES_ASKING_FOR_WEATHER[1]['content'], scene=MM_SCENE_DALLAS)
    return [
        MESSAGES_ASKING_FOR_WEATHER[0],
        build_multimodal_user_message_multi(
            text, [tiger_url, pose_url]),
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
