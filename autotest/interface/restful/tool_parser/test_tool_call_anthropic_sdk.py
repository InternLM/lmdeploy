from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache

import pytest
import requests
from utils.config_utils import get_config
from utils.constant import BASE_URL
from utils.tool_reasoning_definitions import (
    SEARCH_TOOL,
    WEATHER_TOOL,
    WEATHER_TOOL_SINGLE_LOCATION_ANTHROPIC,
    get_async_anthropic_client_and_model,
    openai_chat_messages_to_anthropic_kwargs,
    openai_function_tool_to_anthropic,
)

from lmdeploy.serve.openai.api_client import APIClient

from .conftest import MESSAGES_ASKING_FOR_WEATHER, _apply_marks, _ToolCallTestBase

ANTHROPIC_VERSION = '2023-06-01'

_EVAL_IMAGE_TIGER = 'tiger.jpeg'
_TINY_PNG_BASE64 = (
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
)

_SOLID_COLOR_VLM_PROMPT = (
    'The image is a single solid color (one pixel). '
    'Reply with at most three words: name that color only (e.g. red).'
)

_REDISH_COLOR_KEYWORDS = (
    'red',
    'crimson',
    'scarlet',
    'maroon',
    'ruby',
    'vermilion',
    '红',
    '赤',
    '朱',
    '绯',
)


@lru_cache(maxsize=1)
def _eval_resource_path() -> str:
    cfg = get_config()
    path = cfg.get('resource_path')
    assert isinstance(path, str) and path, 'resource_path must be set in autotest config (e.g. config_h.yml)'
    base = path.rstrip('/')
    assert os.path.isdir(base), f'resource_path is not a directory: {base!r}'
    return base


def _eval_resource_file(filename: str) -> str:
    p = os.path.join(_eval_resource_path(), filename)
    assert os.path.isfile(p), f'missing offline eval resource {filename!r}: {p}'
    return p


def _model_likely_supports_anthropic_vlm(model_name: str) -> bool:
    m = model_name.upper()
    return any(
        needle in m
        for needle in (
            'VL',
            'INTERNVL',
            'INTERN-VL',
            'QWEN-VL',
            'QWEN2-VL',
            'QWEN2.5-VL',
            'QWEN3.5',
            'MINICPM-V',
            'LLAVA',
            'COGVLM',
            'XCOMPOSER',
            'INTERNXCOMPOSER',
            'INTERNS',
        ))


def _http_headers() -> dict[str, str]:
    return {
        'Content-Type': 'application/json',
        'anthropic-version': ANTHROPIC_VERSION,
    }


def _parse_anthropic_sse(raw: str) -> list[tuple[str | None, dict]]:
    pairs: list[tuple[str | None, dict]] = []
    current_event: str | None = None
    for line in raw.splitlines():
        line = line.rstrip('\r')
        if line.startswith('event:'):
            current_event = line[len('event:'):].strip()
        elif line.startswith('data:'):
            data_str = line[len('data:'):].strip()
            if not data_str:
                continue
            pairs.append((current_event, json.loads(data_str)))
            current_event = None
    return pairs


def _aggregate_stream_text(events: list[tuple[str | None, dict]]) -> str:
    text = ''
    for _, obj in events:
        if obj.get('type') != 'content_block_delta':
            continue
        delta = obj.get('delta') or {}
        if delta.get('type') == 'text_delta':
            text += delta.get('text') or ''
    return text


def _sse_tool_use_names(raw: str) -> list[str]:
    names: list[str] = []
    for _, obj in _parse_anthropic_sse(raw):
        if obj.get('type') != 'content_block_start':
            continue
        cb = obj.get('content_block') or {}
        if cb.get('type') == 'tool_use' and cb.get('name'):
            names.append(cb['name'])
    return names


def _assert_redish_color_in_text(assembled: str, *, ctx: str) -> None:
    assert len(assembled.strip()) > 0, (
        f'{ctx}: no text_delta content in stream (prefix {assembled[:300]!r}). '
        'Reasoning models may stream long thinking_delta first; if max_tokens is too low, '
        'the run can end before any visible text block is emitted.'
    )
    al = assembled.lower()
    assert any(k in al for k in _REDISH_COLOR_KEYWORDS), (
        f'{ctx}: expected red-ish color in reply: {assembled[:500]!r}',
    )


def _assistant_text_from_messages_json(data: dict) -> str:
    return ''.join(b.get('text', '') for b in data.get('content', []) if b.get('type') == 'text')


def _http_tool_use_blocks(data: dict) -> list[dict]:
    return [b for b in data.get('content', []) if isinstance(b, dict) and b.get('type') == 'tool_use']


def _sdk_tool_use_blocks(msg) -> list:
    return [b for b in msg.content if getattr(b, 'type', None) == 'tool_use']


def _assert_weather_tool_city_state(inp: dict, *, ctx: str = '') -> None:
    """``get_current_weather`` OpenAI-style args (``city`` / ``state``) after
    parser mapping."""

    assert isinstance(inp, dict), (ctx, type(inp))
    city, state = inp.get('city'), inp.get('state')
    assert isinstance(city, str) and len(city) > 0, (ctx, inp)
    assert isinstance(state, str) and len(state) > 0, (ctx, inp)


def _log_append(path: str, text: str) -> None:
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')
    except OSError:
        pass


def _trace_anthropic_http(
    log_file: str,
    case: str,
    *,
    url: str,
    http_status: int,
    request_json: dict | None,
    response_text: str,
    max_chars: int = 24000,
) -> None:
    """Append one JSON line to ``tool_calls/*.log``.

    Same tree as ``test_tool_call_advanced`` (``setup_log_file``).
    """

    rtxt = (
        response_text
        if len(response_text) <= max_chars
        else response_text[:max_chars] + '\n…[truncated]'
    )
    _log_append(
        log_file,
        json.dumps(
            {
                'anthropic_http_trace': case,
                'url': url,
                'status': http_status,
                'request': request_json,
                'response': rtxt,
            },
            ensure_ascii=False,
            default=str,
        ),
    )


# --- HTTP: ``tool_parser`` / ``--tool-call-parser`` jobs only ----------------------------


@_apply_marks
class TestAnthropicHttpToolMessages(_ToolCallTestBase):
    """``POST /v1/messages`` over HTTP when api_server is launched *with*
    ``--tool-call-parser``.

    Adapter block semantics (``lmdeploy.serve.anthropic.adapter``):

    * ``tool_use`` (assistant): prior tool call replay (``id``, ``name``, ``input``).
    * ``tool_result`` (user): plain-text result for ``tool_use_id``.
    * ``thinking`` / ``redacted_thinking``: replayed reasoning segments (e.g. Claude Code style).
    """

    def test_http_stream_tool_choice_force_named_tool(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        url = f'{BASE_URL}/v1/messages'
        req_json = {
            'model': model_name,
            'max_tokens': 512,
            'temperature': 0,
            'stream': True,
            'tool_choice': {
                'type': 'tool',
                'name': WEATHER_TOOL['function']['name'],
            },
            'messages': [{'role': 'user', 'content': 'What is the weather in Dallas, TX?'}],
            'tools': [
                openai_function_tool_to_anthropic(WEATHER_TOOL),
                openai_function_tool_to_anthropic(SEARCH_TOOL),
            ],
        }
        resp = requests.post(
            url,
            headers=_http_headers(),
            json=req_json,
            stream=True,
            timeout=180,
        )
        assert resp.status_code == 200, resp.text
        raw = ''.join(chunk.decode('utf-8') for chunk in resp.iter_content(chunk_size=None) if chunk)
        _trace_anthropic_http(
            self._log_file,
            'test_http_stream_tool_choice_force_named_tool',
            url=url,
            http_status=resp.status_code,
            request_json=req_json,
            response_text=raw,
        )
        names = _sse_tool_use_names(raw)
        assert names, f'no tool_use content_block_start in SSE (first 800 chars): {raw[:800]!r}'
        assert WEATHER_TOOL['function']['name'] in names, names

    def test_http_stream_single_location_weather_tool(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        url = f'{BASE_URL}/v1/messages'
        req_json = {
            'model': model_name,
            'max_tokens': 512,
            'temperature': 0,
            'stream': True,
            'messages': [{'role': 'user', 'content': "What's the weather like in New York today?"}],
            'tools': [WEATHER_TOOL_SINGLE_LOCATION_ANTHROPIC],
        }
        resp = requests.post(
            url,
            headers=_http_headers(),
            json=req_json,
            stream=True,
            timeout=180,
        )
        assert resp.status_code == 200, resp.text
        raw = ''.join(chunk.decode('utf-8') for chunk in resp.iter_content(chunk_size=None) if chunk)
        _trace_anthropic_http(
            self._log_file,
            'test_http_stream_single_location_weather_tool',
            url=url,
            http_status=resp.status_code,
            request_json=req_json,
            response_text=raw,
        )
        names = _sse_tool_use_names(raw)
        assert names, f'no tool_use content_block_start in SSE (first 800 chars): {raw[:800]!r}'
        assert 'get_current_weather' in names, names

    def test_http_history_tool_use_and_tool_result_blocks(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        url = f'{BASE_URL}/v1/messages'
        req_json = {
            'model': model_name,
            'max_tokens': 8192,
            'temperature': 0.01,
            'messages': [
                {'role': 'user', 'content': 'What is the weather in San Francisco?'},
                {
                    'role': 'assistant',
                    'content': [
                        {
                            'type': 'tool_use',
                            'id': 'toolu_hist_01',
                            'name': 'get_current_weather',
                            'input': {'location': 'San Francisco'},
                        },
                    ],
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'tool_result',
                            'tool_use_id': 'toolu_hist_01',
                            'content': '72F and sunny.',
                        },
                    ],
                },
                {'role': 'user', 'content': 'In one short phrase, was it warm? Answer yes or no.'},
            ],
        }
        resp = requests.post(
            url,
            headers=_http_headers(),
            json=req_json,
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        _trace_anthropic_http(
            self._log_file,
            'test_http_history_tool_use_and_tool_result_blocks',
            url=url,
            http_status=resp.status_code,
            request_json=req_json,
            response_text=json.dumps(data, ensure_ascii=False, default=str),
        )
        text = _assistant_text_from_messages_json(data)
        assert len(text) > 0, (
            'expected non-empty assistant text blocks; '
            f'stop_reason={data.get("stop_reason")!r} content={data.get("content")!r}'
        )

    def test_http_history_thinking_block_replay(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        url = f'{BASE_URL}/v1/messages'
        req_json = {
            'model': model_name,
            'max_tokens': 8192,
            'temperature': 0.01,
            'messages': [
                {'role': 'user', 'content': 'Hi.'},
                {
                    'role': 'assistant',
                    'content': [
                        {'type': 'thinking', 'thinking': '(internal scratchpad)'},
                        {'type': 'text', 'text': 'Hello — how can I help?'},
                    ],
                },
                {'role': 'user', 'content': 'Reply with exactly: ACK'},
            ],
        }
        resp = requests.post(
            url,
            headers=_http_headers(),
            json=req_json,
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        _trace_anthropic_http(
            self._log_file,
            'test_http_history_thinking_block_replay',
            url=url,
            http_status=resp.status_code,
            request_json=req_json,
            response_text=json.dumps(data, ensure_ascii=False, default=str),
        )
        text = _assistant_text_from_messages_json(data)
        assert len(text) > 0, (
            'expected non-empty assistant text blocks; '
            f'stop_reason={data.get("stop_reason")!r} content={data.get("content")!r}'
        )

    def test_http_non_stream_tools_with_user_image_url(self, backend, model_case):
        """``tools`` + user ``content`` blocks with ``image`` (VLM matrix only;
        same tool contract as text-only)."""

        model_name = APIClient(BASE_URL).available_models[0]
        if not _model_likely_supports_anthropic_vlm(model_name):
            pytest.skip(f'model {model_name!r} is not treated as vision-capable for this test')

        image_path = _eval_resource_file(_EVAL_IMAGE_TIGER)
        url = f'{BASE_URL}/v1/messages'
        req_json = {
            'model': model_name,
            'max_tokens': 512,
            'temperature': 0,
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': (
                            'What is the weather in Dallas, TX? '
                            'Use the tools; the attached image is unrelated decoration.'
                        ),
                    },
                    {'type': 'image', 'source': {'type': 'url', 'url': image_path}},
                ],
            }],
            'tools': [
                openai_function_tool_to_anthropic(WEATHER_TOOL),
                openai_function_tool_to_anthropic(SEARCH_TOOL),
            ],
        }
        resp = requests.post(url, headers=_http_headers(), json=req_json, timeout=180)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        _trace_anthropic_http(
            self._log_file,
            'test_http_non_stream_tools_with_user_image_url',
            url=url,
            http_status=resp.status_code,
            request_json=req_json,
            response_text=json.dumps(data, ensure_ascii=False, default=str),
        )
        assert data.get('stop_reason') == 'tool_use', data
        blocks = _http_tool_use_blocks(data)
        assert len(blocks) >= 1, data.get('content')
        assert blocks[0].get('name') == WEATHER_TOOL['function']['name']
        inp = blocks[0].get('input')
        assert isinstance(inp, dict), inp
        _assert_weather_tool_city_state(inp, ctx='test_http_non_stream_tools_with_user_image_url')

    def test_http_stream_tools_with_user_image_url(self, backend, model_case):
        """Streaming ``tools`` + user image URL (VLM): SSE must still surface
        ``tool_use``."""

        model_name = APIClient(BASE_URL).available_models[0]
        if not _model_likely_supports_anthropic_vlm(model_name):
            pytest.skip(f'model {model_name!r} is not treated as vision-capable for this test')

        image_path = _eval_resource_file(_EVAL_IMAGE_TIGER)
        url = f'{BASE_URL}/v1/messages'
        req_json = {
            'model': model_name,
            'max_tokens': 512,
            'temperature': 0,
            'stream': True,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'What is the weather in Dallas, TX? Use tools.'},
                    {'type': 'image', 'source': {'type': 'url', 'url': image_path}},
                ],
            }],
            'tools': [
                openai_function_tool_to_anthropic(WEATHER_TOOL),
                openai_function_tool_to_anthropic(SEARCH_TOOL),
            ],
        }
        resp = requests.post(url, headers=_http_headers(), json=req_json, stream=True, timeout=180)
        assert resp.status_code == 200, resp.text
        raw = ''.join(chunk.decode('utf-8') for chunk in resp.iter_content(chunk_size=None) if chunk)
        _trace_anthropic_http(
            self._log_file,
            'test_http_stream_tools_with_user_image_url',
            url=url,
            http_status=resp.status_code,
            request_json=req_json,
            response_text=raw,
        )
        names = _sse_tool_use_names(raw)
        assert names, f'no tool_use in SSE (first 800 chars): {raw[:800]!r}'
        assert WEATHER_TOOL['function']['name'] in names, names

    def test_http_stream_user_image_base64_solid_color_vlm(self, backend, model_case):
        """Align with RESTful ``test_messages_user_image_base64_stream``: SSE
        text names the solid color."""

        model_name = APIClient(BASE_URL).available_models[0]
        if not _model_likely_supports_anthropic_vlm(model_name):
            pytest.skip(f'model {model_name!r} is not treated as vision-capable for this test')

        url = f'{BASE_URL}/v1/messages'
        req_json = {
            'model': model_name,
            'max_tokens': 16384,
            'temperature': 0.01,
            'stream': True,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': _SOLID_COLOR_VLM_PROMPT},
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': 'image/png',
                            'data': _TINY_PNG_BASE64,
                        },
                    },
                ],
            }],
        }
        resp = requests.post(url, headers=_http_headers(), json=req_json, stream=True, timeout=180)
        assert resp.status_code == 200, resp.text
        raw = ''.join(chunk.decode('utf-8') for chunk in resp.iter_content(chunk_size=None) if chunk)
        _trace_anthropic_http(
            self._log_file,
            'test_http_stream_user_image_base64_solid_color_vlm',
            url=url,
            http_status=resp.status_code,
            request_json=req_json,
            response_text=raw,
        )
        events = _parse_anthropic_sse(raw)
        types = [obj.get('type') for _, obj in events]
        assert 'message_start' in types
        assert 'message_stop' in types
        assembled = _aggregate_stream_text(events)
        _assert_redish_color_in_text(assembled, ctx='test_http_stream_user_image_base64_solid_color_vlm')


def _event_to_dict(event) -> dict:
    if hasattr(event, 'model_dump'):
        return event.model_dump()
    if isinstance(event, dict):
        return event
    return {'repr': repr(event)}


async def _async_weather_tool_single_location_non_stream(log_file: str):
    client, model_name = get_async_anthropic_client_and_model()
    msg = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        messages=[{'role': 'user', 'content': "What's the weather like in New York today?"}],
        tools=[WEATHER_TOOL_SINGLE_LOCATION_ANTHROPIC],
    )
    try:
        dumped = msg.model_dump_json()
    except Exception:
        try:
            dumped = json.dumps(msg.model_dump())
        except Exception:
            dumped = repr(msg)
    _log_append(log_file, dumped)
    return msg


async def _async_tool_choice_force_named_tool(log_file: str):
    client, model_name = get_async_anthropic_client_and_model()
    kwargs = openai_chat_messages_to_anthropic_kwargs(MESSAGES_ASKING_FOR_WEATHER)
    tools = [
        openai_function_tool_to_anthropic(WEATHER_TOOL),
        openai_function_tool_to_anthropic(SEARCH_TOOL),
    ]
    msg = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        tools=tools,
        tool_choice={'type': 'tool', 'name': WEATHER_TOOL['function']['name']},
        **kwargs,
    )
    try:
        dumped = msg.model_dump_json()
    except Exception:
        try:
            dumped = json.dumps(msg.model_dump())
        except Exception:
            dumped = repr(msg)
    _log_append(log_file, dumped)
    return msg


async def _async_tool_choice_any(log_file: str):
    client, model_name = get_async_anthropic_client_and_model()
    kwargs = openai_chat_messages_to_anthropic_kwargs(MESSAGES_ASKING_FOR_WEATHER)
    tools = [
        openai_function_tool_to_anthropic(WEATHER_TOOL),
        openai_function_tool_to_anthropic(SEARCH_TOOL),
    ]
    msg = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        tools=tools,
        tool_choice={'type': 'any'},
        **kwargs,
    )
    try:
        dumped = msg.model_dump_json()
    except Exception:
        try:
            dumped = json.dumps(msg.model_dump())
        except Exception:
            dumped = repr(msg)
    _log_append(log_file, dumped)
    return msg


async def _async_messages_tool_non_stream_with_user_image(log_file: str, image_url: str):
    client, model_name = get_async_anthropic_client_and_model()
    tools = [
        openai_function_tool_to_anthropic(WEATHER_TOOL),
        openai_function_tool_to_anthropic(SEARCH_TOOL),
    ]
    msg = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        tools=tools,
        messages=[{
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': (
                        'What is the weather in Dallas, TX? '
                        'Use tools; the image is unrelated context.'
                    ),
                },
                {'type': 'image', 'source': {'type': 'url', 'url': image_url}},
            ],
        }],
    )
    try:
        dumped = msg.model_dump_json()
    except Exception:
        try:
            dumped = json.dumps(msg.model_dump())
        except Exception:
            dumped = repr(msg)
    _log_append(log_file, dumped)
    return msg


async def _async_messages_tool_non_stream_with_user_image_base64(log_file: str):
    client, model_name = get_async_anthropic_client_and_model()
    tools = [
        openai_function_tool_to_anthropic(WEATHER_TOOL),
        openai_function_tool_to_anthropic(SEARCH_TOOL),
    ]
    msg = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        tools=tools,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'What is the weather in Dallas, TX? Use tools.'},
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': _TINY_PNG_BASE64,
                    },
                },
            ],
        }],
    )
    try:
        dumped = msg.model_dump_json()
    except Exception:
        try:
            dumped = json.dumps(msg.model_dump())
        except Exception:
            dumped = repr(msg)
    _log_append(log_file, dumped)
    return msg


async def _async_messages_tool_non_stream(log_file: str):
    client, model_name = get_async_anthropic_client_and_model()
    kwargs = openai_chat_messages_to_anthropic_kwargs(MESSAGES_ASKING_FOR_WEATHER)
    tools = [
        openai_function_tool_to_anthropic(WEATHER_TOOL),
        openai_function_tool_to_anthropic(SEARCH_TOOL),
    ]
    msg = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        tools=tools,
        **kwargs,
    )
    try:
        dumped = msg.model_dump_json()
    except Exception:
        try:
            dumped = json.dumps(msg.model_dump())
        except Exception:
            dumped = repr(msg)
    _log_append(log_file, dumped)
    return msg


async def _async_messages_tool_stream(log_file: str):
    client, model_name = get_async_anthropic_client_and_model()
    kwargs = openai_chat_messages_to_anthropic_kwargs(MESSAGES_ASKING_FOR_WEATHER)
    tools = [
        openai_function_tool_to_anthropic(WEATHER_TOOL),
        openai_function_tool_to_anthropic(SEARCH_TOOL),
    ]
    stream = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        tools=tools,
        stream=True,
        **kwargs,
    )
    events = []
    async for event in stream:
        events.append(event)
    _log_append(log_file, f'anthropic_stream_event_count={len(events)}')

    final_msg = None
    getter = getattr(stream, 'get_final_message', None)
    if callable(getter):
        try:
            final_msg = await getter()
        except Exception as err:  # noqa: BLE001 — SDK / server variance
            _log_append(log_file, f'get_final_message_failed: {err!r}')

    if final_msg is not None:
        try:
            dumped = final_msg.model_dump_json()
        except Exception:
            try:
                dumped = json.dumps(final_msg.model_dump())
            except Exception:
                dumped = repr(final_msg)
        _log_append(log_file, dumped)
        return 'final', final_msg, events

    blob = json.dumps([_event_to_dict(e) for e in events], default=str)
    _log_append(log_file, blob[:16000])
    return 'raw', blob, events


async def _async_vlm_base64_solid_color_stream(log_file: str) -> tuple[str, str]:
    """Returns ``(kind, text_or_blob)`` where ``kind`` is ``'final'`` or
    ``'raw'``."""

    client, model_name = get_async_anthropic_client_and_model()
    stream = await client.messages.create(
        model=model_name,
        max_tokens=16384,
        temperature=0.01,
        stream=True,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': _SOLID_COLOR_VLM_PROMPT},
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': _TINY_PNG_BASE64,
                    },
                },
            ],
        }],
    )
    events: list = []
    async for event in stream:
        events.append(event)
    _log_append(log_file, f'vlm_color_sdk_stream_events={len(events)}')

    final_msg = None
    getter = getattr(stream, 'get_final_message', None)
    if callable(getter):
        try:
            final_msg = await getter()
        except Exception as err:  # noqa: BLE001
            _log_append(log_file, f'get_final_message_failed: {err!r}')

    if final_msg is not None:
        text = ''.join(
            (getattr(b, 'text', None) or '')
            for b in final_msg.content
            if getattr(b, 'type', None) == 'text'
        )
        try:
            _log_append(log_file, final_msg.model_dump_json())
        except Exception:
            _log_append(log_file, repr(final_msg))
        return 'final', text

    blob = json.dumps([_event_to_dict(e) for e in events], default=str)
    _log_append(log_file, blob[:16000])
    return 'raw', blob


@_apply_marks
class TestAnthropicSdkToolCall(_ToolCallTestBase):
    """Anthropic Messages + tools via official async SDK (end-to-end
    integration)."""

    @pytest.fixture(autouse=True)
    def _require_anthropic_sdk(self):
        pytest.importorskip('anthropic')

    def test_tool_non_stream_weather(self, backend, model_case):
        msg = asyncio.run(_async_messages_tool_non_stream(self._log_file))

        assert msg.stop_reason == 'tool_use'
        assert msg.role == 'assistant'
        tool_blocks = _sdk_tool_use_blocks(msg)
        assert len(tool_blocks) >= 1
        block = tool_blocks[0]
        assert block.name == WEATHER_TOOL['function']['name']

        _assert_weather_tool_city_state(block.input, ctx='test_tool_non_stream_weather')

        assert msg.usage is not None
        assert msg.usage.input_tokens > 0
        assert msg.usage.output_tokens > 0

    def test_tool_stream_weather(self, backend, model_case):
        kind, payload, events = asyncio.run(_async_messages_tool_stream(self._log_file))

        assert len(events) > 0, 'expected at least one stream event'

        if kind == 'final':
            assert payload.stop_reason == 'tool_use'
            tool_blocks = _sdk_tool_use_blocks(payload)
            assert len(tool_blocks) >= 1
            assert tool_blocks[0].name == WEATHER_TOOL['function']['name']
            _assert_weather_tool_city_state(tool_blocks[0].input, ctx='test_tool_stream_weather/final')
            return

        blob = payload
        assert WEATHER_TOOL['function']['name'] in blob
        assert 'tool_use' in blob
        assert 'Dallas' in blob or 'dallas' in blob.lower()

    def test_tool_non_stream_weather_single_location_schema(self, backend, model_case):
        msg = asyncio.run(_async_weather_tool_single_location_non_stream(self._log_file))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        assert len(tool_blocks) >= 1
        assert tool_blocks[0].name == 'get_current_weather'
        inp = tool_blocks[0].input
        assert isinstance(inp, dict)
        loc = inp.get('location', '')
        assert isinstance(loc, str) and len(loc) > 0
        loc_low = loc.lower()
        assert 'new york' in loc_low or 'nyc' in loc_low

    def test_tool_non_stream_tool_choice_force_named(self, backend, model_case):
        msg = asyncio.run(_async_tool_choice_force_named_tool(self._log_file))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        assert len(tool_blocks) >= 1
        assert tool_blocks[0].name == WEATHER_TOOL['function']['name']

    def test_tool_non_stream_tool_choice_any(self, backend, model_case):
        msg = asyncio.run(_async_tool_choice_any(self._log_file))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        assert len(tool_blocks) >= 1
        names = {b.name for b in tool_blocks}
        assert WEATHER_TOOL['function']['name'] in names, names

    def test_tool_non_stream_weather_with_user_image_url(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        if not _model_likely_supports_anthropic_vlm(model_name):
            pytest.skip(f'model {model_name!r} is not treated as vision-capable for this test')

        image_path = _eval_resource_file(_EVAL_IMAGE_TIGER)
        msg = asyncio.run(_async_messages_tool_non_stream_with_user_image(self._log_file, image_path))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        assert len(tool_blocks) >= 1
        assert tool_blocks[0].name == WEATHER_TOOL['function']['name']
        _assert_weather_tool_city_state(tool_blocks[0].input, ctx='test_tool_non_stream_weather_with_user_image_url')

    def test_tool_non_stream_weather_with_user_image_base64(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        if not _model_likely_supports_anthropic_vlm(model_name):
            pytest.skip(f'model {model_name!r} is not treated as vision-capable for this test')

        msg = asyncio.run(_async_messages_tool_non_stream_with_user_image_base64(self._log_file))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        assert len(tool_blocks) >= 1
        assert tool_blocks[0].name == WEATHER_TOOL['function']['name']
        _assert_weather_tool_city_state(
            tool_blocks[0].input,
            ctx='test_tool_non_stream_weather_with_user_image_base64',
        )

    def test_sdk_stream_vlm_user_image_base64_solid_color(self, backend, model_case):
        """SDK streaming + 1×1 red PNG: final text (or raw event blob) should
        mention a red-ish color."""

        model_name = APIClient(BASE_URL).available_models[0]
        if not _model_likely_supports_anthropic_vlm(model_name):
            pytest.skip(f'model {model_name!r} is not treated as vision-capable for this test')

        kind, payload = asyncio.run(_async_vlm_base64_solid_color_stream(self._log_file))
        ctx = f'test_sdk_stream_vlm_user_image_base64_solid_color/{kind}'
        _assert_redish_color_in_text(payload, ctx=ctx)
