from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache

import pytest
import requests
from utils.anthropic_messages import (
    ANTHROPIC_MESSAGES_ASKING_FOR_WEATHER,
    ANTHROPIC_MESSAGES_HISTORY_THINKING_REPLAY,
    ANTHROPIC_MESSAGES_PARALLEL_MIXED,
    ANTHROPIC_MESSAGES_PARALLEL_WEATHER,
    ANTHROPIC_SYSTEM_PARALLEL_MIXED,
    ANTHROPIC_SYSTEM_PARALLEL_WEATHER,
    ANTHROPIC_SYSTEM_WEATHER,
    CALCULATOR_TOOL_ANTHROPIC,
    SEARCH_TOOL_ANTHROPIC,
    USER_ASK_WEATHER_DALLAS,
    USER_ASK_WEATHER_DALLAS_VLM,
    WEATHER_TOOL_ANTHROPIC,
    WEATHER_TOOL_SINGLE_LOCATION_ANTHROPIC,
    assert_parallel_weather_tool_inputs,
    assert_tool_use_message,
    assert_warm_yes_answer,
    assert_weather_tool_city_state,
    build_anthropic_messages_after_tool_use,
    build_anthropic_messages_history_tool_result,
    get_async_anthropic_client_and_model,
)
from utils.config_utils import get_config
from utils.constant import BASE_URL

from lmdeploy.serve.openai.api_client import APIClient

from .conftest import _apply_marks, _ToolCallTestBase

ANTHROPIC_VERSION = '2023-06-01'

_EVAL_IMAGE_TIGER = 'tiger.jpeg'
_TINY_PNG_BASE64 = (
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
)

# Reasoning+vision models may emit long thinking before tool_use.
_VLM_TOOL_MAX_TOKENS = 8192

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
    path = cfg['resource_path']
    assert path, 'resource_path must be set in autotest config (e.g. config_h.yml)'
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
            'INTERN-S',
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
        if obj['type'] != 'content_block_delta':
            continue
        delta = obj['delta']
        if delta['type'] == 'text_delta':
            text += delta['text']
    return text


def _sse_tool_use_names(raw: str) -> list[str]:
    return [b['name'] for b in _sse_tool_use_blocks(raw)]


def _sse_tool_use_blocks(raw: str) -> list[dict]:
    blocks: list[dict] = []
    for _, obj in _parse_anthropic_sse(raw):
        if obj['type'] != 'content_block_start':
            continue
        cb = obj['content_block']
        if cb['type'] == 'tool_use':
            blocks.append(cb)
    return blocks


def _sse_aggregate_tool_use_inputs(raw: str) -> list[dict]:
    """Rebuild each streamed ``tool_use`` ``input`` from
    ``input_json_delta``."""

    meta_by_index: dict[int, dict] = {}
    json_bufs: dict[int, str] = {}
    for _, obj in _parse_anthropic_sse(raw):
        evt = obj['type']
        if evt == 'content_block_start':
            idx = int(obj['index'])
            cb = obj['content_block']
            if cb['type'] != 'tool_use':
                continue
            meta_by_index[idx] = {
                'id': cb['id'],
                'name': cb['name'],
            }
            if isinstance(cb['input'], dict) and cb['input']:
                json_bufs[idx] = json.dumps(cb['input'])
        elif evt == 'content_block_delta':
            idx = int(obj['index'])
            delta = obj['delta']
            if delta['type'] == 'input_json_delta':
                json_bufs[idx] = json_bufs.get(idx, '') + delta['partial_json']

    inputs: list[dict] = []
    for idx in sorted(meta_by_index):
        inp: dict = {}
        buf = json_bufs.get(idx, '').strip()
        if buf:
            try:
                parsed = json.loads(buf)
                if isinstance(parsed, dict):
                    inp = parsed
            except json.JSONDecodeError:
                inp = {'_parse_error': buf}
        inputs.append(inp)
    return inputs


def _tool_inputs_from_sdk_blocks(tool_blocks: list) -> list[dict]:
    inputs: list[dict] = []
    for block in tool_blocks:
        inp = block.input
        if hasattr(inp, 'model_dump'):
            inp = inp.model_dump()
        inputs.append(inp)
    return inputs


def _tool_inputs_from_http_blocks(tool_blocks: list[dict]) -> list[dict]:
    return [b['input'] for b in tool_blocks]


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
    return ''.join(b['text'] for b in data['content'] if b['type'] == 'text')


def _http_tool_use_blocks(data: dict) -> list[dict]:
    return [b for b in data['content'] if b['type'] == 'tool_use']


def _sdk_tool_use_blocks(msg) -> list:
    return [b for b in msg.content if b.type == 'tool_use']


def _sdk_tool_use_block_to_dict(block) -> dict:
    inp = block.input
    if hasattr(inp, 'model_dump'):
        inp = inp.model_dump()
    return {
        'id': block.id,
        'name': block.name,
        'input': inp,
    }


def _assert_parallel_tool_use_blocks(blocks: list, *, min_count: int = 2, ctx: str = '') -> None:
    assert len(blocks) >= min_count, (ctx, len(blocks), blocks)
    ids = [b.id if not isinstance(b, dict) else b['id'] for b in blocks]
    assert len(set(ids)) == len(ids), (ctx, ids)


def _sdk_assistant_text(msg) -> str:
    return ''.join(b.text for b in msg.content if b.type == 'text')


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
                'name': WEATHER_TOOL_ANTHROPIC['name'],
            },
            'messages': [{'role': 'user', 'content': USER_ASK_WEATHER_DALLAS}],
            'tools': [WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
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
        assert WEATHER_TOOL_ANTHROPIC['name'] in names, names
        inputs = _sse_aggregate_tool_use_inputs(raw)
        assert inputs, raw[:800]
        assert_weather_tool_city_state(inputs[0], ctx='test_http_stream_tool_choice_force_named_tool')

    def test_http_stream_single_location_weather_tool(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        url = f'{BASE_URL}/v1/messages'
        req_json = {
            'model': model_name,
            'max_tokens': 512,
            'temperature': 0,
            'stream': True,
            'messages': [{'role': 'user', 'content': USER_ASK_WEATHER_DALLAS}],
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
        inputs = _sse_aggregate_tool_use_inputs(raw)
        assert inputs, raw[:800]
        loc = str(inputs[0]['location']).lower()
        assert 'dallas' in loc, inputs

    def test_http_parallel_same_tool_stream(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        url = f'{BASE_URL}/v1/messages'
        req_json = {
            'model': model_name,
            'max_tokens': 1024,
            'temperature': 0,
            'stream': True,
            'system': ANTHROPIC_SYSTEM_PARALLEL_WEATHER,
            'messages': ANTHROPIC_MESSAGES_PARALLEL_WEATHER,
            'tools': [WEATHER_TOOL_ANTHROPIC],
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
            'test_http_parallel_same_tool_stream',
            url=url,
            http_status=resp.status_code,
            request_json=req_json,
            response_text=raw,
        )
        blocks = _sse_tool_use_blocks(raw)
        _assert_parallel_tool_use_blocks(blocks, min_count=2, ctx='http_parallel_same_tool')
        names = {b['name'] for b in blocks}
        assert WEATHER_TOOL_ANTHROPIC['name'] in names, names
        assert_parallel_weather_tool_inputs(
            _sse_aggregate_tool_use_inputs(raw),
            ctx='test_http_parallel_same_tool_stream',
        )

    def test_http_full_roundtrip_single_tool_result(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        url = f'{BASE_URL}/v1/messages'
        turn1 = {
            'model': model_name,
            'max_tokens': 1024,
            'temperature': 0,
            'system': ANTHROPIC_SYSTEM_WEATHER,
            'messages': ANTHROPIC_MESSAGES_ASKING_FOR_WEATHER,
            'tools': [WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
        }
        resp1 = requests.post(url, headers=_http_headers(), json=turn1, timeout=180)
        assert resp1.status_code == 200, resp1.text
        data1 = resp1.json()
        _trace_anthropic_http(
            self._log_file,
            'test_http_full_roundtrip_single_tool_result/turn1',
            url=url,
            http_status=resp1.status_code,
            request_json=turn1,
            response_text=json.dumps(data1, ensure_ascii=False, default=str),
        )
        tool_blocks = assert_tool_use_message(data1)
        tb = tool_blocks[0]
        tool_result_text = json.dumps({
            'temperature': 98,
            'unit': 'fahrenheit',
            'description': 'Sunny with clear skies',
        })
        turn2_messages = build_anthropic_messages_after_tool_use(
            ANTHROPIC_MESSAGES_ASKING_FOR_WEATHER,
            [tb],
            [(tb['id'], tool_result_text)],
        )
        turn2 = {
            'model': model_name,
            'max_tokens': 1024,
            'temperature': 0,
            'system': ANTHROPIC_SYSTEM_WEATHER,
            'messages': turn2_messages,
            'tools': [WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
        }
        resp2 = requests.post(url, headers=_http_headers(), json=turn2, timeout=180)
        assert resp2.status_code == 200, resp2.text
        data2 = resp2.json()
        _trace_anthropic_http(
            self._log_file,
            'test_http_full_roundtrip_single_tool_result/turn2',
            url=url,
            http_status=resp2.status_code,
            request_json=turn2,
            response_text=json.dumps(data2, ensure_ascii=False, default=str),
        )
        assert data2['stop_reason'] in ('end_turn', 'max_tokens'), data2
        text = _assistant_text_from_messages_json(data2)
        assert len(text) > 0, data2['content']
        assert '98' in text or 'Dallas' in text or 'sunny' in text.lower(), text[:500]

    def test_http_history_tool_use_and_tool_result_blocks(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        url = f'{BASE_URL}/v1/messages'
        req_json = {
            'model': model_name,
            'max_tokens': 8192,
            'temperature': 0.01,
            'messages': build_anthropic_messages_history_tool_result(),
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
        assert_warm_yes_answer(
            text,
            stop_reason=data['stop_reason'],
            ctx='test_http_history_tool_use_and_tool_result_blocks',
        )

    def test_http_history_thinking_block_replay(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        url = f'{BASE_URL}/v1/messages'
        req_json = {
            'model': model_name,
            'max_tokens': 8192,
            'temperature': 0.01,
            'messages': ANTHROPIC_MESSAGES_HISTORY_THINKING_REPLAY,
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
            f'stop_reason={data["stop_reason"]!r} content={data["content"]!r}'
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
            'max_tokens': _VLM_TOOL_MAX_TOKENS,
            'temperature': 0,
            'system': ANTHROPIC_SYSTEM_WEATHER,
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': USER_ASK_WEATHER_DALLAS_VLM,
                    },
                    {'type': 'image', 'source': {'type': 'url', 'url': image_path}},
                ],
            }],
            'tools': [WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
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
        blocks = assert_tool_use_message(data, tool_name=WEATHER_TOOL_ANTHROPIC['name'])
        inp = blocks[0]['input']
        assert_weather_tool_city_state(inp, ctx='test_http_non_stream_tools_with_user_image_url')

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
            'max_tokens': _VLM_TOOL_MAX_TOKENS,
            'temperature': 0,
            'stream': True,
            'system': ANTHROPIC_SYSTEM_WEATHER,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': USER_ASK_WEATHER_DALLAS_VLM},
                    {'type': 'image', 'source': {'type': 'url', 'url': image_path}},
                ],
            }],
            'tools': [WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
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
        assert WEATHER_TOOL_ANTHROPIC['name'] in names, names

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
        types = [obj['type'] for _, obj in events]
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
        messages=[{'role': 'user', 'content': USER_ASK_WEATHER_DALLAS}],
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
    msg = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        system=ANTHROPIC_SYSTEM_WEATHER,
        messages=ANTHROPIC_MESSAGES_ASKING_FOR_WEATHER,
        tools=[WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
        tool_choice={'type': 'tool', 'name': WEATHER_TOOL_ANTHROPIC['name']},
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
    msg = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        system=ANTHROPIC_SYSTEM_WEATHER,
        messages=ANTHROPIC_MESSAGES_ASKING_FOR_WEATHER,
        tools=[WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
        tool_choice={'type': 'any'},
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
    msg = await client.messages.create(
        model=model_name,
        max_tokens=_VLM_TOOL_MAX_TOKENS,
        temperature=0,
        system=ANTHROPIC_SYSTEM_WEATHER,
        tools=[WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
        messages=[{
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': USER_ASK_WEATHER_DALLAS_VLM,
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
    msg = await client.messages.create(
        model=model_name,
        max_tokens=_VLM_TOOL_MAX_TOKENS,
        temperature=0,
        system=ANTHROPIC_SYSTEM_WEATHER,
        tools=[WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': USER_ASK_WEATHER_DALLAS_VLM},
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
    msg = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        system=ANTHROPIC_SYSTEM_WEATHER,
        messages=ANTHROPIC_MESSAGES_ASKING_FOR_WEATHER,
        tools=[WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
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
    stream = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        system=ANTHROPIC_SYSTEM_WEATHER,
        messages=ANTHROPIC_MESSAGES_ASKING_FOR_WEATHER,
        tools=[WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC],
        stream=True,
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


async def _async_parallel_same_tool_non_stream(log_file: str):
    client, model_name = get_async_anthropic_client_and_model()
    msg = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        system=ANTHROPIC_SYSTEM_PARALLEL_WEATHER,
        messages=ANTHROPIC_MESSAGES_PARALLEL_WEATHER,
        tools=[WEATHER_TOOL_ANTHROPIC],
    )
    _log_append(log_file, _safe_dump_message(msg))
    return msg


async def _async_parallel_same_tool_stream(log_file: str):
    client, model_name = get_async_anthropic_client_and_model()
    stream = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        system=ANTHROPIC_SYSTEM_PARALLEL_WEATHER,
        messages=ANTHROPIC_MESSAGES_PARALLEL_WEATHER,
        tools=[WEATHER_TOOL_ANTHROPIC],
        stream=True,
    )
    events = []
    async for event in stream:
        events.append(event)
    _log_append(log_file, f'parallel_stream_event_count={len(events)}')
    final_msg = await _try_get_final_message(stream, log_file)
    if final_msg is not None:
        _log_append(log_file, _safe_dump_message(final_msg))
        return 'final', final_msg, events
    blob = json.dumps([_event_to_dict(e) for e in events], default=str)
    _log_append(log_file, blob[:16000])
    return 'raw', blob, events


async def _async_parallel_mixed_tools_non_stream(log_file: str):
    client, model_name = get_async_anthropic_client_and_model()
    msg = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        system=ANTHROPIC_SYSTEM_PARALLEL_MIXED,
        messages=ANTHROPIC_MESSAGES_PARALLEL_MIXED,
        tools=[WEATHER_TOOL_ANTHROPIC, CALCULATOR_TOOL_ANTHROPIC],
    )
    _log_append(log_file, _safe_dump_message(msg))
    return msg


async def _async_full_roundtrip_single_tool_result(log_file: str):
    client, model_name = get_async_anthropic_client_and_model()
    tools = [WEATHER_TOOL_ANTHROPIC, SEARCH_TOOL_ANTHROPIC]
    msg1 = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        system=ANTHROPIC_SYSTEM_WEATHER,
        messages=ANTHROPIC_MESSAGES_ASKING_FOR_WEATHER,
        tools=tools,
    )
    _log_append(log_file, 'roundtrip_turn1=' + _safe_dump_message(msg1))
    tool_blocks = _sdk_tool_use_blocks(msg1)
    tb = _sdk_tool_use_block_to_dict(tool_blocks[0])
    tool_result_text = json.dumps({
        'temperature': 98,
        'unit': 'fahrenheit',
        'description': 'Sunny with clear skies',
    })
    turn2_messages = build_anthropic_messages_after_tool_use(
        ANTHROPIC_MESSAGES_ASKING_FOR_WEATHER,
        [tb],
        [(tb['id'], tool_result_text)],
    )
    msg2 = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        system=ANTHROPIC_SYSTEM_WEATHER,
        messages=turn2_messages,
        tools=tools,
    )
    _log_append(log_file, 'roundtrip_turn2=' + _safe_dump_message(msg2))
    return msg1, msg2


async def _async_full_roundtrip_parallel_tool_results(log_file: str):
    client, model_name = get_async_anthropic_client_and_model()
    msg1 = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        system=ANTHROPIC_SYSTEM_PARALLEL_WEATHER,
        messages=ANTHROPIC_MESSAGES_PARALLEL_WEATHER,
        tools=[WEATHER_TOOL_ANTHROPIC],
    )
    _log_append(log_file, 'parallel_roundtrip_turn1=' + _safe_dump_message(msg1))
    tool_blocks = _sdk_tool_use_blocks(msg1)
    _assert_parallel_tool_use_blocks(tool_blocks, min_count=2, ctx='parallel_roundtrip_turn1')
    assert_parallel_weather_tool_inputs(
        _tool_inputs_from_sdk_blocks(tool_blocks),
        ctx='parallel_roundtrip_turn1',
    )
    block_dicts = [_sdk_tool_use_block_to_dict(b) for b in tool_blocks]
    tool_results = [
        (block_dicts[0]['id'], json.dumps({
            'temperature': 98,
            'unit': 'fahrenheit',
            'description': 'Sunny',
        })),
        (block_dicts[1]['id'], json.dumps({
            'temperature': 65,
            'unit': 'fahrenheit',
            'description': 'Foggy',
        })),
    ]
    turn2_messages = build_anthropic_messages_after_tool_use(
        ANTHROPIC_MESSAGES_PARALLEL_WEATHER,
        block_dicts,
        tool_results,
    )
    msg2 = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0,
        system=ANTHROPIC_SYSTEM_PARALLEL_WEATHER,
        messages=turn2_messages,
        tools=[WEATHER_TOOL_ANTHROPIC],
    )
    _log_append(log_file, 'parallel_roundtrip_turn2=' + _safe_dump_message(msg2))
    return msg1, msg2


def _safe_dump_message(msg) -> str:
    try:
        return msg.model_dump_json()
    except Exception:
        try:
            return json.dumps(msg.model_dump())
        except Exception:
            return repr(msg)


async def _try_get_final_message(stream, log_file: str):
    getter = getattr(stream, 'get_final_message', None)
    if not callable(getter):
        return None
    try:
        return await getter()
    except Exception as err:  # noqa: BLE001
        _log_append(log_file, f'get_final_message_failed: {err!r}')
        return None


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
        text = ''.join(b.text for b in final_msg.content if b.type == 'text')
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
        block = tool_blocks[0]
        assert block.name == WEATHER_TOOL_ANTHROPIC['name']

        assert_weather_tool_city_state(block.input, ctx='test_tool_non_stream_weather')

        assert msg.usage is not None
        assert msg.usage.input_tokens > 0
        assert msg.usage.output_tokens > 0

    def test_tool_stream_weather(self, backend, model_case):
        kind, payload, events = asyncio.run(_async_messages_tool_stream(self._log_file))

        assert len(events) > 0, 'expected at least one stream event'

        if kind == 'final':
            assert payload.stop_reason == 'tool_use'
            tool_blocks = _sdk_tool_use_blocks(payload)
            assert tool_blocks[0].name == WEATHER_TOOL_ANTHROPIC['name']
            assert_weather_tool_city_state(tool_blocks[0].input, ctx='test_tool_stream_weather/final')
            return

        blob = payload
        assert WEATHER_TOOL_ANTHROPIC['name'] in blob
        assert 'tool_use' in blob
        assert 'Dallas' in blob or 'dallas' in blob.lower()

    def test_tool_non_stream_weather_single_location_schema(self, backend, model_case):
        msg = asyncio.run(_async_weather_tool_single_location_non_stream(self._log_file))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        inp = tool_blocks[0].input
        loc_low = inp['location'].lower()
        assert 'dallas' in loc_low

    def test_tool_non_stream_tool_choice_force_named(self, backend, model_case):
        msg = asyncio.run(_async_tool_choice_force_named_tool(self._log_file))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        assert tool_blocks[0].name == WEATHER_TOOL_ANTHROPIC['name']

    def test_tool_non_stream_tool_choice_any(self, backend, model_case):
        msg = asyncio.run(_async_tool_choice_any(self._log_file))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        names = {b.name for b in tool_blocks}
        assert WEATHER_TOOL_ANTHROPIC['name'] in names, names

    def test_tool_non_stream_weather_with_user_image_url(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        if not _model_likely_supports_anthropic_vlm(model_name):
            pytest.skip(f'model {model_name!r} is not treated as vision-capable for this test')

        image_path = _eval_resource_file(_EVAL_IMAGE_TIGER)
        msg = asyncio.run(_async_messages_tool_non_stream_with_user_image(self._log_file, image_path))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        assert tool_blocks[0].name == WEATHER_TOOL_ANTHROPIC['name']
        assert_weather_tool_city_state(tool_blocks[0].input, ctx='test_tool_non_stream_weather_with_user_image_url')

    def test_tool_non_stream_weather_with_user_image_base64(self, backend, model_case):
        model_name = APIClient(BASE_URL).available_models[0]
        if not _model_likely_supports_anthropic_vlm(model_name):
            pytest.skip(f'model {model_name!r} is not treated as vision-capable for this test')

        msg = asyncio.run(_async_messages_tool_non_stream_with_user_image_base64(self._log_file))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        assert tool_blocks[0].name == WEATHER_TOOL_ANTHROPIC['name']
        assert_weather_tool_city_state(
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

    def test_parallel_same_tool_non_stream(self, backend, model_case):
        msg = asyncio.run(_async_parallel_same_tool_non_stream(self._log_file))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        _assert_parallel_tool_use_blocks(tool_blocks, min_count=2, ctx='parallel_same_tool_non_stream')
        for block in tool_blocks:
            assert block.name == WEATHER_TOOL_ANTHROPIC['name']
        assert_parallel_weather_tool_inputs(
            _tool_inputs_from_sdk_blocks(tool_blocks),
            ctx='parallel_same_tool_non_stream',
        )

    def test_parallel_same_tool_stream(self, backend, model_case):
        kind, payload, events = asyncio.run(_async_parallel_same_tool_stream(self._log_file))
        assert len(events) > 0
        if kind == 'final':
            assert payload.stop_reason == 'tool_use'
            tool_blocks = _sdk_tool_use_blocks(payload)
            _assert_parallel_tool_use_blocks(tool_blocks, min_count=2, ctx='parallel_same_tool_stream/final')
            assert_parallel_weather_tool_inputs(
                _tool_inputs_from_sdk_blocks(tool_blocks),
                ctx='parallel_same_tool_stream/final',
            )
            return
        assert WEATHER_TOOL_ANTHROPIC['name'] in payload
        assert payload.count('tool_use') >= 2, payload[:800]

    def test_parallel_mixed_tools_non_stream(self, backend, model_case):
        msg = asyncio.run(_async_parallel_mixed_tools_non_stream(self._log_file))
        assert msg.stop_reason == 'tool_use'
        tool_blocks = _sdk_tool_use_blocks(msg)
        _assert_parallel_tool_use_blocks(tool_blocks, min_count=2, ctx='parallel_mixed_tools')
        names = {b.name for b in tool_blocks}
        assert WEATHER_TOOL_ANTHROPIC['name'] in names, names
        assert CALCULATOR_TOOL_ANTHROPIC['name'] in names, names

    def test_full_roundtrip_single_tool_result(self, backend, model_case):
        msg1, msg2 = asyncio.run(_async_full_roundtrip_single_tool_result(self._log_file))
        assert msg1.stop_reason == 'tool_use'
        assert msg2.stop_reason in ('end_turn', 'max_tokens')
        assert len(_sdk_tool_use_blocks(msg2)) == 0
        text = _sdk_assistant_text(msg2)
        assert len(text) > 0, msg2.content
        assert '98' in text or 'Dallas' in text or 'sunny' in text.lower(), text[:500]

    def test_full_roundtrip_parallel_tool_results(self, backend, model_case):
        msg1, msg2 = asyncio.run(_async_full_roundtrip_parallel_tool_results(self._log_file))
        assert msg1.stop_reason == 'tool_use'
        assert msg2.stop_reason in ('end_turn', 'max_tokens')
        text = _sdk_assistant_text(msg2)
        assert len(text) > 0, msg2.content
        has_dallas = 'Dallas' in text or '98' in text
        has_sf = 'San Francisco' in text or '65' in text
        assert has_dallas or has_sf, text[:500]
