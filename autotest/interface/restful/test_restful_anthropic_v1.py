from __future__ import annotations

import json
import os
from functools import lru_cache

import pytest
import requests
from utils.config_utils import get_config
from utils.constant import BACKEND_LIST, BASE_URL, RESTFUL_MODEL_LIST
from utils.tool_reasoning_definitions import WEATHER_TOOL, openai_function_tool_to_anthropic

from lmdeploy.serve.openai.api_client import APIClient

ANTHROPIC_VERSION = '2023-06-01'

_MESSAGES_URL = f'{BASE_URL}/v1/messages'
_COUNT_TOKENS_URL = f'{BASE_URL}/v1/messages/count_tokens'

_EVAL_IMAGE_TIGER = 'tiger.jpeg'

# 1×1 PNG (red), for ``source: {type: base64}`` smoke without relying on ``resource_path`` files.
_TINY_PNG_BASE64 = (
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
)


@pytest.fixture(scope='class')
def deployed_model_name() -> str:
    """Single model id exposed by the RESTFUL api_server."""

    return APIClient(BASE_URL).available_models[0]


@lru_cache(maxsize=1)
def _eval_resource_path() -> str:
    """``resource_path`` from active autotest YAML (``TEST_ENV`` →
    ``autotest/config_{tag}.yml``)."""

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


def _anthropic_headers() -> dict[str, str]:
    return {
        'Content-Type': 'application/json',
        'anthropic-version': ANTHROPIC_VERSION,
    }


def _assistant_text_from_message_payload(data: dict) -> str:
    return ''.join(b.get('text', '') for b in data.get('content', []) if b.get('type') == 'text')


def _model_likely_supports_anthropic_vlm(model_name: str) -> bool:
    """Heuristic for RESTFUL matrix: skip image HTTP when the served id is clearly text-only."""

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


def _parse_anthropic_sse(raw: str) -> list[tuple[str | None, dict]]:
    """Parse Anthropic-style SSE (``event:`` / ``data:`` lines) into
    (event_name, json_payload) pairs."""

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


def _assert_count_tokens_json(data: dict) -> int:
    assert set(data.keys()) == {'input_tokens'}, data
    n = data['input_tokens']
    assert isinstance(n, int) and n > 0, n
    return n


def _assert_success_message_json(data: dict, *, model: str) -> dict:
    """Non-stream ``/v1/messages`` success body: Anthropic message + usage
    invariants."""

    assert data.get('type') == 'message', data
    assert data.get('role') == 'assistant'
    assert data.get('model') == model
    mid = data.get('id')
    assert isinstance(mid, str) and mid.startswith('msg_'), mid
    content = data.get('content')
    assert isinstance(content, list) and len(content) >= 1, content
    usage = data.get('usage')
    assert isinstance(usage, dict), data
    assert 'input_tokens' in usage and 'output_tokens' in usage, usage
    assert isinstance(usage['input_tokens'], int) and usage['input_tokens'] >= 0
    assert isinstance(usage['output_tokens'], int) and usage['output_tokens'] > 0
    assert data.get('stop_reason') in ('end_turn', 'max_tokens', 'stop_sequence', 'tool_use', None)
    return data


def _assert_anthropic_error_envelope(body: dict) -> None:
    assert body.get('type') == 'error', body
    err = body.get('error')
    assert isinstance(err, dict) and 'type' in err and 'message' in err, err


def _assert_fastapi_validation_error(resp: requests.Response) -> dict:
    """FastAPI ``RequestValidationError`` payload (not Anthropic ``type:

    error``).
    """

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert isinstance(body.get('detail'), list), body
    return body


def _assert_tool_parser_required_message(resp: requests.Response) -> None:
    assert resp.status_code == 400, resp.text
    body = resp.json()
    _assert_anthropic_error_envelope(body)
    assert body['error']['type'] == 'invalid_request_error'
    err = body['error']['message']
    assert '--tool-call-parser' in err


@pytest.mark.order(8)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', RESTFUL_MODEL_LIST)
class TestRestfulAnthropicV1:

    def test_list_models(self, backend, model_case, deployed_model_name: str):
        url = f'{BASE_URL}/anthropic/v1/models'
        resp = requests.get(url, timeout=30)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data.get('has_more'), bool)
        assert 'data' in data
        assert isinstance(data['data'], list)
        for m in data['data']:
            assert isinstance(m, dict)
            assert m.get('type') == 'model'
            assert isinstance(m.get('id'), str) and len(m['id']) > 0
            assert isinstance(m.get('display_name'), str)
        ids = [m['id'] for m in data['data']]
        assert deployed_model_name in ids, (deployed_model_name, ids)
        if ids:
            assert data.get('first_id') == ids[0]
            assert data.get('last_id') == ids[-1]

    @pytest.mark.parametrize(
        'endpoint_url,body_without_model',
        [
            pytest.param(
                _MESSAGES_URL,
                {'max_tokens': 8, 'messages': [{'role': 'user', 'content': 'Say hi in one word.'}]},
                id='messages',
            ),
            pytest.param(
                _COUNT_TOKENS_URL,
                {'messages': [{'role': 'user', 'content': 'Hi'}]},
                id='count_tokens',
            ),
        ],
    )
    def test_messages_and_count_tokens_missing_version_header(
            self, backend, model_case, deployed_model_name: str, endpoint_url: str, body_without_model: dict):
        resp = requests.post(
            endpoint_url,
            headers={'Content-Type': 'application/json'},
            json={'model': deployed_model_name, **body_without_model},
            timeout=60,
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        _assert_anthropic_error_envelope(body)
        assert body['error']['type'] == 'invalid_request_error'
        assert body['error']['message'] == 'Missing required header: anthropic-version'

    @pytest.mark.parametrize(
        'endpoint_url,request_json',
        [
            pytest.param(
                _MESSAGES_URL,
                {
                    'model': 'definitely-not-a-deployed-model-name',
                    'max_tokens': 8,
                    'messages': [{'role': 'user', 'content': 'Hi'}],
                },
                id='messages',
            ),
            pytest.param(
                _COUNT_TOKENS_URL,
                {
                    'model': 'definitely-not-a-deployed-model-name',
                    'messages': [{'role': 'user', 'content': 'Hi'}],
                },
                id='count_tokens',
            ),
        ],
    )
    def test_messages_and_count_tokens_unknown_model(
            self, backend, model_case, endpoint_url: str, request_json: dict):
        resp = requests.post(
            endpoint_url,
            headers=_anthropic_headers(),
            json=request_json,
            timeout=30,
        )
        assert resp.status_code == 404, resp.text
        body = resp.json()
        _assert_anthropic_error_envelope(body)
        assert body['error']['type'] == 'not_found_error'
        assert 'does not exist' in body['error']['message']

    def test_messages_with_system(self, backend, model_case, deployed_model_name: str):
        """Anthropic ``system`` field (Messages API).

        Some chat models prefix visible chain-of-thought before the final reply; keep
        ``max_tokens`` high enough that the instructed answer still fits the budget.
        """

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 2048,
                'temperature': 0.01,
                'system': 'You reply only with the single word: Acknowledged.',
                'messages': [{'role': 'user', 'content': 'What is your instruction?'}],
            },
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        text = _assistant_text_from_message_payload(data)
        assert 'acknowledged' in text.lower(), text[:500]

    def test_messages_user_content_as_blocks(self, backend, model_case, deployed_model_name: str):
        """``messages[].content`` as a list of ``{type: text}`` blocks
        (Anthropic-native shape)."""

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 24,
                'temperature': 0.01,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Answer with one word: color of grass? '},
                        {'type': 'text', 'text': 'Just the color name.'},
                    ],
                }],
            },
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        text = _assistant_text_from_message_payload(data)
        tl = text.lower()
        assert any(
            k in tl
            for k in ('green', 'grass', '青', '綠', '绿')), f'expected color-of-grass hint in reply: {text[:500]!r}'

    def test_messages_system_as_content_blocks(self, backend, model_case, deployed_model_name: str):
        """``system`` as Anthropic block list (concatenated server-side for the
        chat template)."""

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 256,
                'temperature': 0.01,
                'system': [
                    {'type': 'text', 'text': 'You reply only with the single word: Confirmed.'},
                    {'type': 'text', 'text': ' No extra words.'},
                ],
                'messages': [{'role': 'user', 'content': 'Acknowledge with your required reply.'}],
            },
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        text = _assistant_text_from_message_payload(data).lower()
        assert 'confirmed' in text, text[:500]

    def test_messages_history_tool_use_and_tool_result_without_request_tools(
            self, backend, model_case, deployed_model_name: str):
        """Replay ``tool_use`` / ``tool_result`` blocks without top-level
        ``tools`` (parserless RESTFUL)."""

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 2048,
                'temperature': 0.01,
                'messages': [
                    {'role': 'user', 'content': 'What is the weather in San Francisco?'},
                    {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'tool_use',
                                'id': 'toolu_hist_restful_01',
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
                                'tool_use_id': 'toolu_hist_restful_01',
                                'content': '72F and sunny.',
                            },
                        ],
                    },
                    {
                        'role': 'user',
                        'content': 'In one short phrase, was it warm? Answer yes or no.',
                    },
                ],
            },
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        text = _assistant_text_from_message_payload(data)
        tl = text.lower()
        assert 'yes' in tl or '是' in text or '温暖' in text or '暖和' in text, (
            'expected warm/yes style answer given 72F sunny tool result; '
            f'stop_reason={data.get("stop_reason")!r} text={text[:500]!r}'
        )

    def test_messages_history_thinking_and_text_blocks(self, backend, model_case, deployed_model_name: str):
        """Assistant history with ``thinking`` + ``text`` (reasoning replay
        path)."""

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 2048,
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
            },
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        text = _assistant_text_from_message_payload(data)
        assert 'ack' in text.lower(), (
            'expected literal ACK from final user instruction; '
            f'stop_reason={data.get("stop_reason")!r} text={text[:500]!r}'
        )

    def test_messages_user_image_file_from_config_resource(self, backend, model_case, deployed_model_name: str):
        """``user`` message with Anthropic ``image`` + local ``resource_path``
        file (``config_h.yml``)."""

        if not _model_likely_supports_anthropic_vlm(deployed_model_name):
            pytest.skip(f'model {deployed_model_name!r} is not treated as vision-capable for this test')

        image_path = _eval_resource_file(_EVAL_IMAGE_TIGER)
        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 128,
                'temperature': 0.01,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'In one word, name the animal in the image.'},
                        {
                            'type': 'image',
                            'source': {'type': 'url', 'url': image_path},
                        },
                    ],
                }],
            },
            timeout=180,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        text = _assistant_text_from_message_payload(data).lower()
        assert any(
            k in text
            for k in ('tiger', 'cat', 'big cat', '虎', '猫', 'feline')), text[:800]

    def test_count_tokens_user_image_block_exceeds_text_only(self, backend, model_case, deployed_model_name: str):
        """``count_tokens`` flattens ``image`` blocks in
        ``to_lmdeploy_messages``; count should exceed text-only."""

        image_path = _eval_resource_file(_EVAL_IMAGE_TIGER)
        base = {
            'model': deployed_model_name,
            'messages': [{
                'role': 'user',
                'content': [{'type': 'text', 'text': 'Describe briefly.'}],
            }],
        }
        r0 = requests.post(_COUNT_TOKENS_URL, headers=_anthropic_headers(), json=base, timeout=120)
        assert r0.status_code == 200, r0.text
        n0 = _assert_count_tokens_json(r0.json())

        r1 = requests.post(
            _COUNT_TOKENS_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': 'Describe briefly.'},
                        {'type': 'image', 'source': {'type': 'url', 'url': image_path}},
                    ],
                }],
            },
            timeout=120,
        )
        assert r1.status_code == 200, r1.text
        n1 = _assert_count_tokens_json(r1.json())
        assert n1 > n0, ('image-bearing user message should tokenize longer than text-only', n1, n0)

    def test_messages_user_image_interleaved_text_blocks(self, backend, model_case, deployed_model_name: str):
        """Multimodal user turn: ``text`` → ``image`` → ``text`` (ordering + VLM path)."""

        if not _model_likely_supports_anthropic_vlm(deployed_model_name):
            pytest.skip(f'model {deployed_model_name!r} is not treated as vision-capable for this test')

        image_path = _eval_resource_file(_EVAL_IMAGE_TIGER)
        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 128,
                'temperature': 0.01,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'The next block is an image. After it, follow the final instruction only.',
                        },
                        {'type': 'image', 'source': {'type': 'url', 'url': image_path}},
                        {'type': 'text', 'text': 'In one word, name the animal in the image.'},
                    ],
                }],
            },
            timeout=180,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        text = _assistant_text_from_message_payload(data).lower()
        assert any(
            k in text
            for k in ('tiger', 'cat', 'big cat', '虎', '猫', 'feline')), text[:800]

    def test_messages_user_image_base64_stream(self, backend, model_case, deployed_model_name: str):
        """Tiny PNG via ``base64`` source + ``stream: true`` (VLM + SSE
        path)."""

        if not _model_likely_supports_anthropic_vlm(deployed_model_name):
            pytest.skip(f'model {deployed_model_name!r} is not treated as vision-capable for this test')

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                # Same as tool_parser HTTP solid-color VLM test: leave room after thinking_delta.
                'max_tokens': 16384,
                'temperature': 0.01,
                'stream': True,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': (
                                'The image is a single solid color (one pixel). '
                                'Reply with at most three words: name that color only (e.g. red).'
                            ),
                        },
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
            },
            stream=True,
            timeout=180,
        )
        assert resp.status_code == 200, resp.text
        raw = ''.join(chunk.decode('utf-8') for chunk in resp.iter_content(chunk_size=None) if chunk)
        events = _parse_anthropic_sse(raw)
        types = [obj.get('type') for _, obj in events]
        assert 'message_start' in types
        assert 'message_stop' in types
        assembled = _aggregate_stream_text(events)
        assert len(assembled.strip()) > 0, repr(assembled[:300])
        al = assembled.lower()
        assert any(
            k in al
            for k in (
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
            )), f'expected red-ish color name in streamed reply: {assembled[:500]!r}'

    def test_messages_multi_turn(self, backend, model_case, deployed_model_name: str):
        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 2048,
                'temperature': 0.01,
                'messages': [
                    {'role': 'user', 'content': 'Remember the code word: banana.'},
                    {'role': 'assistant', 'content': 'Understood, the code word is banana.'},
                    {'role': 'user', 'content': 'What was the code word? Reply with that word only.'},
                ],
            },
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        text = _assistant_text_from_message_payload(data).lower()
        assert 'banana' in text, text[:500]

    def test_messages_max_tokens_budget(self, backend, model_case, deployed_model_name: str):
        """Tight ``max_tokens`` should cap generation (``stop_reason`` often
        ``max_tokens``)."""

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 6,
                'temperature': 0.01,
                'messages': [{
                    'role': 'user',
                    'content': 'Write a very long essay about world history. Do not stop early.',
                }],
            },
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        out = data['usage']['output_tokens']
        assert out <= 8
        assert out >= 4
        assert data['stop_reason'] in ('max_tokens', 'end_turn')
        assert _assistant_text_from_message_payload(data), data['content']

    def test_messages_stop_sequences(self, backend, model_case, deployed_model_name: str):
        """Maps to LMDeploy ``stop_sequences`` /
        ``GenerationConfig.stop_words`` (cf.

        chat completion stop tests).
        """

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 200,
                'temperature': 0.01,
                'stop_sequences': [' Shanghai', ' city', ' China'],
                'messages': [{'role': 'user', 'content': 'Shanghai is'}],
            },
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        text = _assistant_text_from_message_payload(data)
        assert ' Shanghai' not in text
        assert ' city' not in text
        assert ' China' not in text
        assert data['stop_reason'] in ('end_turn', 'max_tokens', 'stop_sequence')
        assert len(text) > 0, 'stop_sequence should still yield visible assistant text before the stop'

    def test_messages_non_stream(self, backend, model_case, deployed_model_name: str):
        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 32,
                'temperature': 0.01,
                'messages': [{'role': 'user', 'content': 'Reply with a single short greeting.'}],
            },
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        assert data['content'][0]['type'] == 'text'
        assert len(_assistant_text_from_message_payload(data).strip()) > 0

    def test_messages_stream(self, backend, model_case, deployed_model_name: str):
        """SSE lifecycle including ``message_start`` shape (usage zero until
        ``message_delta``)."""

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 48,
                'temperature': 0.01,
                'stream': True,
                'messages': [{'role': 'user', 'content': 'Count from 1 to 3, one number per line.'}],
            },
            stream=True,
            timeout=120,
        )
        assert resp.status_code == 200, resp.text
        raw = ''.join(chunk.decode('utf-8') for chunk in resp.iter_content(chunk_size=None) if chunk)
        events = _parse_anthropic_sse(raw)
        types = [obj.get('type') for _, obj in events]
        assert 'message_start' in types
        assert 'message_delta' in types
        assert 'message_stop' in types
        start_evt = next((obj for _, obj in events if obj.get('type') == 'message_start'), None)
        assert start_evt is not None
        m0 = start_evt['message']
        assert m0.get('type') == 'message'
        assert m0.get('role') == 'assistant'
        assert m0.get('model') == deployed_model_name
        assert isinstance(m0.get('id'), str) and m0['id'].startswith('msg_')
        assert m0.get('usage', {}).get('input_tokens') == 0
        assert m0.get('usage', {}).get('output_tokens') == 0
        assembled = _aggregate_stream_text(events)
        assert len(assembled) > 0
        assert sum(1 for d in ('1', '2', '3') if d in assembled) >= 2, (
            'expected at least two of the digits 1–3 in streamed text', repr(assembled[:200])
        )
        delta_evt = next((obj for _, obj in events if obj.get('type') == 'message_delta'), None)
        assert delta_evt is not None
        du = delta_evt['usage']
        assert 'output_tokens' in du and isinstance(du['output_tokens'], int)
        assert du['output_tokens'] > 0
        assert 'input_tokens' in du and isinstance(du['input_tokens'], int) and du['input_tokens'] >= 0
        assert any(obj.get('type') == 'message_stop' for _, obj in events)

    def test_count_tokens(self, backend, model_case, deployed_model_name: str):
        r_short = requests.post(
            _COUNT_TOKENS_URL,
            headers=_anthropic_headers(),
            json={'model': deployed_model_name, 'messages': [{'role': 'user', 'content': 'Hi'}]},
            timeout=60,
        )
        assert r_short.status_code == 200, r_short.text
        short = _assert_count_tokens_json(r_short.json())
        r_long = requests.post(
            _COUNT_TOKENS_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'messages': [{'role': 'user', 'content': 'Hello, estimate my token count.'}],
            },
            timeout=60,
        )
        assert r_long.status_code == 200, r_long.text
        long = _assert_count_tokens_json(r_long.json())
        assert long > short, (long, short)

    @pytest.mark.parametrize(
        'endpoint_url',
        [_MESSAGES_URL, _COUNT_TOKENS_URL],
        ids=['messages', 'count_tokens'],
    )
    def test_messages_and_count_tokens_invalid_json_body(
            self, backend, model_case, deployed_model_name: str, endpoint_url: str):
        resp = requests.post(
            endpoint_url,
            headers=_anthropic_headers(),
            data='{"model":',
            timeout=30,
        )
        _assert_fastapi_validation_error(resp)

    def test_count_tokens_rejects_tools(self, backend, model_case, deployed_model_name: str):
        """``count_tokens`` rejects Anthropic ``tools`` until supported (400 +
        fixed message)."""

        base_json = {
            'model': deployed_model_name,
            'messages': [{'role': 'user', 'content': 'Hi'}],
        }
        r_base = requests.post(
            _COUNT_TOKENS_URL,
            headers=_anthropic_headers(),
            json=base_json,
            timeout=30,
        )
        assert r_base.status_code == 200, r_base.text
        _assert_count_tokens_json(r_base.json())

        resp = requests.post(
            _COUNT_TOKENS_URL,
            headers=_anthropic_headers(),
            json={
                **base_json,
                'tools': [{
                    'name': 'demo',
                    'description': 'x',
                    'input_schema': {
                        'type': 'object',
                        'properties': {}
                    },
                }],
            },
            timeout=30,
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        _assert_anthropic_error_envelope(body)
        assert body['error']['type'] == 'invalid_request_error'
        assert body['error']['message'] == 'Anthropic tool fields are temporarily unsupported.'

    def test_count_tokens_with_system_content_blocks(self, backend, model_case, deployed_model_name: str):
        """``count_tokens`` with ``system`` as block list
        (``to_lmdeploy_messages`` flattens text)."""

        messages = [{'role': 'user', 'content': 'Hello, estimate my token count.'}]
        resp_base = requests.post(
            _COUNT_TOKENS_URL,
            headers=_anthropic_headers(),
            json={'model': deployed_model_name, 'messages': messages},
            timeout=60,
        )
        assert resp_base.status_code == 200, resp_base.text
        base_data = resp_base.json()
        assert set(base_data.keys()) == {'input_tokens'}, base_data
        baseline = base_data['input_tokens']
        assert isinstance(baseline, int) and baseline > 0

        resp = requests.post(
            _COUNT_TOKENS_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'system': [
                    {'type': 'text', 'text': 'You are helpful.'},
                    {'type': 'text', 'text': 'Answer briefly.'},
                ],
                'messages': messages,
            },
            timeout=60,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert set(data.keys()) == {'input_tokens'}, data
        with_system = data['input_tokens']
        assert isinstance(with_system, int)
        assert with_system > baseline, (
            'system blocks should increase tokenized prompt vs same messages alone',
            with_system,
            baseline,
        )

    def test_messages_wrong_content_type(self, backend, model_case, deployed_model_name: str):
        resp = requests.post(
            _MESSAGES_URL,
            headers={
                'Content-Type': 'text/plain',
                'anthropic-version': ANTHROPIC_VERSION,
            },
            data='{}',
            timeout=30,
        )
        _assert_fastapi_validation_error(resp)

    def test_messages_invalid_message_role(self, backend, model_case, deployed_model_name: str):
        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 8,
                'messages': [{'role': 'system', 'content': 'not allowed here'}],
            },
            timeout=30,
        )
        _assert_fastapi_validation_error(resp)

    def test_messages_message_missing_role(self, backend, model_case, deployed_model_name: str):
        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 8,
                'messages': [{'content': 'Hi'}],
            },
            timeout=30,
        )
        _assert_fastapi_validation_error(resp)

    def test_messages_max_tokens_zero(self, backend, model_case, deployed_model_name: str):
        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 0,
                'messages': [{'role': 'user', 'content': 'Hi'}],
            },
            timeout=30,
        )
        _assert_fastapi_validation_error(resp)

    def test_messages_messages_not_list(self, backend, model_case, deployed_model_name: str):
        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 8,
                'messages': {'role': 'user', 'content': 'Hi'},
            },
            timeout=30,
        )
        _assert_fastapi_validation_error(resp)

    def test_messages_stream_validation_error_returns_json(self, backend, model_case, deployed_model_name: str):
        """Invalid bodies must not upgrade to ``text/event-stream``; FastAPI
        returns JSON 422."""

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': -1,
                'stream': True,
                'messages': [{'role': 'user', 'content': 'Hi'}],
            },
            stream=True,
            timeout=30,
        )
        _assert_fastapi_validation_error(resp)
        ctype = (resp.headers.get('content-type') or '').lower()
        assert 'application/json' in ctype
        assert 'text/event-stream' not in ctype

    def test_count_tokens_empty_messages(self, backend, model_case, deployed_model_name: str):
        """Pydantic allows ``messages: []``; counting should still return a
        positive estimate."""

        resp = requests.post(
            _COUNT_TOKENS_URL,
            headers=_anthropic_headers(),
            json={'model': deployed_model_name, 'messages': []},
            timeout=60,
        )
        assert resp.status_code == 200, resp.text
        _assert_count_tokens_json(resp.json())

    def test_messages_large_user_payload(self, backend, model_case, deployed_model_name: str):
        """Regression guard for large JSON bodies (CI-sized payload, not
        stress-test scale)."""

        big = 'x' * (128 * 1024)
        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 8,
                'temperature': 0.01,
                'messages': [{'role': 'user', 'content': f'Reply with one word: OK. Context:\n{big}'}],
            },
            timeout=180,
        )
        assert resp.status_code == 200, resp.text
        data = _assert_success_message_json(resp.json(), model=deployed_model_name)
        assert len(_assistant_text_from_message_payload(data).strip()) > 0

    def test_messages_rejects_tools_without_tool_call_parser(self, backend, model_case, deployed_model_name: str):
        """``RESTFUL`` jobs start api_server *without* ``--tool-call-parser``;
        ``tools`` must yield 400."""

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 64,
                'temperature': 0,
                'messages': [{'role': 'user', 'content': 'What is the weather in Dallas, TX?'}],
                'tools': [openai_function_tool_to_anthropic(WEATHER_TOOL)],
            },
            timeout=120,
        )
        _assert_tool_parser_required_message(resp)

    def test_messages_rejects_tool_choice_with_tools_without_tool_call_parser(
            self, backend, model_case, deployed_model_name: str):
        """``tool_choice`` is only meaningful with ``tools``; still blocked
        without ``--tool-call-parser``."""

        resp = requests.post(
            _MESSAGES_URL,
            headers=_anthropic_headers(),
            json={
                'model': deployed_model_name,
                'max_tokens': 64,
                'temperature': 0,
                'messages': [{'role': 'user', 'content': 'What is the weather in Dallas, TX?'}],
                'tools': [openai_function_tool_to_anthropic(WEATHER_TOOL)],
                'tool_choice': {'type': 'auto'},
            },
            timeout=120,
        )
        _assert_tool_parser_required_message(resp)
