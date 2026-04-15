import json
import random
import threading
import time

import pytest
import requests
from utils.constant import BACKEND_LIST, DEFAULT_PORT, DEFAULT_SERVER, RESTFUL_MODEL_LIST
from utils.restful_return_check import assert_chat_completions_batch_return

from lmdeploy.serve.openai.api_client import APIClient

BASE_URL = f'http://{DEFAULT_SERVER}:{DEFAULT_PORT}'
JSON_HEADERS = {'Content-Type': 'application/json'}
_REQUEST_TIMEOUT = 300
_ABORT_TIMEOUT = 60
_SESSION_RETRY = 25
_SESSION_RETRY_INTERVAL = 0.3
_NONSTREAM_ABORT_LEAD_S = 2.0
_THREAD_JOIN_EXTRA_S = 30


def _post_abort_request(payload: dict) -> requests.Response:
    return requests.post(
        f'{BASE_URL}/abort_request',
        headers=JSON_HEADERS,
        json=payload,
        timeout=_ABORT_TIMEOUT,
    )


def _chat_non_stream(model_name: str, session_id: int, *, max_tokens: int = 32) -> requests.Response:
    return requests.post(
        f'{BASE_URL}/v1/chat/completions',
        headers=JSON_HEADERS,
        json={
            'model': model_name,
            'messages': [{'role': 'user', 'content': 'Say OK in one word.'}],
            'max_tokens': max_tokens,
            'temperature': 0.01,
            'stream': False,
            'session_id': session_id,
        },
        timeout=_REQUEST_TIMEOUT,
    )


def _consume_first_nonempty_sse_data_line(resp: requests.Response) -> None:
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith('data:'):
            continue
        chunk = raw[5:].strip()
        if chunk == '[DONE]':
            break
        if not chunk:
            continue
        try:
            json.loads(chunk)
        except json.JSONDecodeError:
            continue
        return
    assert False, 'expected at least one parsable SSE data line before abort'


def _post_abort_explicit_session_or_skip(session_id: int) -> None:
    abort_r = _post_abort_request({'session_id': session_id, 'abort_all': False})
    if abort_r.status_code == 501:
        pytest.skip('api_server started without --enable-abort-handling')
    assert abort_r.status_code == 200, f'abort_request: {abort_r.status_code} {abort_r.text!r}'


def _post_abort_all_or_skip() -> None:
    abort_r = _post_abort_request({'abort_all': True})
    if abort_r.status_code == 501:
        pytest.skip('api_server started without --enable-abort-handling')
    assert abort_r.status_code == 200, f'abort_request abort_all: {abort_r.status_code} {abort_r.text!r}'


def _assert_session_reusable_after_abort(model_name: str, session_id: int) -> None:
    last = None
    for _ in range(_SESSION_RETRY):
        last = _chat_non_stream(model_name, session_id, max_tokens=16)
        if last.status_code == 200:
            data = last.json()
            assert 'choices' in data and data['choices'], last.text
            assert_chat_completions_batch_return(data, model_name)
            return
        if last.status_code == 400 and 'occupied' in (last.text or '').lower():
            time.sleep(_SESSION_RETRY_INTERVAL)
            continue
        break
    assert False, f'session {session_id} not reusable after abort: last={last.status_code} {last.text!r}'


def _long_user_prompt() -> str:
    return 'Write a long numbered list from 1 to 500, one number per line, no other text.'


@pytest.mark.order(9)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', RESTFUL_MODEL_LIST)
class TestRestfulAbortRequest:

    def test_abort_request_releases_explicit_session_mid_stream(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 8_000_000 + random.randint(0, 99_999)

        stream_payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': _long_user_prompt()}],
            'max_tokens': 2048,
            'temperature': 0.3,
            'stream': True,
            'session_id': session_id,
        }
        resp = requests.post(
            f'{BASE_URL}/v1/chat/completions',
            headers=JSON_HEADERS,
            json=stream_payload,
            stream=True,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        try:
            _consume_first_nonempty_sse_data_line(resp)
            _post_abort_explicit_session_or_skip(session_id)
        finally:
            resp.close()

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_request_releases_explicit_session_mid_stream_generate(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 7_000_000 + random.randint(0, 99_999)

        stream_payload = {
            'prompt': _long_user_prompt(),
            'max_tokens': 2048,
            'temperature': 0.3,
            'stream': True,
            'session_id': session_id,
        }
        resp = requests.post(
            f'{BASE_URL}/generate',
            headers=JSON_HEADERS,
            json=stream_payload,
            stream=True,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        try:
            _consume_first_nonempty_sse_data_line(resp)
            _post_abort_explicit_session_or_skip(session_id)
        finally:
            resp.close()

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_request_releases_explicit_session_mid_stream_completions(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 6_000_000 + random.randint(0, 99_999)

        stream_payload = {
            'model': model_name,
            'prompt': _long_user_prompt(),
            'max_tokens': 2048,
            'temperature': 0.3,
            'stream': True,
            'session_id': session_id,
        }
        resp = requests.post(
            f'{BASE_URL}/v1/completions',
            headers=JSON_HEADERS,
            json=stream_payload,
            stream=True,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        try:
            _consume_first_nonempty_sse_data_line(resp)
            _post_abort_explicit_session_or_skip(session_id)
        finally:
            resp.close()

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_request_releases_explicit_session_non_stream_chat_thread(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 5_000_000 + random.randint(0, 99_999)

        def worker(out: dict) -> None:
            try:
                out['resp'] = requests.post(
                    f'{BASE_URL}/v1/chat/completions',
                    headers=JSON_HEADERS,
                    json={
                        'model': model_name,
                        'messages': [{'role': 'user', 'content': _long_user_prompt()}],
                        'max_tokens': 2048,
                        'temperature': 0.3,
                        'stream': False,
                        'session_id': session_id,
                    },
                    timeout=_REQUEST_TIMEOUT,
                )
            except Exception as e:
                out['exc'] = e

        holder: dict = {}
        t = threading.Thread(target=worker, args=(holder,), daemon=True)
        t.start()
        time.sleep(_NONSTREAM_ABORT_LEAD_S)
        abort_r = _post_abort_request({'session_id': session_id, 'abort_all': False})
        if abort_r.status_code == 501:
            t.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)
            pytest.skip('api_server started without --enable-abort-handling')
        assert abort_r.status_code == 200, f'abort_request: {abort_r.status_code} {abort_r.text!r}'
        t.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)
        assert not t.is_alive(), 'non-stream chat thread should finish after abort'

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_request_releases_explicit_session_non_stream_generate_thread(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 4_000_000 + random.randint(0, 99_999)

        def worker(out: dict) -> None:
            try:
                out['resp'] = requests.post(
                    f'{BASE_URL}/generate',
                    headers=JSON_HEADERS,
                    json={
                        'prompt': _long_user_prompt(),
                        'max_tokens': 2048,
                        'temperature': 0.3,
                        'stream': False,
                        'session_id': session_id,
                    },
                    timeout=_REQUEST_TIMEOUT,
                )
            except Exception as e:
                out['exc'] = e

        holder: dict = {}
        t = threading.Thread(target=worker, args=(holder,), daemon=True)
        t.start()
        time.sleep(_NONSTREAM_ABORT_LEAD_S)
        abort_r = _post_abort_request({'session_id': session_id, 'abort_all': False})
        if abort_r.status_code == 501:
            t.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)
            pytest.skip('api_server started without --enable-abort-handling')
        assert abort_r.status_code == 200, f'abort_request: {abort_r.status_code} {abort_r.text!r}'
        t.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)
        assert not t.is_alive(), 'non-stream generate thread should finish after abort'

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_request_releases_explicit_session_non_stream_completions_thread(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 3_000_000 + random.randint(0, 99_999)

        def worker(out: dict) -> None:
            try:
                out['resp'] = requests.post(
                    f'{BASE_URL}/v1/completions',
                    headers=JSON_HEADERS,
                    json={
                        'model': model_name,
                        'prompt': _long_user_prompt(),
                        'max_tokens': 2048,
                        'temperature': 0.3,
                        'stream': False,
                        'session_id': session_id,
                    },
                    timeout=_REQUEST_TIMEOUT,
                )
            except Exception as e:
                out['exc'] = e

        holder: dict = {}
        t = threading.Thread(target=worker, args=(holder,), daemon=True)
        t.start()
        time.sleep(_NONSTREAM_ABORT_LEAD_S)
        abort_r = _post_abort_request({'session_id': session_id, 'abort_all': False})
        if abort_r.status_code == 501:
            t.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)
            pytest.skip('api_server started without --enable-abort-handling')
        assert abort_r.status_code == 200, f'abort_request: {abort_r.status_code} {abort_r.text!r}'
        t.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)
        assert not t.is_alive(), 'non-stream completions thread should finish after abort'

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_streaming_client_close_releases_session_without_abort_request(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 2_000_000 + random.randint(0, 99_999)

        resp = requests.post(
            f'{BASE_URL}/v1/chat/completions',
            headers=JSON_HEADERS,
            json={
                'model': model_name,
                'messages': [{'role': 'user', 'content': _long_user_prompt()}],
                'max_tokens': 2048,
                'temperature': 0.3,
                'stream': True,
                'session_id': session_id,
            },
            stream=True,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        try:
            _consume_first_nonempty_sse_data_line(resp)
        finally:
            resp.close()

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_streaming_client_close_completions_releases_session(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 1_000_000 + random.randint(0, 99_999)

        resp = requests.post(
            f'{BASE_URL}/v1/completions',
            headers=JSON_HEADERS,
            json={
                'model': model_name,
                'prompt': _long_user_prompt(),
                'max_tokens': 2048,
                'temperature': 0.3,
                'stream': True,
                'session_id': session_id,
            },
            stream=True,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        try:
            _consume_first_nonempty_sse_data_line(resp)
        finally:
            resp.close()

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_streaming_client_close_generate_releases_session(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 500_000 + random.randint(0, 99_999)

        resp = requests.post(
            f'{BASE_URL}/generate',
            headers=JSON_HEADERS,
            json={
                'prompt': _long_user_prompt(),
                'max_tokens': 2048,
                'temperature': 0.3,
                'stream': True,
                'session_id': session_id,
            },
            stream=True,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        try:
            _consume_first_nonempty_sse_data_line(resp)
        finally:
            resp.close()

        _assert_session_reusable_after_abort(model_name, session_id)


@pytest.mark.order(10)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', RESTFUL_MODEL_LIST)
class TestRestfulAbortRequestAbortAll:
    def test_abort_request_abort_all_then_chat_ok(self, backend, model_case):
        _post_abort_all_or_skip()
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        last = None
        for out in api_client.chat_completions_v1(
                model=model_name,
                messages=[{'role': 'user', 'content': 'Reply with one word: OK'}],
                max_tokens=16,
                temperature=0.01,
                stream=False):
            last = out
        assert last is not None
        assert_chat_completions_batch_return(last, model_name)
