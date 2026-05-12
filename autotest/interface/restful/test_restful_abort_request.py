import json
import os
import random
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime

import allure
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
_POST_ABORT_LOGPROBS_NUM = 10
_MAX_LOG_TEXT = 2000
_LOG_HOOK: Callable[[dict], None] | None = None


def _set_log_hook(hook: Callable[[dict], None] | None) -> None:
    global _LOG_HOOK
    _LOG_HOOK = hook


def _truncate_text(text: str, limit: int = _MAX_LOG_TEXT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f'...<truncated {len(text) - limit} chars>'


def _response_snapshot(resp: requests.Response) -> dict:
    snap = {'status_code': resp.status_code}
    try:
        snap['json'] = resp.json()
    except Exception:
        snap['text'] = _truncate_text(resp.text or '')
    return snap


def _emit_log(event: str, **kwargs) -> None:
    if _LOG_HOOK is None:
        return
    payload = {'timestamp': datetime.now().isoformat(), 'event': event, **kwargs}
    try:
        _LOG_HOOK(payload)
    except Exception:
        # Logging should never break test assertions.
        pass


def _post_abort_request(payload: dict) -> requests.Response:
    resp = requests.post(
        f'{BASE_URL}/abort_request',
        headers=JSON_HEADERS,
        json=payload,
        timeout=_ABORT_TIMEOUT,
    )
    _emit_log('post_abort_request', request={'payload': payload}, response=_response_snapshot(resp))
    return resp


def _chat_non_stream(
        model_name: str,
        session_id: int,
        *,
        max_tokens: int = 32,
        logprobs: bool = False,
        top_logprobs: int = _POST_ABORT_LOGPROBS_NUM,
) -> requests.Response:
    body: dict = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': 'Say OK in one word.'}],
        'max_tokens': max_tokens,
        'temperature': 0.01,
        'stream': False,
        'session_id': session_id,
    }
    if logprobs:
        body['logprobs'] = True
        body['top_logprobs'] = top_logprobs
    resp = requests.post(
        f'{BASE_URL}/v1/chat/completions',
        headers=JSON_HEADERS,
        json=body,
        timeout=_REQUEST_TIMEOUT,
    )
    _emit_log('chat_non_stream', request={'payload': body}, response=_response_snapshot(resp))
    return resp


def _consume_first_nonempty_sse_data_line(resp: requests.Response) -> None:
    idx = 0
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith('data:'):
            continue
        chunk = raw[5:].strip()
        if chunk == '[DONE]':
            _emit_log('stream_chunk_done_before_abort')
            break
        if not chunk:
            continue
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            _emit_log('stream_chunk_parse_error', raw_preview=_truncate_text(chunk))
            continue
        idx += 1
        _emit_log('stream_chunk_before_abort',
                  chunk_index=idx,
                  chunk_preview=_truncate_text(chunk),
                  finish_reason=(parsed.get('choices') or [{}])[0].get('finish_reason')
                  if isinstance(parsed, dict) else None,
                  meta_finish_reason=(parsed.get('meta_info') or {}).get('finish_reason')
                  if isinstance(parsed, dict) else None)
        return
    assert False, 'expected at least one parsable SSE data line before abort'


def _post_abort_explicit_session_or_skip(session_id: int) -> None:
    abort_r = _post_abort_request({'session_id': session_id, 'abort_all': False})
    if abort_r.status_code == 501:
        pytest.skip('api_server started without --enable-abort-handling')
    assert abort_r.status_code == 200, f'abort_request: {abort_r.status_code} {abort_r.text!r}'


def _post_abort_nonexistent_session(session_id: int) -> requests.Response:
    return _post_abort_request({'session_id': session_id, 'abort_all': False})


def _post_abort_all_or_skip() -> None:
    abort_r = _post_abort_request({'abort_all': True})
    if abort_r.status_code == 501:
        pytest.skip('api_server started without --enable-abort-handling')
    assert abort_r.status_code == 200, f'abort_request abort_all: {abort_r.status_code} {abort_r.text!r}'


def _assert_session_reusable_after_abort(model_name: str, session_id: int) -> None:
    last = None
    for _ in range(_SESSION_RETRY):
        last = _chat_non_stream(
            model_name,
            session_id,
            max_tokens=16,
            logprobs=True,
            top_logprobs=_POST_ABORT_LOGPROBS_NUM,
        )
        if last.status_code == 200:
            data = last.json()
            assert 'choices' in data and data['choices'], last.text
            assert_chat_completions_batch_return(
                data,
                model_name,
                check_logprobs=True,
                logprobs_num=_POST_ABORT_LOGPROBS_NUM,
            )
            return
        if last.status_code == 400 and 'occupied' in (last.text or '').lower():
            time.sleep(_SESSION_RETRY_INTERVAL)
            continue
        break
    assert False, f'session {session_id} not reusable after abort: last={last.status_code} {last.text!r}'


def _long_user_prompt() -> str:
    return 'Write a long numbered list from 1 to 500, one number per line, no other text.'


def _finish_reason_indicates_abort(finish_reason) -> bool:
    """LMDeploy may use OpenAI-style ``'abort'`` or nested ``{'type':
    'abort'}``."""
    if finish_reason == 'abort':
        return True
    if isinstance(finish_reason, dict) and finish_reason.get('type') == 'abort':
        return True
    return False


def _sse_chunk_indicates_abort(chunk: dict) -> bool:
    choices = chunk.get('choices') or []
    if choices:
        if _finish_reason_indicates_abort(choices[0].get('finish_reason')):
            return True
    meta = chunk.get('meta_info') or {}
    return _finish_reason_indicates_abort(meta.get('finish_reason'))


def _verify_stream_abort_finish_reason(resp: requests.Response) -> None:
    found_abort = False
    chunk_idx = 0
    chunk_summaries: list[dict] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith('data:'):
            continue
        chunk_str = raw[5:].strip()
        if chunk_str == '[DONE]':
            _emit_log('stream_chunk_done_after_abort', seen_chunks=chunk_idx)
            break
        if chunk_str:
            try:
                chunk = json.loads(chunk_str)
            except json.JSONDecodeError:
                _emit_log('stream_chunk_parse_error_after_abort', raw_preview=_truncate_text(chunk_str))
                continue
            chunk_idx += 1
            choice_fr = None
            choices = chunk.get('choices') or []
            if choices and isinstance(choices[0], dict):
                choice_fr = choices[0].get('finish_reason')
            meta_fr = (chunk.get('meta_info') or {}).get('finish_reason')
            summary = {
                'idx': chunk_idx,
                'choice_finish_reason': choice_fr,
                'meta_finish_reason': meta_fr,
                'preview': _truncate_text(chunk_str),
            }
            chunk_summaries.append(summary)
            _emit_log('stream_chunk_after_abort', **summary)
            if _sse_chunk_indicates_abort(chunk):
                found_abort = True
                break
    if not found_abort:
        _emit_log('stream_abort_not_found',
                  seen_chunks=chunk_idx,
                  chunk_summaries=chunk_summaries[-10:])
    assert found_abort, "Expected finish_reason 'abort' in stream response"


def _verify_non_stream_abort_finish_reason(resp: requests.Response) -> None:
    data = resp.json()
    if 'choices' in data and data['choices']:
        finish_reason = data['choices'][0].get('finish_reason')
        assert _finish_reason_indicates_abort(finish_reason), (
            f'Expected abort finish_reason, got {finish_reason!r}')
        return
    # Legacy ``/generate`` body: ``text`` + ``meta_info.finish_reason``
    meta = data.get('meta_info') or {}
    fr = meta.get('finish_reason')
    assert _finish_reason_indicates_abort(fr), (
        f'Expected abort in meta_info.finish_reason, got {fr!r}; keys={list(data)!r}')


def _send_nonstream_request_with_abort(model_name: str, session_id: int, endpoint: str) -> requests.Response:
    payload: dict
    if endpoint == '/v1/chat/completions':
        payload = {
            'model': model_name,
            'messages': [{'role': 'user', 'content': _long_user_prompt()}],
            'max_tokens': 2048,
            'temperature': 0.3,
            'stream': False,
            'session_id': session_id,
        }
    elif endpoint == '/generate':
        payload = {
            'prompt': _long_user_prompt(),
            'max_tokens': 2048,
            'temperature': 0.3,
            'stream': False,
            'session_id': session_id,
        }
    else:
        payload = {
            'model': model_name,
            'prompt': _long_user_prompt(),
            'max_tokens': 2048,
            'temperature': 0.3,
            'stream': False,
            'session_id': session_id,
        }
    resp = requests.post(
        f'{BASE_URL}{endpoint}',
        headers=JSON_HEADERS,
        json=payload,
        timeout=_REQUEST_TIMEOUT,
    )
    _emit_log('send_nonstream_request_with_abort',
              request={'endpoint': endpoint, 'payload': payload},
              response=_response_snapshot(resp))
    return resp


@pytest.mark.order(9)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', RESTFUL_MODEL_LIST)
class TestRestfulAbortRequest:

    @pytest.fixture(autouse=True)
    def setup_abort_log(self, request, config, backend, model_case):
        test_name = re.sub(r'[^\w\.-]', '_', request.node.name)
        model_name = str(model_case).replace('/', '_')
        log_base = config.get('log_path', './logs')
        log_dir = os.path.join(log_base, model_name)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'restful_abort_{backend}_{test_name}_{timestamp}.log')

        def _writer(entry: dict) -> None:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        _set_log_hook(_writer)
        _emit_log('test_start', test=test_name, backend=backend, model_case=model_case, base_url=BASE_URL)
        yield
        _emit_log('test_end', test=test_name)
        _set_log_hook(None)
        if os.path.isfile(self.log_file):
            allure.attach.file(self.log_file, name=os.path.basename(self.log_file), attachment_type=allure.attachment_type.TEXT)

    def test_abort_running_stream_chat_request_returns_abort_finish_reason(self, backend, model_case):
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
            _verify_stream_abort_finish_reason(resp)
        finally:
            resp.close()

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_running_stream_generate_request_returns_abort_finish_reason(self, backend, model_case):
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
            _verify_stream_abort_finish_reason(resp)
        finally:
            resp.close()

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_running_stream_completions_request_returns_abort_finish_reason(self, backend, model_case):
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
            _verify_stream_abort_finish_reason(resp)
        finally:
            resp.close()

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_running_non_stream_chat_request_returns_abort_finish_reason(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 5_000_000 + random.randint(0, 99_999)

        results = {'resp': None, 'exc': None, 'completed': False}

        def worker(out: dict, sid: int) -> None:
            try:
                out['resp'] = _send_nonstream_request_with_abort(model_name, sid, '/v1/chat/completions')
                out['completed'] = True
            except Exception as e:
                out['exc'] = e

        thread = threading.Thread(target=worker, args=(results, session_id), daemon=True)
        thread.start()

        time.sleep(_NONSTREAM_ABORT_LEAD_S)

        abort_r = _post_abort_request({'session_id': session_id, 'abort_all': False})
        if abort_r.status_code == 501:
            thread.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)
            pytest.skip('api_server started without --enable-abort-handling')

        assert abort_r.status_code == 200
        thread.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)

        assert not thread.is_alive()
        assert results['resp'] is not None, 'Request should complete even after abort'
        _verify_non_stream_abort_finish_reason(results['resp'])

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_running_non_stream_generate_request_returns_abort_finish_reason(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 5_000_001 + random.randint(0, 99_999)

        results = {'resp': None, 'exc': None, 'completed': False}

        def worker(out: dict, sid: int) -> None:
            try:
                out['resp'] = _send_nonstream_request_with_abort(model_name, sid, '/generate')
                out['completed'] = True
            except Exception as e:
                out['exc'] = e

        thread = threading.Thread(target=worker, args=(results, session_id), daemon=True)
        thread.start()

        time.sleep(_NONSTREAM_ABORT_LEAD_S)

        abort_r = _post_abort_request({'session_id': session_id, 'abort_all': False})
        if abort_r.status_code == 501:
            thread.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)
            pytest.skip('api_server started without --enable-abort-handling')

        assert abort_r.status_code == 200
        thread.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)

        assert not thread.is_alive()
        assert results['resp'] is not None
        _verify_non_stream_abort_finish_reason(results['resp'])

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_running_non_stream_completions_request_returns_abort_finish_reason(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 5_000_002 + random.randint(0, 99_999)

        results = {'resp': None, 'exc': None, 'completed': False}

        def worker(out: dict, sid: int) -> None:
            try:
                out['resp'] = _send_nonstream_request_with_abort(model_name, sid, '/v1/completions')
                out['completed'] = True
            except Exception as e:
                out['exc'] = e

        thread = threading.Thread(target=worker, args=(results, session_id), daemon=True)
        thread.start()

        time.sleep(_NONSTREAM_ABORT_LEAD_S)

        abort_r = _post_abort_request({'session_id': session_id, 'abort_all': False})
        if abort_r.status_code == 501:
            thread.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)
            pytest.skip('api_server started without --enable-abort-handling')

        assert abort_r.status_code == 200
        thread.join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)

        assert not thread.is_alive()
        assert results['resp'] is not None
        _verify_non_stream_abort_finish_reason(results['resp'])

        _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_nonexistent_session_returns_bad_request(self, backend, model_case):
        nonexistent_session_id = 999_999_999

        abort_r = _post_abort_nonexistent_session(nonexistent_session_id)

        if abort_r.status_code == 501:
            pytest.skip('api_server started without --enable-abort-handling')

        assert abort_r.status_code == 400
        error_data = abort_r.json()
        assert 'error' in error_data or 'message' in error_data

    def test_abort_invalid_session_id_format_returns_bad_request(self, backend, model_case):
        invalid_session_ids = [-1, 'invalid', None, 3.14]

        for invalid_id in invalid_session_ids:
            abort_r = _post_abort_request({'session_id': invalid_id, 'abort_all': False})

            if abort_r.status_code == 501:
                pytest.skip('api_server started without --enable-abort-handling')

            assert abort_r.status_code in (400, 422), (
                f'expected 400 or 422 for invalid session_id, got {abort_r.status_code}')

    def test_abort_all_terminates_multiple_requests_with_abort_finish_reason(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]

        sessions = [5_000_000 + random.randint(0, 99_999) for _ in range(3)]
        responses = []

        for session_id in sessions:
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
            responses.append(resp)
            _consume_first_nonempty_sse_data_line(resp)

        _post_abort_all_or_skip()

        for i, resp in enumerate(responses):
            try:
                _verify_stream_abort_finish_reason(resp)
            finally:
                resp.close()

        for session_id in sessions:
            _assert_session_reusable_after_abort(model_name, session_id)

    def test_abort_all_terminates_non_stream_requests_with_abort_finish_reason(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        sessions = [3_000_000 + random.randint(0, 99_999) for _ in range(2)]
        results_list = []

        for session_id in sessions:
            results = {'resp': None, 'exc': None, 'completed': False}
            results_list.append(results)

            def worker(out: dict, sid: int):
                try:
                    out['resp'] = _send_nonstream_request_with_abort(model_name, sid, '/v1/chat/completions')
                    out['completed'] = True
                except Exception as e:
                    out['exc'] = e

            thread = threading.Thread(target=worker, args=(results, session_id), daemon=True)
            thread.start()
            results['thread'] = thread

        time.sleep(_NONSTREAM_ABORT_LEAD_S)

        _post_abort_all_or_skip()

        for results in results_list:
            results['thread'].join(timeout=_REQUEST_TIMEOUT + _THREAD_JOIN_EXTRA_S)
            assert not results['thread'].is_alive()
            assert results['resp'] is not None
            _verify_non_stream_abort_finish_reason(results['resp'])

        for session_id in sessions:
            _assert_session_reusable_after_abort(model_name, session_id)

    def test_session_immediately_reusable_after_abort(self, backend, model_case):
        api_client = APIClient(BASE_URL)
        model_name = api_client.available_models[0]
        session_id = 4_000_000 + random.randint(0, 99_999)

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

        new_resp = _chat_non_stream(model_name, session_id, max_tokens=16)
        assert new_resp.status_code == 200, f'Session should be immediately reusable, got {new_resp.status_code}'
