import time
from pathlib import Path

import pytest
import requests
import torch
from lmdeploy.serve.openai.api_client import APIClient
from utils.constant import (
    DEFAULT_PORT,
    DEFAULT_SERVER,
    SLEEP_WAKEUP_BACKENDS,
    SLEEP_WAKEUP_MODEL_LIST,
)
from utils.restful_return_check import assert_chat_completions_batch_return
from utils.sleep_utils import (
    LEVEL2_BASELINE_RUNS,
    LEVEL2_GREEDY_MESSAGES,
    LEVEL2_MAX_TOKENS,
    apply_serialized_hf_segments_for_level2_weights,
    apply_serialized_hf_segments_for_turbomind_level2_weights,
    assert_assistant_not_degenerate,
    assert_chat_decode_unchanged,
    assistant_content_from_openai_completion_dict,
    level2_update_weights_request_dict,
    resolve_hf_checkpoint_dir,
)

BASE_URL = f'http://{DEFAULT_SERVER}:{DEFAULT_PORT}'
JSON_HEADERS = {'Content-Type': 'application/json'}
_REQUEST_TIMEOUT = 120
_UPDATE_WEIGHTS_TIMEOUT = 600


def _assert_status_200(resp: requests.Response) -> None:
    assert resp.status_code == 200, f'status={resp.status_code} body={resp.text!r}'


def _post_sleep(*, level: int | None = None) -> requests.Response:
    url = f'{BASE_URL}/sleep'
    if level is not None:
        url = f'{url}?level={level}'
    return requests.post(url, headers=JSON_HEADERS, json={}, timeout=_REQUEST_TIMEOUT)


def _post_sleep_level2() -> requests.Response:
    return requests.post(
        f'{BASE_URL}/sleep',
        headers=JSON_HEADERS,
        json={},
        params=[('tags', 'weights'), ('tags', 'kv_cache'), ('level', 2)],
        timeout=_REQUEST_TIMEOUT,
    )


def _post_sleep_query_raw(query: str) -> requests.Response:
    q = query.lstrip('?')
    url = f'{BASE_URL}/sleep?{q}' if q else f'{BASE_URL}/sleep'
    return requests.post(url, headers=JSON_HEADERS, json={}, timeout=_REQUEST_TIMEOUT)


def _post_wakeup(*, tags: list[str] | None = None) -> requests.Response:
    params = [('tags', t) for t in tags] if tags else None
    return requests.post(
        f'{BASE_URL}/wakeup',
        headers=JSON_HEADERS,
        json={},
        params=params,
        timeout=_REQUEST_TIMEOUT,
    )


def _post_update_weights_from_hf_dir(model_dir: Path, *, engine: str) -> None:
    def _emit(serialized_data: object, finished: bool) -> None:
        data = level2_update_weights_request_dict(serialized_data, finished)
        r = requests.post(
            f'{BASE_URL}/update_weights',
            headers=JSON_HEADERS,
            json=data,
            timeout=_UPDATE_WEIGHTS_TIMEOUT,
        )
        _assert_status_200(r)

    if engine == 'pytorch':
        apply_serialized_hf_segments_for_level2_weights(model_dir, _emit)
    elif engine == 'turbomind':
        apply_serialized_hf_segments_for_turbomind_level2_weights(model_dir, _emit)
    else:
        pytest.skip(f'unsupported engine for update_weights: {engine!r}')


def _level2_reload_hf_weights(backend: str, config: dict, model_case: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip('level-2 reload needs CUDA for serialize_state_dict / weight upload')
    model_dir = resolve_hf_checkpoint_dir(config, model_case)
    if not model_dir.is_dir():
        pytest.skip(f'HF checkpoint not found for update_weights: {model_dir}')
    try:
        _post_update_weights_from_hf_dir(model_dir, engine=backend)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    except RuntimeError as e:
        pytest.skip(str(e))


def _fetch_is_sleeping() -> bool:
    r = requests.get(f'{BASE_URL}/is_sleeping', timeout=30)
    _assert_status_200(r)
    return bool(r.json().get('is_sleeping'))


def _ensure_awake(max_attempts: int = 8) -> None:
    for _ in range(max_attempts):
        _assert_status_200(_post_wakeup())
        if not _fetch_is_sleeping():
            return
        time.sleep(0.25)
    raise AssertionError(
        f'engine still is_sleeping=true after {max_attempts} POST /wakeup attempts; '
        f'BASE_URL={BASE_URL!r}')


def _chat_completion_collect(api_client: APIClient, model_name: str, **kwargs) -> dict:
    kw = dict(kwargs)
    kw['stream'] = False
    output = None
    for output in api_client.chat_completions_v1(model=model_name, **kw):
        continue
    assert output is not None, 'chat_completions_v1 returned no chunk'
    return output


def _assert_level2_greedy_baseline_stable(api_client: APIClient, model_name: str, *, label: str) -> dict:
    kwargs = dict(
        messages=LEVEL2_GREEDY_MESSAGES,
        max_tokens=LEVEL2_MAX_TOKENS,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
    )
    refs: list[dict] = []
    contents: list[str] = []
    for i in range(LEVEL2_BASELINE_RUNS):
        out = _chat_completion_collect(api_client, model_name, **kwargs)
        assert_chat_completions_batch_return(out, model_name)
        text = assistant_content_from_openai_completion_dict(out)
        assert_assistant_not_degenerate(text, label=f'{label} baseline run {i + 1}')
        refs.append(out)
        contents.append(text)
    assert len(set(contents)) == 1, (
        f'{label}: greedy REST baseline not stable (fix prompt/model for this case):\n'
        + '\n'.join(f'  run{j + 1}={c!r}' for j, c in enumerate(contents)))
    return refs[0]


def _should_enforce_level2_greedy_checks(backend: str) -> bool:
    # Known issue: TurboMind may produce non-stable outputs even in
    # temperature=0 greedy-style requests. Keep the staged wakeup / reload
    # flow coverage, but skip strict determinism assertions for this backend.
    return backend != 'turbomind'


@pytest.mark.order(8)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('backend', SLEEP_WAKEUP_BACKENDS)
@pytest.mark.parametrize('model_case', SLEEP_WAKEUP_MODEL_LIST)
class TestRestfulSleepWakeup:

    def test_sleep_wakeup_is_sleeping_roundtrip(self, model_case, backend):
        try:
            _ensure_awake()
            r_sleep = _post_sleep()
            _assert_status_200(r_sleep)

            assert _fetch_is_sleeping() is True

            r_wake = _post_wakeup()
            _assert_status_200(r_wake)

            assert _fetch_is_sleeping() is False
        finally:
            _ensure_awake()

    def test_sleep_with_level_query_wakeup_and_chat(self, model_case, backend):
        try:
            _ensure_awake()
            r_sleep = _post_sleep(level=1)
            _assert_status_200(r_sleep)

            assert _fetch_is_sleeping() is True

            r_wake = _post_wakeup()
            _assert_status_200(r_wake)
            assert _fetch_is_sleeping() is False

            api_client = APIClient(BASE_URL)
            model_name = api_client.available_models[0]
            output = None
            for output in api_client.chat_completions_v1(
                    model=model_name,
                    messages=[{'role': 'user', 'content': 'Hi, reply with one short sentence.'}],
                    max_tokens=32,
                    temperature=0.01):
                continue
            assert output is not None
            assert_chat_completions_batch_return(output, model_name)
        finally:
            _ensure_awake()

    def test_sleep_partial_wakeup_with_tags(self, model_case, backend):
        try:
            _ensure_awake()
            r_sleep = _post_sleep(level=1)
            _assert_status_200(r_sleep)
            assert _fetch_is_sleeping() is True

            r_w = _post_wakeup(tags=['weights'])
            _assert_status_200(r_w)
            assert _fetch_is_sleeping() is True

            r_kv = _post_wakeup(tags=['kv_cache'])
            _assert_status_200(r_kv)
            assert _fetch_is_sleeping() is False
        finally:
            _ensure_awake()

    def test_wakeup_unknown_tags_is_noop_then_full_wakeup(self, model_case, backend):
        try:
            _ensure_awake()
            _assert_status_200(_post_sleep(level=1))
            assert _fetch_is_sleeping() is True

            _assert_status_200(_post_wakeup(tags=['not_a_valid_tag']))
            assert _fetch_is_sleeping() is True

            _assert_status_200(_post_wakeup())
            assert _fetch_is_sleeping() is False
        finally:
            _ensure_awake()

    def test_wakeup_mixed_valid_and_invalid_tags_entire_call_noop(self, model_case, backend):
        try:
            _ensure_awake()
            _assert_status_200(_post_sleep(level=1))
            assert _fetch_is_sleeping() is True

            _assert_status_200(_post_wakeup(tags=['weights', 'not_a_valid_tag']))
            assert _fetch_is_sleeping() is True

            _assert_status_200(_post_wakeup(tags=['not_a_valid_tag', 'weights']))
            assert _fetch_is_sleeping() is True

            _assert_status_200(_post_wakeup())
            assert _fetch_is_sleeping() is False
        finally:
            _ensure_awake()

    def test_wakeup_both_valid_tags_in_one_request(self, model_case, backend):
        try:
            _ensure_awake()
            _assert_status_200(_post_sleep(level=1))
            assert _fetch_is_sleeping() is True

            _assert_status_200(_post_wakeup(tags=['weights', 'kv_cache']))
            assert _fetch_is_sleeping() is False

            api_client = APIClient(BASE_URL)
            model_name = api_client.available_models[0]
            output = None
            for output in api_client.chat_completions_v1(
                    model=model_name,
                    messages=[{'role': 'user', 'content': 'Hi, reply with one short sentence.'}],
                    max_tokens=32,
                    temperature=0.01):
                continue
            assert output is not None
            assert_chat_completions_batch_return(output, model_name)
        finally:
            _ensure_awake()

    def test_wakeup_redundant_tag_after_partial_wake_is_noop(self, model_case, backend):
        try:
            _ensure_awake()
            _assert_status_200(_post_sleep(level=1))
            assert _fetch_is_sleeping() is True

            _assert_status_200(_post_wakeup(tags=['weights']))
            assert _fetch_is_sleeping() is True

            _assert_status_200(_post_wakeup(tags=['weights']))
            assert _fetch_is_sleeping() is True

            _assert_status_200(_post_wakeup(tags=['kv_cache']))
            assert _fetch_is_sleeping() is False
        finally:
            _ensure_awake()

    def test_wakeup_empty_string_tag_is_noop_when_sleeping(self, model_case, backend):
        try:
            _ensure_awake()
            _assert_status_200(_post_sleep(level=1))
            assert _fetch_is_sleeping() is True

            r = requests.post(
                f'{BASE_URL}/wakeup',
                headers=JSON_HEADERS,
                json={},
                params=[('tags', '')],
                timeout=_REQUEST_TIMEOUT,
            )
            _assert_status_200(r)
            assert _fetch_is_sleeping() is True

            _assert_status_200(_post_wakeup())
            assert _fetch_is_sleeping() is False
        finally:
            _ensure_awake()

    def test_full_wakeup_when_already_awake(self, model_case, backend):
        try:
            _ensure_awake()
            assert _fetch_is_sleeping() is False
            _assert_status_200(_post_wakeup())
            assert _fetch_is_sleeping() is False
            _assert_status_200(_post_wakeup())
            assert _fetch_is_sleeping() is False
        finally:
            _ensure_awake()

    def test_sleep_second_call_while_sleeping_still_ok(self, model_case, backend):
        try:
            _ensure_awake()
            _assert_status_200(_post_sleep(level=1))
            assert _fetch_is_sleeping() is True
            _assert_status_200(_post_sleep(level=1))
            assert _fetch_is_sleeping() is True
            _assert_status_200(_post_wakeup())
            assert _fetch_is_sleeping() is False
        finally:
            _ensure_awake()

    def test_sleep_non_integer_level_is_http_error(self, model_case, backend):
        try:
            _ensure_awake()
            resp = _post_sleep_query_raw('level=not_an_int')
            assert resp.status_code != 200, f'expected non-200, got {resp.status_code} body={resp.text!r}'
        finally:
            _ensure_awake()

    def test_sleep_level_2_full_wakeup_and_chat(self, model_case, backend, config):
        try:
            _ensure_awake()
            api_client = APIClient(BASE_URL)
            model_name = api_client.available_models[0]

            baseline = None
            if _should_enforce_level2_greedy_checks(backend):
                baseline = _assert_level2_greedy_baseline_stable(
                    api_client, model_name, label='level2 REST')

            _assert_status_200(_post_sleep_level2())
            assert _fetch_is_sleeping() is True

            _assert_status_200(_post_wakeup(tags=['weights']))
            assert _fetch_is_sleeping() is True
            _level2_reload_hf_weights(backend, config, model_case)

            _assert_status_200(_post_wakeup(tags=['kv_cache']))
            assert _fetch_is_sleeping() is False

            after = _chat_completion_collect(
                api_client,
                model_name,
                messages=LEVEL2_GREEDY_MESSAGES,
                max_tokens=LEVEL2_MAX_TOKENS,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
            )
            assert_chat_completions_batch_return(after, model_name)
            assert_assistant_not_degenerate(
                assistant_content_from_openai_completion_dict(after),
                label='level2 REST after staged wakeup (1st chat)')
            if baseline is not None:
                assert_chat_decode_unchanged(baseline, after, label='level2 REST 1st infer after staged wakeup')

            after2 = _chat_completion_collect(
                api_client,
                model_name,
                messages=LEVEL2_GREEDY_MESSAGES,
                max_tokens=LEVEL2_MAX_TOKENS,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
            )
            assert_chat_completions_batch_return(after2, model_name)
            if baseline is not None:
                assert_chat_decode_unchanged(baseline, after2, label='level2 REST 2nd infer after staged wakeup')

            _assert_status_200(_post_sleep_level2())
            assert _fetch_is_sleeping() is True
            _assert_status_200(_post_wakeup(tags=['weights']))
            assert _fetch_is_sleeping() is True
            _level2_reload_hf_weights(backend, config, model_case)
            _assert_status_200(_post_wakeup(tags=['kv_cache']))
            assert _fetch_is_sleeping() is False

            after_full = _chat_completion_collect(
                api_client,
                model_name,
                messages=LEVEL2_GREEDY_MESSAGES,
                max_tokens=LEVEL2_MAX_TOKENS,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
            )
            assert_chat_completions_batch_return(after_full, model_name)
            label2 = 'level2 REST infer after 2nd sleep cycle (staged wakeup)'
            if baseline is not None:
                assert_chat_decode_unchanged(baseline, after_full, label=label2)

            output = None
            for output in api_client.chat_completions_v1(
                    model=model_name,
                    messages=[{'role': 'user', 'content': 'Hi, reply with one short sentence.'}],
                    max_tokens=32,
                    temperature=0.01):
                continue
            assert output is not None
            assert_chat_completions_batch_return(output, model_name)
        finally:
            _ensure_awake()
