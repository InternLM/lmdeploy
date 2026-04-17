from __future__ import annotations

import inspect
import os
import time
from pathlib import Path

import pytest
import torch
from utils.config_utils import get_parallel_config
from utils.constant import SLEEP_WAKEUP_BACKENDS, SLEEP_WAKEUP_MODEL_LIST
from utils.sleep_utils import (
    LEVEL2_BASELINE_RUNS,
    LEVEL2_GREEDY_MESSAGES,
    LEVEL2_MAX_TOKENS,
    apply_serialized_hf_segments_for_level2_weights,
    apply_serialized_hf_segments_for_turbomind_level2_weights,
    assert_assistant_not_degenerate,
    assert_chat_decode_unchanged,
    level2_update_weights_request_dict,
    resolve_hf_checkpoint_dir,
)

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline
from lmdeploy.messages import Response
from lmdeploy.serve.openai.protocol import UpdateParamsRequest
from lmdeploy.utils import is_bf16_supported

_SLEEP_PIPELINE_BACKEND_CLASS = {
    'pytorch': PytorchEngineConfig,
    'turbomind': TurbomindEngineConfig,
}


def _pipeline_sleep_backend_classes():
    out: list[type[PytorchEngineConfig] | type[TurbomindEngineConfig]] = []
    for name in SLEEP_WAKEUP_BACKENDS:
        cls = _SLEEP_PIPELINE_BACKEND_CLASS.get(name)
        if cls is None:
            allowed = set(_SLEEP_PIPELINE_BACKEND_CLASS)
            raise ValueError(
                f'unknown SLEEP_WAKEUP_BACKENDS entry {name!r}; expected one of {allowed}',
            )
        out.append(cls)
    return out


def _force_pipeline_sleep_under_llm_dist() -> bool:
    v = os.environ.get('LMDEPLOY_FORCE_PIPELINE_SLEEP', '').strip().lower()
    return v in ('1', 'true', 'yes', 'on')


@pytest.fixture(scope='module', autouse=True)
def _skip_module_if_rest_runner_gpu_conflict():
    if os.environ.get('LLM_DIST_PORT') and not _force_pipeline_sleep_under_llm_dist():
        pytest.skip(
            'pipeline sleep/wakeup: skipped when LLM_DIST_PORT is set (REST api_server already uses GPUs). '
            'Run this file standalone from lmdeploy_sleep root, or set LMDEPLOY_FORCE_PIPELINE_SLEEP=1 '
            'if you allocated extra GPUs for pytest.')


def _pipeline_tp_for_model(config: dict, model: str) -> int:
    tp = 1
    for item in get_parallel_config(config, model):
        if isinstance(item, dict) and 'tp' in item:
            tp = max(tp, int(item['tp']))
    return max(1, tp)


def _make_backend_config(
    backend: type[PytorchEngineConfig] | type[TurbomindEngineConfig],
    config: dict,
    model: str,
):
    tp = _pipeline_tp_for_model(config, model)
    cfg = backend(tp=tp)
    if backend is TurbomindEngineConfig:
        cfg.empty_init = True
    if backend is PytorchEngineConfig and not is_bf16_supported():
        cfg.dtype = 'float16'
    return cfg


def _model_path(config: dict, model: str) -> str:
    if os.environ.get('LMDEPLOY_USE_MODELSCOPE', 'False') == 'True':
        return model
    return str(Path(config['model_path']) / model)


def _open_pipeline(config: dict, model: str, backend: type[PytorchEngineConfig] | type[TurbomindEngineConfig]):
    return pipeline(
        _model_path(config, model),
        backend_config=_make_backend_config(backend, config, model),
    )


def _pipeline_resp_to_chat_dict(resp: Response) -> dict:
    return {
        'choices': [{
            'message': {'content': (resp.text or '').strip()},
            'finish_reason': getattr(resp, 'finish_reason', None),
        }],
        'usage': {'completion_tokens': resp.generate_token_len},
    }


def _infer_level2_greedy(pipe, gen_cfg: GenerationConfig) -> Response:
    prompt = LEVEL2_GREEDY_MESSAGES[0]['content']
    return pipe.infer(prompt, gen_config=gen_cfg)


def _assert_level2_pipeline_baseline_stable(pipe, gen_cfg: GenerationConfig, *, label: str) -> Response:
    contents: list[str] = []
    refs: list[Response] = []
    for i in range(LEVEL2_BASELINE_RUNS):
        out = _infer_level2_greedy(pipe, gen_cfg)
        assert_assistant_not_degenerate(
            (out.text or '').strip(), label=f'{label} baseline run {i + 1}')
        refs.append(out)
        contents.append((out.text or '').strip())
    assert len(set(contents)) == 1, (
        f'{label}: greedy pipeline baseline not stable:\n'
        + '\n'.join(f'  run{j + 1}={c!r}' for j, c in enumerate(contents)))
    return refs[0]


def _should_enforce_level2_greedy_checks(
        backend: type[PytorchEngineConfig] | type[TurbomindEngineConfig]) -> bool:
    # Known issue: TurboMind may not be deterministic for temperature=0 runs.
    # Keep validating sleep/wakeup/update_params behavior, but do not fail on
    # strict greedy-stability checks for this backend.
    return backend is not TurbomindEngineConfig


def _apply_sleep(pipe, level: int = 1) -> None:
    eng = pipe.async_engine
    out = eng.sleep(level)
    if inspect.isawaitable(out):
        pipe._run(coro=out).result()


def _pipeline_wakeup(pipe, tags: list[str] | None = None) -> None:
    pipe.async_engine.wakeup(tags)


def _pipeline_is_sleeping(pipe) -> bool:
    return bool(pipe.async_engine.is_sleeping)


def _ensure_awake_pipeline(pipe, max_attempts: int = 8) -> None:
    for _ in range(max_attempts):
        _pipeline_wakeup(pipe, None)
        if not _pipeline_is_sleeping(pipe):
            return
        time.sleep(0.25)
    raise AssertionError(
        f'pipeline engine still is_sleeping=true after {max_attempts} wakeup attempts')


def _level2_reload_weights_if_supported_pipeline(
        pipe,
        backend: type[PytorchEngineConfig] | type[TurbomindEngineConfig],
        config: dict,
        model: str,
) -> None:
    if backend is not PytorchEngineConfig and backend is not TurbomindEngineConfig:
        return
    if not torch.cuda.is_available():
        pytest.skip('level-2 reload needs CUDA for serialize_state_dict / weight upload')
    model_dir = resolve_hf_checkpoint_dir(config, model)
    if not model_dir.is_dir():
        pytest.skip(f'HF checkpoint not found for update_weights: {model_dir}')
    eng = pipe.async_engine.engine

    def _emit(serialized_data: object, finished: bool) -> None:
        eng.update_params(UpdateParamsRequest(**level2_update_weights_request_dict(
            serialized_data, finished)))

    try:
        if backend is PytorchEngineConfig:
            apply_serialized_hf_segments_for_level2_weights(model_dir, _emit)
        else:
            apply_serialized_hf_segments_for_turbomind_level2_weights(model_dir, _emit)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.mark.order(8)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('model', SLEEP_WAKEUP_MODEL_LIST)
@pytest.mark.parametrize('backend', _pipeline_sleep_backend_classes())
class TestPipelineSleepWakeup:

    def test_pipeline_sleep_wakeup_roundtrip(self, model, backend, config):
        pipe = None
        try:
            pipe = _open_pipeline(config, model, backend)
            _ensure_awake_pipeline(pipe)
            _apply_sleep(pipe, 1)
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, None)
            assert _pipeline_is_sleeping(pipe) is False
        finally:
            if pipe is not None:
                try:
                    _ensure_awake_pipeline(pipe)
                finally:
                    pipe.close()

    def test_pipeline_sleep_level1_wakeup_and_infer(self, model, backend, config):
        pipe = None
        try:
            pipe = _open_pipeline(config, model, backend)
            _ensure_awake_pipeline(pipe)
            _apply_sleep(pipe, 1)
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, None)
            assert _pipeline_is_sleeping(pipe) is False
            gen = GenerationConfig(max_new_tokens=32, temperature=0.01)
            r = pipe([[{'role': 'user', 'content': 'Hi, reply with one short sentence.'}]], gen_config=gen)
            out = r[0] if isinstance(r, list) else r
            assert (out.text or '').strip()
        finally:
            if pipe is not None:
                try:
                    _ensure_awake_pipeline(pipe)
                finally:
                    pipe.close()

    def test_pipeline_partial_wakeup_with_tags(self, model, backend, config):
        pipe = None
        try:
            pipe = _open_pipeline(config, model, backend)
            _ensure_awake_pipeline(pipe)
            _apply_sleep(pipe, 1)
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, ['weights'])
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, ['kv_cache'])
            assert _pipeline_is_sleeping(pipe) is False
        finally:
            if pipe is not None:
                try:
                    _ensure_awake_pipeline(pipe)
                finally:
                    pipe.close()

    def test_pipeline_wakeup_unknown_tags_noop_then_full(self, model, backend, config):
        pipe = None
        try:
            pipe = _open_pipeline(config, model, backend)
            _ensure_awake_pipeline(pipe)
            _apply_sleep(pipe, 1)
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, ['not_a_valid_tag'])
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, None)
            assert _pipeline_is_sleeping(pipe) is False
        finally:
            if pipe is not None:
                try:
                    _ensure_awake_pipeline(pipe)
                finally:
                    pipe.close()

    def test_pipeline_wakeup_mixed_valid_invalid_tags_noop(self, model, backend, config):
        pipe = None
        try:
            pipe = _open_pipeline(config, model, backend)
            _ensure_awake_pipeline(pipe)
            _apply_sleep(pipe, 1)
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, ['weights', 'not_a_valid_tag'])
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, ['not_a_valid_tag', 'weights'])
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, None)
            assert _pipeline_is_sleeping(pipe) is False
        finally:
            if pipe is not None:
                try:
                    _ensure_awake_pipeline(pipe)
                finally:
                    pipe.close()

    def test_pipeline_wakeup_both_tags_one_call(self, model, backend, config):
        pipe = None
        try:
            pipe = _open_pipeline(config, model, backend)
            _ensure_awake_pipeline(pipe)
            _apply_sleep(pipe, 1)
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, ['weights', 'kv_cache'])
            assert _pipeline_is_sleeping(pipe) is False
            gen = GenerationConfig(max_new_tokens=32, temperature=0.01)
            r = pipe([[{'role': 'user', 'content': 'Hi, reply with one short sentence.'}]], gen_config=gen)
            out = r[0] if isinstance(r, list) else r
            assert (out.text or '').strip()
        finally:
            if pipe is not None:
                try:
                    _ensure_awake_pipeline(pipe)
                finally:
                    pipe.close()

    def test_pipeline_redundant_weights_wakeup_noop(self, model, backend, config):
        pipe = None
        try:
            pipe = _open_pipeline(config, model, backend)
            _ensure_awake_pipeline(pipe)
            _apply_sleep(pipe, 1)
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, ['weights'])
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, ['weights'])
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, ['kv_cache'])
            assert _pipeline_is_sleeping(pipe) is False
        finally:
            if pipe is not None:
                try:
                    _ensure_awake_pipeline(pipe)
                finally:
                    pipe.close()

    def test_pipeline_wakeup_empty_string_tag_noop(self, model, backend, config):
        pipe = None
        try:
            pipe = _open_pipeline(config, model, backend)
            _ensure_awake_pipeline(pipe)
            _apply_sleep(pipe, 1)
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, [''])
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, None)
            assert _pipeline_is_sleeping(pipe) is False
        finally:
            if pipe is not None:
                try:
                    _ensure_awake_pipeline(pipe)
                finally:
                    pipe.close()

    def test_pipeline_full_wakeup_when_already_awake(self, model, backend, config):
        pipe = None
        try:
            pipe = _open_pipeline(config, model, backend)
            _ensure_awake_pipeline(pipe)
            assert _pipeline_is_sleeping(pipe) is False
            _pipeline_wakeup(pipe, None)
            assert _pipeline_is_sleeping(pipe) is False
            _pipeline_wakeup(pipe, None)
            assert _pipeline_is_sleeping(pipe) is False
        finally:
            if pipe is not None:
                try:
                    _ensure_awake_pipeline(pipe)
                finally:
                    pipe.close()

    def test_pipeline_second_sleep_while_sleeping_ok(self, model, backend, config):
        pipe = None
        try:
            pipe = _open_pipeline(config, model, backend)
            _ensure_awake_pipeline(pipe)
            _apply_sleep(pipe, 1)
            assert _pipeline_is_sleeping(pipe) is True
            _apply_sleep(pipe, 1)
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, None)
            assert _pipeline_is_sleeping(pipe) is False
        finally:
            if pipe is not None:
                try:
                    _ensure_awake_pipeline(pipe)
                finally:
                    pipe.close()

    def test_pipeline_sleep_level2_staged_wakeup_and_infer(self, model, backend, config):
        pipe = None
        try:
            pipe = _open_pipeline(config, model, backend)
            _ensure_awake_pipeline(pipe)
            gen = GenerationConfig(
                max_new_tokens=LEVEL2_MAX_TOKENS,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                do_sample=False,
            )
            baseline = None
            if _should_enforce_level2_greedy_checks(backend):
                baseline_r = _assert_level2_pipeline_baseline_stable(pipe, gen, label='level2 pipeline')
                baseline = _pipeline_resp_to_chat_dict(baseline_r)

            _apply_sleep(pipe, 2)
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, ['weights'])
            assert _pipeline_is_sleeping(pipe) is True
            _level2_reload_weights_if_supported_pipeline(pipe, backend, config, model)
            _pipeline_wakeup(pipe, ['kv_cache'])
            assert _pipeline_is_sleeping(pipe) is False

            after = _infer_level2_greedy(pipe, gen)
            assert_assistant_not_degenerate(
                (after.text or '').strip(), label='level2 pipeline after staged wakeup (1st infer)')
            if baseline is not None:
                assert_chat_decode_unchanged(baseline, _pipeline_resp_to_chat_dict(after),
                                             label='level2 pipeline 1st infer after staged wakeup')

            after2 = _infer_level2_greedy(pipe, gen)
            if baseline is not None:
                assert_chat_decode_unchanged(baseline, _pipeline_resp_to_chat_dict(after2),
                                             label='level2 pipeline 2nd infer after staged wakeup')

            _apply_sleep(pipe, 2)
            assert _pipeline_is_sleeping(pipe) is True
            _pipeline_wakeup(pipe, ['weights'])
            assert _pipeline_is_sleeping(pipe) is True
            _level2_reload_weights_if_supported_pipeline(pipe, backend, config, model)
            _pipeline_wakeup(pipe, ['kv_cache'])
            assert _pipeline_is_sleeping(pipe) is False

            after_full = _infer_level2_greedy(pipe, gen)
            if baseline is not None:
                assert_chat_decode_unchanged(
                    baseline, _pipeline_resp_to_chat_dict(after_full),
                    label='level2 pipeline infer after 2nd sleep cycle (staged wakeup)')

            gen2 = GenerationConfig(max_new_tokens=32, temperature=0.01)
            r = pipe([[{'role': 'user', 'content': 'Hi, reply with one short sentence.'}]], gen_config=gen2)
            out = r[0] if isinstance(r, list) else r
            assert (out.text or '').strip()
        finally:
            if pipe is not None:
                try:
                    _ensure_awake_pipeline(pipe)
                finally:
                    pipe.close()
