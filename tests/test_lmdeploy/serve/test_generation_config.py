# Copyright (c) OpenMMLab. All rights reserved.
import json
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.core.generation_config import (
    build_generation_config,
    extract_request_gen_config,
    merge_gen_config,
    resolve_default_gen_config,
)
from lmdeploy.serve.openai.chat_completions.validation import check_request as check_chat_request
from lmdeploy.serve.openai.endpoints.completions import check_request as check_completion_request
from lmdeploy.serve.openai.endpoints.generate import check_request as check_generate_request
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest, GenerateReqInput
from lmdeploy.serve.openai.responses import ResponsesRequest

_DEFAULTS = GenerationConfig()


class _FakeEngineConfig:
    logprobs_mode = None


class _FakeSessionManager:

    def has(self, session_id):
        return False


class _FakeServerContext:

    response_parser_cls = None

    def __init__(self):
        self.async_engine = SimpleNamespace(
            backend_config=_FakeEngineConfig(),
            session_mgr=_FakeSessionManager(),
        )

    @property
    def engine_config(self):
        return self.async_engine.backend_config

    @property
    def session_manager(self):
        return self.async_engine.session_mgr


def test_merge_gen_config_priority():
    merged = merge_gen_config(
        {'temperature': 0.2},
        {'temperature': 0.5, 'top_k': 10},
    )
    assert merged == {'temperature': 0.2, 'top_k': 10}


def test_merge_gen_config_uses_server_defaults():
    merged = merge_gen_config({}, {'temperature': 0.5})
    assert merged == {'temperature': 0.5}


def test_extract_request_gen_config_only_explicit_fields():
    request = ChatCompletionRequest(model='test', messages=[{'role': 'user', 'content': 'hi'}], temperature=0.3)
    values = extract_request_gen_config(request)
    assert values == {'temperature': 0.3}


class _RequestWithExtraAllow(BaseModel):
    model_config = ConfigDict(extra='allow')

    temperature: float | None = None


def test_extract_request_gen_config_ignores_undeclared_fields():
    request = _RequestWithExtraAllow(temperature=0.3, return_ppl=True)
    assert extract_request_gen_config(request) == {'temperature': 0.3}


def test_responses_request_ignores_unknown_generation_fields():
    request = ResponsesRequest(model='fake-model', input='hi', return_ppl=True)
    assert 'return_ppl' not in request.model_dump()
    gen_config = build_generation_config(request, {})
    assert gen_config.return_ppl is False


def test_build_generation_config_from_merged_values():
    request = ChatCompletionRequest(model='test', messages=[{'role': 'user', 'content': 'hi'}], temperature=0.2)
    gen_config = build_generation_config(
        request,
        {'top_k': 5},
        max_new_tokens=32,
    )
    assert gen_config.temperature == 0.2
    assert gen_config.top_k == 5
    assert gen_config.max_new_tokens == 32
    assert gen_config.do_sample is True


def test_build_generation_config_max_new_tokens_defaults_to_none():
    request = CompletionRequest(model='test', prompt='hello')
    gen_config = build_generation_config(request, {})
    assert gen_config.max_new_tokens is None


def test_build_generation_config_uses_generation_config_defaults():
    request = CompletionRequest(model='test', prompt='hello')
    gen_config = build_generation_config(request, {})
    assert gen_config.temperature == _DEFAULTS.temperature
    assert gen_config.top_k == _DEFAULTS.top_k


@pytest.mark.parametrize('request_cls, payload, check_request', [
    (ChatCompletionRequest, {'model': 'test', 'messages': [{'role': 'user', 'content': 'hi'}]}, check_chat_request),
    (CompletionRequest, {'model': 'test', 'prompt': 'hi'}, check_completion_request),
    (GenerateReqInput, {'prompt': 'hi'}, check_generate_request),
])
@pytest.mark.parametrize('top_k', [None, -2, -1, 0, 1, 20])
def test_request_top_k_validation_and_merge(request_cls, payload, check_request, top_k):
    request = request_cls(**payload, top_k=top_k)
    error = check_request(request, _FakeServerContext())
    if top_k == -2:
        assert 'top_k' in error
        return
    assert error == ''
    config = build_generation_config(request, {'top_k': 10, 'temperature': 0.6})
    assert config.top_k == (10 if top_k is None else max(0, top_k))
    assert config.temperature == 0.6


@pytest.mark.parametrize('request_temperature, defaults, expected', [
    (None, {}, 1.0),
    (None, {'temperature': 0.6}, 0.6),
    (0.0, {'temperature': 0.6}, 0.0),
    (0.8, {'temperature': 0.6}, 0.8),
    (1.0, {'temperature': 0.6}, 1.0),
])
def test_temperature_fallback_and_overrides(request_temperature, defaults, expected):
    request = CompletionRequest(model='test', prompt='hi', temperature=request_temperature)
    assert build_generation_config(request, defaults).temperature == expected


@pytest.mark.parametrize('source', ['auto', 'path'])
def test_model_sampling_defaults_and_request_overrides(tmp_path, source):
    (tmp_path / 'generation_config.json').write_text(json.dumps({
        'do_sample': True, 'temperature': 0.6, 'top_k': -1,
    }))
    defaults = resolve_default_gen_config('auto' if source == 'auto' else str(tmp_path), str(tmp_path), False)
    request = CompletionRequest(model='test', prompt='hi')
    config = build_generation_config(request, defaults)
    assert config.top_k == 0
    assert config.temperature == 0.6

    override = CompletionRequest(model='test', prompt='hi', top_k=5, temperature=0.8)
    config = build_generation_config(override, defaults)
    assert config.top_k == 5
    assert config.temperature == 0.8


def test_build_generation_config_ignores_unsupported_defaults():
    request = CompletionRequest(model='test', prompt='hello')
    gen_config = build_generation_config(
        request,
        {
            'temperature': 0.6,
            'eos_token_id': 2,
            'pad_token_id': 0,
            'transformers_version': '5.12.1',
        },
    )
    assert gen_config.temperature == 0.6


def test_completion_request_max_tokens_is_optional():
    request = CompletionRequest(model='test', prompt='hello')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        assert request.max_tokens is None


def test_generate_request_sampling_defaults_match_chat_request():
    chat_request = ChatCompletionRequest(model='test', messages=[{'role': 'user', 'content': 'hello'}])
    generate_request = GenerateReqInput(prompt='hello')
    for name in ('temperature', 'top_p', 'top_k', 'min_p'):
        assert getattr(generate_request, name) == getattr(chat_request, name)


def test_generate_request_accepts_none_sampling_defaults():
    request = GenerateReqInput(prompt='hello')
    assert check_generate_request(request, _FakeServerContext()) == ''


def test_generate_request_sampling_merge_uses_server_defaults():
    request = GenerateReqInput(prompt='hello')
    gen_config = build_generation_config(
        request,
        {
            'temperature': 0.2,
            'top_p': 0.3,
            'top_k': 7,
            'min_p': 0.1,
        },
        max_new_tokens=request.max_tokens,
    )
    assert gen_config.temperature == 0.2
    assert gen_config.top_p == 0.3
    assert gen_config.top_k == 7
    assert gen_config.min_p == 0.1


@patch('lmdeploy.serve.core.generation_config._load_hf_generation_config')
def test_resolve_default_gen_config_auto(mock_load):
    mock_load.return_value = {
        'temperature': 0.6,
        'top_p': 0.8,
        'max_new_tokens': 2048,
        'eos_token_id': 2,
        'transformers_version': '5.12.1',
    }
    config = resolve_default_gen_config('auto', '/fake/model', False)
    assert config == {
        'temperature': 0.6,
        'top_p': 0.8,
        'max_new_tokens': 2048,
    }
    mock_load.assert_called_once_with('/fake/model', False)


def test_resolve_default_gen_config_lmdeploy():
    config = resolve_default_gen_config('lmdeploy', '/fake/model', False)
    assert config == {}


def test_completion_request_sampling_merge():
    request = CompletionRequest(model='test', prompt='hello')
    gen_config = build_generation_config(request, {'temperature': 0.9})
    assert gen_config.temperature == 0.9
    assert gen_config.top_k == _DEFAULTS.top_k
