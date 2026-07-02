# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from unittest.mock import patch

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.core.generation_config import (
    build_generation_config,
    extract_request_gen_config,
    merge_gen_config,
    resolve_default_gen_config,
)
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest, GenerateReqInput
from lmdeploy.serve.openai.serving_generate import check_request as check_generate_request

_DEFAULTS = GenerationConfig()


class _FakeEngineConfig:
    logprobs_mode = None


class _FakeSessionManager:

    def has(self, session_id):
        return False


class _FakeServerContext:

    def get_engine_config(self):
        return _FakeEngineConfig()

    def get_session_manager(self):
        return _FakeSessionManager()


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
    request = ChatCompletionRequest(model='test', messages='hi', temperature=0.3)
    values = extract_request_gen_config(request)
    assert values == {'temperature': 0.3}


def test_build_generation_config_from_merged_values():
    request = ChatCompletionRequest(model='test', messages='hi', temperature=0.2)
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
    chat_request = ChatCompletionRequest(model='test', messages='hello')
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
