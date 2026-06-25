# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

from lmdeploy.serve.core.generation_config import (
    PROTOCOL_FALLBACKS,
    build_generation_config,
    extract_request_sampling_values,
    merge_sampling_params,
    resolve_max_new_tokens,
    resolve_server_sampling_defaults,
)
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest


def test_merge_sampling_params_priority():
    merged = merge_sampling_params(
        {'temperature': 0.2},
        {'temperature': 0.5, 'top_k': 10},
        PROTOCOL_FALLBACKS,
    )
    assert merged['temperature'] == 0.2
    assert merged['top_k'] == 10
    assert merged['top_p'] == PROTOCOL_FALLBACKS['top_p']


def test_merge_sampling_params_uses_server_then_fallback():
    merged = merge_sampling_params({}, {'temperature': 0.5}, PROTOCOL_FALLBACKS)
    assert merged['temperature'] == 0.5
    assert merged['top_k'] == PROTOCOL_FALLBACKS['top_k']


def test_extract_request_sampling_values_only_non_null():
    request = ChatCompletionRequest(model='test', messages='hi', temperature=0.3)
    values = extract_request_sampling_values(request)
    assert values == {'temperature': 0.3}


def test_resolve_max_new_tokens_uses_server_default():
    assert resolve_max_new_tokens(None, None, 128) == 128


def test_resolve_max_new_tokens_caps_request_value():
    assert resolve_max_new_tokens(256, None, 128) == 128
    assert resolve_max_new_tokens(None, 256, 128) == 128


def test_resolve_max_new_tokens_prefers_max_completion_tokens():
    assert resolve_max_new_tokens(64, 256, None) == 64


def test_build_generation_config_from_merged_values():
    gen_config = build_generation_config(
        {'temperature': 0.2},
        {'top_k': 5},
        max_completion_tokens=32,
        override_max_new_tokens=64,
    )
    assert gen_config.temperature == 0.2
    assert gen_config.top_k == 5
    assert gen_config.max_new_tokens == 32
    assert gen_config.do_sample is True


@patch('lmdeploy.serve.core.generation_config._load_hf_generation_config')
def test_resolve_server_sampling_defaults_auto(mock_load):
    mock_load.return_value = {
        'temperature': 0.6,
        'top_p': 0.8,
        'max_new_tokens': 2048,
    }
    defaults, cap = resolve_server_sampling_defaults('auto', None, '/fake/model', False)
    assert defaults == {'temperature': 0.6, 'top_p': 0.8}
    assert cap == 2048
    mock_load.assert_called_once_with('/fake/model', False)


def test_resolve_server_sampling_defaults_lmdeploy():
    defaults, cap = resolve_server_sampling_defaults('lmdeploy', None, '/fake/model', False)
    assert defaults == {}
    assert cap is None


@patch('lmdeploy.serve.core.generation_config._load_hf_generation_config')
def test_resolve_server_sampling_defaults_with_override(mock_load):
    mock_load.return_value = {'temperature': 0.6, 'top_k': 20}
    defaults, cap = resolve_server_sampling_defaults(
        'auto',
        {'temperature': 0.5, 'max_new_tokens': 100},
        '/fake/model',
        False,
    )
    assert defaults == {'temperature': 0.5, 'top_k': 20}
    assert cap == 100


def test_completion_request_sampling_merge():
    request = CompletionRequest(model='test', prompt='hello')
    gen_config = build_generation_config(
        extract_request_sampling_values(request),
        {'temperature': 0.9},
    )
    assert gen_config.temperature == 0.9
    assert gen_config.top_k == PROTOCOL_FALLBACKS['top_k']
