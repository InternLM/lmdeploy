# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from lmdeploy.messages import TurbomindEngineConfig
from lmdeploy.turbomind.converter import _check_fp8_capability, get_tm_config


@pytest.mark.parametrize('major', [7, 8])
def test_check_fp8_capability_rejects_pre_hopper(major):
    with patch('torch.cuda.is_available', return_value=True), \
            patch('torch.cuda.current_device', return_value=0), \
            patch('torch.cuda.get_device_capability', return_value=(major, 0)):
        with pytest.raises(RuntimeError, match='requires a GPU with compute capability >= 9.0'):
            _check_fp8_capability()


@pytest.mark.parametrize('major', [9, 10])
def test_check_fp8_capability_accepts_hopper_and_newer(major):
    with patch('torch.cuda.is_available', return_value=True), \
            patch('torch.cuda.current_device', return_value=0), \
            patch('torch.cuda.get_device_capability', return_value=(major, 0)):
        _check_fp8_capability()


def test_check_fp8_capability_rejects_without_cuda():
    """No CUDA device visible: fail with the same clear RuntimeError rather
    than a raw torch.cuda init error (2026-08-17 Copilot review on PR #4875:
    the original guard called torch.cuda.current_device() unconditionally)."""
    with patch('torch.cuda.is_available', return_value=False):
        with pytest.raises(RuntimeError, match='no CUDA device is available'):
            _check_fp8_capability()


def test_get_tm_config_rejects_fp8_on_sm75_before_building_resolver():
    """Reproduces #4863: a user forcing model_format='fp8' (the CLI's `--model-
    format fp8`) on an sm75 GPU must fail fast with a clear RuntimeError
    instead of proceeding into weight-format resolution and crashing later
    inside turbomind's C++ gemm dispatch."""
    hf_cfg = SimpleNamespace(to_dict=lambda: {'architectures': ['LlamaForCausalLM']},
                              text_config=None,
                              llm_config=None,
                              dtype=torch.float16,
                              torch_dtype=None)
    engine_config = TurbomindEngineConfig(model_format='fp8')

    with patch('lmdeploy.turbomind.converter.get_model_arch', return_value=('LlamaForCausalLM', hf_cfg)), \
            patch('lmdeploy.turbomind.converter.search_nested_config', return_value=None), \
            patch('torch.cuda.is_available', return_value=True), \
            patch('torch.cuda.current_device', return_value=0), \
            patch('torch.cuda.get_device_capability', return_value=(7, 5)):
        with pytest.raises(RuntimeError, match='requires a GPU with compute capability >= 9.0'):
            get_tm_config('fake/model/path', engine_config)


def test_get_tm_config_does_not_reject_non_fp8_format_on_sm75():
    """The new guard must not touch unrelated formats (regression: it lives
    right before `_build_resolver`, so a broad placement mistake would reject
    every model_format on old hardware, not only fp8)."""
    hf_cfg = SimpleNamespace(to_dict=lambda: {'architectures': ['LlamaForCausalLM']},
                              text_config=None,
                              llm_config=None,
                              dtype=torch.float16,
                              torch_dtype=None)
    engine_config = TurbomindEngineConfig(model_format=None)

    with patch('lmdeploy.turbomind.converter.get_model_arch', return_value=('LlamaForCausalLM', hf_cfg)), \
            patch('lmdeploy.turbomind.converter.search_nested_config', return_value=None), \
            patch('torch.cuda.is_available', return_value=True), \
            patch('torch.cuda.current_device', return_value=0), \
            patch('torch.cuda.get_device_capability', return_value=(7, 5)), \
            patch('lmdeploy.turbomind.converter.get_registered_name', return_value='fake'), \
            patch('lmdeploy.turbomind.converter.INPUT_MODELS') as mock_models, \
            patch('lmdeploy.turbomind.converter._get_and_verify_max_len', return_value=2048), \
            patch('lmdeploy.turbomind.converter.source_model_config', return_value={}):
        mock_model_cls = mock_models.get.return_value
        mock_model_cls._vision = False

        get_tm_config('fake/model/path', engine_config)

        mock_model_cls.assert_called_once()


def test_get_tm_config_rejects_fp8_from_checkpoint_quant_config_on_sm75():
    """Reproduces the other #4863 entry point (2026-08-17 Copilot review on
    PR #4875): a checkpoint whose own quantization_config declares
    quant_method='fp8' must trip the same guard as the user-forced CLI flag,
    even though engine_config.model_format starts out None here and is only
    set to 'fp8' by search_nested_config's result inside get_tm_config."""
    hf_cfg = SimpleNamespace(to_dict=lambda: {'architectures': ['LlamaForCausalLM']},
                              text_config=None,
                              llm_config=None,
                              dtype=torch.float16,
                              torch_dtype=None)
    engine_config = TurbomindEngineConfig(model_format=None)
    quant_config = {'quant_method': 'fp8'}

    with patch('lmdeploy.turbomind.converter.get_model_arch', return_value=('LlamaForCausalLM', hf_cfg)), \
            patch('lmdeploy.turbomind.converter.search_nested_config', return_value=quant_config), \
            patch('torch.cuda.is_available', return_value=True), \
            patch('torch.cuda.current_device', return_value=0), \
            patch('torch.cuda.get_device_capability', return_value=(7, 5)):
        with pytest.raises(RuntimeError, match='requires a GPU with compute capability >= 9.0'):
            get_tm_config('fake/model/path', engine_config)

        assert engine_config.model_format == 'fp8'
