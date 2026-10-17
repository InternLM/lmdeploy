# Copyright (c) OpenMMLab. All rights reserved.
import builtins
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def config_module(monkeypatch):
    # Exercise old installations even when the test environment has GLM-OCR.
    original_import = builtins.__import__

    def import_without_glm_ocr(name, *args, **kwargs):
        if name.startswith('transformers.models.glm_ocr'):
            raise ModuleNotFoundError(f'No module named {name!r}', name=name)
        return original_import(name, *args, **kwargs)

    path = Path(__file__).resolve().parents[2] / 'lmdeploy/hf_configs/configuration_glm5_next.py'
    spec = importlib.util.spec_from_file_location('glm5_next_config_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as context:
        context.setattr(builtins, '__import__', import_without_glm_ocr)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('vision_config', [None, {}, {'depth': 12, 'out_hidden_size': 4096, 'swiglu_limit': 8.0}])
def test_glm5_config_roundtrip_without_glm_ocr(config_module, tmp_path, vision_config):
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config_cls = config_module.Glm5NextConfig
    config = config_cls(text_config={'hidden_size': 256, 'num_hidden_layers': 5}, vision_config=vision_config)
    config.save_pretrained(tmp_path)
    # Register the local config, as LMDeploy does for an unknown model type.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(CONFIG_MAPPING._extra_content, 'glm5_next', config_cls)
        restored = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False, local_files_only=True)
    assert isinstance(restored, config_cls)
    assert restored.hidden_size == restored.text_config.hidden_size == 256
    assert restored.text_config.num_hidden_layers == 5
    assert restored.text_config.full_attention_layer_ids == [3]
    if vision_config is None:
        assert restored.vision_config is None
    else:
        assert isinstance(restored.vision_config, config_module.Glm5NextVisionConfig)
        assert restored.vision_config.to_dict() == config.vision_config.to_dict()
        for name, value in vision_config.items():
            assert getattr(restored.vision_config, name) == value


@pytest.mark.parametrize('overrides', [{}, {'depth': 12, 'image_size': [336, 448], 'out_hidden_size': 4096,
                                         'attention_bias': False, 'custom_vision_field': 'preserved'}])
def test_glm5_vision_config_matches_native_defaults(config_module, overrides):
    native_module = pytest.importorskip('transformers.models.glm_ocr.configuration_glm_ocr')
    expected = native_module.GlmOcrVisionConfig(**overrides).to_dict()
    actual = config_module.Glm5NextVisionConfig(**overrides).to_dict()
    assert actual.pop('swiglu_limit') == 10.0
    assert actual.pop('model_type') == 'glm5_next_vision'
    expected.pop('model_type')
    assert actual == expected


def test_glm5_vision_config_positional_swiglu_limit(config_module):
    config = config_module.Glm5NextVisionConfig(8.0)
    assert config.swiglu_limit == 8.0
    assert config.depth == 24
