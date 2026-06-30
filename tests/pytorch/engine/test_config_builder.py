# Copyright (c) OpenMMLab. All rights reserved.
import json
from types import SimpleNamespace

import pytest

from lmdeploy.pytorch.config import CacheConfig, DistConfig
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder


def _cache_config(block_size=64):
    return CacheConfig(max_batches=4, block_size=block_size, num_cpu_blocks=0, num_gpu_blocks=0)


def _hf_config(vocab_size=32000):
    return SimpleNamespace(
        architectures=['Qwen3ForCausalLM'],
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=1024,
        model_type='qwen3',
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=2,
        torch_dtype='float16',
        vocab_size=vocab_size,
    )


def _engine_config(hf_overrides):
    return SimpleNamespace(
        hf_overrides=hf_overrides,
        dtype='float16',
        model_format='awq',
        device_type='cpu',
        download_dir=None,
        revision=None,
    )


def test_build_memdecode_config_consumes_flat_hf_overrides(monkeypatch):
    returned_hf_configs = {}

    def fake_config_from_pretrained(model_path, trust_remote_code=False):
        assert model_path == 'memory'
        returned_hf_configs[model_path] = _hf_config(vocab_size=30000)
        return returned_hf_configs[model_path]

    monkeypatch.setattr('lmdeploy.pytorch.transformers.config_from_pretrained', fake_config_from_pretrained)
    monkeypatch.setattr('lmdeploy.pytorch.engine.config_builder.get_model', lambda model, *_: model)
    engine_config = _engine_config({
        'memory_model_path': 'memory',
        'lambda_value': 0.25,
        'rope_scaling_factor': 2.0,
    })

    memdecode_config = ConfigBuilder.build_memdecode_config(
        target_model='base',
        engine_config=engine_config,
        cache_config=_cache_config(block_size=32),
        dist_config=DistConfig(tp=2),
        trust_remote_code=True,
    )

    assert memdecode_config.memory_model_path == 'memory'
    assert memdecode_config.lambda_value == 0.25
    assert memdecode_config.adaptive_router is False
    assert memdecode_config.router_path is None
    assert memdecode_config.lambda_base_only_threshold == -1.0
    assert memdecode_config.memory_model_config.hf_config is returned_hf_configs['memory']
    assert memdecode_config.memory_model_config.dist_config.tp == 2
    assert engine_config.hf_overrides == {'rope_scaling_factor': 2.0}


def test_build_memdecode_config_returns_none_without_memory_model_path():
    engine_config = SimpleNamespace(hf_overrides={'rope_scaling_factor': 2.0})

    memdecode_config = ConfigBuilder.build_memdecode_config(
        target_model='base',
        engine_config=engine_config,
        cache_config=_cache_config(),
        dist_config=DistConfig(),
    )

    assert memdecode_config is None
    assert engine_config.hf_overrides == {'rope_scaling_factor': 2.0}


def test_build_memdecode_config_uses_packaged_default_router(monkeypatch, tmp_path):
    memory_path = tmp_path / 'memory'
    fusion_dir = memory_path / 'memory_fusion'
    router_dir = fusion_dir / 'routers' / 'qwen3-8b'
    router_dir.mkdir(parents=True)
    (fusion_dir / 'config.json').write_text(
        json.dumps({
            'lambda_value': 0.25,
            'adaptive_router': True,
            'default_router': 'qwen3-8b',
            'routers': {
                'qwen3-8b': {
                    'path': 'routers/qwen3-8b',
                },
                'qwen3-14b': {
                    'path': 'routers/qwen3-14b',
                },
            },
            'lambda_base_only_threshold': 0.1,
        }))
    returned_hf_configs = {}

    def fake_config_from_pretrained(model_path, trust_remote_code=False):
        assert model_path == str(memory_path)
        returned_hf_configs[model_path] = _hf_config(vocab_size=30000)
        return returned_hf_configs[model_path]

    monkeypatch.setattr('lmdeploy.pytorch.transformers.config_from_pretrained', fake_config_from_pretrained)
    engine_config = _engine_config({
        'memory_model_path': str(memory_path),
        'rope_scaling_factor': 2.0,
    })

    memdecode_config = ConfigBuilder.build_memdecode_config(
        target_model='base',
        engine_config=engine_config,
        cache_config=_cache_config(block_size=32),
        dist_config=DistConfig(tp=2),
        trust_remote_code=True,
    )

    assert memdecode_config.memory_model_path == str(memory_path)
    assert memdecode_config.lambda_value == 0.25
    assert memdecode_config.adaptive_router is True
    assert memdecode_config.router_path == str(router_dir)
    assert memdecode_config.lambda_base_only_threshold == 0.1
    assert memdecode_config.memory_model_config.hf_config is returned_hf_configs[str(memory_path)]
    assert memdecode_config.memory_model_config.dist_config.tp == 2
    assert engine_config.hf_overrides == {'rope_scaling_factor': 2.0}


def test_build_memdecode_config_router_name_selects_packaged_router(monkeypatch, tmp_path):
    memory_path = tmp_path / 'memory'
    fusion_dir = memory_path / 'memory_fusion'
    router_8b = fusion_dir / 'routers' / 'qwen3-8b'
    router_14b = fusion_dir / 'routers' / 'qwen3-14b'
    router_8b.mkdir(parents=True)
    router_14b.mkdir(parents=True)
    (fusion_dir / 'config.json').write_text(
        json.dumps({
            'lambda_value': 0.75,
            'adaptive_router': True,
            'default_router': 'qwen3-8b',
            'routers': {
                'qwen3-8b': {
                    'path': 'routers/qwen3-8b',
                },
                'qwen3-14b': {
                    'path': 'routers/qwen3-14b',
                },
            },
            'lambda_base_only_threshold': 0.4,
        }))

    def fake_config_from_pretrained(model_path, trust_remote_code=False):
        assert model_path == str(memory_path)
        return _hf_config(vocab_size=30000)

    monkeypatch.setattr('lmdeploy.pytorch.transformers.config_from_pretrained', fake_config_from_pretrained)
    engine_config = _engine_config({
        'memory_model_path': str(memory_path),
        'lambda_value': 0.25,
        'router_name': 'qwen3-14b',
        'lambda_base_only_threshold': 0.1,
    })

    memdecode_config = ConfigBuilder.build_memdecode_config(
        target_model='base',
        engine_config=engine_config,
        cache_config=_cache_config(),
        dist_config=DistConfig(),
    )

    assert memdecode_config.lambda_value == 0.25
    assert memdecode_config.adaptive_router is True
    assert memdecode_config.router_path == str(router_14b)
    assert memdecode_config.lambda_base_only_threshold == 0.1
    assert 'router_name' not in engine_config.hf_overrides


def test_build_memdecode_config_rejects_packaged_router_without_name_or_default(monkeypatch, tmp_path):
    memory_path = tmp_path / 'memory'
    fusion_dir = memory_path / 'memory_fusion'
    fusion_dir.mkdir(parents=True)
    (fusion_dir / 'config.json').write_text(
        json.dumps({
            'adaptive_router': True,
            'routers': {
                'qwen3-8b': {
                    'path': 'routers/qwen3-8b',
                },
                'qwen3-14b': {
                    'path': 'routers/qwen3-14b',
                },
            },
        }))

    def fake_config_from_pretrained(model_path, trust_remote_code=False):
        assert model_path == str(memory_path)
        return _hf_config(vocab_size=30000)

    monkeypatch.setattr('lmdeploy.pytorch.transformers.config_from_pretrained', fake_config_from_pretrained)
    engine_config = _engine_config({'memory_model_path': str(memory_path)})

    with pytest.raises(ValueError, match='router_name or default_router is required'):
        ConfigBuilder.build_memdecode_config(
            target_model='base',
            engine_config=engine_config,
            cache_config=_cache_config(),
            dist_config=DistConfig(),
        )


def test_build_memdecode_config_rejects_unknown_packaged_router_name(monkeypatch, tmp_path):
    memory_path = tmp_path / 'memory'
    fusion_dir = memory_path / 'memory_fusion'
    fusion_dir.mkdir(parents=True)
    (fusion_dir / 'config.json').write_text(
        json.dumps({
            'adaptive_router': True,
            'default_router': 'qwen3-8b',
            'routers': {
                'qwen3-8b': {
                    'path': 'routers/qwen3-8b',
                },
            },
        }))

    def fake_config_from_pretrained(model_path, trust_remote_code=False):
        assert model_path == str(memory_path)
        return _hf_config(vocab_size=30000)

    monkeypatch.setattr('lmdeploy.pytorch.transformers.config_from_pretrained', fake_config_from_pretrained)
    engine_config = _engine_config({
        'memory_model_path': str(memory_path),
        'router_name': 'qwen3-14b',
    })

    with pytest.raises(ValueError, match='unknown MemDecode router_name'):
        ConfigBuilder.build_memdecode_config(
            target_model='base',
            engine_config=engine_config,
            cache_config=_cache_config(),
            dist_config=DistConfig(),
        )


def test_build_memdecode_config_rejects_invalid_packaged_routers(monkeypatch, tmp_path):
    memory_path = tmp_path / 'memory'
    fusion_dir = memory_path / 'memory_fusion'
    fusion_dir.mkdir(parents=True)
    (fusion_dir / 'config.json').write_text(
        json.dumps({
            'adaptive_router': True,
            'default_router': 'qwen3-8b',
            'routers': None,
        }))

    def fake_config_from_pretrained(model_path, trust_remote_code=False):
        assert model_path == str(memory_path)
        return _hf_config(vocab_size=30000)

    monkeypatch.setattr('lmdeploy.pytorch.transformers.config_from_pretrained', fake_config_from_pretrained)
    engine_config = _engine_config({'memory_model_path': str(memory_path)})

    with pytest.raises(ValueError, match='field "routers" must be a JSON object'):
        ConfigBuilder.build_memdecode_config(
            target_model='base',
            engine_config=engine_config,
            cache_config=_cache_config(),
            dist_config=DistConfig(),
        )
