# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

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


def test_build_memdecode_config_consumes_flat_hf_overrides(monkeypatch):
    returned_hf_configs = {}

    def fake_config_from_pretrained(model_path, trust_remote_code=False):
        assert model_path == 'memory'
        returned_hf_configs[model_path] = _hf_config(vocab_size=30000)
        return returned_hf_configs[model_path]

    monkeypatch.setattr('lmdeploy.pytorch.transformers.config_from_pretrained', fake_config_from_pretrained)
    monkeypatch.setattr('lmdeploy.pytorch.engine.config_builder.get_model', lambda model, *_: model)
    engine_config = SimpleNamespace(
        hf_overrides={
            'memory_model_path': 'memory',
            'lambda_value': 0.25,
            'adaptive_router': True,
            'router_path': 'router.pt',
            'lambda_base_only_threshold': 0.1,
            'rope_scaling_factor': 2.0,
        },
        dtype='float16',
        model_format='awq',
        device_type='cpu',
        download_dir=None,
        revision=None,
    )

    memdecode_config = ConfigBuilder.build_memdecode_config(
        target_model='base',
        engine_config=engine_config,
        cache_config=_cache_config(block_size=32),
        dist_config=DistConfig(tp=2),
        trust_remote_code=True,
    )

    assert memdecode_config.memory_model_path == 'memory'
    assert memdecode_config.lambda_value == 0.25
    assert memdecode_config.adaptive_router is True
    assert memdecode_config.router_path == 'router.pt'
    assert memdecode_config.lambda_base_only_threshold == 0.1
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
