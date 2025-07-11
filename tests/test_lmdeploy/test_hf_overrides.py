import pytest

from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, pipeline


@pytest.mark.parametrize('model_path', ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen3-30B-A3B'])
def test_hf_overrides_turbomind(model_path):
    # Define a custom rope_scaling configuration to override the model's default settings.
    rope_scaling_override = {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
    hf_overrides = {'rope_scaling': rope_scaling_override}

    backend_config = TurbomindEngineConfig(hf_overrides=hf_overrides)
    with pipeline(model_path, backend_config=backend_config) as pipe:
        processed_config = pipe.engine.config.attention_config

        assert getattr(processed_config, 'rope_param') is not None
        for key, value in rope_scaling_override.items():
            # Adjust key for compatibility with Turbomind's config
            if key == 'rope_type':
                key = 'type'
            if key == 'original_max_position_embeddings':
                key = 'max_position_embeddings'

            assert getattr(processed_config.rope_param, key) == value, \
                f'Expected {key} to be {value}, but got {getattr(processed_config.rope_param, key)}'


@pytest.mark.parametrize('model_path', ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen3-30B-A3B'])
def test_hf_overrides_pytorch(model_path):
    # Define a custom rope_scaling configuration to override the model's default settings.
    rope_scaling_override = {'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
    hf_overrides = {'rope_scaling': rope_scaling_override}

    backend_config = PytorchEngineConfig(hf_overrides=hf_overrides)
    with pipeline(model_path, backend_config=backend_config) as pipe:
        processed_config = pipe.engine.get_model_config()

        assert processed_config.hf_config.rope_scaling is not None
        for key, value in rope_scaling_override.items():
            assert processed_config.hf_config.rope_scaling.get(key) == value, \
                f'Expected {key} to be {value}, but got {processed_config.hf_config.rope_scaling.get(key)}'
