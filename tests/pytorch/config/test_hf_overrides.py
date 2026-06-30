import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, DistConfig, ModelConfig, SpecDecodeConfig
from lmdeploy.pytorch.model_inputs import BuildModelContext


class TestHFOverrides:

    @pytest.fixture
    def hf_config(self):
        from transformers.models.llava import LlavaConfig
        yield LlavaConfig()

    def test_hf_overrides(self, hf_config):
        from lmdeploy.pytorch.config import override_hf_config

        # update root
        assert hf_config.model_type == 'llava'
        overrides_dict = dict(model_type='llava_custom', )
        override_hf_config(hf_config, overrides_dict)
        assert hf_config.model_type == 'llava_custom'

        # update rope_parameters (renamed from rope_scaling in newer transformers)
        assert hf_config.text_config.model_type == 'llama'
        assert hf_config.text_config.rope_parameters['rope_type'] == 'default'
        overrides_dict = dict(text_config=dict(rope_parameters=dict(rope_type='yarn', )))
        override_hf_config(hf_config, overrides_dict)
        assert hf_config.text_config.model_type == 'llama'
        assert hf_config.text_config.rope_parameters['rope_type'] == 'yarn'

        # update both
        overrides_dict = dict(model_type='llava_custom2', text_config=dict(rope_parameters=dict(rope_type='yarn2', )))
        override_hf_config(hf_config, overrides_dict)
        assert hf_config.model_type == 'llava_custom2'
        assert hf_config.text_config.model_type == 'llama'
        assert hf_config.text_config.rope_parameters['rope_type'] == 'yarn2'


def test_fp32_lm_head_hf_override_survives_mtp_draft_config(monkeypatch):
    from transformers import PretrainedConfig

    from lmdeploy.pytorch import transformers as pytorch_transformers
    from lmdeploy.pytorch.configurations import AutoModelConfigBuilder

    def fake_config_from_pretrained(pretrained_model_name_or_path, **kwargs):
        return PretrainedConfig(model_type='fake', torch_dtype='float16', tie_word_embeddings=False)

    def fake_build(cls, hf_config, model_path=None, **kwargs):
        return ModelConfig(hidden_size=16,
                           num_layers=1,
                           num_attention_heads=2,
                           num_key_value_heads=2,
                           bos_token_id=1,
                           eos_token_id=[2],
                           head_dim=8,
                           dtype=torch.float16,
                           vocab_size=32,
                           hf_config=hf_config,
                           llm_config=hf_config)

    monkeypatch.setattr(pytorch_transformers, 'config_from_pretrained', fake_config_from_pretrained)
    monkeypatch.setattr(AutoModelConfigBuilder, 'build', classmethod(fake_build))

    hf_overrides = {'fp32_lm_head': True}
    target_cache_cfg = CacheConfig(max_batches=1,
                                   block_size=64,
                                   num_cpu_blocks=0,
                                   num_gpu_blocks=0)

    specdecode_config = SpecDecodeConfig.from_config(method='qwen3_5_mtp',
                                                     num_speculative_tokens=1,
                                                     model=None,
                                                     target_model='fake-model',
                                                     target_cache_cfg=target_cache_cfg,
                                                     hf_overrides=hf_overrides,
                                                     dist_config=DistConfig())
    main_model_config = ModelConfig.from_pretrained('fake-model',
                                                   hf_overrides=hf_overrides,
                                                   spec_method=specdecode_config.method,
                                                   num_spec_tokens=specdecode_config.num_speculative_tokens)

    assert specdecode_config.model_config.fp32_lm_head
    assert main_model_config.fp32_lm_head
    assert BuildModelContext(fp32_lm_head=specdecode_config.model_config.fp32_lm_head).fp32_lm_head
    assert BuildModelContext(fp32_lm_head=main_model_config.fp32_lm_head).fp32_lm_head
    assert hf_overrides['fp32_lm_head']
