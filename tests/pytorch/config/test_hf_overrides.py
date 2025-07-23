import pytest


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

        # update rope
        assert hf_config.text_config.model_type == 'llama'
        assert hf_config.text_config.rope_scaling is None
        overrides_dict = dict(text_config=dict(rope_scaling=dict(rope_type='yarn', )))
        override_hf_config(hf_config, overrides_dict)
        assert hf_config.text_config.model_type == 'llama'
        assert hf_config.text_config.rope_scaling['rope_type'] == 'yarn'

        # update both
        overrides_dict = dict(model_type='llava_custom2', text_config=dict(rope_scaling=dict(rope_type='yarn2', )))
        override_hf_config(hf_config, overrides_dict)
        assert hf_config.model_type == 'llava_custom2'
        assert hf_config.text_config.model_type == 'llama'
        assert hf_config.text_config.rope_scaling['rope_type'] == 'yarn2'
