# Copyright (c) OpenMMLab. All rights reserved.
from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


class LlamaModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.architectures[0] in ['LlamaForCausalLM']

    @classmethod
    def build(cls, hf_config, model_path: str = None, is_draft_model: bool = False, spec_method: str = None, **kwargs):
        """Build llama."""
        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, **kwargs)

        if is_draft_model:
            # update draft model arch
            assert spec_method is not None
            hf_config.architectures[0] = spec_method.capitalize() + hf_config.architectures[0]
            cfg.vocab_size = getattr(hf_config, 'draft_vocab_size', hf_config.vocab_size)
            cfg.model_paradigm = 'ar_spec'
        elif spec_method is not None:
            # add aux_hidden_state_layers for eagle3
            if spec_method == 'eagle3':
                num_layers = cfg.num_layers
                hf_config.aux_hidden_state_layers = (2, num_layers // 2, num_layers - 3)
            cfg.model_paradigm = 'ar_spec'
        cfg.hf_config = hf_config
        return cfg
