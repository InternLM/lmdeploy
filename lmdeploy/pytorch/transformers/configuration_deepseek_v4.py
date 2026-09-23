# Copyright (c) OpenMMLab. All rights reserved.
# DeepSeek-V4 HF config (mirrors vllm.transformers_utils.configs.deepseek_v4).
# Subclasses PretrainedConfig; all V4 fields in config.json become attributes
# via **kwargs.  max_position_embeddings / rope_parameters are set explicitly so
# transformers>=5 RoPE standardization (convert_rope_params_to_dict) succeeds.

from transformers import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = 'deepseek_v4'

    def __init__(
        self,
        max_position_embeddings: int = 1048576,
        rope_scaling: dict | None = None,
        rope_parameters: dict | None = None,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.rope_parameters = rope_scaling or rope_parameters
        super().__init__(**kwargs)
