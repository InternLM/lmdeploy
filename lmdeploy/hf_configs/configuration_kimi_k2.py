# Copyright (c) OpenMMLab. All rights reserved.

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


class KimiK2Config(DeepseekV3Config):
    """DeepSeek-V3 config alias used by standalone Kimi-K2 checkpoints."""

    model_type = 'kimi_k2'
