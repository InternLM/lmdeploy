# Copyright (c) OpenMMLab. All rights reserved.

from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


class DeepseekV32Config(DeepseekV3Config):
    model_type = 'deepseek_v32'

    def __init__(self, index_head_dim=128, index_n_heads=64, index_topk=2048, **kwargs):
        super().__init__(**kwargs)
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
