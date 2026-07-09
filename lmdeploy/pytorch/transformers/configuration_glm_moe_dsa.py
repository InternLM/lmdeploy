# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.pytorch.transformers.configuration_deepseek_v32 import DeepseekV32Config


class GlmMoeDsaConfig(DeepseekV32Config):
    model_type = 'glm_moe_dsa'
    attribute_map = {
        'num_local_experts': 'n_routed_experts',
    }

    def __init__(
        self,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2048,
        index_topk_pattern=None,
        index_topk_freq=1,
        index_skip_topk_offset=2,
        rope_interleave=True,
        indexer_rope_interleave=True,
        indexer_types=None,
        **kwargs,
    ):
        super().__init__(
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_topk=index_topk,
            **kwargs,
        )
        self.index_topk_pattern = index_topk_pattern
        self.index_topk_freq = index_topk_freq
        self.index_skip_topk_offset = index_skip_topk_offset
        self.rope_interleave = rope_interleave
        self.indexer_rope_interleave = indexer_rope_interleave
        self.indexer_types = indexer_types

        if hasattr(self, 'qk_nope_head_dim') and hasattr(self, 'qk_rope_head_dim'):
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.head_dim = self.qk_rope_head_dim
