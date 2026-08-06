# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any

import torch

from lmdeploy.pytorch.model_inputs import StepContextManager

from .deepseek_mtp import DeepseekMTPModel
from .deepseek_v32 import DeepseekV32DecoderLayer, _load_fused_qkv_a_weight


class DeepseekV32MTPModel(DeepseekMTPModel):
    """DeepSeek-V3.2 multi-token predictor with sparse DSA attention."""

    def __init__(self,
                 config: Any,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(config,
                         ctx_mgr,
                         dtype=dtype,
                         device=device,
                         decoder_layer_cls=DeepseekV32DecoderLayer)

    def _load_weight_attention(self, name, loaded_weight, params_dict,
                               update_pe_mapping):
        if _load_fused_qkv_a_weight(name, loaded_weight, params_dict,
                                    self.config):
            return
        return super()._load_weight_attention(name, loaded_weight, params_dict,
                                              update_pe_mapping)
