# Copyright (c) OpenMMLab. All rights reserved.
"""Dense Qwen3.5 with FP32 zero-centered norm scales for Intern-Decision."""
from collections.abc import Iterable

import torch
from transformers import PretrainedConfig

from lmdeploy.pytorch.config import QuantizationConfig
from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.nn import RMSNorm

from .patch import get_build_model_context
from .qwen3_5 import Qwen3_5ForConditionalGeneration


class InternDecisionForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Reuse Qwen3.5 execution, preserving only text norm scales in FP32."""

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 prefix: str = ''):
        if (QuantizationConfig.from_config(config).quant_method is not None
                or get_build_model_context().quant_config.quant_method is not None):
            raise ValueError('Intern-Decision FP32 norms require an unquantized dense Qwen3.5 model.')
        super().__init__(config, ctx_mgr, dtype=dtype, device=device, prefix=prefix)

        # Gated norms use direct scales and are not RMSNorm instances. Restrict
        # this policy to the text model so vision norms keep their original dtype.
        self._fp32_norm_weights = set()
        for name, module in self.model.language_model.named_modules():
            if isinstance(module, RMSNorm):
                module.float()
                self._fp32_norm_weights.add(f'model.language_model.{name}.weight')

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Cast offsets before the parent loader folds +1 and copies
        weights."""
        super().load_weights((name, weight.float() if name in self._fp32_norm_weights else weight)
                             for name, weight in weights)
