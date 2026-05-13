# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn

from .base import MODELS, Base


class MixtralMoeMLP(nn.Module):
    """Use unfused MoE expert MLP after splitting fused expert weights."""

    def __init__(self, hidden_size, intermediate_size, dtype=None, device=None):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def load_weight(self, w1_weight: torch.Tensor, w2_weight: torch.Tensor, w3_weight: torch.Tensor):
        """Load weights for the MoE expert MLP."""
        self.w1.weight = nn.Parameter(w1_weight.detach(), requires_grad=False)
        self.w2.weight = nn.Parameter(w2_weight.detach(), requires_grad=False)
        self.w3.weight = nn.Parameter(w3_weight.detach(), requires_grad=False)


@MODELS.register_module(name='MixtralForCausalLM')
class Mixtral(Base):

    def __init__(self, model: nn.Module):
        # super().__init__(model, MixtralMoeMLP)
        self.convert_moe_parameters(model, MixtralMoeMLP)

    @classmethod
    def skipped_modules(cls):
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse('5.0.0'):
            return ['block_sparse_moe.gate']
        else:
            return ['mlp.gate']
