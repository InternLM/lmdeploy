# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn

from .base import CONVERT_MOE_MODELS


@CONVERT_MOE_MODELS.register_module(name='qwen3-moe')
@CONVERT_MOE_MODELS.register_module(name='qwen3_5-moe')
class QwenMoeMLP(nn.Module):
    """Use unfused MoE expert MLP after splitting fused expert weights."""

    def __init__(self, hidden_size, intermediate_size, dtype=None, device=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def load_weight(self, gate_proj_weight: torch.Tensor, down_proj_weight: torch.Tensor, up_proj_weight: torch.Tensor):
        """Load weights for the MoE expert MLP."""
        self.gate_proj.weight = nn.Parameter(gate_proj_weight.detach(), requires_grad=False)
        self.up_proj.weight = nn.Parameter(up_proj_weight.detach(), requires_grad=False)
        self.down_proj.weight = nn.Parameter(down_proj_weight.detach(), requires_grad=False)
