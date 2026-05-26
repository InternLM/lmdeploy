# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn

from .base import MODELS, Base


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


@MODELS.register_module(name='Qwen3MoeForCausalLM')
class Qwen3_Moe(Base):

    def __init__(self, model: nn.Module):
        self.convert_moe_parameters(model, QwenMoeMLP)

    @classmethod
    def skipped_modules(cls):
        return ['mlp.gate']


@MODELS.register_module(name='Qwen3_5ForConditionalGeneration')
class Qwen3_5(Base):

    @classmethod
    def skipped_modules(cls):
        return ['visual', 'linear_attn', 'self_attn', 'model.layers.0.', 'mtp']


@MODELS.register_module(name='Qwen3_5MoeForConditionalGeneration')
class Qwen3_5Moe(Qwen3_Moe):

    @classmethod
    def skipped_modules(cls):
        return ['mlp.gate', 'visual', 'linear_attn', 'self_attn',
                'model.layers.0.', 'mtp', 'shared_expert']


@MODELS.register_module(name='InternS2PreviewForConditionalGeneration')
class InternS2PreviewForConditionalGeneration(Qwen3_5Moe):

    @classmethod
    def skipped_modules(cls):
        return ['mlp.gate', 'visual', 'linear_attn', 'self_attn',
                'model.layers.0.', 'time_series', 'shared_expert']
