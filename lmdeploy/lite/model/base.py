# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
from mmengine import Registry

MODELS = Registry('model', locations=['lmdeploy.lite.model'])

class Base:

    def __init__(self, model: nn.Module):
        pass

    def convert_gate(self, gate_mod: nn.Module):
        num_experts, hidden_size = gate_mod.weight.shape
        dtype = gate_mod.weight.dtype
        device = gate_mod.weight.device
        gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype, device=device)
        gate.weight = nn.Parameter(gate_mod.weight.data.detach(), requires_grad=False)
        return gate

    def convert_experts(self, experts_mod: nn.Module, cls) -> nn.ModuleList:
        """Convert fused MoE expert weights into a ModuleList of MLP experts
        without copying."""
        num_experts, intermediate_size_2, hidden_size = experts_mod.gate_up_proj.shape
        intermediate_size = intermediate_size_2 // 2

        dtype = experts_mod.gate_up_proj.dtype

        weight_gate_up = experts_mod.gate_up_proj.data
        weight_down = experts_mod.down_proj.data

        MoeExpert_list = nn.ModuleList()

        for e in range(num_experts):
            mod_mlp_instance = cls(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=dtype,
                device='meta'
            )
            mod_mlp_instance.load_weight(
                weight_gate_up[e, :intermediate_size],
                weight_down[e],
                weight_gate_up[e, intermediate_size:]
            )
            MoeExpert_list.append(mod_mlp_instance)

        return MoeExpert_list

    def convert_moe_parameters(self, model: nn.Module, cls):
        """Replace fused MoE experts with expert ModuleList if transformers >=
        5.0."""
        parent_target = getattr(model, 'mlp', None)
        if parent_target is None:
            return

        target = getattr(parent_target, 'experts', None)
        if target is not None and not isinstance(target, nn.ModuleList):
            parent_target.experts = self.convert_experts(target, cls)

        gate_target = getattr(parent_target, 'gate', None)
        if gate_target is not None and not isinstance(gate_target, nn.Linear):
            parent_target.gate = self.convert_gate(gate_target)

    @classmethod
    def skipped_modules(cls):
        pass
