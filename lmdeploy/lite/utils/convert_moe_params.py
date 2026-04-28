# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn

from lmdeploy.lite.mlp_moe_modules.base import CONVERT_MOE_MODELS
from lmdeploy.turbomind.deploy.converter import get_input_model_registered_name

PARENT_NAME_LIST = ['mlp', 'block_sparse_moe']


def find_moe_parent(model: nn.Module):
    """Return the first module that may own fused MoE experts."""
    return next(
        (mod for name, mod in model.named_modules() if name in PARENT_NAME_LIST),
        None
    )


def convert_experts(experts_mod: nn.Module, moemlp_cls) -> nn.ModuleList:
    """Convert fused MoE expert weights into a ModuleList of MLP experts
    without copying."""
    num_experts, intermediate_size_2, hidden_size = experts_mod.gate_up_proj.shape
    intermediate_size = intermediate_size_2 // 2

    dtype = experts_mod.gate_up_proj.dtype

    weight_gate_up = experts_mod.gate_up_proj.data
    weight_down = experts_mod.down_proj.data

    MoeExpert_list = nn.ModuleList()

    for e in range(num_experts):
        mod_mlp_instance = moemlp_cls(
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


def convert_moe_parameters(model_path: str, model: nn.Module):
    """Replace fused MoE experts with expert ModuleList if transformers >=
    5.0."""
    model_name = get_input_model_registered_name(model_path, 'awq')
    parent_target = find_moe_parent(model)
    if parent_target is None:
        return

    target = getattr(parent_target, 'experts', None)
    if target is None or isinstance(target, nn.ModuleList):
        return

    moemlp_cls = CONVERT_MOE_MODELS.get(model_name)
    if moemlp_cls is None:
        return
    parent_target.experts = convert_experts(target, moemlp_cls)
