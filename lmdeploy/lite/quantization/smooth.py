# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

# Maps that describe the structure of your model.
NORM_FCS_MAP = {
    'LlamaDecoderLayer': {
        'input_layernorm':
        ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    },
    'InternLMDecoderLayer': {
        'input_layernorm':
        ['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj'],
        'post_attention_layernorm': ['mlp.gate_proj', 'mlp.up_proj']
    }
}

FC_FCS_MAP = {
    'LlamaDecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    },
    'InternLMDecoderLayer': {
        'self_attn.v_proj': ['self_attn.o_proj'],
        'mlp.up_proj': ['mlp.down_proj']
    }
}


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale


@torch.no_grad()
def smooth_ln_fcs(ln: torch.nn.Module,
                  fcs: List[torch.nn.Module],
                  act_scales: torch.Tensor,
                  group_size: int = -1,
                  alpha: float = 0.5) -> torch.Tensor:
    """Smooth weights of a layer normalization and its fully connected layers.

    :param ln: Layer Normalization module
    :param fcs: List of Fully Connected modules
    :param act_scales: Activation scales
    :param alpha: Scaling factor (default is 0.5)
    :return: Scales
    """
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    concat_w = torch.cat([fc.weight for fc in fcs], dim=0)
    w_scales = get_weight_scale(concat_w, group_size)

    scales = (act_scales.pow(alpha) /
              w_scales.pow(1 - alpha)).clamp(min=1e-4).to(device).to(dtype)
    scales = scales / (scales.max() * scales.min()).sqrt()

    ln.weight.div_(scales)
    if hasattr(ln, 'bias'):
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0
    return scales


@torch.no_grad()
def smooth_fc_fcs(pre_fc: torch.nn.Module,
                  fcs: List[torch.nn.Module],
                  act_scales: torch.Tensor,
                  group_size: int = -1,
                  alpha: float = 0.5) -> torch.Tensor:
    """Smooth weights of a fully connected layer and its downstream layers.

    :param pre_fc: Previous Fully Connected layer
    :param fcs: List of Fully Connected modules
    :param act_scales: Activation scales
    :param alpha: Scaling factor (default is 0.5)
    :return: Scales
    """
    device, dtype = pre_fc.weight.device, pre_fc.weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)

    concat_w = torch.cat([fc.weight for fc in fcs], dim=0)
    w_scales = get_weight_scale(concat_w, group_size)

    scales = (act_scales.pow(alpha) /
              w_scales.pow(1 - alpha)).clamp(min=1e-4).to(device).to(dtype)
    scales = scales / (scales.max() * scales.min()).sqrt()

    pre_fc.weight.div_(scales.view(-1, 1))

    if getattr(pre_fc, 'bias', None) is not None:
        pre_fc.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in pre_fc.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

    return scales
