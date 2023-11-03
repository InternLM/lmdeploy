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
    },
    'QWenBlock': {
        'ln_1': ['attn.c_attn'],
        'ln_2': ['mlp.w1', 'mlp.w2']
    },
    'DecoderLayer': {
        'input_layernorm': ['self_attn.W_pack'],
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
    },
    'QWenBlock': {
        'attn.c_attn': ['attn.c_proj'],
        'mlp.w1': ['mlp.c_proj']
    },
    'DecoderLayer': {
        'self_attn.W_pack': ['self_attn.o_proj'],
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
              w_scales.pow(1 - alpha)).to(device).to(dtype)
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

    size_a = act_scales.size(0)
    size_pre_fc = pre_fc.weight.size(0)

    # (for llama2) use group query attention, pre_fc is v_proj, fc is o_proj
    if size_pre_fc < size_a and size_a % size_pre_fc == 0:
        return

    act_scales = act_scales.to(device=device, dtype=dtype)

    concat_w = torch.cat([fc.weight for fc in fcs], dim=0)
    w_scales = get_weight_scale(concat_w, group_size)

    scales = (act_scales.pow(alpha) /
              w_scales.pow(1 - alpha)).to(device).to(dtype)
    scales = scales / (scales.max() * scales.min()).sqrt()

    # (for qwen&baichuan) pre_fc is packed QKV, only V needs to scale
    if size_pre_fc > size_a and size_pre_fc % size_a == 0 \
            and size_pre_fc // size_a == 3:

        pre_fc.weight[-size_a:].div_(scales.view(-1, 1))

        if getattr(pre_fc, 'bias', None) is not None:
            pre_fc.bias[-size_a:].div_(scales)
    else:
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


def check_awq_supported(layer_type):
    """Check if the smooth function is supported by inspecting layer type."""
    norm_fcs_found = False
    fc_fcs_found = False

    if isinstance(layer_type, str):
        if layer_type in NORM_FCS_MAP:
            norm_fcs_found = True
        if layer_type in FC_FCS_MAP:
            fc_fcs_found = True

    elif isinstance(layer_type, type):
        if layer_type.__name__ in NORM_FCS_MAP:
            norm_fcs_found = True
        if layer_type.__name__ in FC_FCS_MAP:
            fc_fcs_found = True

    else:
        raise NotImplementedError

    if not norm_fcs_found:
        raise NotImplementedError

    if not fc_fcs_found:
        raise NotImplementedError


def quant_weights(model, fcs, bits, symmetry, group_size=-1, device='cuda'):
    """Quantize the weights of the target model's linear layers."""
    from lmdeploy.lite.quantization import WeightQuantizer
    from lmdeploy.pytorch.modules import WeightOnlyQLinear
    for name, fc in fcs.items():
        fc.to(device)
        quantizer = WeightQuantizer(bits, symmetry, 'per_group', group_size)
        q_linear = WeightOnlyQLinear.from_linear(fc, quantizer)

        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name)
        fc.to('cpu')
        setattr(parent, child_name, q_linear)

        print(f'{name} weight packed.')


def smooth_layers(layers,
                  fc2fcs,
                  norm2fcs,
                  a_scales,
                  group_size=-1,
                  device='cuda'):
    """Apply weight smoothing based on input scales."""

    for l_name, layer in layers.items():
        layer.to(device)
        for ln_name, fc_names in norm2fcs.items():
            a_name = [f'{l_name}.{n}' for n in fc_names][0]

            ln = layer.get_submodule(ln_name)
            fcs = [layer.get_submodule(n) for n in fc_names]
            smooth_ln_fcs(ln, fcs, a_scales[a_name], group_size)

        for f_name, fc_names in fc2fcs.items():
            a_name = [f'{l_name}.{n}' for n in fc_names][0]

            fc = layer.get_submodule(f_name)
            fcs = [layer.get_submodule(n) for n in fc_names]

            smooth_fc_fcs(fc, fcs, a_scales[a_name], group_size)

        layer.to('cpu')
        print(f'{l_name} smooth weight done.')
