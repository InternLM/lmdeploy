"""Private vision-projector LayerNorm contract; default off, no global patch.

On constructs the same public torch.nn.LayerNorm used by the actual XTuner
projector, with BF16 parameters loaded normally from the unchanged HF names.
Only this vision merger receives the factory. Language LayerNorm is untouched.
"""
import os
import torch
from torch import nn

FLAG = 'LMDEPLOY_GLM53_VISION_XT_PROJECTOR_LN'
ENABLED = os.environ.get(FLAG, '0') == '1'


def build_vision_projector_norm(original_class, hidden_size, *, dtype, device):
    if not ENABLED:
        return original_class(hidden_size, eps=1e-6, dtype=torch.float32, device=device)
    if hidden_size != 4096 or dtype != torch.bfloat16:
        raise ValueError('Private vision projector LayerNorm requires BF16 D4096')
    result = nn.LayerNorm(hidden_size, eps=1e-5, elementwise_affine=True, bias=True,
                          dtype=dtype, device=device)
    result.requires_grad_(False)
    result._glm53_vision_projector_ln_chengqinxiu = True
    return result


def vision_projector_norm_input(value):
    if not ENABLED:
        return value.float()
    if value.dtype != torch.bfloat16 or value.ndim != 2 or value.shape[-1] != 4096:
        raise ValueError('Private vision projector LayerNorm requires [N,4096] BF16 input')
    return value
