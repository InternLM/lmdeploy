"""Opt-in vision-only XT RMS value contract; default preserves original classes.

Intended only for a separately reviewed -12h-chengqinxiu_1115 candidate.
Set the flag before importing the vision model; fresh workers are required
when changing it. Language RMSNorm is never replaced by this helper.
"""
from __future__ import annotations

import os

import torch
from torch import nn


ENV_NAME = 'LMDEPLOY_GLM53_VISION_XT_RMS_NORM'
ENABLED = os.environ.get(ENV_NAME, '0').strip().lower() in {'1', 'true', 'yes', 'on'}


class Glm53VisionXtRMSNorm(nn.Module):
    """Same FP32-statistics, cast-before-affine expression as XT vision.

The optional residual implements the actual native vision block's norm2 ABI:
first form its BF16 residual sum, then normalize that sum and return it.
This matches XT's explicit ``hidden_states + attention_output`` boundary.
Only unquantized, unsharded vision norms are admitted.
"""

    def __init__(self, hidden_size, eps=1e-6, dtype=None, device=None,
                 quant_config=None, tp=False, align=1, prefix='', all_reduce_group=None):
        super().__init__()
        if hidden_size not in (64, 1024) or quant_config is not None or tp or all_reduce_group is not None:
            raise ValueError('Vision XT RMS supports only unquantized TP1 D64/D1024 vision norms')
        if align != 1:
            raise ValueError('Vision XT RMS does not shard/alignment-pad norm weights')
        dtype = torch.float16 if dtype is None else dtype
        device = 'cuda' if device is None else device
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device), requires_grad=False)
        self.eps = eps
        self.variance_epsilon = eps

    def forward(self, x, residual=None):
        if x.shape[-1] != self.weight.numel():
            raise ValueError('Vision XT RMS last dimension does not match its HF weight')
        if residual is not None:
            x = x + residual
        input_dtype = x.dtype
        hidden_states = x.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states.to(input_dtype)
        return output if residual is None else (output, x)


def get_vision_rms_norm_class(original_class):
    """Return the identical original class object when the flag is off."""
    return Glm53VisionXtRMSNorm if ENABLED else original_class


def vision_qk_norm_eps(config_eps, original_eps=1e-6):
    return config_eps if ENABLED else original_eps
