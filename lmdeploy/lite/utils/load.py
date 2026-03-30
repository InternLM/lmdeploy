# Copyright (c) OpenMMLab. All rights reserved.

from typing import Literal

import torch
from transformers import AutoConfig, AutoModelForCausalLM


class LoadNoInit:
    """Initialize model without parameter initialization."""

    def __init__(self):
        self.constant_ = torch.nn.init.constant_
        self.zeros_ = torch.nn.init.zeros_
        self.ones_ = torch.nn.init.ones_
        self.uniform_ = torch.nn.init.uniform_
        self.normal_ = torch.nn.init.normal_
        self.kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        self.kaiming_normal_ = torch.nn.init.kaiming_normal_
        self.tensor_normal_ = torch.Tensor.normal_

    def __enter__(self, *args, **kwargs):
        """Replace initializers with no-op."""

        torch.nn.init.constant_ = lambda *args, **kwargs: None
        torch.nn.init.zeros_ = lambda *args, **kwargs: None
        torch.nn.init.ones_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None
        torch.Tensor.normal_ = lambda *args, **kwargs: None

    def __exit__(self, *args, **kwargs):
        """Recover."""

        torch.nn.init.constant_ = self.constant_
        torch.nn.init.zeros_ = self.zeros_
        torch.nn.init.ones_ = self.ones_
        torch.nn.init.uniform_ = self.uniform_
        torch.nn.init.normal_ = self.normal_
        torch.nn.init.kaiming_uniform_ = self.kaiming_uniform_
        torch.nn.init.kaiming_normal_ = self.kaiming_normal_
        torch.Tensor.normal_ = self.tensor_normal_


def load_hf_from_pretrained(pretrained_model_name_or_path, dtype: Literal['float16', 'bfloat16', 'auto'], **kwargs):

    if dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Your device does not supports bf16(bfloat16), '
                           'please change to fp16(float16)')

    kwargs.pop('config', None)

    hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

    # HACK hard code for qwen, other configs do not have the `fp16` attribute.
    if hasattr(hf_config, 'fp16') or hasattr(hf_config, 'bf16'):
        if dtype == 'bfloat16':
            hf_config.bf16 = True
        else:
            hf_config.fp16 = True

    torch_dtype = getattr(hf_config, 'torch_dtype', torch.float16)
    if dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif dtype == 'float16':
        torch_dtype = torch.float16
    elif dtype == 'auto' and torch_dtype == torch.bfloat16:
        print('Warning: we cast model to float16 to prevent OOM. '
              'You may enforce it bfloat16 by `--dtype bfloat16`')
        torch_dtype = torch.float16

    with LoadNoInit():
        # Load model
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path,
                                                     config=hf_config,
                                                     torch_dtype=torch_dtype,
                                                     **kwargs)
        model.config.use_cache = False

    return model
