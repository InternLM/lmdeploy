# Copyright (c) OpenMMLab. All rights reserved.

from typing import Literal

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.pytorch.accel import LoadNoInit


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
