# Copyright (c) OpenMMLab. All rights reserved.

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.pytorch.accel import LoadNoInit


def load_hf_from_pretrained(pretrained_model_name_or_path,
                            dtype=torch.float16,
                            **kwargs):

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Your device does not supports bf16(bfloat16), '
                           'please change to fp16(float16)')

    kwargs.pop('config', None)

    hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                           torch_dtype=dtype,
                                           trust_remote_code=True)

    # HACK hard code for qwen, other configs do not have the `fp16` attribute.
    if dtype == torch.float16:
        hf_config.fp16 = True
    elif dtype == torch.bfloat16:
        hf_config.bf16 = True

    with LoadNoInit():
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, config=hf_config, **kwargs)
        model.config.use_cache = False

    return model
