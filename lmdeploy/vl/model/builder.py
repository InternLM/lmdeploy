# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.turbomind.utils import get_hf_config_content

from .llava import LlavaVLModel
from .qwen import QwenVLModel


def load_vl_model(model_path):
    config = get_hf_config_content(model_path)
    arch = config['architectures'][0]
    if arch == 'QWenLMHeadModel':
        return QwenVLModel(model_path)
    elif arch == 'LlavaLlamaForCausalLM':
        return LlavaVLModel(model_path)
    raise ValueError(f'unsupported val model with arch {arch}')
