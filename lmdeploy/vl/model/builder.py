# Copyright (c) OpenMMLab. All rights reserved.
import os

from lmdeploy.turbomind.utils import get_hf_config_content
from lmdeploy.utils import get_model

from .llava import LlavaVLModel
from .qwen import QwenVLModel
from .yi import YiVLModel


def load_vl_model(model_path: str):
    """load visual model."""
    config = get_hf_config_content(model_path)
    if not os.path.exists(model_path):
        model_path = get_model(model_path)
    arch = config['architectures'][0]
    if arch == 'QWenLMHeadModel':
        return QwenVLModel(model_path)
    elif arch == 'LlavaLlamaForCausalLM':
        projector_type = config.get('mm_projector_type', 'linear')
        if '_Norm' in projector_type:
            return YiVLModel(model_path)
        else:
            return LlavaVLModel(model_path)
    raise ValueError(f'unsupported val model with arch {arch}')
