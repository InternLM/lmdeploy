# Copyright (c) OpenMMLab. All rights reserved.
import os

from lmdeploy.utils import get_hf_config_content, get_model

from .deepseek import DeepSeekVisionModel
from .internvl import InternVLVisionModel
from .llava import LlavaVisionModel
from .qwen import QwenVisionModel
from .yi import YiVisionModel


def load_vl_model(model_path: str):
    """load visual model."""
    if not os.path.exists(model_path):
        model_path = get_model(model_path)
    config = get_hf_config_content(model_path)
    arch = config['architectures'][0]
    if arch == 'QWenLMHeadModel':
        return QwenVisionModel(model_path)
    elif arch == 'LlavaLlamaForCausalLM':
        projector_type = config.get('mm_projector_type', 'linear')
        if '_Norm' in projector_type:
            return YiVisionModel(model_path)
        else:
            return LlavaVisionModel(model_path)
    if arch == 'MultiModalityCausalLM':
        return DeepSeekVisionModel(model_path)
    if arch == 'InternVLChatModel':
        return InternVLVisionModel(model_path)
    raise ValueError(f'unsupported vl model with arch {arch}')
