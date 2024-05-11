# Copyright (c) OpenMMLab. All rights reserved.
import os

from lmdeploy.utils import get_hf_config_content, get_model

from .deepseek import DeepSeekVisionModel
from .internvl import InternVLVisionModel
from .internvl_llava import InternVLLlavaVisionModel
from .llava import LlavaVisionModel
from .mini_gemeni import MiniGeminiVisionModel
from .qwen import QwenVisionModel
from .xcomposer2 import Xcomposer2VisionModel
from .yi import YiVisionModel


def load_vl_model(model_path: str):
    """load visual model."""
    if not os.path.exists(model_path):
        model_path = get_model(model_path)
    config = get_hf_config_content(model_path)
    arch = config['architectures'][0]
    if 'auto_map' in config:
        for _, v in config['auto_map'].items():
            if 'InternLMXComposer2ForCausalLM' in v:
                arch = 'InternLMXComposer2ForCausalLM'
    if arch == 'QWenLMHeadModel':
        return QwenVisionModel(model_path)
    elif arch in ['LlavaLlamaForCausalLM', 'LlavaMistralForCausalLM']:
        projector_type = config.get('mm_projector_type', 'linear')
        mm_vision_tower = config.get('mm_vision_tower', '')
        if '_Norm' in projector_type:
            return YiVisionModel(model_path)
        elif 'OpenGVLab' in mm_vision_tower:
            return InternVLLlavaVisionModel(model_path)
        else:
            return LlavaVisionModel(model_path, arch=arch)
    elif arch == 'MultiModalityCausalLM':
        return DeepSeekVisionModel(model_path)
    elif arch == 'InternLMXComposer2ForCausalLM':
        return Xcomposer2VisionModel(model_path)
    elif arch == 'InternVLChatModel':
        return InternVLVisionModel(model_path)
    elif arch in ['MiniGeminiLlamaForCausalLM', 'MGMLlamaForCausalLM']:
        return MiniGeminiVisionModel(model_path)
    raise ValueError(f'unsupported vl model with arch {arch}')
