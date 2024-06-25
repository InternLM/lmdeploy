# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional, Union

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.utils import get_hf_config_content, get_model

from .cogvlm import CogVLMVisionModel
from .deepseek import DeepSeekVisionModel
from .internvl import InternVLVisionModel
from .internvl_llava import InternVLLlavaVisionModel
from .llava import LlavaVisionModel
from .llava_hf import LlavaHfVisionModel
from .llava_next import LlavaNextVisionModel
from .mini_gemeni import MiniGeminiVisionModel
from .minicpmv import MiniCPMVModel
from .qwen import QwenVisionModel
from .xcomposer2 import Xcomposer2VisionModel
from .yi import YiVisionModel


def load_vl_model(model_path: str,
                  with_llm: bool = False,
                  backend_config: Optional[Union[TurbomindEngineConfig,
                                                 PytorchEngineConfig]] = None):
    """load visual model."""
    if not os.path.exists(model_path):
        revision = getattr(backend_config, 'revision', None)
        download_dir = getattr(backend_config, 'download_dir', None)
        model_path = get_model(model_path,
                               revision=revision,
                               download_dir=download_dir)
    config = get_hf_config_content(model_path)
    arch = config['architectures'][0]
    if 'auto_map' in config:
        for _, v in config['auto_map'].items():
            if 'InternLMXComposer2ForCausalLM' in v:
                arch = 'InternLMXComposer2ForCausalLM'
    if arch == 'QWenLMHeadModel':
        return QwenVisionModel(model_path, with_llm)
    elif arch in ['LlavaLlamaForCausalLM', 'LlavaMistralForCausalLM']:
        projector_type = config.get('mm_projector_type', 'linear')
        mm_vision_tower = config.get('mm_vision_tower', '')
        if '_Norm' in projector_type:
            return YiVisionModel(model_path, with_llm)
        elif 'OpenGVLab' in mm_vision_tower:
            return InternVLLlavaVisionModel(model_path, with_llm)
        else:
            return LlavaVisionModel(model_path, with_llm=with_llm, arch=arch)
    if arch == 'MultiModalityCausalLM':
        return DeepSeekVisionModel(model_path, with_llm)
    elif arch == 'CogVLMForCausalLM':
        return CogVLMVisionModel(model_path, with_llm)
    if arch == 'InternLMXComposer2ForCausalLM':
        return Xcomposer2VisionModel(model_path, with_llm)
    if arch == 'InternVLChatModel':
        return InternVLVisionModel(model_path, with_llm)
    if arch in ['MiniGeminiLlamaForCausalLM', 'MGMLlamaForCausalLM']:
        return MiniGeminiVisionModel(model_path, with_llm)
    if arch == 'MiniCPMV':
        return MiniCPMVModel(model_path, with_llm)
    if arch == 'LlavaForConditionalGeneration':
        return LlavaHfVisionModel(model_path, with_llm)
    if arch == 'LlavaNextForConditionalGeneration':
        return LlavaNextVisionModel(model_path, with_llm)
    raise ValueError(f'unsupported vl model with arch {arch}')


def vl_model_with_tokenizer(model_path: str, with_llm: bool = True):
    """load visual model."""
    vl_model = load_vl_model(model_path, with_llm).vl_model
    llm = vl_model
    if hasattr(vl_model, 'language_model'):  # deepseek vl
        llm = vl_model.language_model
    if hasattr(vl_model, 'llm'):  # MiniCPMV
        llm = vl_model.llm
    llm.config.use_cache = False
    llm.half().eval()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    return vl_model, llm, tokenizer
