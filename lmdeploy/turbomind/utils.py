# Copyright (c) OpenMMLab. All rights reserved.
import os

from transformers.utils import ExplicitEnum

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class ModelSource(ExplicitEnum):
    """Turbomind model source."""
    WORKSPACE = 'workspace'
    HF_MODEL = 'hf_model'


def get_model_source(pretrained_model_name_or_path: str,
                     **kwargs) -> ModelSource:
    """Get model source."""
    triton_model_path = os.path.join(pretrained_model_name_or_path,
                                     'triton_models')
    if os.path.exists(triton_model_path):
        return ModelSource.WORKSPACE
    return ModelSource.HF_MODEL


def get_model_from_config(model_dir: str):
    import json
    config_file = os.path.join(model_dir, 'config.json')
    default = 'llama'
    if not os.path.exists(config_file):
        return default

    with open(config_file) as f:
        config = json.load(f)

    ARCH_MAP = {
        'LlavaLlamaForCausalLM': default,
        'LlamaForCausalLM': default,
        'InternLM2ForCausalLM': 'internlm2',
        'MultiModalityCausalLM': 'deepseekvl',
        'InternLMForCausalLM': default,
        'BaiChuanForCausalLM': 'baichuan',  # Baichuan-7B
        'BaichuanForCausalLM': 'baichuan2',  # not right for Baichuan-13B-Chat
        'QWenLMHeadModel': 'qwen',
    }

    arch = 'LlamaForCausalLM'
    if 'auto_map' in config:
        arch = config['auto_map']['AutoModelForCausalLM'].split('.')[-1]
    elif 'architectures' in config:
        arch = config['architectures'][0]

    return ARCH_MAP[arch]
