# Copyright (c) OpenMMLab. All rights reserved.
import json
import os

from huggingface_hub import hf_hub_download
from transformers.utils import ExplicitEnum

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class ModelSource(ExplicitEnum):
    """Turbomind model source."""
    WORKSPACE = 'workspace'
    HF_MODEL = 'hf_model'


def create_hf_download_args(**kwargs) -> dict:
    download_kwargs = {
        'revision': None,
        'cache_dir': None,
        'proxies': None,
        'resume_download': True,
        'force_download': False,
        'token': None,
        'local_files_only': False
    }
    for k in download_kwargs.keys():
        if k in kwargs:
            download_kwargs[k] = kwargs[k]
    return download_kwargs


def get_hf_config_path(pretrained_model_name_or_path, **kwargs) -> str:
    """Get local hf config local file path."""
    if os.path.exists(pretrained_model_name_or_path):
        config_path = os.path.join(pretrained_model_name_or_path,
                                   'config.json')
    else:
        download_kwargs = create_hf_download_args(**kwargs)
        config_path = hf_hub_download(pretrained_model_name_or_path,
                                      'config.json', **download_kwargs)
    return config_path


def get_hf_config_content(pretrained_model_name_or_path, **kwargs) -> dict:
    """Get config content of a hf model."""
    config_path = get_hf_config_path(pretrained_model_name_or_path, **kwargs)
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_model_source(pretrained_model_name_or_path: str,
                     **kwargs) -> ModelSource:
    """Get model source."""
    triton_model_path = os.path.join(pretrained_model_name_or_path,
                                     'triton_models')
    if os.path.exists(triton_model_path):
        return ModelSource.WORKSPACE
    return ModelSource.HF_MODEL


def check_tm_model_input(pretrained_model_name_or_path, **kwargs):
    """Check if single input pretrained_model_name_or_path is enough to use."""
    if kwargs.get('model_name', None):
        return

    model_source = get_model_source(pretrained_model_name_or_path, **kwargs)
    if model_source == ModelSource.WORKSPACE:
        return

    config = get_hf_config_content(pretrained_model_name_or_path, **kwargs)
    if 'turbomind' in config and config['turbomind']['model_name'] != '':
        return

    assert (0), '\nCan not get model name from input model, '\
        'please supply model name with arg --model-name,' \
        'you can list supported models by `lmdeploy list`'


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
