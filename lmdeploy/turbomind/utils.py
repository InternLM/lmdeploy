# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import json
import logging
import os

from huggingface_hub import hf_hub_download
from transformers.utils import ExplicitEnum

logger = logging.getLogger(__name__)


class ModelSource(ExplicitEnum):
    """Turbomind model source."""
    WORKSPACE = 'workspace'
    HF_MODEL = 'hf_model'
    HF_LMDEPLOY = 'hf_lmdeploy'


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
    config = get_hf_config_content(pretrained_model_name_or_path, **kwargs)
    model_source = ModelSource.HF_LMDEPLOY if 'turbomind' in config \
        else ModelSource.HF_MODEL
    return model_source


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


@dataclasses.dataclass
class GenParam:
    top_p: float
    top_k: float
    temperature: float
    repetition_penalty: float
    sequence_start: bool = False
    sequence_end: bool = False
    step: int = 0
    request_output_len: int = 512


def get_gen_param(cap,
                  sampling_param,
                  nth_round,
                  step,
                  request_output_len=512,
                  **kwargs):
    """return parameters used by token generation."""
    gen_param = GenParam(**dataclasses.asdict(sampling_param),
                         request_output_len=request_output_len)
    # Fix me later. turbomind.py doesn't support None top_k
    if gen_param.top_k is None:
        gen_param.top_k = 40

    if cap == 'chat':
        gen_param.sequence_start = (nth_round == 1)
        gen_param.sequence_end = False
        gen_param.step = step
    else:
        gen_param.sequence_start = True
        gen_param.sequence_end = True
        gen_param.step = 0
    return gen_param
