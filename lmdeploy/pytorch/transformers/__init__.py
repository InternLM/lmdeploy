# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache

from transformers import AutoConfig

from lmdeploy.utils import get_logger


@lru_cache()
def register_config(model_type: str):
    if model_type == 'deepseek_v32':
        from lmdeploy.pytorch.transformers.configuration_deepseek_v32 import DeepseekV32Config
        AutoConfig.register(DeepseekV32Config.model_type, DeepseekV32Config)
    else:
        logger.debug(f'Can not register config for model_type: {model_type}')


logger = get_logger('lmdeploy')


def config_from_pretrained(pretrained_model_name_or_path: str, **kwargs):
    try:
        return AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    except ValueError as e:
        logger.debug(f'AutoConfig.from_pretrained failed: {e}, try register config manually.')
        # some models (dsv32) does not provide auto map for config
        from transformers import PretrainedConfig
        trust_remote_code = kwargs.pop('trust_remote_code', None)
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        model_type = config_dict.get('model_type', None)
        if trust_remote_code is not None:
            kwargs['trust_remote_code'] = trust_remote_code
        register_config(model_type)

    return AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
