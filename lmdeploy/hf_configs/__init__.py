# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache

from transformers import AutoConfig

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


@lru_cache
def register_config(model_type: str):
    """Register an LMDeploy-owned Transformers config when available."""
    if model_type == 'kimi_k2':
        # Standalone Kimi EAGLE checkpoints do not provide an auto_map.
        from .configuration_kimi_k2 import KimiK2Config
        AutoConfig.register(KimiK2Config.model_type, KimiK2Config)
    else:
        logger.debug(f'Can not register config for model_type: {model_type}')


def config_from_pretrained(pretrained_model_name_or_path: str, **kwargs):
    try:
        return AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    except Exception as error:
        logger.debug(f'AutoConfig.from_pretrained failed: {error}, try register config manually.')

    # Some models neither provide an auto_map nor have a native Transformers
    # config. Read model_type without executing remote code so LMDeploy can
    # register a local config, then fall back to the generic config when no
    # local implementation exists.
    from transformers import PretrainedConfig
    trust_remote_code = kwargs.pop('trust_remote_code', None)
    config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
    model_type = config_dict.get('model_type')
    if trust_remote_code is not None:
        kwargs['trust_remote_code'] = trust_remote_code
    register_config(model_type)
    try:
        return AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    except Exception:
        return PretrainedConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
