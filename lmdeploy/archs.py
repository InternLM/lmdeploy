# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Literal, Optional, Union

from transformers import AutoConfig

from .messages import PytorchEngineConfig, TurbomindEngineConfig
from .utils import get_logger

logger = get_logger('lmdeploy')


def autoget_backend(model_path: str) -> Literal['turbomind', 'pytorch']:
    """Get backend type in auto backend mode.

    Args:
         model_path (str): the path of a model.
            It could be one of the following options:
                - i) A local directory path of a turbomind model which is
                    converted by `lmdeploy convert` command or download from
                    ii) and iii).
                - ii) The model_id of a lmdeploy-quantized model hosted
                    inside a model repo on huggingface.co, such as
                    "InternLM/internlm-chat-20b-4bit",
                    "lmdeploy/llama2-chat-70b-4bit", etc.
                - iii) The model_id of a model hosted inside a model repo
                    on huggingface.co, such as "internlm/internlm-chat-7b",
                    "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                    and so on.

    Returns:
        str: the backend type.
    """

    turbomind_has = False
    is_turbomind_installed = True
    try:
        from lmdeploy.turbomind.supported_models import is_supported as is_supported_turbomind
        turbomind_has = is_supported_turbomind(model_path)
    except ImportError:
        is_turbomind_installed = False

    if is_turbomind_installed:
        if not turbomind_has:
            logger.warning('Fallback to pytorch engine because '
                           f'`{model_path}` not supported by turbomind'
                           ' engine.')
    else:
        logger.warning('Fallback to pytorch engine because turbomind engine is not '
                       'installed correctly. If you insist to use turbomind engine, '
                       'you may need to reinstall lmdeploy from pypi or build from '
                       'source and try again.')

    backend = 'turbomind' if turbomind_has else 'pytorch'
    return backend


def autoget_backend_config(
    model_path: str,
    backend_config: Optional[Union[PytorchEngineConfig, TurbomindEngineConfig]] = None
) -> Union[PytorchEngineConfig, TurbomindEngineConfig]:
    """Get backend config automatically.

    Args:
        model_path (str): The input model path.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): The
            input backend config. Default to None.

    Returns:
        (PytorchEngineConfig | TurbomindEngineConfig): The auto-determined
            backend engine config.
    """
    from dataclasses import asdict

    backend = autoget_backend(model_path)
    if backend == 'pytorch':
        config = PytorchEngineConfig()
    else:
        config = TurbomindEngineConfig()
    if backend_config is not None:
        if type(backend_config) == type(config):
            return backend_config
        else:
            data = asdict(backend_config)
            for k, v in data.items():
                if v and hasattr(config, k):
                    setattr(config, k, v)
            # map attributes with different names
            if type(backend_config) is TurbomindEngineConfig:
                config.block_size = backend_config.cache_block_seq_len
            else:
                config.cache_block_seq_len = backend_config.block_size
    return config


def check_vl_llm(config: dict) -> bool:
    """Check if the model is a vl model from model config."""
    if 'auto_map' in config:
        for _, v in config['auto_map'].items():
            if 'InternLMXComposer2ForCausalLM' in v:
                return True

    if 'language_config' in config and 'vision_config' in config and config['language_config'].get(
            'architectures', [None])[0] == 'DeepseekV2ForCausalLM':
        return True

    arch = config['architectures'][0]
    supported_archs = set([
        'LlavaLlamaForCausalLM', 'LlavaMistralForCausalLM', 'CogVLMForCausalLM', 'InternLMXComposer2ForCausalLM',
        'InternVLChatModel', 'MiniCPMV', 'LlavaForConditionalGeneration', 'LlavaNextForConditionalGeneration',
        'Phi3VForCausalLM', 'Qwen2VLForConditionalGeneration', 'Qwen2_5_VLForConditionalGeneration',
        'Qwen3VLForConditionalGeneration', 'Qwen3VLMoeForConditionalGeneration', 'MllamaForConditionalGeneration',
        'MolmoForCausalLM', 'Gemma3ForConditionalGeneration', 'Llama4ForConditionalGeneration',
        'InternVLForConditionalGeneration', 'InternS1ForConditionalGeneration', 'Glm4vForConditionalGeneration'
    ])
    if arch == 'QWenLMHeadModel' and 'visual' in config:
        return True
    elif arch == 'MultiModalityCausalLM' and 'language_config' in config:
        return True
    elif arch in ['ChatGLMModel', 'ChatGLMForConditionalGeneration'] and 'vision_config' in config:
        return True
    elif arch in supported_archs:
        return True
    return False


def get_task(model_path: str):
    """Get pipeline type and pipeline class from model config."""
    from lmdeploy.serve.async_engine import AsyncEngine

    if os.path.exists(os.path.join(model_path, 'triton_models', 'weights')):
        # workspace model
        return 'llm', AsyncEngine
    _, config = get_model_arch(model_path)
    if check_vl_llm(config.to_dict()):
        from lmdeploy.serve.vl_async_engine import VLAsyncEngine
        return 'vlm', VLAsyncEngine

    # default task, pipeline_class
    return 'llm', AsyncEngine


def get_model_arch(model_path: str):
    """Get a model's architecture and configuration.

    Args:
        model_path(str): the model path
    """
    if os.path.exists(os.path.join(model_path, 'triton_models', 'weights')):
        # the turbomind model
        import yaml
        config_file = os.path.join(model_path, 'triton_models', 'weights', 'config.yaml')
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        from .turbomind.deploy.config import TurbomindModelConfig
        tm_config = TurbomindModelConfig.from_dict(config)

        return tm_config.model_config.model_arch, tm_config
    else:
        # transformers model
        try:
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:  # noqa
            from transformers import PretrainedConfig
            cfg = PretrainedConfig.from_pretrained(model_path, trust_remote_code=True)

        _cfg = cfg.to_dict()
        if _cfg.get('architectures', None):
            arch = _cfg['architectures'][0]
            if _cfg.get('auto_map'):
                for _, v in _cfg['auto_map'].items():
                    if 'InternLMXComposer2ForCausalLM' in v:
                        arch = 'InternLMXComposer2ForCausalLM'
        elif _cfg.get('auto_map', None) and 'AutoModelForCausalLM' in _cfg['auto_map']:
            arch = _cfg['auto_map']['AutoModelForCausalLM'].split('.')[-1]
        elif _cfg.get('language_config', None) and _cfg['language_config'].get(
                'auto_map', None) and 'AutoModelForCausalLM' in _cfg['language_config']['auto_map']:
            arch = _cfg['language_config']['auto_map']['AutoModelForCausalLM'].split('.')[-1]
        else:
            raise RuntimeError(f'Could not find model architecture from config: {_cfg}')
        return arch, cfg


def search_nested_config(config, key):
    """Recursively searches for the value associated with the given key in a
    nested configuration of a model."""
    if isinstance(config, Dict):
        for k, v in config.items():
            if k == key:
                return v
            if isinstance(v, (Dict, List)):
                result = search_nested_config(v, key)
                if result is not None:
                    return result
    elif isinstance(config, List):
        for item in config:
            result = search_nested_config(item, key)
            if result is not None:
                return result
    return None
