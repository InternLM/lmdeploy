# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Literal, Optional, Union

from lmdeploy.serve.async_engine import AsyncEngine
from lmdeploy.serve.vl_async_engine import VLAsyncEngine
from lmdeploy.utils import get_hf_config_content

from .messages import PytorchEngineConfig, TurbomindEngineConfig
from .utils import get_logger

SUPPORTED_TASKS = {'llm': AsyncEngine, 'vlm': VLAsyncEngine}

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
    from lmdeploy.pytorch.supported_models import \
        is_supported as is_supported_pytorch

    pytorch_has, turbomind_has = False, False
    is_turbomind_installed = True
    try:
        from lmdeploy.turbomind.supported_models import \
            is_supported as is_supported_turbomind
        turbomind_has = is_supported_turbomind(model_path)
    except ImportError:
        is_turbomind_installed = False

    pytorch_has = is_supported_pytorch(model_path)

    try_run_msg = (f'Try to run with pytorch engine because `{model_path}`'
                   ' is not explicitly supported by lmdeploy. ')
    if is_turbomind_installed:
        if not turbomind_has:
            if pytorch_has:
                logger.warning('Fallback to pytorch engine because '
                               f'`{model_path}` not supported by turbomind'
                               ' engine.')
            else:
                logger.warning(try_run_msg)
    else:
        logger.warning(
            'Fallback to pytorch engine because turbomind engine is not '
            'installed correctly. If you insist to use turbomind engine, '
            'you may need to reinstall lmdeploy from pypi or build from '
            'source and try again.')
        if not pytorch_has:
            logger.warning(try_run_msg)

    backend = 'turbomind' if turbomind_has else 'pytorch'
    return backend


def autoget_backend_config(
    model_path: str,
    backend_config: Optional[Union[PytorchEngineConfig,
                                   TurbomindEngineConfig]] = None
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
    """check if the model is a vl model from model config."""
    if 'auto_map' in config:
        for _, v in config['auto_map'].items():
            if 'InternLMXComposer2ForCausalLM' in v:
                return True
    arch = config['architectures'][0]
    if arch == 'LlavaLlamaForCausalLM':
        return True
    elif arch == 'QWenLMHeadModel' and 'visual' in config:
        return True
    elif arch == 'MultiModalityCausalLM' and 'language_config' in config:
        return True
    elif arch == 'InternLMXComposer2ForCausalLM':
        return True
    elif arch == 'InternVLChatModel':
        return True
    elif arch in ['MiniGeminiLlamaForCausalLM', 'MGMLlamaForCausalLM']:
        return True
    return False


def get_task(model_path: str):
    """get pipeline type and pipeline class from model config."""
    if os.path.exists(os.path.join(model_path, 'triton_models', 'weights')):
        # workspace model
        return 'llm', AsyncEngine
    config = get_hf_config_content(model_path)
    if check_vl_llm(config):
        return 'vlm', VLAsyncEngine

    # default task, pipeline_class
    return 'llm', AsyncEngine
