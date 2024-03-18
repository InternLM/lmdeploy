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


def autoget_backend(model_path: str) -> Union[Literal['turbomind', 'pytorch']]:
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
    try:
        from lmdeploy.turbomind.supported_models import \
            is_supported as is_supported_turbomind
        turbomind_has = is_supported_turbomind(model_path)
    except ImportError:
        logger.warning(
            'Lmdeploy with turbomind engine is not installed correctly. '
            'You may need to install lmdeploy from pypi or build from source '
            'for turbomind engine.')

    pytorch_has = is_supported_pytorch(model_path)

    if not (pytorch_has or turbomind_has):
        logger.warning(f'{model_path} is not explicitly supported by lmdeploy.'
                       f' Try to run with lmdeploy pytorch engine.')
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
        data = asdict(backend_config)
        for k, v in data.items():
            if v and hasattr(config, k):
                setattr(config, k, v)
    return config


def check_vl_llm(config: dict) -> bool:
    """check if the model is a vl model from model config."""
    arch = config['architectures'][0]
    if arch == 'LlavaLlamaForCausalLM':
        return True
    elif arch == 'QWenLMHeadModel' and 'visual' in config:
        return True
    return False


def get_task(model_path: str):
    """get pipeline type and pipeline class from model config."""
    if os.path.exists(model_path) and os.path.exists(
            os.path.join(model_path, 'triton_models', 'weights')):
        # workspace model
        return 'llm', AsyncEngine
    config = get_hf_config_content(model_path)
    if check_vl_llm(config):
        return 'vlm', VLAsyncEngine

    # default task, pipeline_class
    return 'llm', AsyncEngine
