# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional, Union

from ..messages import PytorchEngineConfig, TurbomindEngineConfig
from ..model import ChatTemplateConfig


def pipeline(model_path: str,
             model_name: Optional[str] = None,
             backend_config: Optional[Union[TurbomindEngineConfig,
                                            PytorchEngineConfig]] = None,
             chat_template_config: Optional[ChatTemplateConfig] = None,
             log_level='ERROR',
             **kwargs):
    """
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
        model_name (str): needed when model_path is a pytorch model on
            huggingface.co, such as "internlm/internlm-chat-7b",
            "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat" and so on.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): beckend
            config instance. Default to None.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
        log_level(str): set log level whose value among [CRITICAL, ERROR, WARNING, INFO, DEBUG]

    Examples:
        >>> import lmdeploy
        >>> pipe = lmdeploy.pipeline('internlm/internlm-chat-7b')
        >>> response = pipe(['hi','say this is a test'])
        >>> print(response)
    """ # noqa E501
    from lmdeploy.serve.vl_async_engine import VLAsyncEngine
    if os.getenv('TM_LOG_LEVEL') is None:
        os.environ['TM_LOG_LEVEL'] = log_level
    from lmdeploy.utils import get_logger
    logger = get_logger('lmdeploy')
    logger.setLevel(log_level)
    backend = 'pytorch' if type(
        backend_config) is PytorchEngineConfig else 'turbomind'
    if 'tp' in kwargs:
        logger.warning(
            'The argument "tp" is deprecated and will be removed soon. '
            'Please set "tp" in "backend_config"')
        tp = kwargs['tp']
        kwargs.pop('tp')
    else:
        tp = 1 if backend_config is None else backend_config.tp
    return VLAsyncEngine(model_path,
                         model_name=model_name,
                         backend=backend,
                         backend_config=backend_config,
                         chat_template_config=chat_template_config,
                         tp=tp,
                         **kwargs)
