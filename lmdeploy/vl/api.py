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
        model_path (str): the path of a vl model like 'liuhaotian/llava-v1.5-7b'
            'liuhaotian/llava-v1.6-vicuna-7b', 'Qwen/Qwen-VL-Chat'.
        model_name (str): deprecated, use chat_template_config instead.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): beckend
            config instance. Default to None.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
        log_level(str): set log level whose value among [CRITICAL, ERROR, WARNING, INFO, DEBUG]

    Examples:
        >>> from lmdeploy.vl import pipeline, load_image_from_url
        >>> from lmdeploy import TurbomindEngineConfig, ChatTemplateConfig
        >>> pipe = pipeline('liuhaotian/llava-v1.5-7b',
        ...                 backend_config=TurbomindEngineConfig(session_len=8192),
        ...                 chat_template_config=ChatTemplateConfig(model_name='vicuna'))
        >>> im = load_image_from_url('https://bkimg.cdn.bcebos.com/pic/b8014a90f603738da97755563251a751f81986184626')
        >>> response = pipe([('describe this image', [im])])
        >>> print(response)
    """ # noqa E501
    from lmdeploy.serve.vl_async_engine import VLAsyncEngine
    if os.getenv('TM_LOG_LEVEL') is None:
        os.environ['TM_LOG_LEVEL'] = log_level
    from lmdeploy.utils import get_logger
    logger = get_logger('lmdeploy')
    logger.setLevel(log_level)
    assert type(backend_config) is TurbomindEngineConfig, \
        'Only turbomind backend is supported.'
    backend = 'turbomind'
    return VLAsyncEngine(model_path,
                         model_name=model_name,
                         backend=backend,
                         backend_config=backend_config,
                         chat_template_config=chat_template_config,
                         **kwargs)
