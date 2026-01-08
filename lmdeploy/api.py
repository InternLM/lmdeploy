# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import TYPE_CHECKING, List, Literal, Optional, Union

from .pipeline import Pipeline

if TYPE_CHECKING:
    from .messages import PytorchEngineConfig, SpeculativeConfig, TurbomindEngineConfig
    from .model import ChatTemplateConfig


def pipeline(model_path: str,
             backend_config: Optional[Union['TurbomindEngineConfig', 'PytorchEngineConfig']] = None,
             chat_template_config: Optional['ChatTemplateConfig'] = None,
             log_level: str = 'WARNING',
             max_log_len: int = None,
             speculative_config: 'SpeculativeConfig' = None,
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
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): backend
            config instance. Default to None.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
        log_level(str): set log level whose value among [CRITICAL, ERROR,
            WARNING, INFO, DEBUG]
        max_log_len(int): Max number of prompt characters or prompt tokens
            being printed in log

    Examples:
        >>> # LLM
        >>> import lmdeploy
        >>> pipe = lmdeploy.pipeline('internlm/internlm-chat-7b')
        >>> response = pipe(['hi','say this is a test'])
        >>> print(response)
        >>>
        >>> # VLM
        >>> from lmdeploy.vl import load_image
        >>> from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
        >>> pipe = pipeline('liuhaotian/llava-v1.5-7b',
        ...                 backend_config=TurbomindEngineConfig(session_len=8192),
        ...                 chat_template_config=ChatTemplateConfig(model_name='vicuna'))
        >>> im = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
        >>> response = pipe([('describe this image', [im])])
        >>> print(response)
    """ # noqa E501

    return Pipeline(model_path,
                    backend_config=backend_config,
                    chat_template_config=chat_template_config,
                    log_level=log_level,
                    max_log_len=max_log_len,
                    speculative_config=speculative_config,
                    **kwargs)


def serve(model_path: str,
          model_name: Optional[str] = None,
          backend: Literal['turbomind', 'pytorch'] = 'turbomind',
          backend_config: Optional[Union['TurbomindEngineConfig', 'PytorchEngineConfig']] = None,
          chat_template_config: Optional['ChatTemplateConfig'] = None,
          server_name: str = '0.0.0.0',
          server_port: int = 23333,
          log_level: str = 'ERROR',
          api_keys: Optional[Union[List[str], str]] = None,
          ssl: bool = False,
          **kwargs):
    """This function is deprecated and no longer available.

    .. deprecated::
        This function has been removed. Please use alternative methods.

    This will run the api_server in a subprocess.
    """ # noqa E501
    warnings.warn("The 'serve' function is deprecated and no longer available. "
                  'Please use alternative methods.',
                  DeprecationWarning,
                  stacklevel=2)
    raise NotImplementedError("The 'serve' function is no longer available. "
                              'This function has been deprecated and removed.')


def client(api_server_url: str = 'http://0.0.0.0:23333', api_key: Optional[str] = None, **kwargs):
    """
    Args:
        api_server_url (str): communicating address 'http://<ip>:<port>' of
            api_server
        api_key (str | None): api key. Default to None, which means no
            api key will be used.
    Return:
        Chatbot for LLaMA series models with turbomind as inference engine.
    """
    from lmdeploy.serve.openai.api_client import APIClient
    return APIClient(api_server_url, api_key, **kwargs)
