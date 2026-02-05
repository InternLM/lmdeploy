# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal

from typing_extensions import deprecated

from .pipeline import Pipeline

if TYPE_CHECKING:
    from .messages import PytorchEngineConfig, SpeculativeConfig, TurbomindEngineConfig
    from .model import ChatTemplateConfig


def pipeline(model_path: str,
             backend_config: 'TurbomindEngineConfig' | 'PytorchEngineConfig' | None = None,
             chat_template_config: 'ChatTemplateConfig' | None = None,
             log_level: str = 'WARNING',
             max_log_len: int | None = None,
             speculative_config: 'SpeculativeConfig' | None = None,
             **kwargs):
    """
    Args:
        model_path: the path of a model. It could be one of the following options:

            - i) A local directory path of a turbomind model which is
              converted by ``lmdeploy convert`` command or download from
              ii) and iii).
            - ii) The model_id of a lmdeploy-quantized model hosted
              inside a model repo on huggingface.co, such as
              ``InternLM/internlm-chat-20b-4bit``,
              ``lmdeploy/llama2-chat-70b-4bit``, etc.
            - iii) The model_id of a model hosted inside a model repo
              on huggingface.co, such as ``internlm/internlm-chat-7b``,
              ``Qwen/Qwen-7B-Chat``, ``baichuan-inc/Baichuan2-7B-Chat``
              and so on.
        backend_config: backend
            config instance. Default to None.
        chat_template_config: chat template configuration.
            Default to None.
        log_level: set log level whose value among [``CRITICAL``, ``ERROR``,
            ``WARNING``, ``INFO``, ``DEBUG``]
        max_log_len: Max number of prompt characters or prompt tokens
            being printed in log

    Examples:

        .. code-block:: python

            # LLM
            import lmdeploy
            pipe = lmdeploy.pipeline('internlm/internlm-chat-7b')
            response = pipe(['hi','say this is a test'])
            print(response)

            # VLM
            from lmdeploy.vl import load_image
            from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
            pipe = pipeline('liuhaotian/llava-v1.5-7b',
                            backend_config=TurbomindEngineConfig(session_len=8192),
                            chat_template_config=ChatTemplateConfig(model_name='vicuna'))
            im = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg')
            response = pipe([('describe this image', [im])])
            print(response)

    """ # noqa E501

    return Pipeline(model_path,
                    backend_config=backend_config,
                    chat_template_config=chat_template_config,
                    log_level=log_level,
                    max_log_len=max_log_len,
                    speculative_config=speculative_config,
                    **kwargs)


@deprecated('This function is no longer available. Please use CLI command "lmdeploy serve api_server" instead.')
def serve(model_path: str,
          model_name: str | None = None,
          backend: Literal['turbomind', 'pytorch'] = 'turbomind',
          backend_config: 'TurbomindEngineConfig' | 'PytorchEngineConfig' | None = None,
          chat_template_config: 'ChatTemplateConfig' | None = None,
          server_name: str = '0.0.0.0',
          server_port: int = 23333,
          log_level: str = 'ERROR',
          api_keys: List[str] | str | None = None,
          ssl: bool = False,
          **kwargs):
    """This function is deprecated and no longer available.

    .. deprecated::
        This function has been removed. Please use alternative methods.

    This will run the api_server in a subprocess.
    """ # noqa E501
    raise NotImplementedError("The 'serve' function is no longer available. "
                              'This function has been deprecated and removed.')


@deprecated('This function is no longer available. Please use "from lmdeploy.serve import APIClient" instead.')
def client(api_server_url: str = 'http://0.0.0.0:23333', api_key: str | None = None, **kwargs):
    """This function is deprecated and no longer available.

    .. deprecated::
        This function has been removed. Please use ``from lmdeploy.serve import APIClient`` instead.

    Args:
        api_server_url: communicating address ``http://<ip>:<port>`` of
            api_server
        api_key: api key. Default to None, which means no
            api key will be used.
    Return:
        Chatbot for LLaMA series models with turbomind as inference engine.
    """
    raise NotImplementedError("The 'client' function is no longer available. This function has been deprecated. "
                              ' Please use "from lmdeploy.serve import APIClient" instead.')
