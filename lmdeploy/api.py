# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Literal, Optional, Union

from lmdeploy.model import ChatTemplateConfig
from lmdeploy.pytorch import EngineConfig as PytorchEngineConfig
from lmdeploy.turbomind import EngineConfig as TurbomindEngineConfig


def pipeline(model_path: str,
             model_name: Optional[str] = None,
             backend: Literal['turbomind', 'pytorch'] = 'turbomind',
             backend_config: Optional[Union[TurbomindEngineConfig,
                                            PytorchEngineConfig]] = None,
             chat_template_config: Optional[ChatTemplateConfig] = None,
             instance_num: int = 32,
             tp: int = 1,
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
        backend (str): either `turbomind` or `pytorch` backend. Default to
            `turbomind` backend.
        backend_config (EngineConfig): beckend config. Default to none.
        instance_num (int): instance numbers to be created
        tp (int): tensor parallel
        log_level(str): set log level whose value among [CRITICAL, ERROR, WARNING, INFO, DEBUG]

    Examples:
        >>> import lmdeploy
        >>> pipe = lmdeploy.pipeline('internlm/internlm-chat-7b')
        >>> response = pipe(['hi','say this is a test'])
        >>> print(response)
    """ # noqa E501
    from lmdeploy.serve.async_engine import AsyncEngine
    os.environ['TM_LOG_LEVEL'] = log_level
    return AsyncEngine(model_path,
                       model_name=model_name,
                       backend=backend,
                       backend_config=backend_config,
                       chat_template_config=chat_template_config,
                       instance_num=instance_num,
                       tp=tp,
                       **kwargs)


def serve(model_path: str,
          model_name: Optional[str] = None,
          server_name: str = '0.0.0.0',
          server_port: int = 23333,
          instance_num: int = 64,
          tp: int = 1,
          log_level: str = 'ERROR',
          **kwargs):
    """This will run the api_server in a subprocess.

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
        server_name (str): host ip for serving
        server_port (int): server port
        instance_num (int): number of instances of turbomind model
        tp (int): tensor parallel
        log_level(str): set log level whose value among [CRITICAL, ERROR, WARNING, INFO, DEBUG]

    Return:
        APIClient: A client chatbot for LLaMA series models.

    Examples:
        >>> import lmdeploy
        >>> client = lmdeploy.serve('internlm/internlm-chat-7b', 'internlm-chat-7b')
        >>> for output in client.chat('hi', 1):
        ...    print(output)
    """ # noqa E501
    import time
    from multiprocessing import Process

    from lmdeploy.serve.openai.api_client import APIClient
    from lmdeploy.serve.openai.api_server import serve
    task = Process(target=serve,
                   args=(model_path, ),
                   kwargs=dict(model_name=model_name,
                               server_name=server_name,
                               server_port=server_port,
                               instance_num=instance_num,
                               tp=tp,
                               log_level=log_level,
                               **kwargs))
    task.start()
    client = APIClient(f'http://{server_name}:{server_port}')
    while True:
        time.sleep(1)
        try:
            client.available_models
            return client
        except:  # noqa
            pass


def client(api_server_url: str = 'http://0.0.0.0:23333', **kwargs):
    """
    Args:
        api_server_url (str): communicating address 'http://<ip>:<port>' of
            api_server
    Return:
        Chatbot for LLaMA series models with turbomind as inference engine.
    """
    from lmdeploy.serve.openai.api_client import APIClient
    return APIClient(api_server_url, **kwargs)
