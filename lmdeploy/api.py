# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import List, Literal, Optional, Union

from .archs import autoget_backend_config, get_task
from .messages import PytorchEngineConfig, SpeculativeConfig, TurbomindEngineConfig
from .model import ChatTemplateConfig


def pipeline(model_path: str,
             backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
             chat_template_config: Optional[ChatTemplateConfig] = None,
             log_level: str = 'WARNING',
             max_log_len: int = None,
             speculative_config: SpeculativeConfig = None,
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
    if os.getenv('TM_LOG_LEVEL') is None:
        os.environ['TM_LOG_LEVEL'] = log_level
    from lmdeploy.utils import get_logger, get_model
    logger = get_logger('lmdeploy')
    logger.setLevel(log_level)

    # model_path is not local path.
    if not os.path.exists(model_path):
        download_dir = backend_config.download_dir \
            if backend_config is not None else None
        revision = backend_config.revision \
            if backend_config is not None else None
        model_path = get_model(model_path, download_dir, revision)

    # spec model
    if speculative_config is not None and speculative_config.model and not os.path.exists(speculative_config.model):
        download_dir = backend_config.download_dir \
            if backend_config is not None else None
        speculative_config.model = get_model(speculative_config.model, download_dir)

    _, pipeline_class = get_task(model_path)
    if not isinstance(backend_config, PytorchEngineConfig):
        # set auto backend mode
        backend_config = autoget_backend_config(model_path, backend_config)
    backend = 'pytorch' if isinstance(backend_config, PytorchEngineConfig) else 'turbomind'
    logger.info(f'Using {backend} engine')

    return pipeline_class(model_path,
                          backend=backend,
                          backend_config=backend_config,
                          chat_template_config=chat_template_config,
                          max_log_len=max_log_len,
                          speculative_config=speculative_config,
                          **kwargs)


def serve(model_path: str,
          model_name: Optional[str] = None,
          backend: Literal['turbomind', 'pytorch'] = 'turbomind',
          backend_config: Optional[Union[TurbomindEngineConfig, PytorchEngineConfig]] = None,
          chat_template_config: Optional[ChatTemplateConfig] = None,
          server_name: str = '0.0.0.0',
          server_port: int = 23333,
          log_level: str = 'ERROR',
          api_keys: Optional[Union[List[str], str]] = None,
          ssl: bool = False,
          **kwargs):
    """This will run the api_server in a subprocess.

    Args:
        model_path: the path of a model.
            It could be one of the following options:

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

        model_name: the name of the served model. It can be accessed
            by the RESTful API ``/v1/models``. If it is not specified,
            ``model_path`` will be adopted
        backend: either ``turbomind`` or ``pytorch`` backend. Default to
            ``turbomind`` backend.
        backend_config: backend
            config instance. Default to none.
        chat_template_config: chat template configuration.
            Default to None.
        server_name: host ip for serving
        server_port: server port
        log_level: set log level whose value among
            [``CRITICAL``, ``ERROR``, ``WARNING``, ``INFO``, ``DEBUG``]
        api_keys: Optional list of API keys. Accepts string type as
            a single api_key. Default to None, which means no api key applied.
        ssl: Enable SSL. Requires OS Environment variables
            ``SSL_KEYFILE`` and ``SSL_CERTFILE``.

    Return:
        APIClient: A client chatbot for LLaMA series models.

    Examples:

        .. code-block:: python

            from lmdeploy.api import serve
            client = serve('internlm/internlm-chat-7b', 'internlm-chat-7b')
            for output in client.chat('hi', 1):
               print(output)
    """ # noqa E501
    import time
    from multiprocessing import Process

    from lmdeploy.serve.openai.api_client import APIClient
    from lmdeploy.serve.openai.api_server import serve

    if type(backend_config) is not PytorchEngineConfig:
        # set auto backend mode
        backend_config = autoget_backend_config(model_path, backend_config)
    backend = 'pytorch' if type(backend_config) is PytorchEngineConfig else 'turbomind'

    task = Process(target=serve,
                   args=(model_path, ),
                   kwargs=dict(model_name=model_name,
                               backend=backend,
                               backend_config=backend_config,
                               chat_template_config=chat_template_config,
                               server_name=server_name,
                               server_port=server_port,
                               log_level=log_level,
                               api_keys=api_keys,
                               ssl=ssl,
                               **kwargs),
                   daemon=True)
    task.start()
    client = APIClient(f'http://{server_name}:{server_port}')
    while True:
        time.sleep(1)
        try:
            client.available_models
            print(f'Launched the api_server in process {task.pid}, user can '
                  f'kill the server by:\nimport os,signal\nos.kill({task.pid}, '
                  'signal.SIGKILL)')
            return client
        except:  # noqa
            pass


def client(api_server_url: str = 'http://0.0.0.0:23333', api_key: Optional[str] = None, **kwargs):
    """
    Args:
        api_server_url: communicating address ``http://<ip>:<port>`` of
            api_server
        api_key: api key. Default to None, which means no
            api key will be used.
    Return:
        Chatbot for LLaMA series models with turbomind as inference engine.
    """
    from lmdeploy.serve.openai.api_client import APIClient
    return APIClient(api_server_url, api_key, **kwargs)
