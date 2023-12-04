# Copyright (c) OpenMMLab. All rights reserved.
import os


def pipeline(model_path, instance_num=32, tp=1, log_level='ERROR', **kwargs):
    """
    Args:
        model_path (str): the path of the deployed model
        instance_num (int): instance numbers to be created
        tp (int): tensor parallel
        log_level(str): set log level whose value among
            [CRITICAL, ERROR, WARNING, INFO, DEBUG]
    """
    from lmdeploy.serve.async_engine import AsyncEngine
    os.environ['TM_LOG_LEVEL'] = log_level
    return AsyncEngine(model_path, instance_num=instance_num, tp=tp, **kwargs)


def serve(model_path: str,
          server_name: str = '0.0.0.0',
          server_port: int = 23333,
          instance_num: int = 64,
          tp: int = 1,
          log_level: str = 'ERROR',
          **kwargs):
    """This will run the api_server in a subprocess.

    Args:
        model_path (str): the path of the deployed model
        server_name (str): host ip for serving
        server_port (int): server port
        instance_num (int): number of instances of turbomind model
        tp (int): tensor parallel
        log_level(str): set log level whose value among [CRITICAL, ERROR, WARNING, INFO, DEBUG]
    """ # noqa E501
    from multiprocessing import Process

    from lmdeploy.serve.openai.api_server import serve
    task = Process(target=serve,
                   args=(model_path, ),
                   kwargs=dict(server_name=server_name,
                               server_port=server_port,
                               instance_num=instance_num,
                               tp=tp,
                               log_level=log_level,
                               **kwargs))
    task.start()


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
