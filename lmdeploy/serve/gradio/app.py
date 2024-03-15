# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal, Optional, Union

from lmdeploy.archs import get_task
from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.model import ChatTemplateConfig


def run(model_path_or_server: str,
        server_name: str = '0.0.0.0',
        server_port: int = 6006,
        batch_size: int = 32,
        backend: Literal['turbomind', 'pytorch'] = 'turbomind',
        backend_config: Optional[Union[PytorchEngineConfig,
                                       TurbomindEngineConfig]] = None,
        chat_template_config: Optional[ChatTemplateConfig] = None,
        tp: int = 1,
        model_name: str = None,
        **kwargs):
    """chat with AI assistant through web ui.

    Args:
        model_path_or_server (str): the path of the deployed model or the
            tritonserver URL or restful api URL. For example:
            - ./workspace
            - 0.0.0.0:23333
            - http://0.0.0.0:23333
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
        batch_size (int): batch size for running Turbomind directly
        backend (str): either `turbomind` or `pytorch` backend. Default to
            `turbomind` backend.
        backend_config (TurbomindEngineConfig | PytorchEngineConfig): beckend
            config instance. Default to none.
        chat_template_config (ChatTemplateConfig): chat template configuration.
            Default to None.
        tp (int): tensor parallel for Turbomind
    """
    if ':' in model_path_or_server:
        if 'http:' in model_path_or_server:
            from lmdeploy.serve.gradio.api_server_backend import run_api_server
            run_api_server(model_path_or_server, server_name, server_port,
                           batch_size)
        else:
            from lmdeploy.serve.gradio.triton_server_backend import \
                run_triton_server
            run_triton_server(model_path_or_server, server_name, server_port)
    else:
        pipeline_type, _ = get_task(model_path_or_server)
        if pipeline_type == 'vlm':
            from lmdeploy.serve.gradio.vl import run_local
            assert backend == 'turbomind', 'vlm only support turbomind backend'
            if backend_config is not None and \
                    backend_config.session_len is None:
                backend_config.session_len = 8192
        else:
            from lmdeploy.serve.gradio.turbomind_coupled import run_local
        run_local(model_path_or_server,
                  server_name=server_name,
                  server_port=server_port,
                  backend=backend,
                  backend_config=backend_config,
                  chat_template_config=chat_template_config,
                  model_name=model_name,
                  batch_size=batch_size,
                  tp=tp,
                  **kwargs)


if __name__ == '__main__':
    import fire

    fire.Fire(run)
