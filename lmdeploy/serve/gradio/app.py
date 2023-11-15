# Copyright (c) OpenMMLab. All rights reserved.


def run(model_path_or_server: str,
        server_name: str = '0.0.0.0',
        server_port: int = 6006,
        batch_size: int = 32,
        tp: int = 1,
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
        from lmdeploy.serve.gradio.turbomind_coupled import run_local
        run_local(model_path_or_server, server_name, server_port, batch_size,
                  tp, **kwargs)


if __name__ == '__main__':
    import fire

    fire.Fire(run)
