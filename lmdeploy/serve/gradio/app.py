# Copyright (c) OpenMMLab. All rights reserved.
import fire


def run(model_path_or_server: str,
        server_name: str = 'localhost',
        server_port: int = 6006,
        batch_size: int = 32,
        tp: int = 1,
        restful_api: bool = False):
    """chat with AI assistant through web ui.

    Args:
        model_path_or_server (str): the path of the deployed model or the
            tritonserver URL or restful api URL. The former is for directly
            running service with gradio. The latter is for running with
            tritonserver by default. If the input URL is restful api. Please
            enable another flag `restful_api`.
        server_name (str): the ip address of gradio server
        server_port (int): the port of gradio server
        batch_size (int): batch size for running Turbomind directly
        tp (int): tensor parallel for Turbomind
        restufl_api (bool): a flag for model_path_or_server
    """
    if ':' in model_path_or_server:
        if restful_api:
            from lmdeploy.serve.gradio.api_server_decoupled import \
                run_api_server
            run_api_server(model_path_or_server, server_name, server_port,
                           batch_size)
        else:
            from lmdeploy.serve.gradio.triton_server_decoupled import \
                run_triton_server
            run_triton_server(model_path_or_server, server_name, server_port)
    else:
        from lmdeploy.serve.gradio.turbomind_coupled import run_local
        run_local(model_path_or_server, server_name, server_port, batch_size,
                  tp)


if __name__ == '__main__':
    fire.Fire(run)
