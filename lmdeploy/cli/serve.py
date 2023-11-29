# Copyright (c) OpenMMLab. All rights reserved.
from typing import List


class SubCliServe(object):
    """Serve LLMs and interact on terminal or web UI."""

    def gradio(self,
               model_path_or_server: str,
               server_name: str = '0.0.0.0',
               server_port: int = 6006,
               batch_size: int = 32,
               tp: int = 1,
               **kwargs):
        """Serve LLMs with web ui using gradio.

        Example 1:
            lmdeploy serve gradio ./workspace

        Example 2:
            lmdeploy serve gradio http://0.0.0.0:23333
            --server_name 0.0.0.0
            --server_port 6006

        Example 3:
            lmdeploy serve gradio ${triton_server_ip_addresss}:33337

        Args:
            model_path_or_server (str): the path of the deployed model or the
                tritonserver URL or restful api URL. The former is for directly
                running service with gradio. The latter is for running with
                tritonserver by default.
            server_name (str): the ip address of gradio server
            server_port (int): the port of gradio server
            batch_size (int): batch size for running Turbomind directly
            tp (int): tensor parallel for Turbomind
            kwargs (dict): extra params to init
        """
        from lmdeploy.serve.gradio.app import run
        run(model_path_or_server,
            server_name=server_name,
            server_port=server_port,
            batch_size=batch_size,
            tp=tp,
            **kwargs)

    def api_server(self,
                   model_path: str,
                   server_name: str = '0.0.0.0',
                   server_port: int = 23333,
                   instance_num: int = 32,
                   tp: int = 1,
                   allow_origins: List[str] = ['*'],
                   allow_credentials: bool = True,
                   allow_methods: List[str] = ['*'],
                   allow_headers: List[str] = ['*'],
                   **kwargs):
        """Serve LLMs with restful api using fastapi.

        Args:
            model_path (str): the path of the deployed model
            server_name (str): host ip for serving
            server_port (int): server port
            instance_num (int): number of instances of turbomind model
            tp (int): tensor parallel
            allow_origins (List[str]): a list of allowed origins for CORS
            allow_credentials (bool): whether to allow credentials for CORS
            allow_methods (List[str]): a list of allowed HTTP methods for CORS
            allow_headers (List[str]): a list of allowed HTTP headers for CORS
            kwargs (dict) extra params to init api server
        """
        from lmdeploy.serve.openai.api_server import main as run_api_server

        run_api_server(model_path,
                       server_name=server_name,
                       server_port=server_port,
                       instance_num=instance_num,
                       tp=tp,
                       allow_origins=allow_origins,
                       allow_credentials=allow_credentials,
                       allow_methods=allow_methods,
                       allow_headers=allow_headers,
                       **kwargs)

    def api_client(self, restful_api_url: str, session_id: int = 0):
        """Interact with restful api server in terminal.

        Args:
            restful_api_url: The restful api URL.
            session_id: The identical id of a session.
        """
        from lmdeploy.serve.openai.api_client import main as run_api_client
        run_api_client(restful_api_url, session_id=session_id)

    def triton_client(self,
                      tritonserver_addr: str,
                      session_id: int = 1,
                      cap: str = 'chat',
                      stream_output: bool = True,
                      **kwargs):
        """Interact with Triton Server using gRPC protocol.

        Args:
            tritonserver_addr (str): the address in format "ip:port" of
              triton inference server
            session_id (int): the identical id of a session
            cap (str): the capability of a model. For example, codellama
                has the ability among ['completion', 'infill', 'instruct',
                'python']
            stream_output (bool): indicator for streaming output or not
            **kwargs (dict): other arguments for initializing model's
                chat template
        """

        from lmdeploy.serve.client import main as run_triton_client

        run_triton_client(
            tritonserver_addr,
            session_id=session_id,
            cap=cap,
            stream_output=stream_output,
            **kwargs,
        )
