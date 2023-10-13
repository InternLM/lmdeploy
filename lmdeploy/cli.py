# Copyright (c) OpenMMLab. All rights reserved.
import fire


class CLI(object):
    """CLI."""


class SubLite(object):
    """This command group include."""


class SubServe(object):
    """SubServe."""


class SubChat(object):
    """SubChat."""
    pass


def run():
    from lmdeploy.lite.apis.auto_awq import auto_awq
    from lmdeploy.lite.apis.calibrate import calibrate
    from lmdeploy.lite.apis.kv_qparams import main as run_kv_qparams
    from lmdeploy.pytorch.chat import main as chat_pytorch
    from lmdeploy.serve.client import main as run_triton_client
    from lmdeploy.serve.gradio.app import run as run_gradio
    from lmdeploy.serve.openai.api_client import main as run_api_client
    from lmdeploy.serve.openai.api_server import main as run_api_server
    from lmdeploy.serve.turbomind.deploy import main as convert
    from lmdeploy.turbomind.chat import main as chat_turbomind

    cli = CLI()
    lite = SubLite()
    lite.calibrate = calibrate
    lite.auto_awq = auto_awq
    lite.kv_qparams = run_kv_qparams

    serve = SubServe()
    serve.api_server = run_api_server
    serve.api_client = run_api_client
    serve.gradio = run_gradio
    serve.triton_client = run_triton_client

    chat = SubChat()
    chat.pytorch = chat_pytorch
    chat.turbomind = chat_turbomind

    cli.convert = convert
    cli.lite = lite
    cli.chat = chat
    cli.serve = serve
    fire.Fire(cli, name='lmdeploy')
