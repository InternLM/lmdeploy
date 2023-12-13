import inspect


def compare_func(class_method, function):
    """Compare if a class method has same arguments as a function."""

    argspec_cls = inspect.getfullargspec(class_method)
    argspec_func = inspect.getfullargspec(function)
    assert argspec_cls.args[1:] == argspec_func.args
    assert argspec_cls.defaults == argspec_func.defaults
    assert argspec_cls.annotations == argspec_func.annotations


def test_cli():

    from lmdeploy.cli.cli import CLI
    from lmdeploy.serve.turbomind.deploy import main as convert
    compare_func(CLI.convert, convert)


def test_subcli_chat():
    from lmdeploy.cli.chat import SubCliChat
    from lmdeploy.pytorch.chat import main as run_torch_model
    from lmdeploy.turbomind.chat import main as run_turbomind_model

    compare_func(SubCliChat.torch, run_torch_model)
    compare_func(SubCliChat.turbomind, run_turbomind_model)


def test_subcli_lite():
    from lmdeploy.cli.lite import SubCliLite
    from lmdeploy.lite.apis.auto_awq import auto_awq
    from lmdeploy.lite.apis.calibrate import calibrate
    from lmdeploy.lite.apis.kv_qparams import main as run_kv_qparams

    compare_func(SubCliLite.auto_awq, auto_awq)
    compare_func(SubCliLite.calibrate, calibrate)
    compare_func(SubCliLite.kv_qparams, run_kv_qparams)


def test_subcli_serve():
    from lmdeploy.cli.serve import SubCliServe
    from lmdeploy.serve.client import main as run_triton_client
    from lmdeploy.serve.gradio.app import run as run_gradio
    from lmdeploy.serve.openai.api_client import main as run_api_client
    from lmdeploy.serve.openai.api_server import serve as run_api_server

    compare_func(SubCliServe.gradio, run_gradio)
    compare_func(SubCliServe.api_server, run_api_server)
    compare_func(SubCliServe.api_client, run_api_client)
    compare_func(SubCliServe.triton_client, run_triton_client)
