import pytest

from lmdeploy.messages import PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.model import ChatTemplateConfig
from lmdeploy.serve.async_engine import deduce_a_name


@pytest.mark.parametrize(
    'backend_config',
    [TurbomindEngineConfig('internlm'),
     PytorchEngineConfig(None), None])
@pytest.mark.parametrize(
    'chat_template_config',
    [ChatTemplateConfig('internlm'),
     ChatTemplateConfig(None), None])
@pytest.mark.parametrize('model_name', ['internlm', None])
@pytest.mark.parametrize('model_path', ['internlm/internlm2-chat-7b'])
def test_deduce_a_name(model_path, model_name, chat_template_config,
                       backend_config):
    name = deduce_a_name(model_path, model_name, chat_template_config,
                         backend_config)
    if model_name or getattr(backend_config, 'model_name', None) or getattr(
            chat_template_config, 'model_name', None):
        assert name == 'internlm'
    else:
        assert name == model_path
