from transformers import AutoConfig

from lmdeploy.turbomind.deploy.config import (ModelConfig,
                                              TurbomindModelConfig,
                                              config_from_dict)
from lmdeploy.utils import _get_and_verify_max_len


def test_get_and_verify_max_len():
    # with PretrainedConfig
    config = AutoConfig.from_pretrained('OpenGVLab/InternVL-Chat-V1-5-AWQ',
                                        trust_remote_code=True)
    assert (_get_and_verify_max_len(config, None) == 32768)
    assert (_get_and_verify_max_len(config, 1024) == 1024)
    assert (_get_and_verify_max_len(config, 102400) == 102400)

    # with PretrainedConfig
    config = AutoConfig.from_pretrained('internlm/internlm2-chat-7b',
                                        trust_remote_code=True)
    assert (_get_and_verify_max_len(config, None) == 32768)
    assert (_get_and_verify_max_len(config, 1024) == 1024)
    assert (_get_and_verify_max_len(config, 102400) == 102400)

    # with TurbomindModelConfig
    config = config_from_dict(TurbomindModelConfig, {})
    config.model_config = config_from_dict(ModelConfig, dict(session_len=4096))
    assert (_get_and_verify_max_len(config, None) == config.session_len)
    assert (_get_and_verify_max_len(config, 1024) == 1024)
