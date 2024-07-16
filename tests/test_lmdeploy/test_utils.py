from transformers import AutoConfig

from lmdeploy.turbomind.deploy.target_model.base import TurbomindModelConfig
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
    config = TurbomindModelConfig.from_dict({}, allow_none=True)
    config.session_len = 4096
    assert (_get_and_verify_max_len(config, None) == config.session_len)
    assert (_get_and_verify_max_len(config, 1024) == 1024)
