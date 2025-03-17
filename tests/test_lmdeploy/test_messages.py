from typing import List

import pytest

from lmdeploy import GenerationConfig, Tokenizer
from lmdeploy.utils import get_hf_gen_cfg


def test_engine_generation_config():
    tokenizer = Tokenizer('internlm/internlm-chat-7b')
    config = GenerationConfig(n=3, stop_words=['<eoa>'])
    stop_token_ids = tokenizer.encode('<eoa>', add_bos=False)
    config.convert_stop_bad_words_to_ids(tokenizer)
    assert stop_token_ids == config.stop_token_ids
    assert isinstance(config.stop_token_ids, List) and \
        isinstance(config.stop_token_ids[0], int)


@pytest.mark.parametrize('model_path', [
    'deepseek-ai/DeepSeek-V3',
    'Qwen/Qwen2.5-32B-Instruct',
    'internlm/internlm3-8b-instruct',
])
def test_update_from_hf_gen_cfg(model_path):
    tokenizer = Tokenizer(model_path)
    model_cfg = get_hf_gen_cfg(model_path)

    generation_config = GenerationConfig()
    generation_config.update_from_hf_gen_cfg(model_cfg, tokenizer.eos_token_id)
    assert generation_config.stop_token_ids is not None
