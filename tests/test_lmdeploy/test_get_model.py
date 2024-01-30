import os

import pytest

from lmdeploy.turbomind.utils import get_model_from_config


@pytest.mark.parametrize('item',
                         [('baichuan-inc/Baichuan-7B', 'baichuan'),
                          ('baichuan-inc/Baichuan2-7B-Base', 'baichuan2'),
                          ('internlm/internlm2-7b', 'internlm2'),
                          ('internlm/internlm2-chat-7b', 'internlm2'),
                          ('internlm/internlm2-math-20b', 'internlm2'),
                          ('internlm/internlm-20b', 'llama'),
                          ('NousResearch/Llama-2-7b-chat-hf', 'llama'),
                          ('Qwen/Qwen-7B-Chat', 'qwen'),
                          ('Qwen/Qwen-14B', 'qwen'),
                          ('NousResearch/Nous-Hermes-2-SOLAR-10.7B', 'llama'),
                          ('01-ai/Yi-34B-Chat', 'llama')])
def test_get_model_from_config(item):
    from transformers.utils import cached_file
    model_id, result = item
    local_file = cached_file(model_id, 'config.json')
    local_dir = os.path.dirname(local_file)
    print(get_model_from_config(local_dir))
    assert get_model_from_config(local_dir) == result
