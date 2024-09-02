from typing import List

from lmdeploy import GenerationConfig, Tokenizer


def test_engine_generation_config():
    tokenizer = Tokenizer('internlm/internlm-chat-7b')
    config = GenerationConfig(n=3, stop_words=['<eoa>'])
    stop_token_ids = tokenizer.encode('<eoa>', add_bos=False)
    config.convert_stop_bad_words_to_ids(tokenizer)
    assert stop_token_ids == config.stop_token_ids
    assert isinstance(config.stop_token_ids, List) and \
        isinstance(config.stop_token_ids[0], int)
