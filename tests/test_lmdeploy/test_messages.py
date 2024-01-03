from typing import List

from lmdeploy import EngineGenerationConfig, GenerationConfig, Tokenizer


def test_engine_generation_config():
    tokenizer = Tokenizer('internlm/internlm-chat-7b')
    config = GenerationConfig(n=3, stop_words=['<eoa>'])
    _config = EngineGenerationConfig.From(config, tokenizer)

    assert _config.n == config.n == 3 and \
        _config.max_new_tokens == config.max_new_tokens and \
        _config.temperature == config.temperature
    assert isinstance(_config.stop_words, List) and \
        isinstance(_config.stop_words[0], int)
