
import pytest
from pydantic import ValidationError

from lmdeploy import GenerationConfig, Tokenizer
from lmdeploy.messages import Response
from lmdeploy.pytorch.messages import SamplingParam
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.utils import get_hf_gen_cfg


def test_generation_config_repetition_ngram_clamped():
    c = GenerationConfig(repetition_ngram_size=-1, repetition_ngram_threshold=-2)
    assert c.repetition_ngram_size == 0
    assert c.repetition_ngram_threshold == 0


def test_input_logprobs_contract_and_sampling_conversion():
    assert GenerationConfig().logprob_start_len == -1

    for start in (-1, 0, 3):
        config = GenerationConfig(logprobs=0, logprob_start_len=start)
        param = SamplingParam.from_gen_config(config)
        assert param.num_logprobs == 0
        assert param.logprob_start_len == start

    # Preserve upstream generated-only validation behavior.
    GenerationConfig(logprobs=-1)
    with pytest.raises(ValueError, match='logprobs must be non-negative'):
        GenerationConfig(logprobs=-1, logprob_start_len=0)
    with pytest.raises(ValueError, match='greater than or equal to -1'):
        GenerationConfig(logprobs=0, logprob_start_len=-2)
    with pytest.raises(ValueError, match='logprobs must be non-negative'):
        GenerationConfig(logprob_start_len=0)


@pytest.mark.parametrize('carrier', [[], [{2: -0.5}]])
def test_response_extend_preserves_logprob_carrier(carrier):
    response = Response('', 0, 3)
    assert not hasattr(response, 'input_logprobs')
    response.extend(Response('', 0, 3, logprobs=carrier))
    assert response.logprobs == carrier


def test_chat_completion_request_repetition_ngram_ge_zero():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model='m',
            messages=[{'role': 'user', 'content': 'hi'}],
            repetition_ngram_size=-1,
        )


def test_engine_generation_config():
    tokenizer = Tokenizer('internlm/internlm2-chat-7b', trust_remote_code=True)
    config = GenerationConfig(n=3, stop_words=['<|im_end|>'])
    stop_token_ids = tokenizer.encode('<|im_end|>', add_bos=False)
    config.convert_stop_bad_words_to_ids(tokenizer)
    assert stop_token_ids == config.stop_token_ids
    assert isinstance(config.stop_token_ids, list) and \
        isinstance(config.stop_token_ids[0], int)


@pytest.mark.parametrize('model_path', [
    'deepseek-ai/DeepSeek-V3',
    'Qwen/Qwen2.5-32B-Instruct',
    'internlm/internlm3-8b-instruct',
])
def test_update_from_hf_gen_cfg(model_path):
    tokenizer = Tokenizer(model_path, trust_remote_code=True)
    model_cfg = get_hf_gen_cfg(model_path)

    generation_config = GenerationConfig()
    generation_config.update_from_hf_gen_cfg(model_cfg, tokenizer.eos_token_id)
    assert generation_config.stop_token_ids is not None
