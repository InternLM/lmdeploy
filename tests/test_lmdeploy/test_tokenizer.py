import random

import pytest

from lmdeploy.tokenizer import DetokenizeState, HuggingFaceTokenizer


@pytest.mark.parametrize('model_path', [
    'internlm/internlm-chat-7b', 'Qwen/Qwen-7B-Chat',
    'baichuan-inc/Baichuan2-7B-Chat', 'upstage/SOLAR-0-70b-16bit',
    'baichuan-inc/Baichuan-7B', 'codellama/CodeLlama-7b-hf',
    'THUDM/chatglm2-6b', '01-ai/Yi-6B-200k', '01-ai/Yi-34B-Chat',
    '01-ai/Yi-6B-Chat', 'WizardLM/WizardLM-70B-V1.0',
    'codellama/CodeLlama-34b-Instruct-hf', 'tiiuae/falcon-7b'
])
@pytest.mark.parametrize('input', [
    'hi, this is a test 😆😆! ' * 5, '為什麼我還在用繁體字 😆😆 gg! ' * 5,
    ' License at\n#\n#' + ' ' * 100 + 'ht', '   '
])
@pytest.mark.parametrize('interval', [1, 3])
@pytest.mark.parametrize('skip_special_tokens', [True, False])
def test_tokenizer(model_path, input, interval, skip_special_tokens):
    tokenizer = HuggingFaceTokenizer(model_path)
    encoded = tokenizer.encode(input, False, add_special_tokens=False)
    output = ''
    input = tokenizer.decode(encoded, skip_special_tokens=skip_special_tokens)
    state = DetokenizeState()
    for i in range(0, len(encoded), interval):
        offset = i + interval
        if offset < len(encoded):
            # lmdeploy may decode nothing when concurrency is high
            if random.randint(1, 10) < 4:
                offset -= interval
        decoded, state = tokenizer.detokenize_incrementally(
            encoded[:offset], state, skip_special_tokens)
        output += decoded
    assert input == output, 'input string should equal to output after enc-dec'


@pytest.mark.parametrize('model_path', [
    'internlm/internlm-chat-7b', 'Qwen/Qwen-7B-Chat',
    'baichuan-inc/Baichuan2-7B-Chat', 'codellama/CodeLlama-7b-hf',
    'upstage/SOLAR-0-70b-16bit'
])
@pytest.mark.parametrize('stop_words', ['.', ' ', '?', ''])
def test_tokenizer_with_stop_words(model_path, stop_words):
    tokenizer = HuggingFaceTokenizer(model_path)
    indexes = tokenizer.indexes_containing_token(stop_words)
    assert indexes is not None


def test_qwen_vl_decode_special():
    from lmdeploy.tokenizer import Tokenizer
    tok = Tokenizer('Qwen/Qwen-VL-Chat')
    try:
        tok.decode([151857])
        assert (0)
    except Exception as e:
        assert str(e) == 'Unclosed image token'
