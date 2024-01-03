import pytest

from lmdeploy.tokenizer import HuggingFaceTokenizer


@pytest.mark.parametrize('model_path', [
    'internlm/internlm-chat-7b', 'Qwen/Qwen-7B-Chat',
    'baichuan-inc/Baichuan2-7B-Chat', 'codellama/CodeLlama-7b-hf',
    'upstage/SOLAR-0-70b-16bit'
])
@pytest.mark.parametrize(
    'input', ['hi, this is a test 😆😆! ' * 5, '為什麼我還在用繁體字 😆😆 gg! ' * 5])
def test_tokenizer(model_path, input):
    tokenizer = HuggingFaceTokenizer(model_path)
    encoded = tokenizer.encode(input, False)
    output = ''
    offset = 0
    for i in range(1, len(encoded) + 1):
        decoded = tokenizer.decode(encoded[:i], offset)
        if decoded.endswith('�'):
            continue
        output += decoded
        offset = i
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
