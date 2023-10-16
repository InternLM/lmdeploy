import pytest

from lmdeploy.model import MODELS, SamplingParam


def test_base_model():
    model = MODELS.get('llama')()
    assert model is not None
    assert model.capability == 'chat'
    assert model.get_prompt('test') == 'test'
    assert model.stop_words is None

    model = MODELS.get('internlm')(capability='completion')
    assert model.capability == 'completion'
    assert model.get_prompt('hi') == 'hi'
    assert model.messages2prompt('test') == 'test'


def test_vicuna():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('vicuna')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None
    assert model.system is not None

    model = MODELS.get('vicuna')(capability='chat',
                                 system='Provide answers in Python')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt
    assert model.system == 'Provide answers in Python'

    model = MODELS.get('vicuna')(capability='voice')
    _prompt = None
    with pytest.raises(AssertionError):
        _prompt = model.get_prompt(prompt, sequence_start=True)
    assert _prompt is None


def test_internlm_chat():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('internlm-chat-7b')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is not None
    assert model.system == ''
    assert model.session_len == 2048

    model = MODELS.get('internlm-chat-7b')(capability='chat',
                                           system='Provide answers in Python')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt
    assert model.system == 'Provide answers in Python'

    model = MODELS.get('internlm-chat-7b')(capability='voice')
    _prompt = None
    with pytest.raises(AssertionError):
        _prompt = model.get_prompt(prompt, sequence_start=True)
    assert _prompt is None

    model = MODELS.get('internlm-chat-7b-8k')()
    assert model.session_len == 8192


def test_baichuan():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('baichuan-7b')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None
    assert model.repetition_penalty == 1.1

    model = MODELS.get('baichuan-7b')(capability='chat')
    _prompt = model.get_prompt(prompt, sequence_start=True)
    assert _prompt == prompt


def test_llama2():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('llama2')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None
    assert model.default_sys_prompt is not None

    model = MODELS.get('llama2')(capability='chat',
                                 system='Provide answers in Python')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt
    assert model.default_sys_prompt == 'Provide answers in Python'

    model = MODELS.get('llama2')(capability='voice')
    _prompt = None
    with pytest.raises(AssertionError):
        _prompt = model.get_prompt(prompt, sequence_start=True)
    assert _prompt is None


def test_qwen():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('qwen-7b')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is not None

    model = MODELS.get('qwen-7b')(capability='chat')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt

    model = MODELS.get('qwen-7b')(capability='voice')
    _prompt = None
    with pytest.raises(AssertionError):
        _prompt = model.get_prompt(prompt, sequence_start=True)
    assert _prompt is None


def test_codellama_completion():
    model = MODELS.get('codellama')(capability='completion')
    prompt = """\
import socket

def ping_exponential_backoff(host: str):"""
    assert model.get_prompt(prompt) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None


def test_codellama_infilling():
    model = MODELS.get('codellama')(capability='infilling')
    prompt = '''def remove_non_ascii(s: str) -> str:
    """ <FILL>
    return result
'''
    _prompt = model.get_prompt(prompt)
    assert _prompt.find('<FILL>') == -1
    assert model.stop_words == ['<EOT>']

    model = MODELS.get('codellama')(capability='infilling', suffix_first=True)
    _prompt = model.get_prompt(prompt)
    assert _prompt.find('<FILL>') == -1


def test_codellama_chat():
    model = MODELS.get('codellama')(capability='chat',
                                    system='Provide answers in Python')
    prompt = 'Write a function that computes the set of sums of all contiguous sublists of a given list.'  # noqa: E501
    _prompt = model.get_prompt(prompt, sequence_start=True)
    assert _prompt.find('Provide answers in Python') != -1

    _prompt = model.get_prompt(prompt, sequence_start=False)
    assert _prompt.find('Provide answers in Python') == -1
    assert model.stop_words is None


def test_codellama_python_specialist():
    model = MODELS.get('codellama')(capability='python')
    prompt = """
    def remove_non_ascii(s: str) -> str:
"""
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None


def test_codellama_others():
    model = None
    with pytest.raises(AssertionError):
        model = MODELS.get('codellama')(capability='java')
    assert model is None


def test_sampling_param():
    model = MODELS.get('llama')()
    default_sampling_param = SamplingParam()
    assert model.sampling_param == default_sampling_param

    model = MODELS.get('llama')(top_p=0.1, top_k=10)
    assert model.sampling_param.top_p == 0.1 and \
        model.sampling_param.top_k == 10
    assert model.sampling_param.temperature == 0.8 and \
        model.sampling_param.repetition_penalty == 1.0

    model = MODELS.get('codellama')(capability='completion')
    assert model.sampling_param.top_p == 0.9 and \
        model.sampling_param.top_k is None and \
        model.sampling_param.temperature == 0.2 and \
        model.sampling_param.repetition_penalty == 1.0

    model = MODELS.get('codellama')(capability='chat')
    assert model.sampling_param.top_p == 0.95 and \
        model.sampling_param.top_k is None and \
        model.sampling_param.temperature == 0.2 and \
        model.sampling_param.repetition_penalty == 1.0

    model = MODELS.get('codellama')(capability='infilling')
    assert model.sampling_param.top_p == 0.9 and \
        model.sampling_param.top_k is None and \
        model.sampling_param.temperature == 0.0 and \
        model.sampling_param.repetition_penalty == 1.0

    model = MODELS.get('codellama')(capability='python')
    assert model.sampling_param.top_p == 0.9 and \
        model.sampling_param.top_k is None and \
        model.sampling_param.temperature == 0.2 and \
        model.sampling_param.repetition_penalty == 1.0
