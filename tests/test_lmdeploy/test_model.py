import pytest

from lmdeploy.model import MODELS, best_match_model


@pytest.mark.parametrize(
    'model_path_and_name',
    [('internlm/internlm-chat-7b', ['internlm']),
     ('models--internlm--internlm-chat-7b/snapshots/1234567', ['internlm']),
     ('Qwen/Qwen-7B-Chat', ['qwen']),
     ('codellama/CodeLlama-7b-hf', ['codellama']),
     ('upstage/SOLAR-0-70b', ['solar', 'solar-70b']),
     ('meta-llama/Llama-2-7b-chat-hf', ['llama2']),
     ('THUDM/chatglm2-6b', ['chatglm']),
     ('01-ai/Yi-6B-200k', ['yi', 'yi-200k']), ('01-ai/Yi-34B-Chat', ['yi']),
     ('01-ai/Yi-6B-Chat', ['yi', 'yi-chat']),
     ('WizardLM/WizardLM-70B-V1.0', ['wizardlm']),
     ('codellama/CodeLlama-34b-Instruct-hf', ['codellama']),
     ('tiiuae/falcon-7b', ['falcon']), ('workspace', [None])])
@pytest.mark.parametrize('suffix', ['', '-w4', '-4bit', '-16bit'])
def test_best_match_model(model_path_and_name, suffix):
    deduced_name = best_match_model(model_path_and_name[0] + suffix)
    if deduced_name is not None:
        assert deduced_name in model_path_and_name[
            1], f'expect {model_path_and_name[1]}, but got {deduced_name}'
    else:
        assert deduced_name in model_path_and_name[
            1], f'expect {model_path_and_name[1]}, but got {deduced_name}'


@pytest.mark.parametrize('model_name',
                         ['llama2', 'base', 'yi', 'qwen-7b', 'vicuna'])
@pytest.mark.parametrize('meta_instruction', ['[fake meta_instruction]'])
def test_model_config(model_name, meta_instruction):
    from lmdeploy.model import ChatTemplateConfig
    chat_template = ChatTemplateConfig(
        model_name, meta_instruction=meta_instruction).chat_template
    prompt = chat_template.get_prompt('')
    if model_name == 'base':
        assert prompt == ''
    else:
        assert meta_instruction in prompt


def test_base_model():
    model = MODELS.get('base')()
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
    assert model.system == '<|System|>:'
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

    model = MODELS.get('baichuan-7b')(capability='chat')
    _prompt = model.get_prompt(prompt, sequence_start=True)
    assert _prompt == prompt


def test_llama2():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('llama2')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None
    assert model.meta_instruction is not None

    model = MODELS.get('llama2')(capability='chat',
                                 meta_instruction='Provide answers in Python')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt
    assert model.meta_instruction == 'Provide answers in Python'

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
