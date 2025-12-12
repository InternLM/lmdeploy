import pytest

from lmdeploy.model import MODELS, best_match_model

HF_MODELS_WITH_CHAT_TEMPLATES = [
    'Qwen/Qwen1.5-7B-Chat',
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen3-8B',
    'Qwen/QwQ-32B',
    'Qwen/QwQ-32B-Preview',
    'Qwen/QwQ-32B-AWQ',
    'Qwen/Qwen2.5-VL-7B-Instruct',
    'Qwen/Qwen2-VL-7B-Instruct',
    'internlm/internlm2-chat-7b',
    'internlm/internlm2_5-7b-chat',
    'internlm/internlm3-8b-instruct',
    'internlm/Intern-S1',
    'internlm/Intern-S1-mini',
    'OpenGVLab/InternVL-Chat-V1-2',
    'OpenGVLab/InternVL-Chat-V1-5',
    'OpenGVLab/Mini-InternVL-Chat-2B-V1-5',
    'OpenGVLab/InternVL2-2B',
    'OpenGVLab/InternVL2-4B',
    'OpenGVLab/InternVL2-8B',
    'OpenGVLab/InternVL2_5-2B',
    'OpenGVLab/InternVL2_5-4B',
    'OpenGVLab/InternVL2_5-8B',
    'OpenGVLab/InternVL3-2B',
    'OpenGVLab/InternVL3-8B',
    'OpenGVLab/InternVL3-9B',
    'OpenGVLab/InternVL3_5-1B',
    'OpenGVLab/InternVL3_5-4B',
    'OpenGVLab/InternVL3_5-8B',
    'OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview',
    'AI4Chem/ChemVLM-8B',
    'deepseek-ai/DeepSeek-V2-Lite',
    'deepseek-ai/DeepSeek-V3',
    'deepseek-ai/DeepSeek-R1',
    'deepseek-ai/DeepSeek-R1-Zero',
    'deepseek-ai/DeepSeek-V3.1',
    'deepseek-ai/deepseek-coder-1.3b-instruct',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'zai-org/chatglm3-6b',
    'zai-org/glm-4-9b-chat',
    'zai-org/codegeex4-all-9b',
    'zai-org/cogvlm2-llama3-chat-19B',
    'microsoft/Phi-3-mini-128k-instruct',
    'microsoft/Phi-3-vision-128k-instruct',
    'microsoft/Phi-3.5-mini-instruct',
    'microsoft/Phi-3.5-vision-instruct',
    'microsoft/Phi-3.5-MoE-instruct',
    '01-ai/Yi-1.5-34B-Chat',
    # Accessing the following models is supposed to be authenticated
    # 'openbmb/MiniCPM-V-2_6',
    # 'google/gemma-3-4b-it',
]


@pytest.mark.parametrize('model_path', HF_MODELS_WITH_CHAT_TEMPLATES)
def test_HFChatTemplate_get_prompt_sequence_start_True(model_path):
    model = MODELS.get('hf')(model_path=model_path)
    prompt = 'How to apply chat template using transformers?'
    messages = [{'role': 'user', 'content': prompt}]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    expected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert model.get_prompt(prompt, sequence_start=True) == expected


@pytest.mark.parametrize('model_path', HF_MODELS_WITH_CHAT_TEMPLATES)
def test_HFChatTemplate_message2prompt_sequence_start_True(model_path):
    model = MODELS.get('hf')(model_path=model_path)
    prompt = 'How to apply chat template using transformers?'
    messages = [{'role': 'user', 'content': prompt}]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    expected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert model.messages2prompt(prompt, sequence_start=True) == expected
    assert model.messages2prompt(messages, sequence_start=True) == expected


@pytest.mark.parametrize('model_path', HF_MODELS_WITH_CHAT_TEMPLATES)
def test_best_match_model_hf(model_path):
    assert best_match_model(model_path) == 'hf'


@pytest.mark.parametrize('model_path_and_name', [
    ('internlm/internlm-chat-7b', ['internlm']),
    ('internlm/internlm2-1_8b', ['base']),
    ('codellama/CodeLlama-7b-hf', ['codellama']),
    ('meta-llama/Llama-2-7b-chat-hf', ['llama2']),
    ('THUDM/chatglm2-6b', ['chatglm']),
    ('codellama/CodeLlama-34b-Instruct-hf', ['codellama']),
    ('deepseek-ai/deepseek-vl-7b-chat', ['deepseek-vl']),
])
def test_best_match_model(model_path_and_name):
    deduced_name = best_match_model(model_path_and_name[0])
    if deduced_name is not None:
        assert deduced_name in model_path_and_name[1], f'expect {model_path_and_name[1]}, but got {deduced_name}'
    else:
        assert deduced_name in model_path_and_name[1], f'expect {model_path_and_name[1]}, but got {deduced_name}'


def test_base_model():
    model = MODELS.get('internlm')(capability='completion')
    assert model.capability == 'completion'
    assert model.get_prompt('hi') == 'hi'
    assert model.messages2prompt('test') == 'test'


def test_vicuna():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('vicuna')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt

    model = MODELS.get('vicuna')(capability='chat', system='Provide answers in Python')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt
    assert model.system == 'Provide answers in Python'

    model = MODELS.get('vicuna')(capability='voice')
    _prompt = None
    with pytest.raises(AssertionError):
        _prompt = model.get_prompt(prompt, sequence_start=True)
        assert _prompt is None


def test_prefix_response():
    model = MODELS.get('hf')(model_path='Qwen/Qwen3-8B')
    messages = [dict(role='assistant', content='prefix test')]
    prompt = model.messages2prompt(messages)
    assert prompt[-len('prefix test'):] == 'prefix test'


def test_internlm_chat():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('internlm')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is not None
    assert model.system == '<|System|>:'

    model = MODELS.get('internlm')(capability='chat', system='Provide answers in Python')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt
    assert model.system == 'Provide answers in Python'

    model = MODELS.get('internlm')(capability='voice')
    _prompt = None
    with pytest.raises(AssertionError):
        _prompt = model.get_prompt(prompt, sequence_start=True)
        assert _prompt is None


def test_baichuan():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('baichuan2')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None

    model = MODELS.get('baichuan2')(capability='chat')
    _prompt = model.get_prompt(prompt, sequence_start=True)
    assert _prompt == '<reserved_106>' + prompt + '<reserved_107>'


def test_llama2():
    prompt = 'hello, can u introduce yourself'
    model = MODELS.get('llama2')(capability='completion')
    assert model.get_prompt(prompt, sequence_start=True) == prompt
    assert model.get_prompt(prompt, sequence_start=False) == prompt
    assert model.stop_words is None
    assert model.meta_instruction is not None

    model = MODELS.get('llama2')(capability='chat', meta_instruction='Provide answers in Python')
    assert model.get_prompt(prompt, sequence_start=True) != prompt
    assert model.get_prompt(prompt, sequence_start=False) != prompt
    assert model.meta_instruction == 'Provide answers in Python'

    model = MODELS.get('llama2')(capability='voice')
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
    model = MODELS.get('codellama')(capability='chat', system='Provide answers in Python')
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


@pytest.mark.parametrize(
    'model_path_or_name',
    ['deepseek-ai/deepseek-vl2-tiny', 'deepseek-ai/deepseek-vl2-small', 'deepseek-ai/deepseek-vl2'])
def test_deepseek_vl2(model_path_or_name):
    deduced_name = best_match_model(model_path_or_name)
    assert deduced_name == 'deepseek-vl2'

    chat_template = MODELS.get(deduced_name)()
    messages = [{
        'role': 'user',
        'content': 'This is image_1: <image>\n'
        'This is image_2: <image>\n'
        'This is image_3: <image>\n Can you tell me what are in the images?',
        'images': [
            'images/multi_image_1.jpeg',
            'images/multi_image_2.jpeg',
            'images/multi_image_3.jpeg',
        ],
    }, {
        'role': 'assistant',
        'content': ''
    }]

    ref = '<|User|>: This is image_1: <image>\nThis is image_2: <image>\nThis is image_3: <image>' + \
          '\n Can you tell me what are in the images?\n\n<|Assistant|>:'
    lm_res = chat_template.messages2prompt(messages)
    assert ref == lm_res


@pytest.mark.parametrize('model_path', ['Qwen/Qwen3-30B-A3B', 'Qwen/Qwen2.5-7B-Instruct'])
@pytest.mark.parametrize('enable_thinking', [True, False, None])
def test_qwen3(model_path, enable_thinking):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    chat_template_name = best_match_model(model_path)
    assert chat_template_name == 'hf'
    chat_template = MODELS.get(chat_template_name)(model_path)

    messages = [{
        'role': 'system',
        'content': 'you are a helpful assistant'
    }, {
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'AGI is?'
    }]
    if enable_thinking is None:
        ref = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        ref = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=enable_thinking)
    lm_res = chat_template.messages2prompt(messages, enable_thinking=enable_thinking)
    assert ref == lm_res


@pytest.mark.parametrize('model_path', ['internlm/Intern-S1'])
@pytest.mark.parametrize('enable_thinking', [None, True, False])
@pytest.mark.parametrize('has_user_sys', [True, False])
def test_interns1(model_path, enable_thinking, has_user_sys):
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except OSError:
        pytest.skip(reason=f'{model_path} not exists')

    chat_template_name = best_match_model(model_path)
    chat_template = MODELS.get(chat_template_name)(model_path)

    messages = [{
        'role': 'system',
        'content': 'you are a helpful assistant'
    }, {
        'role': 'user',
        'content': 'who are you'
    }, {
        'role': 'assistant',
        'content': 'I am an AI'
    }, {
        'role': 'user',
        'content': 'AGI is?'
    }]
    if not has_user_sys:
        messages = messages[1:]

    if enable_thinking is None:
        ref = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        ref = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=enable_thinking)
    lm_res = chat_template.messages2prompt(messages, enable_thinking=enable_thinking)
    assert ref == lm_res


@pytest.mark.parametrize('model_path', ['Qwen/Qwen1.5-7B-Chat', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen3-8B'])
def test_HFChatTemplate_get_prompt_sequence_start_False_Qwen(model_path):
    model = MODELS.get('hf')(model_path=model_path)
    assert model.stop_words == ['<|im_end|>']

    prompt = 'How to apply chat template using transformers?'
    assert model.get_prompt(prompt,
                            sequence_start=False) == f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'


@pytest.mark.parametrize('model_path', ['internlm/Intern-S1', 'internlm/Intern-S1-mini'])
def test_InternS1_thinking(model_path):
    pass


@pytest.mark.parametrize('model_path', [''])
def test_InternVL(model_path):
    pass


@pytest.mark.parametrize('model_path', [''])
def test_HFChatTemplate_llama(model_path):
    # TODO: add a huggingface token to github
    pass


@pytest.mark.parametrize('model_path', ['deepseek-ai/DeepSeek-V3'])
def test_HFChatTemplate_DeepSeek_V3(model_path):
    model = MODELS.get('hf')(model_path=model_path)
    assert model.stop_words == ['<｜end▁of▁sentence｜>']

    prompt = 'How to apply chat template using transformers?'
    assert model.get_prompt(prompt, sequence_start=False) == f'<｜User｜>{prompt}<｜Assistant｜>'


@pytest.mark.parametrize('model_path', ['deepseek-ai/DeepSeek-R1'])
def test_HFChatTemplate_DeepSeek_thinking(model_path):
    model = MODELS.get('hf')(model_path=model_path)
    assert model.stop_words == ['<｜end▁of▁sentence｜>']

    prompt = 'How to apply chat template using transformers?'
    assert model.get_prompt(prompt, sequence_start=False) == f'<｜User｜>{prompt}<｜Assistant｜><think>\n'


@pytest.mark.parametrize('model_path', ['Qwen/Qwen3-VL-8B-Instruct'])
def test_HFChatTemplate_Qwen3_VL_with_vision_id(model_path):
    model = MODELS.get('hf')(model_path=model_path)

    # testcase from https://github.com/QwenLM/Qwen3-VL
    messages = [
        {
            'role': 'user',
            'content': [{
                'type': 'image'
            }, {
                'type': 'text',
                'text': 'Hello, how are you?'
            }],
        },
        {
            'role': 'assistant',
            'content': "I'm doing well, thank you for asking. How can I assist you today?",
        },
        {
            'role':
            'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'Can you describe these images and video?'
                },
                {
                    'type': 'image'
                },
                {
                    'type': 'image'
                },
                {
                    'type': 'video'
                },
                {
                    'type': 'text',
                    'text': 'These are from my vacation.'
                },
            ],
        },
        {
            'role':
            'assistant',
            'content':
            """I'd be happy to describe the images and video for you.
                Could you please provide more context about your vacation?""",
        },
        {
            'role': 'user',
            'content': 'It was a trip to the mountains. Can you see the details in the images and video?',
        },
    ]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    expected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
    chat_template_kwargs = dict(add_vision_id=True)
    lm_res = model.messages2prompt(messages, **chat_template_kwargs)
    assert expected == lm_res
