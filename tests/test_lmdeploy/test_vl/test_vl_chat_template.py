import pytest

from lmdeploy.model import HFChatTemplate
from lmdeploy.vl.model.builder import load_vl_model


@pytest.fixture(scope='module')
def mock_messages():
    return [
        dict(role='user',
             content=[
                 dict(type='text', text='Describe the following images in detail'),
                 dict(type='image', url=dict(url='http://images.cocodataset.org/val2017/000000039769.jpg')),
                 dict(type='image', url=dict(url='http://images.cocodataset.org/val2017/000000039769.jpg')),
                 dict(type='text', text='How many cats are there in total?')
             ]),
    ]


@pytest.mark.parametrize('model_path', [
    'OpenGVLab/InternVL3_5-8B-HF', 'internlm/Intern-S1-mini'
    'Qwen/Qwen2-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen3-VL-8B-Instruct'
])
def test_proc_messages(model_path, mock_messages):
    model = load_vl_model(model_path=model_path, with_llm=False, backend='pytorch')
    model.build_preprocessor()
    reference = model.processor.apply_chat_template(mock_messages,
                                                    add_generation_prompt=True,
                                                    tokenize=False,
                                                    return_dict=True)
    chat_template = HFChatTemplate(model_path=model_path)
    model.proc_messages(mock_messages, chat_template, sequence_start=True)
    prompt, _ = model.proc_messages(mock_messages, chat_template, sequence_start=True)
    assert prompt.replace('<IMAGE_TOKEN>', '<IMG_CONTEXT>') == reference
