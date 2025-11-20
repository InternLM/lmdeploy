import pytest

from lmdeploy.model import HFChatTemplate
from lmdeploy.vl.model.internvl3_hf import InternVL3VisionModel

TEST_MODELS = ['OpenGVLab/InternVL3_5-8B-HF', 'internlm/Intern-S1-mini']


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


def test_proc_messages(mock_messages):
    for model_path in TEST_MODELS:
        vision_model = InternVL3VisionModel(model_path=model_path, with_llm=False)
        vision_model.build_preprocessor()
        reference = vision_model.processor.apply_chat_template(mock_messages,
                                                               add_generation_prompt=True,
                                                               tokenize=False,
                                                               return_dict=True)
        chat_template = HFChatTemplate(model_path=model_path)
        vision_model.proc_messages(mock_messages, chat_template, sequence_start=True)
        prompt, _ = vision_model.proc_messages(mock_messages, chat_template, sequence_start=True)
        assert prompt.replace('<IMAGE_TOKEN>', '<IMG_CONTEXT>') == reference
