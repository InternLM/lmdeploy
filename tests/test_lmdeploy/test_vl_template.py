import PIL

from lmdeploy.model import MODELS
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.templates import VLChatTemplateWrapper


def test_prompt_to_messages():
    model = MODELS.get('llava-v1')()
    templtae = VLChatTemplateWrapper(model)
    out = templtae.prompt_to_messages('hi')
    assert isinstance(out, list) and isinstance(out[0], dict)
    im = PIL.Image.new(mode='RGB', size=(200, 200))
    out = templtae.prompt_to_messages(('hi', [im]))
    assert isinstance(out, list) and isinstance(out[0], dict)


def test_messages2prompt():
    model = MODELS.get('llava-v1')()
    templtae = VLChatTemplateWrapper(model)
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text='q1'),
                 dict(type='image_url', image_url=dict(url='xxx'))
             ])
    ]
    prompt = templtae.messages2prompt(messages)
    assert isinstance(prompt, str)
    assert prompt.count(IMAGE_TOKEN) == 1
    expected = (
        'A chat between a curious human and an artificial intelligence '
        'assistant. The assistant gives helpful, detailed, and polite '
        "answers to the human's questions. USER: "
        '<IMAGE_TOKEN>\nq1 ASSISTANT:')
    assert prompt == expected

    messages.append({'role': 'assistant', 'content': 'a1'})
    messages.append({'role': 'user', 'content': 'q2'})
    prompt = templtae.messages2prompt(messages)
    expected = (
        'A chat between a curious human and an artificial intelligence '
        'assistant. The assistant gives helpful, detailed, and polite '
        "answers to the human's questions. USER: "
        '<IMAGE_TOKEN>\nq1 ASSISTANT: a1</s>USER: q2 ASSISTANT:')
    assert prompt == expected


def test_internvl2_conv():
    # https://huggingface.co/OpenGVLab/InternVL2-8B/blob/3bfd3664dea4f3da628785f5125d30f889701253/conversation.py
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    get_conv_template = get_class_from_dynamic_module(
        'conversation.get_conv_template', 'OpenGVLab/InternVL2-8B')
    template = get_conv_template('internlm2-chat')
    question1 = 'question1'
    template.append_message(template.roles[0], question1)
    template.append_message(template.roles[1], None)
    model = MODELS.get('internvl2-internlm2')()
    messages = [dict(role='user', content=question1)]
    assert template.get_prompt() == model.messages2prompt(messages)

    answer1 = 'answer1'
    template.messages[-1][1] = answer1
    question2 = 'question2'
    template.append_message(template.roles[0], question2)
    template.append_message(template.roles[1], None)
    messages.append(dict(role='assistant', content=answer1))
    messages.append(dict(role='user', content=question2))
    assert template.get_prompt() == model.messages2prompt(messages)


def test_llava_conv_chatml_direct():
    model = MODELS.get('llava-chatml')()
    templtae = VLChatTemplateWrapper(model)
    messages = [
        dict(role='user',
             content=[
                 dict(type='text', text='q1'),
                 dict(type='image_url', image_url=dict(url='xxx'))
             ])
    ]

    prompt = templtae.messages2prompt(messages)
    expected = ('<|im_start|>system\nAnswer the questions.<|im_end|>'
                '<|im_start|>user\n<IMAGE_TOKEN>\nq1<|im_end|>'
                '<|im_start|>assistant\n')
    assert prompt == expected

    messages.append({'role': 'assistant', 'content': 'a1'})
    messages.append({'role': 'user', 'content': 'q2'})
    prompt = templtae.messages2prompt(messages)
    expected = ('<|im_start|>system\nAnswer the questions.<|im_end|>'
                '<|im_start|>user\n<IMAGE_TOKEN>\nq1<|im_end|>'
                '<|im_start|>assistant\na1<|im_end|>'
                '<|im_start|>user\nq2<|im_end|>'
                '<|im_start|>assistant\n')
    assert prompt == expected


def test_custom_image_token():
    from lmdeploy.vl.templates import DeepSeekVLChatTemplateWrapper
    model = MODELS.get('deepseek-vl')()
    template = DeepSeekVLChatTemplateWrapper(model)

    def create_user(query: str):
        item = dict(role='user', content=[dict(type='text', text=query)])
        num = query.count(IMAGE_TOKEN)
        for _ in range(num):
            item['content'].append(
                dict(type='image_url', image_url=dict(url='xxx')))
        return item

    def create_assistant(response: str):
        return dict(role='assistant', content=response)

    messages = [create_user(f'{IMAGE_TOKEN} q1')]
    prompt = template.messages2prompt(messages)
    expected = ('You are a helpful language and vision assistant. You are able'
                ' to understand the visual content that the user provides, and'
                ' assist the user with a variety of tasks using natural '
                'language.\n\nUser: <IMAGE_TOKEN> q1\n\nAssistant:')
    assert prompt == expected

    messages.append(create_assistant('a1'))
    messages.append(create_user(f'q2 {IMAGE_TOKEN}'))
    prompt = template.messages2prompt(messages)
    expected = ('You are a helpful language and vision assistant. You are able'
                ' to understand the visual content that the user provides, and'
                ' assist the user with a variety of tasks using natural '
                'language.\n\nUser: <IMAGE_TOKEN> q1\n\nAssistant: '
                'a1<｜end▁of▁sentence｜>User: q2 <IMAGE_TOKEN>\n\nAssistant:')
    assert prompt == expected
