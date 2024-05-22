import PIL

from lmdeploy.model import MODELS
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.templates import VLChatTemplateWrapper


def test_prompt_to_messages():
    model = MODELS.get('vicuna')()
    templtae = VLChatTemplateWrapper(model)
    out = templtae.prompt_to_messages('hi')
    assert isinstance(out, list) and isinstance(out[0], dict)
    im = PIL.Image.new(mode='RGB', size=(200, 200))
    out = templtae.prompt_to_messages(('hi', [im]))
    assert isinstance(out, list) and isinstance(out[0], dict)


def test_messages2prompt():
    model = MODELS.get('vicuna')()
    templtae = VLChatTemplateWrapper(model)
    messages = [{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'hi'
        }, {
            'type': 'image_url',
            'image_url': {
                'url': 'xxx'
            }
        }]
    }]
    prompt = templtae.messages2prompt(messages)
    assert isinstance(prompt, str)
    assert prompt.count(IMAGE_TOKEN) == 1


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
