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
