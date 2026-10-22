# Copyright (c) OpenMMLab. All rights reserved.
import copy
from types import SimpleNamespace

import pytest
from PIL import Image

from lmdeploy.vl.model.base import VisionModel
from lmdeploy.vl.model.internvl import InternVLVisionModel


@pytest.mark.parametrize('name', ['internvl3', 'internvl3_5'])
@pytest.mark.parametrize('legacy', [(), ('user',), ('tool',), ('user', 'tool')])
def test_tool_images_preserve_metadata_and_per_message_placeholders(hf_media_template, name, legacy):
    messages = [
        dict(role='user', content=[
            dict(type='text', text='first'), dict(type='image', data=Image.new('RGB', (28, 28))),
        ]),
        dict(role='tool', tool_call_id='screenshot', name='capture', content=[
            dict(type='text', text='second'), dict(type='image', data=Image.new('RGB', (28, 28))),
        ]),
    ]
    for message in messages:
        if message['role'] in legacy:
            message['content'][0]['text'] += '<IMAGE_TOKEN>'
    original = copy.deepcopy(messages)
    template = hf_media_template(name)
    seen = []
    render = template.messages2prompt

    def capture(prompt_messages, **kwargs):
        seen.extend(prompt_messages)
        return render(prompt_messages, **kwargs)

    template.messages2prompt = capture
    model = InternVLVisionModel.__new__(InternVLVisionModel)
    model.image_token = '<IMG_CONTEXT>'
    model.image_token_id = 13
    prompt, token = model.proc_messages(messages, template)
    assert prompt.count(token) == len(VisionModel.collect_multimodal_items(messages)) == 2
    assert seen[1]['role'] == 'tool'
    assert seen[1]['tool_call_id'] == 'screenshot'
    assert seen[1]['name'] == 'capture'
    assert messages == original
    preprocessed = messages + [dict(role='preprocess', content=[
        dict(image_token_id=13, image_tokens=2) for _ in range(2)
    ])]
    result = model.to_pytorch_aux(preprocessed, prompt, token, SimpleNamespace(encode=lambda text, **kw: [1]))
    assert len(result['multimodal']) == 2


@pytest.mark.parametrize('name', ['internvl3', 'internvl3_5'])
def test_image_only_tool_result(hf_media_template, name):
    messages = [dict(role='tool', tool_call_id='screenshot', content=[
        dict(type='image', data=Image.new('RGB', (28, 28))),
    ])]
    model = InternVLVisionModel.__new__(InternVLVisionModel)
    model.image_token = '<IMG_CONTEXT>'
    prompt, token = model.proc_messages(messages, hf_media_template(name))
    assert prompt.count(token) == 1
