# Copyright (c) OpenMMLab. All rights reserved.
"""Three-protocol conversion and shared preprocessing tests, without HTTP or
generation.

HTTP/SSE tool-image tests live in each protocol's endpoint test directory.
"""
import asyncio
import base64
import copy
import itertools
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from lmdeploy.model import BaseChatTemplate, HFChatTemplate
from lmdeploy.serve.anthropic.adapter import to_openai_messages
from lmdeploy.serve.anthropic.protocol import MessagesRequest
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.openai.responses import ResponsesRequest
from lmdeploy.serve.openai.responses.request import messages_from_input
from lmdeploy.serve.parsers.response_parser import normalize_chat_request
from lmdeploy.serve.processors.multimodal import MultimodalProcessor
from lmdeploy.vl.model.base import VisionModel
from lmdeploy.vl.model.internvl import InternVLVisionModel

CALL_IDS = ['call_z', 'call_a', 'call_m']
COLORS = ['red', 'blue', 'lime']
PIXELS = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]


def _image(color):
    buf = BytesIO()
    Image.new('RGB', (28, 28), color).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _messages(protocol, order, two_images=False):
    calls = [dict(id=call_id, type='function', function=dict(name='screenshot', arguments='{}'))
             for call_id in CALL_IDS]
    contents = []
    for color in COLORS:
        parts = [dict(type='text', text='before'), dict(type='image_url', image_url=dict(url=_image(color))),
                 dict(type='text', text='after')]
        if two_images:
            parts += [dict(type='image_url', image_url=dict(url=_image('white')))]
        contents.append(parts)
    messages = [dict(role='user', content='Read the screenshots.'),
                dict(role='assistant', content=None, tool_calls=calls)]
    messages += [dict(role='tool', tool_call_id=CALL_IDS[i], content=contents[i]) for i in order]
    if protocol == 'chat':
        return normalize_chat_request(ChatCompletionRequest(model='fake-model', messages=messages)).messages
    if protocol == 'messages':
        uses = [dict(type='tool_use', id=call_id, name='screenshot', input={}) for call_id in CALL_IDS]
        results = []
        for i in order:
            parts = [dict(type='image', source=dict(type='url', url=p['image_url']['url']))
                     if p['type'] == 'image_url' else p for p in contents[i]]
            results.append(dict(type='tool_result', tool_use_id=CALL_IDS[i], content=parts))
        request = MessagesRequest(model='fake-model', max_tokens=32, messages=[messages[0],
                                  dict(role='assistant', content=uses), dict(role='user', content=results)])
        return normalize_chat_request(ChatCompletionRequest(model='fake-model',
                                                           messages=to_openai_messages(request))).messages
    items = [messages[0]]
    items += [dict(type='function_call', call_id=call_id, name='screenshot', arguments='{}') for call_id in CALL_IDS]
    for i in order:
        parts = [dict(type='input_image', image_url=p['image_url']['url']) if p['type'] == 'image_url'
                 else dict(type='input_text', text=p['text']) for p in contents[i]]
        items.append(dict(type='function_call_output', call_id=CALL_IDS[i], output=parts))
    return normalize_chat_request(ChatCompletionRequest(model='fake-model', messages=messages_from_input(
        ResponsesRequest(model='fake-model', input=items)))).messages



def _template(glm=True):
    template = HFChatTemplate.__new__(HFChatTemplate)
    template.is_glm5_next = glm
    return template


@pytest.mark.parametrize('protocol', ['chat', 'messages', 'responses'])
@pytest.mark.parametrize('order', list(itertools.permutations(range(3))))
@pytest.mark.parametrize('two_images', [False, True])
@pytest.mark.parametrize('backend', ['pytorch', 'turbomind'])
def test_parallel_tool_images_keep_ownership_through_shared_processor(protocol, order, two_images, backend,
                                                                     hf_media_template):
    messages = _messages(protocol, order, two_images)
    before = copy.deepcopy(messages)
    assert [m['tool_call_id'] for m in messages[2:]] == [CALL_IDS[i] for i in order]
    assert [call['id'] for call in messages[1]['tool_calls']] == CALL_IDS
    # Exercise the actual shared path used before both prompt rendering and
    # media collection, with real decoding and no model/GPU allocation.
    seen = {}

    def get_input_prompt(messages, **kwargs):
        seen['rendered'] = messages
        return VisionModel.get_input_prompt(messages, **kwargs)

    async def preprocess(messages, input_prompt, **kwargs):
        assert messages is seen['rendered']
        assert input_prompt.count('<|image|>') == (6 if two_images else 3)
        seen['collected'] = VisionModel.collect_multimodal_items(messages)
        return messages

    encoder = SimpleNamespace(_uses_new_preprocess=True, model=SimpleNamespace(get_input_prompt=get_input_prompt),
                              preprocess=preprocess)
    processor = MultimodalProcessor(None, hf_media_template(), vl_encoder=encoder, backend=backend)
    result = asyncio.run(processor.get_prompt_input(messages, do_preprocess=True))
    assert [m['tool_call_id'] for m in result[2:]] == CALL_IDS
    pixels = [data.getpixel((0, 0)) for _, data, _ in seen['collected']]
    expected = [p for color in PIXELS for p in ([color, (255, 255, 255)] if two_images else [color])]
    assert pixels == expected
    for message in result[2:]:
        assert [part['type'] for part in message['content']][:3] == ['text', 'image', 'text']
        assert message['content'][0]['text'] == 'before'
        assert message['content'][2]['text'] == 'after'
    assert messages == before


def _parsed():
    return asyncio.run(MultimodalProcessor.async_parse_multimodal_item(_messages('chat', (2, 1, 0))))


@pytest.mark.parametrize('kind', ['missing_result', 'duplicate_result', 'unknown_result',
                                  'missing_call', 'duplicate_call', 'no_calls'])
def test_ambiguous_ids_keep_template_fallback_order(kind):
    messages = _parsed()
    if kind == 'missing_result':
        messages[2].pop('tool_call_id')
    elif kind == 'duplicate_result':
        messages[2]['tool_call_id'] = messages[3]['tool_call_id']
    elif kind == 'unknown_result':
        messages[2]['tool_call_id'] = 'unknown'
    elif kind == 'missing_call':
        messages[1]['tool_calls'][0].pop('id')
    elif kind == 'duplicate_call':
        messages[1]['tool_calls'][0]['id'] = messages[1]['tool_calls'][1]['id']
    else:
        messages[1].pop('tool_calls')
    ordered = _template().prepare_multimodal_messages(messages)
    assert all(left is right for left, right in zip(ordered, messages))


def test_partial_group_id_fallback_and_mixed_text_result():
    messages = _parsed()
    messages.pop(3)  # A subset of the calls has returned.
    messages[2]['id'] = messages[2].pop('tool_call_id')
    messages[2]['content'] = 'plain text result'
    ordered = _template().prepare_multimodal_messages(messages)
    assert ordered[2] is messages[3]
    assert ordered[3] is messages[2]


def test_no_sort_for_other_templates_or_text_only_results():
    messages = _parsed()
    assert _template(glm=False).prepare_multimodal_messages(messages) is messages
    assert BaseChatTemplate().prepare_multimodal_messages(messages) is messages
    for message in messages[2:]:
        message['content'] = 'text only'
    ordered = _template().prepare_multimodal_messages(messages)
    assert all(left is right for left, right in zip(ordered, messages))


def test_user_boundaries_and_separate_tool_rounds():
    first = _parsed()
    second = _parsed()
    messages = first + [dict(role='user', content='next round')] + second
    ordered = _template().prepare_multimodal_messages(messages)
    assert [m['tool_call_id'] for m in ordered if m['role'] == 'tool'] == CALL_IDS * 2
    split = first[:2] + [dict(role='user', content='boundary')] + first[2:]
    ordered = _template().prepare_multimodal_messages(split)
    assert all(left is right for left, right in zip(ordered, split))


def test_messages_preserves_user_content_around_tool_results():
    result = dict(type='tool_result', tool_use_id='call_z', content=[
        dict(type='text', text='inside before'), dict(type='image', source=dict(type='url', url=_image('red'))),
        dict(type='text', text='inside after'),
    ])
    request = MessagesRequest(model='fake-model', max_tokens=32, messages=[dict(role='user', content=[
        dict(type='text', text='before'), result, dict(type='text', text='between'), result,
        dict(type='image', source=dict(type='url', url=_image('blue'))), dict(type='text', text='after'),
    ])])
    messages = to_openai_messages(request)
    assert [m['role'] for m in messages] == ['user', 'tool', 'user', 'tool', 'user']
    assert messages[0]['content'] == 'before'
    assert messages[2]['content'] == 'between'
    assert [p['type'] for p in messages[1]['content']] == ['text', 'image_url', 'text']
    assert [p['type'] for p in messages[4]['content']] == ['image_url', 'text']
    assert messages[4]['content'][1]['text'] == 'after'


def test_responses_does_not_merge_calls_across_rounds_or_messages():
    call = dict(type='function_call', call_id='call_z', name='screenshot', arguments='{}')
    output = dict(type='function_call_output', call_id='call_z', output='text')
    request = ResponsesRequest(model='fake-model', input=[call, output, call,
                              dict(role='user', content='boundary'), call])
    messages = messages_from_input(request)
    assert [m['role'] for m in messages] == ['assistant', 'tool', 'assistant', 'user', 'assistant']
    assert all(len(m['tool_calls']) == 1 for m in messages if m['role'] == 'assistant')


@pytest.mark.parametrize('protocol', ['chat', 'messages', 'responses'])
def test_mixed_tool_results_across_rounds(protocol, hf_media_template):
    first = _messages(protocol, (2, 1, 0))
    first[3]['content'] = 'plain text result'
    messages = first + [dict(role='user', content=[
        dict(type='image_url', image_url=dict(url=_image('white'))),
        dict(type='text', text='next round'),
    ])] + _messages(protocol, (1, 2, 0), two_images=True)
    parsed = asyncio.run(MultimodalProcessor.async_parse_multimodal_item(messages))
    template = hf_media_template()
    parsed = template.prepare_multimodal_messages(parsed)
    prompt = VisionModel.get_input_prompt(parsed, template)
    items = VisionModel.collect_multimodal_items(parsed)
    assert prompt.count('<|image|>') == len(items)
    assert [image.getpixel((0, 0)) for _, image, _ in items] == [
        PIXELS[0], PIXELS[2], (255, 255, 255),
        PIXELS[0], (255, 255, 255), PIXELS[1], (255, 255, 255), PIXELS[2], (255, 255, 255),
    ]


@pytest.mark.parametrize('protocol', ['chat', 'messages', 'responses'])
@pytest.mark.parametrize('name', ['qwen3_vl', 'qwen2_5_vl', 'qwen3_omni'])
def test_other_templates_keep_their_own_tool_rendering_order(protocol, name, hf_media_template):
    messages = asyncio.run(MultimodalProcessor.async_parse_multimodal_item(
        _messages(protocol, (2, 0, 1), two_images=True)))
    prompt = VisionModel.get_input_prompt(messages, hf_media_template(name),
                                         chat_template_kwargs={'add_vision_id': True})
    items = VisionModel.collect_multimodal_items(messages)
    assert prompt.count('<|image_pad|>') == len(items)
    assert [image.getpixel((0, 0)) for _, image, _ in items] == [
        PIXELS[2], (255, 255, 255), PIXELS[0], (255, 255, 255), PIXELS[1], (255, 255, 255),
    ]


@pytest.mark.parametrize('protocol', ['chat', 'messages', 'responses'])
@pytest.mark.parametrize('name', ['internvl3', 'internvl3_5'])
def test_internvl_tool_image_placeholders_across_protocols(protocol, name, hf_media_template):
    messages = asyncio.run(MultimodalProcessor.async_parse_multimodal_item(
        _messages(protocol, (2, 0, 1), two_images=True)))
    model = InternVLVisionModel.__new__(InternVLVisionModel)
    model.image_token = '<IMG_CONTEXT>'
    prompt, token = model.proc_messages(messages, hf_media_template(name))
    assert prompt.count(token) == len(VisionModel.collect_multimodal_items(messages)) == 6


@pytest.mark.parametrize('protocol', ['chat', 'messages', 'responses'])
@pytest.mark.parametrize('name,token', [
    ('glm5_3_flash', '<|image|>'),
    ('qwen3_vl', '<|image_pad|>'),
    ('qwen2_5_vl', '<|image_pad|>'),
    ('qwen3_omni', '<|image_pad|>'),
])
@pytest.mark.parametrize('count', [1, 2])
def test_user_images_keep_official_template_behavior(protocol, name, token, count, hf_media_template):
    urls = [_image(color) for color in COLORS[:count]]
    if protocol == 'messages':
        content = [dict(type='image', source=dict(type='url', url=url)) for url in urls]
        request = MessagesRequest(model='fake-model', max_tokens=16,
                                  messages=[dict(role='user', content=content)])
        messages = to_openai_messages(request)
    elif protocol == 'responses':
        content = [dict(type='input_image', image_url=url) for url in urls]
        messages = messages_from_input(ResponsesRequest(model='fake-model',
                                                       input=[dict(role='user', content=content)]))
    else:
        content = [dict(type='image_url', image_url=dict(url=url)) for url in urls]
        messages = ChatCompletionRequest(model='fake-model', messages=[dict(role='user', content=content)]).messages
    parsed = asyncio.run(MultimodalProcessor.async_parse_multimodal_item(messages))
    template = hf_media_template(name)
    prepared = template.prepare_multimodal_messages(parsed)
    prompt = VisionModel.get_input_prompt(prepared, template)
    assert prompt.count(token) == count
    assert [image.getpixel((0, 0)) for _, image, _ in VisionModel.collect_multimodal_items(prepared)] == PIXELS[:count]
