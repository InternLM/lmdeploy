# Copyright (c) OpenMMLab. All rights reserved.
"""Preserve valid HF rendering without imposing a particular Jinja
traversal."""
import pytest
from PIL import Image

from lmdeploy.vl.model.base import VisionModel


@pytest.mark.parametrize('source', [
    "{% for p in messages[0].content %}{% if p.type == 'image' %}<|image|>{% endif %}{% endfor %}",
    "{% if messages[0].content[0].type == 'image' %}<|image|>{% endif %}",
    '{% for i in range(messages[0].content|length) %}{% set p = messages[0].content[i] %}'
    "{% if p.type == 'image' %}<|image|>{% endif %}{% endfor %}",
    '{% macro emit(content) %}{% for p in content %}  '
    "{% if p.type == 'image' %}<|image|>{% endif %}  {% endfor %}{% endmacro %}"
    '{{ emit(messages[0].content).strip() }}',
    "{% for p in messages[0].content %}{% if p.type == 'image' %}<|image|>"
    '{% break %}{% endif %}{% endfor %}',
    "{% for p in messages[0].content %}{% if p.type == 'image' %}<|image|>"
    '{% continue %}{% endif %}{% endfor %}',
])
def test_custom_image_template_traversals(hf_media_template, source):
    messages = [dict(role='user', content=[dict(type='image', data=Image.new('RGB', (28, 28)))])]
    template = hf_media_template('custom', source=source)
    assert template.prepare_multimodal_messages(messages) is messages
    assert VisionModel.get_input_prompt(messages, template) == '<|image|>'


@pytest.mark.parametrize('modality,token', [
    ('image', '<|image_pad|>'),
    ('video', '<|video_pad|>'),
    ('audio', '<|audio_pad|>'),
])
def test_qwen3_omni_can_accumulate_media_tokens_in_variables(hf_media_template, modality, token):
    # Rendering boundary only; this does not exercise audio/video decoding.
    messages = [dict(role='user', content=[dict(type=modality, data='parsed-media')])]
    template = hf_media_template('qwen3_omni')
    assert template.prepare_multimodal_messages(messages) is messages
    prompt = VisionModel.get_input_prompt(messages, template)
    assert prompt.count(token) == 1
