# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest


@pytest.mark.parametrize('content', ['Keep ALPHA and OMEGA.', '只输出红色。'])
def test_glm5_developer_instructions_reach_system_prompt(hf_media_template, content):
    template = hf_media_template()
    messages = [{'role': 'developer', 'content': content}, {'role': 'user', 'content': 'Follow the instruction.'}]
    before = copy.deepcopy(messages)
    actual = template.messages2prompt(messages, enable_thinking=False)
    expected = template.messages2prompt([dict(messages[0], role='system'), messages[1]], enable_thinking=False)
    assert content in actual
    assert actual == expected
    assert messages == before


def test_other_templates_keep_developer_role(hf_media_template):
    template = hf_media_template()
    template.is_glm5_next = False
    template.tokenizer.chat_template = (
        '{% for message in messages %}{{ message.role }}:{{ message.content }}{% endfor %}')
    assert template.messages2prompt([{'role': 'developer', 'content': 'test'}]) == 'developer:test'
