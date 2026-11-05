# Copyright (c) OpenMMLab. All rights reserved.

import copy
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from lmdeploy.model import HFChatTemplate
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.response_parser import normalize_chat_request
from lmdeploy.serve.processors.multimodal import MultimodalProcessor


@pytest.fixture
def template(monkeypatch):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0}, unk_token='[UNK]')))
    # The original GLM-5.3 template always opens reasoning, even when the
    # caller passes enable_thinking=False. Keep that behavior in the fixture.
    tokenizer.chat_template = (
        "{% for m in messages %}{{ '<|' + m.role + '|>' + m.content }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|assistant|><think>' }}{% endif %}")
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr('lmdeploy.model.get_model_arch',
                        lambda _: ('Glm5NextForConditionalGeneration', SimpleNamespace()))
    return HFChatTemplate(model_path='local-test')


@pytest.mark.parametrize('enable_thinking', [None, True, False])
def test_glm5_generation_prompt_honors_thinking_mode(template, enable_thinking):
    messages = [{'role': 'user', 'content': 'Reply with one word: ok'}]
    result = template.messages2prompt(messages, enable_thinking=enable_thinking)
    expected = '<|user|>Reply with one word: ok<|assistant|><think>'
    if enable_thinking is False:
        expected += '</think>'
    assert result == expected


def test_glm5_does_not_change_assistant_prefill(template):
    messages = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': '<think>'}]
    assert template.messages2prompt(messages, enable_thinking=False) == '<|user|>hi<|assistant|><think>'


def test_glm5_does_not_double_close_supported_template(template):
    template.tokenizer.chat_template += "{% if enable_thinking is false %}{{ '</think>' }}{% endif %}"
    assert template.messages2prompt([{'role': 'user', 'content': 'hi'}], enable_thinking=False).endswith(
        '<think></think>')


def test_other_models_keep_their_template_contract(template):
    template.is_glm5_next = False
    assert template.messages2prompt([{'role': 'user', 'content': 'hi'}], enable_thinking=False).endswith('<think>')


@pytest.mark.parametrize('reasoning_fields, expected', [
    pytest.param({}, '', id='no-reasoning'),
    pytest.param({'reasoning_content': 'legacy thought'}, 'legacy thought', id='legacy-only'),
    pytest.param({'reasoning': 'new thought'}, 'new thought', id='reasoning-only'),
    pytest.param({'reasoning': 'new thought', 'reasoning_content': 'legacy thought'},
                 'new thought', id='reasoning-wins'),
    pytest.param({'reasoning': None, 'reasoning_content': 'legacy thought'}, 'legacy thought', id='null-reasoning'),
    pytest.param({'reasoning': 'new thought', 'reasoning_content': None}, 'new thought', id='null-legacy'),
    pytest.param({'reasoning': '', 'reasoning_content': 'legacy thought'}, '', id='empty-reasoning-wins'),
    pytest.param({'reasoning': 'new thought', 'reasoning_content': ''}, 'new thought', id='empty-legacy'),
    pytest.param({'reasoning': 'same thought', 'reasoning_content': 'same thought'}, 'same thought', id='same-fields'),
])
@pytest.mark.parametrize('tool_round', [False, True], ids=['chat', 'tool'])
def test_glm5_renders_reasoning_alias_in_history(hf_media_template, reasoning_fields, expected, tool_round):
    assistant = {'role': 'assistant', 'content': '4', **reasoning_fields}
    tail = {'role': 'user', 'content': 'Check the result.'}
    if tool_round:
        assistant.update(content=None, tool_calls=[{
            'id': 'call_1', 'type': 'function',
            'function': {'name': 'lookup', 'arguments': '{"key": "test"}'},
        }])
        tail = {'role': 'tool', 'tool_call_id': 'call_1', 'content': '4'}
    messages = [{'role': 'user', 'content': 'Find the result.'}, assistant, tail]
    request = ChatCompletionRequest(model='glm5', messages=messages)
    original = copy.deepcopy(request.model_dump())
    normalized = normalize_chat_request(request)
    rendered_messages = [MultimodalProcessor.merge_message_content(msg) for msg in normalized.messages]
    prompt = hf_media_template().messages2prompt(rendered_messages, clear_thinking=False)

    assert f'<|assistant|><think>{expected}</think>' in prompt
    assert request.model_dump() == original
    assert messages == original['messages']
    if tool_round:
        assert normalized.messages[1]['tool_calls'][0]['function']['arguments'] == {'key': 'test'}
