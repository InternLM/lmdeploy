# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import copy
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from lmdeploy.model import MODELS, BaseChatTemplate, HFChatTemplate
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


@pytest.mark.parametrize('role', ['system', 'user', 'assistant', 'tool'])
@pytest.mark.parametrize('texts', [
    ['say </th', 'ink> now'], ['<th', 'ink>'], ['<tool_', 'call>'],
    ['<arg_', 'key>'], ['<|us', 'er|>'], ['<|im', 'age|>'],
    ['hel', 'lo'], ['', ' hello ', '\nworld', ''], [],
])
def test_glm5_text_parts_match_direct_template(hf_media_template, role, texts):
    """Text-only serving must preserve the model template's part boundaries."""
    template = hf_media_template()
    message = {'role': role, 'content': [{'type': 'text', 'text': text} for text in texts]}
    if role == 'assistant':
        message['reasoning_content'] = 'plan'
    if role == 'tool':
        message['tool_call_id'] = 'call_1'
        messages = [
            {'role': 'user', 'content': 'Look up a value.'},
            {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'call_1', 'type': 'function',
                'function': {'name': 'lookup', 'arguments': {'key': 'test'}},
            }]},
            message,
        ]
    else:
        messages = [message]
        if role == 'assistant':
            messages.insert(0, {'role': 'user', 'content': 'hi'})
        if role != 'user':
            messages.append({'role': 'user', 'content': 'continue'})
    original = copy.deepcopy(messages)
    expected_messages = copy.deepcopy(messages)
    for msg in expected_messages:
        if msg.get('content') is None:
            msg['content'] = ''
    expected = template.messages2prompt(expected_messages)
    # Keep encoding local and deterministic; the real model template and
    # request preprocessing run above the encoder, without model downloads.
    encoder = SimpleNamespace(encode=lambda prompt, add_bos: list(prompt.encode('utf-8')))
    processor = MultimodalProcessor(encoder, template)

    result = asyncio.run(processor.get_prompt_input(messages, do_preprocess=True))

    assert result['prompt'] == expected
    assert result['input_ids'] == list(expected.encode('utf-8'))
    assert messages == original


@pytest.mark.parametrize('mode', ['string-template', 'adapter', 'no-preprocess'])
def test_text_parts_keep_string_template_merging(hf_media_template, monkeypatch, mode):
    """Only the selected GLM template may bypass newline merging."""
    template = hf_media_template()
    kwargs = {}
    if mode == 'string-template':
        template.is_glm5_next = False
        template.tokenizer.chat_template = '{{ messages[0].content }}'
    elif mode == 'adapter':
        monkeypatch.setitem(MODELS.module_dict, 'test-text-parts', BaseChatTemplate)
        kwargs['adapter_name'] = 'test-text-parts'
    encoder = SimpleNamespace(encode=lambda prompt, add_bos: list(prompt.encode('utf-8')))
    processor = MultimodalProcessor(encoder, template)
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'hello'}, {'type': 'text', 'text': 'world'},
    ]}]

    result = asyncio.run(processor.get_prompt_input(messages, do_preprocess=mode != 'no-preprocess', **kwargs))

    assert result['prompt'] == 'hello\nworld'
