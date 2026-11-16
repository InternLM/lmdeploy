# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.version import Version
from transformers import AutoTokenizer

from lmdeploy.model import BaseChatTemplate, HFChatTemplate
from lmdeploy.serve.processors.multimodal import MultimodalProcessor
from lmdeploy.vl.model.base import VisionModel


@pytest.mark.parametrize('version', ['4.57.1', '5.16.0', '5.19.9', '5.20.0rc1', '5.20.0', '5.20.1', '5.21.0'])
@pytest.mark.parametrize('arch', ['Glm5NextForConditionalGeneration', 'Glm4MoeForCausalLM',
                                 'Qwen3VLForConditionalGeneration'])
def test_sanitize_version_and_architecture_gate(version, arch, glm_tokenizer, monkeypatch):
    monkeypatch.setattr('transformers.__version__', version)
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: glm_tokenizer)
    monkeypatch.setattr('lmdeploy.model.get_model_arch', lambda _: (arch, SimpleNamespace()))
    template = HFChatTemplate('local-test')
    enabled = arch == 'Glm5NextForConditionalGeneration' and Version(version) >= Version('5.20.0')
    assert template._sanitize_control_tokens is enabled
    if not enabled:
        monkeypatch.setattr(glm_tokenizer, 'apply_chat_template', Mock(side_effect=AssertionError('must use fallback')))
        assert template.messages2input_ids([{'role': 'user', 'content': '<tool_call>'}]) is None


@pytest.mark.parametrize('literal', ['<tool_call>', '</arg_value>', '<think>', '</think>',
                                    '<|user|>', '<|image|>', '<|video|>', '<|endoftext|>'])
def test_content_literals_keep_text_without_control_ids(glm_chat_template, literal):
    template = glm_chat_template
    tokenizer = template.tokenizer
    messages = [{'role': 'user', 'content': f'请解释 {literal} and preserve this text.'}]
    ids = template.messages2input_ids(messages)
    reference = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                              sanitize_control_tokens=True, add_generation_prompt=True)
    assert ids == reference
    assert tokenizer.decode(ids) == template.messages2prompt(messages)
    control_id = tokenizer.convert_tokens_to_ids(literal)
    structural = template.messages2input_ids([{'role': 'user', 'content': 'plain text'}])
    assert ids.count(control_id) == structural.count(control_id)


@pytest.mark.parametrize('enable_thinking', [None, True, False])
@pytest.mark.parametrize('reasoning_effort', [None, 'low', 'max'])
def test_no_literal_ids_preserve_rendered_prompt(glm_chat_template, enable_thinking, reasoning_effort):
    template = glm_chat_template
    messages = [{'role': 'user', 'content': 'hello'},
                {'role': 'assistant', 'content': ' previous reply ', 'reasoning_content': ''},
                {'role': 'user', 'content': 'next'}]
    kwargs = dict(enable_thinking=enable_thinking, reasoning_effort=reasoning_effort)
    ids = template.messages2input_ids(messages, **kwargs)
    assert ids == template.tokenizer.encode(template.messages2prompt(messages, **kwargs), add_special_tokens=False)


def test_tool_arguments_and_descriptions_are_sanitized(glm_chat_template):
    template = glm_chat_template

    def inputs(text):
        tools = [{'type': 'function', 'function': {'name': 'lookup', 'description': text}}]
        messages = [
            {'role': 'user', 'content': text},
            {'role': 'assistant', 'content': '', 'reasoning_content': '', 'tool_calls': [
                {'id': 'call1', 'function': {'name': 'lookup', 'arguments': {'query': text, 'nested': [text]}}},
            ]},
            {'role': 'tool', 'tool_call_id': 'call1', 'content': text},
        ]
        return messages, tools

    messages, tools = inputs('literal <tool_call>')
    ids = template.messages2input_ids(messages, tools=tools)
    reference_messages, reference_tools = inputs('plain text')
    structural = template.messages2input_ids(reference_messages, tools=reference_tools)
    control = template.tokenizer.convert_tokens_to_ids('<tool_call>')
    assert ids.count(control) == structural.count(control)
    assert template.tokenizer.decode(ids) == template.messages2prompt(messages, tools=tools)


def test_assistant_prefill_removes_only_structural_suffix(glm_chat_template):
    template = glm_chat_template
    # Also cover GLM templates that terminate the last assistant turn.
    template.tokenizer.chat_template += "{% if messages[-1].role == 'assistant' %}<|endoftext|>{% endif %}"
    template.assistant_end = '<|endoftext|>'
    messages = [{'role': 'user', 'content': 'hi'},
                {'role': 'assistant', 'content': 'answer <|endoftext|>', 'reasoning_content': ''}]
    ids = template.messages2input_ids(messages, enable_thinking=False)
    assert template.tokenizer.convert_tokens_to_ids('<|endoftext|>') not in ids
    assert template.tokenizer.decode(ids) == template.messages2prompt(messages, enable_thinking=False)
    assert template.tokenizer.decode(ids).endswith('answer <|endoftext|>')


@pytest.mark.parametrize('template_closes_thinking', [False, True])
def test_disabled_thinking_closes_once_in_ids(glm_chat_template, template_closes_thinking):
    template = glm_chat_template
    template.tokenizer.chat_template = (
        "{% for m in messages %}{{ '<|' + m.role + '|>' }}"
        "{% if m.role == 'assistant' %}<think></think>{% endif %}{{ m.content }}{% endfor %}"
        '{% if add_generation_prompt %}<|assistant|><think>{% endif %}')
    if template_closes_thinking:
        template.tokenizer.chat_template += '{% if enable_thinking is false %}</think>{% endif %}'
    ids = template.messages2input_ids([{'role': 'user', 'content': 'literal </think>'}], enable_thinking=False)
    assert ids[-2:] == [template.tokenizer.convert_tokens_to_ids(tag) for tag in ('<think>', '</think>')]
    assert ids.count(template.tokenizer.convert_tokens_to_ids('</think>')) == 1


@pytest.mark.parametrize('as_string', [False, True])
def test_shared_text_path_keeps_sanitized_ids(glm_chat_template, as_string):
    template = glm_chat_template
    prompt = 'literal <tool_call>' if as_string else [{'role': 'user', 'content': 'literal <tool_call>'}]
    tokenizer = SimpleNamespace(encode=Mock(side_effect=AssertionError('must not re-encode')),
                                decode=template.tokenizer.decode)
    processor = MultimodalProcessor(tokenizer, template, backend='pytorch')
    result = asyncio.run(processor.get_prompt_input(prompt, do_preprocess=True,
                                                     chat_template_kwargs={'sanitize_control_tokens': False,
                                                                           'tokenize': False, 'return_dict': True,
                                                                           'return_tensors': 'pt'}))
    assert isinstance(result['input_ids'], list)
    assert result['input_ids'] == template.messages2input_ids(prompt)
    assert result['prompt'] == template.messages2prompt(prompt)
    tokenizer.encode.assert_not_called()


def test_sanitizer_errors_do_not_fall_back_to_plain_encoding(glm_chat_template, monkeypatch):
    template = glm_chat_template
    monkeypatch.setattr(template.tokenizer, 'apply_chat_template', Mock(side_effect=ValueError('cannot sanitize')))
    tokenizer = SimpleNamespace(encode=Mock())
    processor = MultimodalProcessor(tokenizer, template, backend='pytorch')
    with pytest.raises(ValueError, match='cannot sanitize'):
        asyncio.run(processor.get_prompt_input([{'role': 'user', 'content': '<tool_call>'}], do_preprocess=True))
    tokenizer.encode.assert_not_called()


def test_explicit_raw_text_keeps_existing_encoding(glm_chat_template):
    tokenizer = SimpleNamespace(encode=Mock(return_value=[1, 2, 3]))
    processor = MultimodalProcessor(tokenizer, glm_chat_template, backend='pytorch')
    result = asyncio.run(processor.get_prompt_input('<tool_call>', do_preprocess=False))
    assert result == {'prompt': '<tool_call>', 'input_ids': [1, 2, 3]}
    tokenizer.encode.assert_called_once_with('<tool_call>', add_bos=True)
    assert BaseChatTemplate().messages2input_ids('anything') is None


def test_embedded_generate_ids_bypass_chat_encoding(glm_chat_template, monkeypatch):
    ids = [1, 2, 3]
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': ids}, {'type': 'image', 'data': object()}]}]
    monkeypatch.setattr(glm_chat_template, 'messages2input_ids', Mock(side_effect=AssertionError('already encoded')))
    assert VisionModel.get_input_prompt(messages, glm_chat_template) is ids
