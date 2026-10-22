# Copyright (c) OpenMMLab. All rights reserved.

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from lmdeploy.model import HFChatTemplate


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
