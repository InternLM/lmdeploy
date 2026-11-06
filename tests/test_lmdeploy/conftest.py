# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path

import pytest


@pytest.fixture
def hf_media_template():
    """Load a real chat template without downloading model
    weights/tokenizers."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from lmdeploy.model import HFChatTemplate

    def create(name='glm5_3_flash', *, source=None):
        if source is None:
            source = (Path(__file__).parent / 'test_vl' / 'templates' / f'{name}.jinja').read_text(encoding='utf-8')
        template = HFChatTemplate.__new__(HFChatTemplate)
        template.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({'<unk>': 0}, unk_token='<unk>')), chat_template=source)
        template.assistant_end = ''
        template.is_gpt_oss = False
        template.is_glm5_next = name == 'glm5_3_flash'
        # This fixture exercises rendered strings with an unknown-only vocabulary.
        template._sanitize_control_tokens = False
        return template

    return create


@pytest.fixture
def glm_tokenizer():
    """A lossless local tokenizer with GLM's template and control spellings."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({char: i for i, char in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens([
        '[gMASK]', '<sop>', '<|system|>', '<|user|>', '<|assistant|>', '<|observation|>', '<|endoftext|>',
        '<|begin_of_image|>', '<|image|>', '<|end_of_image|>', '<|begin_of_video|>', '<|video|>', '<|end_of_video|>',
    ])
    # GLM's tool/reasoning controls include added tokens that are not special.
    backend.add_tokens(['<think>', '</think>', '<tool_call>', '</tool_call>',
                        '<arg_key>', '</arg_key>', '<arg_value>', '</arg_value>'])
    source = (Path(__file__).parent / 'test_vl' / 'templates' / 'glm5_3_flash.jinja').read_text()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<|endoftext|>',
                                   pad_token='<|endoftext|>', chat_template=source)


@pytest.fixture
def glm_chat_template(glm_tokenizer, monkeypatch):
    """Exercise the real HF sanitizer without downloading a tokenizer."""
    from types import SimpleNamespace

    pytest.importorskip('transformers', minversion='5.20.0')
    from transformers import AutoTokenizer

    from lmdeploy.model import HFChatTemplate

    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: glm_tokenizer)
    monkeypatch.setattr('lmdeploy.model.get_model_arch',
                        lambda _: ('Glm5NextForConditionalGeneration', SimpleNamespace()))
    return HFChatTemplate('local-glm')
