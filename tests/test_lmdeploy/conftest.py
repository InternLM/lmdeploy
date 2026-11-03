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
        return template

    return create
