# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise real grammar masks with model-specific turn-ending tokens."""
import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from lmdeploy.pytorch.engine.guided_process import GuidedDecodingManager


@pytest.fixture
def tokenizer():
    vocab = {'<unk>': 0, '<eos>': 1, '<user>': 2, '<observation>': 3, 'ok': 4}
    return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token='<unk>')),
                                  unk_token='<unk>', eos_token='<eos>',
                                  additional_special_tokens=['<user>', '<observation>'])


def allowed(manager, matcher, token):
    mask = manager.allocate_batched_bitmap(1)
    manager.fill_bitmap(matcher, mask, 0)
    return bool(int(mask[0, token // 32]) & (1 << (token % 32)))


@pytest.mark.parametrize('stop_token', [1, 2, 3])
def test_model_eos_is_allowed_only_after_complete_output(tokenizer, stop_token):
    manager = GuidedDecodingManager(tokenizer, len(tokenizer), stop_token_ids=[1, 2, 3])
    matcher = manager.get_processor(1, 1, {'type': 'regex_schema', 'regex_schema': 'ok'})
    assert not allowed(manager, matcher, stop_token)
    assert matcher.accept_string('ok')
    assert allowed(manager, matcher, stop_token)
    assert matcher.accept_token(stop_token)
    assert manager.is_terminated(matcher)


def test_default_stop_detection_is_preserved(tokenizer):
    manager = GuidedDecodingManager(tokenizer, len(tokenizer))
    matcher = manager.get_processor(1, 1, {'type': 'regex_schema', 'regex_schema': 'ok'})
    assert matcher.accept_string('ok')
    assert allowed(manager, matcher, 1)
    assert not allowed(manager, matcher, 2)
    assert not allowed(manager, matcher, 3)
