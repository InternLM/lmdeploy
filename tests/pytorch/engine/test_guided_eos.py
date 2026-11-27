# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise real grammar masks with model-specific turn-ending tokens."""
import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from lmdeploy.pytorch.engine.guided_process import GuidedDecodingManager


@pytest.fixture
def tokenizer():
    vocab = {'<unk>': 0, '<eos>': 1, '<user>': 2, '<observation>': 3, 'ok': 4, 'bad': 5}
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
    matcher = manager.get_or_create_processor(1, 1, {'type': 'regex_schema', 'regex_schema': 'ok'})
    assert not allowed(manager, matcher, stop_token)
    assert matcher.accept_string('ok')
    assert allowed(manager, matcher, stop_token)
    assert matcher.accept_token(stop_token)
    assert manager.is_terminated(matcher)


@pytest.mark.parametrize('response_format,output', [
    ({'type': 'regex_schema', 'regex_schema': 'ok'}, 'ok'),
    ({'type': 'json_object'}, '{}'),
])
@pytest.mark.parametrize('stop_token', [1, 2, 3])
def test_model_eos_preserves_distinct_tokenizer_eos(tokenizer, stop_token, response_format, output):
    # Like InternVL3, the model's EOS list need not include the tokenizer EOS.
    model_eos = [2, 3]
    manager = GuidedDecodingManager(tokenizer, len(tokenizer), stop_token_ids=model_eos)
    assert model_eos == [2, 3]
    matcher = manager.get_or_create_processor(1, 1, response_format)
    assert not allowed(manager, matcher, stop_token)
    assert matcher.accept_string(output)
    assert allowed(manager, matcher, stop_token)
    assert matcher.accept_token(stop_token)
    assert manager.is_terminated(matcher)


def test_default_stop_detection_is_preserved(tokenizer):
    manager = GuidedDecodingManager(tokenizer, len(tokenizer))
    matcher = manager.get_or_create_processor(1, 1, {'type': 'regex_schema', 'regex_schema': 'ok'})
    assert matcher.accept_string('ok')
    assert allowed(manager, matcher, 1)
    assert not allowed(manager, matcher, 2)
    assert not allowed(manager, matcher, 3)


@pytest.mark.parametrize('session_id', [1, 2])
@pytest.mark.parametrize('initial_token_id', [1, 5], ids=['premature_eos', 'invalid_text'])
def test_rejected_initial_token_preserves_existing_matchers_and_allows_retry(tokenizer, session_id, initial_token_id):
    manager = GuidedDecodingManager(tokenizer, len(tokenizer))
    response_format = {'type': 'regex_schema', 'regex_schema': 'okok'}
    existing = manager.get_or_create_processor(1, 1, response_format, initial_token_id=4)

    with pytest.raises(ValueError, match=f'Initial output token {initial_token_id} does not match the grammar'):
        manager.get_or_create_processor(session_id, 2, response_format, initial_token_id=initial_token_id)

    assert manager.processors == {1: {1: existing}}
    assert not allowed(manager, existing, 1)
    assert existing.accept_token(4)
    assert allowed(manager, existing, 1)

    matcher = manager.get_or_create_processor(session_id, 2, response_format, initial_token_id=4)
    assert matcher is not existing
    assert not allowed(manager, matcher, 1)
    assert matcher.accept_token(4)
    assert allowed(manager, matcher, 1)


def test_pd_recompute_initializes_grammar_once_from_prefill_token(tokenizer):
    import numpy as np

    from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
    from lmdeploy.pytorch.messages import SequenceMeta, UpdateTokenMode
    from lmdeploy.pytorch.paging.scheduler import Scheduler
    from lmdeploy.pytorch.strategies.ar.sampling import ARSamplingStrategy
    from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy

    scheduler = Scheduler(SchedulerConfig(max_batches=2, max_session_len=32),
                          CacheConfig(max_batches=2, block_size=4, num_gpu_blocks=8, num_cpu_blocks=0),
                          SequenceMeta(4, strategy=ARSpecSequenceStrategy()))
    seq = scheduler.add_session(1).add_sequence(np.arange(8))
    seq.sampling_param.response_format = {'type': 'regex_schema', 'regex_schema': 'okok'}
    seq.update_token_ids(np.array([4]), draft_token_ids=np.array([], dtype=np.int64), mode=UpdateTokenMode.PREFILL)
    # PDCoordinator releases migration_request after installing the accepted first token.
    assert seq.migration_request is None
    # The first D forward can be a recompute prefill, without the PD decode carrier.
    seq.set_step(0)
    inputs = ARSamplingStrategy(0).make_sampling_inputs([seq])
    manager = GuidedDecodingManager(tokenizer, len(tokenizer))
    matcher = manager.get_processors(inputs.session_ctx, inputs.response_formats)[0]
    assert not allowed(manager, matcher, 1)
    assert matcher.accept_token(4)
    assert allowed(manager, matcher, 1)
    # A later preemption reuses the advanced matcher instead of replaying y1 or resetting it.
    assert manager.get_processors(inputs.session_ctx, inputs.response_formats)[0] is matcher
    assert allowed(manager, matcher, 1)
