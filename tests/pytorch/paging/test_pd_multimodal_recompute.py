# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy


def make_replay(*, hybrid=False, checkpoint=24, image_end=12, same_image=True, mtp=False, apc=True):
    config = CacheConfig(max_batches=2, block_size=4, num_cpu_blocks=0, num_gpu_blocks=64,
                         enable_prefix_caching=apc, max_prefill_token_num=64, prefill_save_alignment=4,
                         states_shapes=[((2,), torch.float32)] if hybrid else [],
                         num_state_caches=6 if hybrid else 0, prefix_cache_state_budget=2 if hybrid else 0)
    strategy = ARSpecSequenceStrategy(recompute_blocks=1) if mtp else ARSequenceStrategy()
    scheduler = Scheduler(SchedulerConfig(max_batches=2, max_session_len=128, max_request_output_len=32),
                          config, SequenceMeta(4, strategy=strategy))
    images = {'image': [MultiModalData(torch.tensor([1]), start=4, end=image_end, content_hash='image-a')]}
    cached = scheduler.add_session(0).add_sequence(range(32), multimodals=images)
    assert scheduler.schedule(is_prefill=True).running == [cached]
    if hybrid and checkpoint is not None:
        assert scheduler.state_checkpoints.reserve_save(cached, step=checkpoint) > 0
        assert scheduler.state_checkpoints.publish_save(cached)
    scheduler.end_session(0)

    seq = scheduler.add_session(1).add_sequence(range(33))
    if not same_image:
        images['image'][0].content_hash = 'image-b'
    seq.prefix_cache.multimodal_spans = seq._make_multimodal_spans(images)
    seq.pd_metadata_only = True
    return scheduler, seq


@pytest.mark.parametrize(('options', 'matched_step'), [
    ({}, 32),
    ({'hybrid': True}, 24),
    ({'hybrid': True, 'mtp': True}, 24),
    ({'hybrid': True, 'checkpoint': None}, None),
    ({'hybrid': True, 'checkpoint': 4}, None),
    ({'same_image': False}, None),
    ({'mtp': True, 'image_end': 32}, None),
    ({'mtp': True}, 28),
    ({'apc': False}, None),
])
def test_pd_replay_requires_media_only_outside_the_restored_prefix(options, matched_step):
    scheduler, seq = make_replay(**options)
    free_blocks = scheduler.block_manager.get_num_free_gpu_blocks()
    output = scheduler.schedule(is_prefill=True)
    assert seq.pd_metadata_only and seq.history_multimodals.empty()
    if matched_step is not None:
        assert output.running == [seq]
        assert seq.num_history_ids == matched_step
        assert scheduler.take_multimodal_requests() == []
        assert seq.prefix_cache.restore.is_selected == bool(options.get('hybrid'))
    else:
        assert not output.running
        assert seq.status == MessageStatus.WAITING_FOR_MULTIMODALS
        assert seq.num_blocks == seq.num_history_ids == 0
        assert not seq.prefix_cache.restore.is_selected
        assert scheduler.block_manager.get_num_free_gpu_blocks() == free_blocks
        assert scheduler.take_multimodal_requests() == [seq]
        assert scheduler.take_multimodal_requests() == []
        # Media I/O must not occupy a model-batch slot or block another waiter.
        other = scheduler.add_session(2).add_sequence([99])
        assert scheduler.schedule(is_prefill=True).running == [other]
    scheduler.shutdown()


def test_pd_replay_rechecks_media_after_checkpoint_pin_failure(monkeypatch):
    scheduler, seq = make_replay(hybrid=True)
    monkeypatch.setattr(scheduler.state_checkpoints, 'pin_restore', Mock(return_value=False))
    assert scheduler.schedule(is_prefill=True).running == []
    assert seq.status == MessageStatus.WAITING_FOR_MULTIMODALS
    assert seq.num_blocks == seq.num_history_ids == 0
    assert not seq.prefix_cache.restore.is_selected
    assert scheduler.take_multimodal_requests() == [seq]
    scheduler.shutdown()


def test_pd_replay_can_request_media_after_an_earlier_cache_hit():
    scheduler, seq = make_replay(hybrid=True)
    assert scheduler.schedule(is_prefill=True).running == [seq]
    assert scheduler.take_multimodal_requests() == []
    scheduler.activate_seqs([seq])
    scheduler.preempt_seqs([seq])
    seq.state.release_paging_resources()
    scheduler.state_checkpoints.evict(64)
    scheduler.block_trie.evict(64)
    assert scheduler.schedule(is_prefill=True).running == []
    assert seq.pd_metadata_only and seq.status == MessageStatus.WAITING_FOR_MULTIMODALS
    assert scheduler.take_multimodal_requests() == [seq]
    scheduler.shutdown()


def test_cancelled_pd_media_request_is_not_sent_to_the_frontend():
    scheduler, seq = make_replay(same_image=False)
    assert scheduler.schedule(is_prefill=True).running == []
    scheduler.end_session(seq.session_id)
    assert scheduler.take_multimodal_requests() == []
    scheduler.shutdown()


@pytest.mark.parametrize('same_image', [False, True])
def test_decode_rejection_cleanup_preserves_same_turn_pd_replay_admission(same_image):
    from lmdeploy.pytorch.engine.inputs_maker import InputsMakerAsync

    scheduler, seq = make_replay(same_image=same_image)
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.scheduler = scheduler
    maker.to_evict_seqs = [seq]
    scheduler.schedule(is_prefill=True)
    expected = MessageStatus.READY if same_image else MessageStatus.WAITING_FOR_MULTIMODALS
    assert seq.status == expected
    maker.preempt_invalid_decode_seqs()
    assert seq.status == expected
    assert maker.to_evict_seqs == []
    assert scheduler.take_multimodal_requests() == ([] if same_image else [seq])
    scheduler.shutdown()
