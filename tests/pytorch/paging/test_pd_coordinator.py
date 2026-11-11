# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.disagg.conn.protocol import PDControlRequest
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy


def make_export():
    cache = CacheConfig(max_batches=2, block_size=4, num_gpu_blocks=4, num_cpu_blocks=0)
    scheduler = Scheduler(SchedulerConfig(max_batches=2, max_session_len=32), cache,
                          SequenceMeta(4, strategy=ARSequenceStrategy()))
    seq = scheduler.add_session(1).add_sequence(np.arange(8), preserve_cache=True)
    seq.pd_prompt_tokens = np.arange(9)
    scheduler.block_manager.allocate(seq)
    handoff = scheduler.pd_coordinator.publish(seq, 'schema', [])
    return scheduler, seq, handoff


def test_granted_export_survives_real_session_end_and_rejects_wrong_receiver():
    scheduler, seq, handoff = make_export()
    owner = scheduler.pd_coordinator
    request = dict(transfer_id=handoff.transfer_id, engine_epoch=handoff.engine_epoch, receiver_epoch='decode')
    owner.control(PDControlRequest(action='claim', **request))
    free = scheduler.block_manager.get_num_free_gpu_blocks()
    scheduler.end_session(seq.session_id)
    assert scheduler.get_session(1) is not None
    assert scheduler.block_manager.get_num_free_gpu_blocks() == free
    with pytest.raises(ValueError, match='granted receiver'):
        owner.control(PDControlRequest(action='release', **(request | dict(receiver_epoch='other'))))
    assert owner.control(PDControlRequest(action='release', **request)) is seq
    scheduler.end_session(1)
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4
    assert owner.control(PDControlRequest(action='release', **request)) is None


def test_revoke_before_grant_prevents_late_reader():
    scheduler, seq, handoff = make_export()
    scheduler.end_session(seq.session_id)
    assert scheduler.block_manager.get_num_free_gpu_blocks() == 4
    with pytest.raises(ValueError, match='no longer available'):
        scheduler.pd_coordinator.control(PDControlRequest(
            action='claim', transfer_id=handoff.transfer_id, engine_epoch=handoff.engine_epoch,
            receiver_epoch='late'))
