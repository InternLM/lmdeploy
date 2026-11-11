# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.engine.engine import Engine
from lmdeploy.pytorch.engine.inputs_maker import InputsMakerAsync
from lmdeploy.pytorch.messages import MessageStatus, SamplingParam, SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy


@pytest.mark.parametrize('model_paradigm,stop_pos', [('ar', 0), *[('ar_spec', pos) for pos in range(4)]])
@pytest.mark.parametrize('enable_replayssm', [False, True])
@pytest.mark.parametrize('prefix', [False, True])
def test_mtp_resume_recomputes_replay_state(model_paradigm, enable_replayssm, stop_pos, prefix):
    cache = CacheConfig(max_batches=1, block_size=16, num_cpu_blocks=0,
                        num_gpu_blocks=64, num_state_caches=3,
                        enable_prefix_caching=prefix,
                        enable_replayssm=enable_replayssm,
                        states_shapes=[((1,), torch.float32)])
    strategy = ARSpecSequenceStrategy() if model_paradigm == 'ar_spec' else ARSequenceStrategy()
    scheduler = Scheduler(SchedulerConfig(max_batches=1, max_session_len=1024), cache,
                          SequenceMeta(16, strategy=strategy, use_mrope=True))
    seq = scheduler.add_session(0).add_sequence(torch.arange(16))
    scheduler.schedule(is_prefill=True)
    seq.activate()
    seq.update_token_ids(torch.tensor([17]), draft_token_ids=torch.tensor([18, 19, 20]),
                         mode=UpdateTokenMode.PREFILL)
    slot = seq.logical_state
    seq.update_token_ids(torch.tensor([18, 19, 20, 21] if model_paradigm == 'ar_spec' else [18]),
                         draft_token_ids=torch.tensor([22, 23, 24]),
                         mode=UpdateTokenMode.DECODE, stop_pos=stop_pos)
    seq.finish()
    old_history = seq.num_history_ids
    retained = seq.valid_ids.copy()
    # Finishing alone must not reclaim a slot still referenced by queued work.
    assert seq.status == MessageStatus.STOPPED and seq.logical_state == slot

    engine = Engine.__new__(Engine)
    engine.scheduler = scheduler
    engine.executor = SimpleNamespace(model_config=SimpleNamespace(
        enable_replayssm=enable_replayssm, model_paradigm=model_paradigm))
    engine.engine_config = SimpleNamespace(role=EngineRole.Hybrid)
    engine.max_session_len = 1024
    follow = [30, 31]
    multimodals = None
    if stop_pos == 0:
        follow = [30, 31, 31, 31, 31, 32]
        multimodals = {'image': [MultiModalData(
            data=torch.zeros(4, 2), start=1, end=5,
            mrope_pos_ids=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]))]}
    engine._add_message([SimpleNamespace(data=dict(session_id=0, token_ids=follow,
                                                  input_multimodals=multimodals,
                                                  sampling_param=SamplingParam(max_new_tokens=8)), resp=None)])

    assert seq.status == MessageStatus.WAITING
    assert seq.valid_ids.tolist() == [*retained.tolist(), *follow]
    assert seq.num_new_tokens == 0
    recompute = enable_replayssm and model_paradigm == 'ar_spec'
    assert seq.num_history_ids == old_history and seq.logical_state == slot
    assert seq.replayssm_recompute_pending == recompute
    scheduler.schedule(is_prefill=True)
    assert seq.logical_state >= 0
    assert seq.num_history_ids == (0 if recompute else old_history)
    assert not seq.replayssm_recompute_pending
    assert len(seq.mrope_pos_ids) == seq.num_token_ids
    fresh = scheduler.add_session(1).add_sequence(
        torch.from_numpy(seq.valid_ids.copy()), multimodals=deepcopy(seq.history_multimodals.multimodals))
    np.testing.assert_array_equal(seq.all_ids, fresh.valid_ids)
    np.testing.assert_array_equal(seq.history_mrope_pos_ids.get_real(), fresh.history_mrope_pos_ids.get_real())
    np.testing.assert_array_equal(seq.mrope_pos_ids, fresh.mrope_pos_ids[seq.num_history_ids:])
    # Exercise the actual chunked-prefill consumer, including the MTP lookahead.
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = SimpleNamespace(is_ssm=True, use_mrope=True)
    maker.scheduler = scheduler
    maker.executor = SimpleNamespace(device_type='cuda')
    maker.adapter_manager = SimpleNamespace(num_adapters=lambda: 1)
    maker.kernel_blocks_per_kv = 1
    inputs = maker.create_model_inputs_long_context(seq, 1).to_device('cpu')
    np.testing.assert_array_equal(inputs.input_ids[0], fresh.valid_ids[seq.num_history_ids:seq.num_history_ids + 1])
    np.testing.assert_array_equal(inputs.mrope_pos_ids.T,
                                  fresh.mrope_pos_ids[seq.num_history_ids:seq.num_history_ids + 1])
    if model_paradigm == 'ar_spec':
        assert inputs.prefill_next_token_ids.tolist() == [seq.token_ids[1]]
