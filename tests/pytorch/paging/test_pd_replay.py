# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy


@pytest.mark.parametrize('length', [1, 64, 65, 66])
@pytest.mark.parametrize('mtp', [False, True])
def test_prefill_keeps_complete_prompt_and_cache_identity(length, mtp):
    strategy = ARSpecSequenceStrategy() if mtp else ARSequenceStrategy()
    cache = CacheConfig(max_batches=1, block_size=64, num_gpu_blocks=8, num_cpu_blocks=0,
                        enable_prefix_caching=True, prefix_cache_token_lookahead=int(mtp))
    scheduler = Scheduler(SchedulerConfig(max_batches=1, max_session_len=256), cache,
                          SequenceMeta(64, strategy=strategy))
    seq = scheduler.add_session(1).add_sequence(np.arange(length))
    seq.prepare_pd_prefill()
    assert seq.num_valid_ids == seq.num_token_ids == length
    assert seq.input_end_pos == length
    assert seq.valid_ids.tolist() == list(range(length))
    assert seq.num_new_tokens == 0
    if mtp:
        assert seq.get_prefix_cache_token_dependency(length - 1)[0].token_ids == (length - 1,)


def test_full_prefill_never_splits_a_trailing_image():
    from lmdeploy.pytorch.long_context import plan_long_context_chunk

    cache = CacheConfig(max_batches=1, block_size=16, num_gpu_blocks=16, num_cpu_blocks=0)
    scheduler = Scheduler(SchedulerConfig(max_batches=1, max_session_len=256), cache,
                          SequenceMeta(16, strategy=ARSpecSequenceStrategy()))
    image = MultiModalData(np.ones((32, 4)), 32, 64)
    seq = scheduler.add_session(1).add_sequence(np.arange(64), multimodals={'image': [image]})
    seq.prepare_pd_prefill()
    assert len(seq.pd_prompt_tokens) == seq.input_end_pos == 64
    first = plan_long_context_chunk(seq, 40)
    assert first.chunk_end <= 31
    seq.set_step(first.chunk_end)
    last = plan_long_context_chunk(seq, 40)
    assert last.chunk_end == 64 and last.is_last_chunk
    assert seq.history_multimodals.get_datas(seq.num_history_ids, 64)['image'] == [image]
