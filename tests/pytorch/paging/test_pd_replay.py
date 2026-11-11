# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy


@pytest.mark.parametrize('length', [1, 64, 65, 66])
@pytest.mark.parametrize('mtp', [False, True])
def test_prefill_stops_before_tail_and_preserves_cache_identity(length, mtp):
    strategy = ARSpecSequenceStrategy() if mtp else ARSequenceStrategy()
    cache = CacheConfig(max_batches=1, block_size=64, num_gpu_blocks=8, num_cpu_blocks=0,
                        enable_prefix_caching=True, prefix_cache_token_lookahead=int(mtp))
    scheduler = Scheduler(SchedulerConfig(max_batches=1, max_session_len=256), cache,
                          SequenceMeta(64, strategy=strategy))
    seq = scheduler.add_session(1).add_sequence(np.arange(length))
    seq.prepare_pd_prefill()
    assert seq.num_valid_ids == seq.num_token_ids == seq.pd_prefix_length == length - 1
    assert seq.input_end_pos == length - 1
    assert seq.cache_identity_ids.tolist() == list(range(length))
    assert seq.num_new_tokens == 0
    if mtp:
        assert seq.get_prefix_cache_token_dependency(length - 1)[0].token_ids == (length - 1,)
