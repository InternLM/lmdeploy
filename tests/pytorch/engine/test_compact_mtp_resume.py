# Copyright (c) OpenMMLab. All rights reserved.
"""Truncated MTP output must not resume from the worker's later compact
state."""
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig, SSMStateCapacity
from lmdeploy.pytorch.engine.model_agent.agent import BatchedOutputs
from lmdeploy.pytorch.messages import MessageStatus, SequenceMeta, UpdateTokenMode
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.strategies.ar_spec.sequence import ARSpecSequenceStrategy


@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('input_kind', ['full', 'delta', 'elided_delta'])
@pytest.mark.parametrize('accepted,stop_pos', [(4, 0), (4, 3), (2, 0), (2, 1)])
def test_truncated_mtp_recomputes_on_session_resume(compact, input_kind, accepted, stop_pos):
    strategy = ARSpecSequenceStrategy()
    cache = CacheConfig(max_batches=1, block_size=16, num_cpu_blocks=0, num_gpu_blocks=64,
                        states_shapes=[((2,), torch.float32)], num_state_caches=2)
    if compact:
        cache.compact_state_capacity = SSMStateCapacity(2, 4, 0)
    scheduler = Scheduler(SchedulerConfig(max_batches=1, max_session_len=4096), cache,
                          SequenceMeta(16, strategy=strategy, prefix_cache_token_lookahead=1))
    seq = scheduler.add_session(0).add_sequence(list(range(9)))
    assert scheduler.schedule(is_prefill=True).running == [seq]
    seq.state.activate()
    seq.set_step(8)
    tokens = torch.full((1, 4), -1)
    tokens[0, :accepted] = torch.arange(20, 20 + accepted)
    outputs = BatchedOutputs(next_token_ids=tokens, stopped=torch.tensor([True]),
                              ssm_state_access_id=1 if compact else None,
                              stop_pos=torch.tensor([stop_pos]), model_metas=None, all_routed_experts=None,
                              extra_outputs=SimpleNamespace(draft_token_ids=torch.tensor([[30, 31, 32]])))
    inputs = SimpleNamespace(is_decoding=True, seq_length=torch.tensor([4]),
                             ssm_state_meta=object() if compact else None)
    delta = SimpleNamespace(is_decoding=True, seq_length=torch.tensor([4]),
                            ssm_state_ids=object() if compact and input_kind != 'elided_delta' else None)
    strategy.update_running([seq], outputs, inputs if input_kind == 'full' else None,
                            delta if input_kind != 'full' else None)
    retained = min(accepted, stop_pos + 1) if stop_pos >= 0 else accepted
    assert seq.num_history_ids == 8 + retained
    assert seq.status == MessageStatus.STOPPED
    assert seq.generated_ids.tolist() == list(range(20, 20 + retained))
    # Finishing alone must not discard the current response or paging resources.
    assert seq.logical_state >= 0 and seq.num_blocks > 0
    seq.update_token_ids([40, 41], mode=UpdateTokenMode.INPUTS)
    recompute = compact and retained < accepted
    assert seq.num_history_ids == (0 if recompute else 8 + retained)
    if recompute:
        assert seq.logical_state == -1 and seq.num_blocks == 0
    assert seq.history_cache[:seq.num_valid_ids].tolist() == list(range(9)) + list(range(20, 20 + retained)) + [40, 41]
    assert not seq._recompute_compact_on_resume
    seq.state.activate()
    assert scheduler.schedule(is_prefill=True).running == [seq]
    if recompute:
        assert seq.num_history_ids == 0  # Worker bind will select unit0, not its stale cursor.
    # A later untruncated turn must not inherit the recompute marker.
    seq.state.activate()
    seq.set_step(seq.num_valid_ids - 1)
    outputs.next_token_ids = torch.tensor([[50, 51, 52, 53]])
    outputs.stop_pos = torch.tensor([3])
    strategy.update_running([seq], outputs, inputs if input_kind == 'full' else None,
                            delta if input_kind != 'full' else None)
    history = seq.num_history_ids
    seq.update_token_ids([60], mode=UpdateTokenMode.INPUTS)
    assert seq.num_history_ids == history and not seq._recompute_compact_on_resume
    scheduler.end_session(0)
    scheduler.shutdown()
