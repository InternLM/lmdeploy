# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.model_inputs import ModelInputs


@record_function('make_dummy_input')
def make_dummy_inputs(batch_size: int,
                      max_q_seqlen: int,
                      is_decoding: bool,
                      device: str = 'cpu',
                      dummy_block_id: int = 0,
                      vocab_size: int = 1):
    """Make dummy inputs global implement."""
    num_tokens = batch_size * max_q_seqlen
    max_kv_seqlen = max_q_seqlen
    input_ids = torch.randint(0, vocab_size, (
        1,
        num_tokens,
    ), dtype=torch.long, device=device)
    seq_length = torch.full((batch_size, ), max_q_seqlen, dtype=torch.long, device=device)
    history_lengths = torch.zeros((batch_size, ), dtype=torch.long, device=device)
    block_offsets = torch.full((batch_size, 1), dummy_block_id, dtype=torch.long, device=device)
    num_ignored_history = torch.zeros((batch_size, ), dtype=torch.long, device=device)
    local_adapter_ids = torch.zeros((batch_size, ), dtype=torch.long, device=device)
    state_offsets = torch.full((batch_size, ), -1, dtype=torch.long, device=device)

    return ModelInputs(
        input_ids=input_ids,
        seq_length=seq_length,
        history_lengths=history_lengths,
        block_offsets=block_offsets,
        is_decoding=is_decoding,
        num_ignored_history=num_ignored_history,
        max_q_seqlen=max_q_seqlen,
        max_kv_seqlen=max_kv_seqlen,
        sum_kv_seqlen=num_tokens,
        local_adapter_ids=local_adapter_ids,
        state_offsets=state_offsets,
    )


class ModelInputsStrategy(ABC):

    @abstractmethod
    def make_dummy(self,
                   batch_size: int,
                   is_decoding: bool,
                   device: str = 'cpu',
                   dummy_block_id: int = 0,
                   vocab_size: int = 1) -> ModelInputs:
        """Create dummy model inputs."""
        pass
