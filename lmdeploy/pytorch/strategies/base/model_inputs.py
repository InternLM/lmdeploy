# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch.profiler import record_function

from lmdeploy.pytorch.config import ModelConfig
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta


@dataclass
class MakeDummyMeta:
    """Make dummy meta for model inputs strategy."""
    # Add any fields needed for making dummy inputs
    use_ssm: bool = False
    use_mrope: bool = False


@record_function('make_dummy_input')
def make_dummy_inputs(batch_size: int,
                      max_q_seqlen: int,
                      is_decoding: bool,
                      device: str = 'cpu',
                      dummy_block_id: int = 0,
                      vocab_size: int = 1,
                      meta: MakeDummyMeta | None = None) -> ModelInputs:
    """Make dummy inputs global implement."""
    if meta is None:
        meta = MakeDummyMeta()
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

    state_offsets = None
    if meta.use_ssm:
        state_offsets = torch.full((batch_size, ), -1, dtype=torch.long, device=device)

    mrope_pos_ids = None
    if meta.use_mrope:
        mrope_pos_ids = torch.zeros(3, num_tokens, dtype=torch.long, device=device)

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
        is_dummy=True,
        state_offsets=state_offsets,
        mrope_pos_ids=mrope_pos_ids,
    )


class ModelInputsStrategy(ABC):

    def create_make_dummy_meta(self, model_config: ModelConfig):
        """Create make dummy meta."""
        return MakeDummyMeta(
            use_ssm=len(model_config.states_shapes) > 0,
            use_mrope=model_config.use_mrope,
        )

    @abstractmethod
    def make_dummy(self,
                   batch_size: int,
                   is_decoding: bool,
                   device: str = 'cpu',
                   dummy_block_id: int = 0,
                   vocab_size: int = 1,
                   meta: MakeDummyMeta | None = None) -> ModelInputs:
        """Create dummy model inputs."""
        pass

    @abstractmethod
    def merge(self, inputs: ModelInputs, other: ModelInputs) -> ModelInputs:
        """Merge model inputs."""
        pass

    @abstractmethod
    def update_inputs(self, inputs: ModelInputs, delta: 'ModelInputsDelta') -> ModelInputs:
        """Update model inputs with delta."""
        pass
