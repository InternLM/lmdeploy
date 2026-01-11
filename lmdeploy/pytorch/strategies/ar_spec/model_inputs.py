# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.profiler import record_function

from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta

from ..ar.model_inputs import merge_model_inputs
from ..base.model_inputs import ModelInputsStrategy, make_dummy_inputs


class ARSpecModelInputsStrategy(ModelInputsStrategy):

    def __init__(self, num_spec_tokens: int):
        self.num_spec_tokens = num_spec_tokens

    def make_dummy(
        self,
        batch_size: int,
        is_decoding: bool,
        device: str = 'cpu',
        dummy_block_id: int = 0,
        vocab_size: int = 1,
        max_q_seqlen: int = 1,
        target_hidden_size: int = None,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> ModelInputs:
        """Create dummy model inputs."""
        inputs = make_dummy_inputs(batch_size,
                                   max_q_seqlen=max_q_seqlen,
                                   is_decoding=is_decoding,
                                   device=device,
                                   dummy_block_id=dummy_block_id,
                                   vocab_size=vocab_size)
        if target_hidden_size is not None:
            inputs.target_hidden_states = torch.randn((1, batch_size * max_q_seqlen, target_hidden_size),
                                                      dtype=target_dtype,
                                                      device=device)
        return inputs

    @record_function('ModelInputs.merge')
    def merge(self, inputs: ModelInputs, other: ModelInputs) -> ModelInputs:
        """Merge model inputs."""
        return merge_model_inputs(inputs, other)

    @record_function('ModelInputs.update_inputs')
    def update_inputs(self, inputs: ModelInputs, delta: 'ModelInputsDelta') -> ModelInputs:
        """Update model inputs with delta."""
        assert inputs.is_decoding, 'Only support update_delta in decoding.'
        indices = delta.indices
        indice_cpu = delta.indice_cpu
        block_offsets = delta.block_offsets
        max_q_seqlen = delta.max_q_seqlen
        max_kv_seqlen = delta.max_kv_seqlen
        sum_kv_seqlen = delta.sum_kv_seqlen
        num_ignored_history = delta.num_ignored_history

        # required inputs
        # input_ids = inputs.input_ids[..., indices]
        inputs_ids = inputs.input_ids.reshape(1, -1, self.num_spec_tokens + 1)
        input_ids = inputs_ids[:, indices].reshape(1, -1)
        seq_length = inputs.seq_length[indices]
        history_lengths = inputs.history_lengths[indices]
        if block_offsets is None:
            block_offsets = inputs.block_offsets[indices]
        if num_ignored_history is None:
            num_ignored_history = inputs.num_ignored_history[indices]
        max_q_seqlen = max_q_seqlen or inputs.max_q_seqlen
        max_kv_seqlen = max_kv_seqlen or inputs.max_kv_seqlen
        sum_kv_seqlen = sum_kv_seqlen or inputs.sum_kv_seqlen

        # lora adapter ids
        local_adapter_ids = inputs.local_adapter_ids
        if local_adapter_ids is not None:
            local_adapter_ids = local_adapter_ids[indices]

        # model metas for vl models
        model_metas = inputs.model_metas
        if model_metas is not None and indice_cpu is not None:
            model_metas = [model_metas[i] for i in indice_cpu]

        # for ssm
        state_offsets = inputs.state_offsets
        if state_offsets is not None:
            state_offsets = state_offsets[indices]

        # return new inputs
        return ModelInputs(
            input_ids=input_ids,
            seq_length=seq_length,
            history_lengths=history_lengths,
            block_offsets=block_offsets,
            is_decoding=inputs.is_decoding,
            num_ignored_history=num_ignored_history,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=max_kv_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            local_adapter_ids=local_adapter_ids,
            model_metas=model_metas,
            state_offsets=state_offsets,
        )
