# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.model_inputs import ModelInputs

from ..base.model_inputs import MakeDummyMeta, ModelInputsStrategy, make_dummy_inputs
from .model_agent import ARSpecExtraInputs


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
        meta: MakeDummyMeta | None = None,
    ) -> ModelInputs:
        """Create dummy model inputs."""
        is_draft_model = target_hidden_size is not None

        # warmup decoding for main model
        if not is_draft_model and is_decoding and max_q_seqlen == 1:
            max_q_seqlen = self.num_spec_tokens + 1

        inputs = make_dummy_inputs(batch_size,
                                   max_q_seqlen=max_q_seqlen,
                                   is_decoding=is_decoding,
                                   device=device,
                                   dummy_block_id=dummy_block_id,
                                   vocab_size=vocab_size,
                                   meta=meta)
        if is_draft_model:
            inputs.target_hidden_states = torch.randn((1, batch_size * max_q_seqlen, target_hidden_size),
                                                      dtype=target_dtype,
                                                      device=device)
            inputs.target_position_ids = torch.zeros_like(inputs.input_ids, dtype=torch.long, device=device)

        return inputs


    def make_dummy_extra_inputs(self,
                   inputs: ModelInputs,
                   meta: MakeDummyMeta | None = None) -> ARSpecExtraInputs:
        """Create dummy model inputs."""
        extra_inputs = ARSpecExtraInputs()
        if inputs.is_decoding:
            batch_size = inputs.seq_length.size(0)
            extra_inputs.output_draft_token_ids = inputs.input_ids.new_zeros((batch_size, self.num_spec_tokens))
        return extra_inputs
