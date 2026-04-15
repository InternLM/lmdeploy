# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.model_inputs import ModelInputs

from ..base.model_inputs import MakeDummyMeta, ModelInputsStrategy, make_dummy_inputs


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
