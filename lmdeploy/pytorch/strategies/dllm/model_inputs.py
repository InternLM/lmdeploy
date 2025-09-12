# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.model_inputs import ModelInputs

from ..base.model_inputs import ModelInputsStrategy, make_dummy_inputs


class DLLMModelInputsStrategy(ModelInputsStrategy):

    def __init__(self, block_size: int):
        self.block_size = block_size

    def make_dummy(self,
                   batch_size: int,
                   is_decoding: bool,
                   device: str = 'cpu',
                   dummy_block_id: int = 0,
                   vocab_size: int = 1) -> ModelInputs:
        """Create dummy model inputs."""
        return make_dummy_inputs(batch_size,
                                 max_q_seqlen=self.block_size,
                                 is_decoding=is_decoding,
                                 device=device,
                                 dummy_block_id=dummy_block_id,
                                 vocab_size=vocab_size)
