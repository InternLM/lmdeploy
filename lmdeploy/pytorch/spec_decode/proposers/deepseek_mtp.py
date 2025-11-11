# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch

from lmdeploy.utils import get_logger

from ...model_inputs import ModelInputs
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='deepseek_mtp')
class DeepseekMTP(BaseSpecProposer):

    def get_outputs(self,
                    model_outputs: Dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ARSpecExtraInputs = None):
        """Get outputs."""
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs['model_metas']
        if extra_inputs is not None and extra_inputs.last_token_indices is not None:
            # for long input
            if (not model_inputs.is_decoding) and model_inputs.seq_length.size(0) == 1:
                hidden_states = hidden_states[:, -1:]
            else:
                last_token_loc = extra_inputs.last_token_indices
                hidden_states = hidden_states[:, last_token_loc]

        logits = self.get_logits(hidden_states)[0]
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        return draft_token_ids, model_metas, hidden_states
