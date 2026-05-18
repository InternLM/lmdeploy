# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.utils import get_logger

from ...model_inputs import ModelInputs
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='deepseek_mtp')
class DeepseekMTP(BaseSpecProposer):

    async def get_outputs(self,
                    model_outputs: dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ARSpecExtraInputs = None,
                    guided_processors: dict | None = None):
        """Get outputs."""
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs['model_metas']
        if extra_inputs is not None:
            last_token_loc = extra_inputs.last_token_indices
            hidden_states = hidden_states[:, last_token_loc]
            # use hidden states for draft prefill forward for next step
            target_hidden_states = hidden_states
        else:
            target_hidden_states = hidden_states

        logits = self.get_logits(hidden_states)[0]

        guided_bitmask = await self._apply_guided_bitmask(logits, guided_processors)
        if guided_bitmask is not None:
            self.guided_decoding_manager.apply_batched_bitmap(logits, guided_bitmask)

        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        await self._accept_guided_tokens(draft_token_ids, guided_processors)

        return draft_token_ids, model_metas, target_hidden_states
