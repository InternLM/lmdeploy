# Copyright (c) OpenMMLab. All rights reserved.

import torch

from ...model_inputs import ModelInputs
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer


@SPEC_PROPOSERS.register_module(name='deepseek_mtp')
class DeepseekMTP(BaseSpecProposer):

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None, build_model_ctx=None):
        """Build the draft model and bind the target embedding."""
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)
        draft_model = self.model
        if (getattr(draft_model, 'uses_dsa_topk_buffer', False)
                and hasattr(draft_model, 'set_input_embeddings')):
            draft_model.set_input_embeddings(target_model.get_input_embeddings())

    async def get_outputs(self,
                    model_outputs: dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ARSpecExtraInputs = None,
                    guided_processors: dict | None = None):
        """Get outputs."""
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs['model_metas']
        draft_model = self.model.get_model() if hasattr(self.model, 'get_model') else self.model
        uses_topk_buffer = getattr(draft_model, 'uses_dsa_topk_buffer', False)

        if extra_inputs is not None:
            last_token_loc = extra_inputs.last_token_indices
            hidden_states = hidden_states[:, last_token_loc]
            if uses_topk_buffer:
                draft_model.compact_topk_indices(last_token_loc)

        if hasattr(draft_model, 'prepare_hidden_states_for_logits'):
            hidden_states = draft_model.prepare_hidden_states_for_logits(hidden_states)
            logits = self.target_model.get_logits(hidden_states)
        else:
            logits = self.get_logits(hidden_states)
        target_hidden_states = hidden_states

        if uses_topk_buffer:
            model_metas = model_metas or [None] * hidden_states.size(1)
            model_metas = [dict(meta or {}, skip_topk=True) for meta in model_metas]
        logits = logits[0]

        guided_bitmask = await self.guided_helper.prepare_bitmask(logits, guided_processors)
        if guided_bitmask is not None:
            self.guided_helper.apply_bitmask(logits, guided_bitmask)

        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        await self.guided_helper.accept_draft_tokens(draft_token_ids, guided_processors)

        return draft_token_ids, model_metas, target_hidden_states
