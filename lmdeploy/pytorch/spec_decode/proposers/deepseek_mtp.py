# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.utils import get_logger

from ...model_inputs import ModelInputs
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='deepseek_mtp')
class DeepseekMTP(BaseSpecProposer):

    @staticmethod
    def get_spec_step_idx(model_inputs: ModelInputs) -> int:
        """Read the current draft step from model_metas."""
        model_metas = model_inputs.model_metas
        if not model_metas:
            return 0
        model_meta = model_metas[0]
        if not isinstance(model_meta, dict):
            return 0
        return int(model_meta.get('spec_step_idx', 0))

    @staticmethod
    def with_spec_step_idx(
        model_metas: list[dict] | None,
        batch_size: int,
        spec_step_idx: int,
    ):
        """Attach spec_step_idx to every batch meta entry."""
        if model_metas is None:
            model_metas = [None] * batch_size

        updated = []
        for batch_idx in range(batch_size):
            model_meta = model_metas[batch_idx] if batch_idx < len(model_metas) else None
            if model_meta is None:
                model_meta = {}
            else:
                model_meta = dict(model_meta)
            model_meta['spec_step_idx'] = spec_step_idx
            updated.append(model_meta)
        return updated

    def update_inputs_decoding(
        self,
        model_inputs: ModelInputs,
        extra_inputs: ARSpecExtraInputs,
        next_input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        model_metas: list[dict],
    ):
        """Update decoding inputs with deepseek-style spec step metadata."""
        new_inputs = super().update_inputs_decoding(
            model_inputs,
            extra_inputs,
            next_input_ids,
            target_hidden_states,
            model_metas,
        )
        return new_inputs.clone(
            model_metas=self.with_spec_step_idx(
                model_metas,
                new_inputs.seq_length.size(0),
                0,
            )
        )

    def get_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
        """Get logits of deepseek/qwen-style MTP draft models."""
        draft_model = self.model
        if not isinstance(draft_model, torch.nn.Module):
            draft_model = draft_model.model

        if hasattr(draft_model, 'get_logits'):
            try:
                logits = draft_model.get_logits(hidden_states, spec_step_idx=spec_step_idx)
            except TypeError:
                logits = draft_model.get_logits(hidden_states)
        else:
            logits = self.target_model.get_logits(hidden_states)
        return logits

    def get_outputs(self,
                    model_outputs: dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ARSpecExtraInputs = None):
        """Get outputs."""
        hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs['model_metas']
        if model_metas is None:
            model_metas = model_inputs.model_metas
        spec_step_idx = self.get_spec_step_idx(model_inputs)
        if extra_inputs is not None:
            last_token_loc = extra_inputs.last_token_indices
            hidden_states = hidden_states[:, last_token_loc]
            # use hidden states for draft prefill forward for next step
            target_hidden_states = hidden_states
        else:
            target_hidden_states = hidden_states

        logits = self.get_logits(hidden_states, spec_step_idx=spec_step_idx)[0]
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        return draft_token_ids, model_metas, target_hidden_states
