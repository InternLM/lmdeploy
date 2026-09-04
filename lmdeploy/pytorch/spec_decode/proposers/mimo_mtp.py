# Copyright (c) OpenMMLab. All rights reserved.

import torch

from ...model_inputs import ModelInputs
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from .base import SPEC_PROPOSERS, BaseSpecProposer


@SPEC_PROPOSERS.register_module(name='mimo_mtp')
class MiMoMTP(BaseSpecProposer):
    """MiMo same-position, multi-depth MTP proposer."""

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None, build_model_ctx=None):
        """Build the MiMo draft and bind its target-owned TP embedding."""
        if target_model is None:
            raise ValueError('MiMo MTP requires the target model for its shared token embedding.')
        # MiMo always shares the target vocab-parallel embedding; unlike the
        # DeepSeek DSA optimization this is part of the model architecture.
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)
        draft_model = self.model
        if not hasattr(draft_model, 'set_input_embeddings'):
            raise TypeError(f'{type(draft_model).__name__} cannot bind MiMo shared token embeddings.')
        draft_model.set_input_embeddings(target_model.get_input_embeddings())

    def advance_draft_depth(self,
                            model_inputs: ModelInputs,
                            extra_inputs: ARSpecExtraInputs | None,
                            next_input_ids: torch.Tensor,
                            target_hidden_states: torch.Tensor,
                            model_metas: list,
                            *,
                            first_depth: bool) -> tuple[ModelInputs, ARSpecExtraInputs]:
        """Advance one same-position MiMo depth without advancing time."""
        # Every depth conditions on the same target-model state; only the
        # candidate token/embedding stream and prediction depth change.
        del target_hidden_states, first_depth
        if extra_inputs is None:
            raise ValueError('MiMo MTP depth transitions require AR speculative inputs.')
        last_token_indices = extra_inputs.last_token_indices
        input_ids = model_inputs.input_ids.clone()
        input_ids[:, :-1] = model_inputs.input_ids[:, 1:]
        input_ids[:, last_token_indices] = next_input_ids.flatten()

        target_inputs_embeds = model_inputs.target_inputs_embeds
        if target_inputs_embeds is not None:
            target_inputs_embeds = target_inputs_embeds.clone()
            target_inputs_embeds[:, :-1] = model_inputs.target_inputs_embeds[:, 1:]
            target_inputs_embeds[:, last_token_indices] = self.embed_input_ids(next_input_ids.flatten())

        model_inputs = model_inputs.clone(
            input_ids=input_ids,
            target_inputs_embeds=target_inputs_embeds,
            target_hidden_states=model_inputs.target_hidden_states,
            model_metas=model_metas,
            spec_step_idx=model_inputs.spec_step_idx + 1,
        )
        return model_inputs, extra_inputs

    def get_draft_depth_token_counts(self, dp_meta) -> list[int]:
        """Preserve varlen prompt sizes across same-position MTP depths."""
        num_tokens = dp_meta.dp_draft_num_tokens
        if num_tokens is None:
            raise RuntimeError('MiMo MTP requires per-rank draft token counts for DP depth transitions.')
        return num_tokens

    async def get_outputs(self,
                          model_outputs: dict[str, torch.Tensor],
                          model_inputs: ModelInputs,
                          extra_inputs: ARSpecExtraInputs = None,
                          guided_processors: dict | None = None):
        """Select the active MiMo depth's norm/head and produce its draft."""
        raw_hidden_states = model_outputs['hidden_states']
        model_metas = model_outputs['model_metas']
        if extra_inputs is not None:
            raw_hidden_states = raw_hidden_states[:, extra_inputs.last_token_indices]

        draft_model = self.model.get_model() if hasattr(self.model, 'get_model') else self.model
        logits_hidden_states = draft_model.prepare_hidden_states_for_logits(
            raw_hidden_states, spec_step_idx=model_inputs.spec_step_idx)
        logits = self.target_model.get_logits(logits_hidden_states)[0]

        guided_bitmask = await self.guided_helper.prepare_bitmask(logits, guided_processors)
        if guided_bitmask is not None:
            self.guided_helper.apply_bitmask(logits, guided_bitmask)
        draft_token_ids = logits.argmax(dim=-1, keepdim=True)

        await self.guided_helper.accept_draft_tokens(draft_token_ids, guided_processors)
        return draft_token_ids, model_metas, raw_hidden_states
