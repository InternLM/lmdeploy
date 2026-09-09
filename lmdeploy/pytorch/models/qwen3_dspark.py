# Copyright (c) OpenMMLab. All rights reserved.
"""Qwen/DFlash-backbone DSpark draft model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from lmdeploy.pytorch.nn.embedding import ParallelLMHead
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .dspark_heads import build_dspark_head, compute_dspark_proposal_ids
from .qwen3_dflash import DFlashDraftModel, _normalize_dflash_weight_name


class Qwen3DSparkModel(DFlashDraftModel):
    """DFlash Qwen draft backbone plus a sequential DSpark head."""

    def __init__(self, config, ctx_mgr, dtype=None, device=None, prefix: str = ''):
        super().__init__(config, ctx_mgr, dtype=dtype, device=device, prefix=prefix)
        self.draft_vocab_size = int(getattr(config, 'draft_vocab_size', None)
                                    or config.vocab_size)
        self.dspark_sample_from_anchor = bool(
            getattr(config, 'dspark_sample_from_anchor'))
        self.dspark_draft_query_len = int(
            getattr(config, 'dspark_draft_query_len'))
        self.dspark_num_speculative_tokens = int(
            getattr(config, 'dspark_num_speculative_tokens'))
        self.markov_head = build_dspark_head(config, dtype=dtype, device=device)
        # External Speculators checkpoints may either ship an lm_head or omit
        # it and rely on the target head.  Always construct the checkpoint
        # shape so the loader can accept either form, then replace a missing
        # full-vocabulary head with the target module in ``set_output_head``.
        self.output_head: nn.Module | None = ParallelLMHead(
            self.draft_vocab_size,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False)
        self.has_own_output_head = False
        if self.draft_vocab_size != int(config.vocab_size):
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.draft_vocab_size,
                            dtype=torch.long,
                            device=device),
                requires_grad=False)
        else:
            self.register_parameter('draft_id_to_target_id', None)

    def set_output_head(self, output_head: nn.Module):
        """Share the full-vocabulary target head when the draft omits one."""
        if self.has_own_output_head:
            return
        if self.draft_vocab_size != int(self.config.vocab_size):
            raise RuntimeError('Reduced-vocabulary DSpark checkpoint is missing lm_head.weight.')
        if output_head is not None:
            self.output_head = output_head

    def compute_base_logits(self, hidden_states: torch.Tensor):
        if self.output_head is None:
            raise RuntimeError('DSpark draft model requires a shared or checkpoint output head.')
        return self.output_head(hidden_states)

    def init_sequential_state(self, batch_size: int, dtype: torch.dtype,
                              device: torch.device):
        return self.markov_head.init_state(batch_size, dtype, device)

    def sequential_head_step(self, prev_token_ids: torch.Tensor,
                             hidden_states: torch.Tensor,
                             state: torch.Tensor | None):
        return self.markov_head.step(prev_token_ids, hidden_states, state)

    def map_draft_to_target(self, draft_ids: torch.Tensor):
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                past_key_values: list[list[torch.Tensor]], attn_metadata,
                inputs_embeds: torch.Tensor | None = None, **kwargs):
        """Run the draft backbone and captured sequential proposal epilogue."""
        hidden_states = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            **kwargs)
        if not attn_metadata.is_decoding:
            return hidden_states
        draft_token_ids = compute_dspark_proposal_ids(
            self, hidden_states, input_ids)
        return dict(hidden_states=hidden_states,
                    draft_token_ids=draft_token_ids)

    def get_outputs_cudagraph(self, output_buffers: dict[str, torch.Tensor],
                              input_ids: torch.Tensor, **kwargs):
        """Return only live proposal rows from the captured batch bucket."""
        outputs = super().get_outputs_cudagraph(
            output_buffers, input_ids, **kwargs)
        draft_token_ids = output_buffers.get('draft_token_ids')
        if draft_token_ids is not None:
            batch_size = input_ids.numel() // self.dspark_draft_query_len
            outputs['draft_token_ids'] = draft_token_ids[:batch_size]
        return outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        backbone_weights = []
        own_params = dict(self.named_parameters())
        for raw_name, loaded_weight in weights:
            name = _normalize_dflash_weight_name(raw_name)
            if name is None or 'confidence_head.' in name or 't2d' in name:
                continue
            if 'd2t' in name:
                if self.draft_id_to_target_id is not None:
                    load_weight(self.draft_id_to_target_id, loaded_weight)
                continue
            if name.startswith('markov_head.'):
                load_weight(own_params[name], loaded_weight)
                continue
            if name == 'lm_head.weight' and self.output_head is not None:
                load_weight(self.output_head.weight, loaded_weight)
                self.has_own_output_head = True
                continue
            backbone_weights.append((name, loaded_weight))
        super().load_weights(backbone_weights)


# Raw Speculators and transformed vLLM configs use different architecture names.
DSparkDraftModel = Qwen3DSparkModel
