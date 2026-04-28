# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.utils import get_logger

from ...config import ModelConfig
from ...model_inputs import ModelInputs
from ...strategies.base.model_agent import ExtraInputs
from .base import SPEC_PROPOSERS
from .deepseek_mtp import DeepseekMTP

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='eagle3')
class Eagle3(DeepseekMTP):

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None, build_model_ctx=None):
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)
        self.draft_id_to_target_id = self.model.draft_id_to_target_id
        self._init_bitmask_translate_constants()
        if not self.model.include_embed_tokens:
            logger.info('Using embed_tokens from target model.')
            del self.model.model.embed_tokens
            self.model.model.embed_tokens = target_model.get_input_embeddings()

    def _init_bitmask_translate_constants(self):
        d2t = self.draft_id_to_target_id
        self._d2t_words = d2t // 32
        self._d2t_bits = d2t % 32
        draft_vocab_size = d2t.size(0)
        draft_indices = torch.arange(draft_vocab_size, dtype=torch.int32)
        self._draft_words = draft_indices // 32
        self._draft_bits = draft_indices % 32

    def _translate_bitmask(self, target_bitmask: torch.Tensor) -> torch.Tensor:
        """Translate target-vocab bitmask to draft-vocab bitmask.

        Args:
            target_bitmask: [batch, ceil(target_vocab/32)] int32 bitmask
                produced by xgr.GrammarMatcher.fill_next_token_bitmask.

        Returns:
            draft_bitmask: [batch, ceil(draft_vocab/32)] int32 bitmask
                compatible with apply_batched_bitmap.
        """
        d2t_words = self._d2t_words.to(target_bitmask.device)
        d2t_bits = self._d2t_bits.to(target_bitmask.device)
        draft_words = self._draft_words.to(target_bitmask.device)
        draft_bits = self._draft_bits.to(target_bitmask.device)

        word_vals = target_bitmask[:, d2t_words]
        draft_valid = ((word_vals >> d2t_bits.unsqueeze(0)) & 1).to(torch.int32)

        # scatter_add_ is correct because bit positions within the same word
        # never overlap, so addition ≡ bitwise OR.
        bits_to_set = draft_valid << draft_bits
        n_draft_words = (draft_valid.size(1) + 31) // 32
        out = target_bitmask.new_zeros(target_bitmask.size(0), n_draft_words)
        out.scatter_add_(1, draft_words.unsqueeze(0).expand(target_bitmask.size(0), -1),
                         bits_to_set)
        return out

    def get_target_hidden_size(self, model_config: ModelConfig):
        """Get target hidden size."""
        hf_config = self.specdecode_config.model_config.hf_config
        hidden_size = getattr(hf_config, 'target_hidden_size', hf_config.hidden_size)
        return hidden_size * 3

    def get_outputs(self,
                    model_outputs: dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ExtraInputs = None,
                    guided_processors: dict | None = None):
        """Get outputs."""
        hidden_states = model_outputs['hidden_states']
        hidden_states_prenorm = model_outputs['hidden_states_prenorm']
        model_metas = model_outputs['model_metas']
        if extra_inputs is not None and extra_inputs.last_token_indices is not None:
            # for long input
            if (not model_inputs.is_decoding) and model_inputs.seq_length.size(0) == 1:
                hidden_states = hidden_states[:, -1:]
                hidden_states_prenorm = hidden_states_prenorm[:, -1:]
            else:
                last_token_loc = extra_inputs.last_token_indices
                hidden_states = hidden_states[:, last_token_loc]
                hidden_states_prenorm = hidden_states_prenorm[:, last_token_loc]

        logits = self.get_logits(hidden_states)[0]

        if guided_processors and self.guided_decoding_manager is not None:
            guided_manager = self.guided_decoding_manager
            guided_bitmask = guided_manager.allocate_batched_bitmap(logits.size(0))
            for idx, processor in guided_processors.items():
                guided_manager.fill_bitmap(processor, guided_bitmask, idx)
            draft_bitmask = self._translate_bitmask(guided_bitmask)
            guided_manager.apply_batched_bitmap(logits, draft_bitmask)

        draft_token_ids = logits.argmax(dim=-1, keepdim=True)
        draft_token_ids = self.draft_id_to_target_id[draft_token_ids]

        if guided_processors and self.guided_decoding_manager is not None:
            guided_manager = self.guided_decoding_manager
            for idx, processor in guided_processors.items():
                guided_manager.accept_token(processor, draft_token_ids[idx, 0].item())

        return draft_token_ids, model_metas, hidden_states_prenorm
