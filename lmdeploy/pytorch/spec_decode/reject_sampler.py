# Copyright (c) OpenMMLab. All rights reserved.
from torch import LongTensor, Tensor, nn
from torch.profiler import record_function

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.engine.logits_process import FusedLogitsProcessor, SamplingInputs


class RejectionSampler(nn.Module):
    """Apply speculative rejection through the selected device backend."""

    def __init__(self, backend_type: str | None = None):
        super().__init__()
        backend = get_backend(backend_type)
        impl_builder = backend.get_layer_impl_builder(OpType.RejectionSampling)
        self._impl = impl_builder.build()

    @record_function('rejection_sample')
    def forward(
        self,
        target_logits: Tensor,
        draft_token_ids: LongTensor,
        expanded_sampling_inputs: SamplingInputs,
        draft_probs: Tensor | None = None,
    ):
        """Route processed draft-plus-bonus logits through verification."""
        assert target_logits.ndim == 3
        assert draft_token_ids.ndim == 2
        batch_size, num_tokens = target_logits.shape[:2]
        assert target_logits.shape[:2] == (
            draft_token_ids.size(0), draft_token_ids.size(1) + 1), (
                'target logits must contain each draft position plus one bonus')
        assert expanded_sampling_inputs.batch_size == batch_size * num_tokens, (
            'rejection sampling requires one sampling policy per target token')
        if draft_probs is not None:
            assert draft_probs.shape == (
                batch_size, num_tokens - 1, target_logits.size(-1)), (
                    'draft probabilities must match the draft-position logits')

        if expanded_sampling_inputs.max_top_k == 1:
            target_token_ids = target_logits.argmax(dim=-1)
            return self._impl.forward_greedy(target_token_ids, draft_token_ids)

        bonus_sampling_inputs = expanded_sampling_inputs.select_sampling_rows(
            slice(num_tokens - 1, None, num_tokens))
        bonus_logits = target_logits[:, -1]
        bonus_token_ids = FusedLogitsProcessor(
            bonus_sampling_inputs).sampling(bonus_logits)
        target_draft_logits = target_logits[:, :-1].contiguous()

        is_greedy = None
        if bonus_sampling_inputs.has_greedy:
            assert bonus_sampling_inputs.top_k is not None
            is_greedy = bonus_sampling_inputs.top_k == 1

        if is_greedy is not None:
            assert is_greedy.shape == (batch_size, ), (
                'rejection sampling requires one greedy-policy value per request')

        return self._impl.forward(
            target_draft_logits,
            draft_token_ids,
            bonus_token_ids,
            is_greedy=is_greedy,
            draft_probs=draft_probs,
        )
