# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.kernels.cuda.rejection_sampling import (
    greedy_rejection_sample,
    random_rejection_sample,
    sample_recovered_tokens,
)

from ..rejection_sampling import RejectionSamplingBuilder, RejectionSamplingImpl


def _allocate_outputs(draft_token_ids: torch.LongTensor):
    batch_size, num_spec_tokens = draft_token_ids.shape
    output_token_ids = draft_token_ids.new_empty(
        (batch_size, num_spec_tokens + 1))
    num_rejected_tokens = draft_token_ids.new_empty(batch_size)
    last_token_ids = draft_token_ids.new_empty(batch_size)
    return output_token_ids, num_rejected_tokens, last_token_ids


class CudaRejectionSamplingImpl(RejectionSamplingImpl):
    """Triton rejection-sampling implementation for CUDA."""

    def forward_greedy(
        self,
        target_token_ids: torch.LongTensor,
        draft_token_ids: torch.LongTensor,
    ):
        output_token_ids, num_rejected_tokens, last_token_ids = (
            _allocate_outputs(draft_token_ids))
        greedy_rejection_sample(
            output_token_ids,
            num_rejected_tokens,
            last_token_ids,
            draft_token_ids,
            target_token_ids[:, :-1],
            target_token_ids[:, -1],
            is_greedy=None,
        )
        return output_token_ids, num_rejected_tokens, last_token_ids

    def forward(
        self,
        target_logits: torch.Tensor,
        draft_token_ids: torch.LongTensor,
        bonus_token_ids: torch.LongTensor,
        *,
        is_greedy: torch.Tensor | None,
        draft_probs: torch.Tensor | None = None,
    ):
        assert draft_probs is None or draft_probs.is_contiguous()
        draft_token_ids = draft_token_ids.contiguous()
        if is_greedy is not None:
            is_greedy = is_greedy.contiguous()

        output_token_ids, num_rejected_tokens, last_token_ids = (
            _allocate_outputs(draft_token_ids))
        if is_greedy is not None:
            target_token_ids = target_logits.argmax(dim=-1)
            greedy_rejection_sample(
                output_token_ids,
                num_rejected_tokens,
                last_token_ids,
                draft_token_ids,
                target_token_ids,
                bonus_token_ids,
                is_greedy,
            )

        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
        batch_size, num_spec_tokens = draft_token_ids.shape
        vocab_size = target_logits.shape[-1]
        uniform_probs = torch.rand(
            (batch_size, num_spec_tokens),
            dtype=torch.float64,
            device=target_logits.device,
        )
        q = torch.empty(
            (batch_size, vocab_size),
            dtype=torch.float32,
            device=target_logits.device,
        )
        q.exponential_()
        inv_q = q.reciprocal()
        recovered_token_ids = draft_token_ids.new_empty(
            (batch_size, num_spec_tokens))
        sample_recovered_tokens(
            recovered_token_ids,
            draft_token_ids,
            draft_probs,
            target_probs,
            inv_q,
        )
        random_rejection_sample(
            output_token_ids,
            num_rejected_tokens,
            last_token_ids,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
        )
        return output_token_ids, num_rejected_tokens, last_token_ids


class CudaRejectionSamplingBuilder(RejectionSamplingBuilder):
    """Build the CUDA rejection sampler."""

    @staticmethod
    def build() -> RejectionSamplingImpl:
        return CudaRejectionSamplingImpl()
