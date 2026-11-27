# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..default.rejection_sampling import DefaultRejectionSamplingImpl


class AscendRejectionSamplingImpl(DefaultRejectionSamplingImpl):
    """Select the portable sampler or the optional Ascend Triton kernels."""

    def forward(self, target_logits, draft_token_ids, bonus_token_ids, *, is_greedy,
                random_seeds, random_offsets, draft_probs=None):
        import os
        from types import SimpleNamespace

        # The current LMDeploy BuildSpec path uses the Torch implementation by default.
        # The legacy dlinfer function patch no longer reaches RejectionSampler.
        if os.getenv('DLINFER_ASCEND_REJECT_SAMPLE_USE_TRITON', '0') != '1':
            return super().forward(
                target_logits.float(), draft_token_ids, bonus_token_ids, is_greedy=is_greedy,
                random_seeds=random_seeds, random_offsets=random_offsets, draft_probs=draft_probs)

        from lmdeploy.pytorch.kernels.ascend.triton_ops.reject_sample import rejection_sample

        # BuildSpec supplies a per-request policy after AR filtering, not the old flattened SamplingInputs.
        top_k = None if is_greedy is None else torch.where(is_greedy, 1, 2)
        policy = SimpleNamespace(max_top_k=2, top_k=top_k, random_seeds=random_seeds, random_offsets=random_offsets)
        return rejection_sample(
            target_logits.float().contiguous(),
            draft_token_ids.contiguous(),
            bonus_token_ids,
            policy,
            draft_probs=None if draft_probs is None else draft_probs.contiguous(),
        )
