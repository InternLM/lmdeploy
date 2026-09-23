# Copyright (c) OpenMMLab. All rights reserved.

import torch

from ..multinomial_sampling import MultinomialSamplingBuilder, MultinomialSamplingImpl

# Dedicated stream for the exponential_ RNG draw, mirroring vllm-ascend's
# global_stream (vllm_ascend/sample/sampler.py random_sample). Running
# exponential_ on the DEFAULT stream (eager, between graph replays) produces a
# nondeterministic vector-core exception (507035, "invalid GM address /
# cross-device memory access timeout") on the EP rank -- the aivec RNG op's
# workspace overlaps the captured aclgraph replay's CANN workspace. Isolating
# it on a side stream + wait_stream keeps it off the default stream so it
# cannot race the replay. Lazy-created (one per process).
_SAMPLE_STREAM = None


def _get_sample_stream():
    global _SAMPLE_STREAM
    if _SAMPLE_STREAM is None:
        _SAMPLE_STREAM = torch.npu.Stream()
    return _SAMPLE_STREAM


class DefaultMultinomialSamplingImpl(MultinomialSamplingImpl):
    """Multinomial sampling implementation api."""

    def forward(self,
                scores: torch.Tensor,
                seeds: torch.LongTensor,
                offsets: torch.LongTensor,
                indices: torch.Tensor = None):
        """forward.

        Gumbel-max sampling: argmax(probs / q), q ~ Exponential(1), is
        mathematically equivalent to multinomial(probs) for a single draw. On
        Ascend this replaces torch.multinomial, whose
        MultinomialWithReplacement aicpu kernel nondeterministically crashes
        (errcode 0x2a -> surfaces as 507018 stream-sync timeout) over large
        bf16 vocab distributions. vllm-ascend uses the same Gumbel-max trick
        (vllm_ascend/sample/sampler.py random_sample). Upcast to fp32 for a
        stable divide + exponential_, and run exponential_ on a dedicated
        side stream (see _get_sample_stream) so it does not overlap the
        captured aclgraph replay on the default stream.
        """
        probs = scores.float()
        s = _get_sample_stream()
        with torch.npu.stream(s):
            q = torch.empty_like(probs).exponential_()
        torch.npu.current_stream().wait_stream(s)
        sampled_index = probs.div_(q).argmax(dim=-1, keepdim=True)
        outputs = torch.gather(indices, dim=1, index=sampled_index)
        return outputs.view(-1)


class DefaultMultinomialSamplingBuilder(MultinomialSamplingBuilder):
    """Multinomial sampling implementation builder."""

    @staticmethod
    def build():
        """build."""
        return DefaultMultinomialSamplingImpl()
