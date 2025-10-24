# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.messages import SchedulerSequence

from ..ar.sampling import ARSamplingStrategy

SeqList = List[SchedulerSequence]


class DLLMSamplingStrategy(ARSamplingStrategy):
    """Sampling strategy for autoregressive models."""

    def __init__(self, pad_token_id: int, dllm_block_length: int) -> None:
        super().__init__(pad_token_id)
        self.dllm_block_length = dllm_block_length

    def make_sampling_inputs(self, seqs: SeqList) -> SamplingInputs:
        """Create sampling inputs from the sequences."""
        out = super().make_sampling_inputs(seqs)
        dllm_block_length = self.dllm_block_length

        # repeat tensor
        update_attr_names = [
            'temperature',
            'bad_words',
            'bad_mask',
            'stop_words',
            'stop_mask',
            'repetition_penalty',
            'top_k',
            'top_p',
            'min_p',
            'random_seeds',
            'random_offsets',
            'all_ids',
            'num_ignore_eos',
        ]
        for name in update_attr_names:
            attr = getattr(out, name)
            if attr is None:
                continue
            repeats = (dllm_block_length, ) + (1, ) * (attr.dim())
            attr = attr[None].repeat(*repeats).flatten(0, 1)
            setattr(out, name, attr)

        if len(out.response_formats) > 0:
            new_resp_formats = []
            for resp in out.response_formats:
                new_resp_formats += [resp] * dllm_block_length
            out.response_formats = tuple(new_resp_formats)

        out.batch_size *= dllm_block_length

        return out
