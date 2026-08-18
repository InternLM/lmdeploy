# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
from torch.profiler import record_function

from lmdeploy.pytorch.engine.logits_process import SamplingInputs
from lmdeploy.pytorch.messages import SchedulerSequence

from ..ar.sampling import ARSamplingStrategy

SeqList = list[SchedulerSequence]


class DLLMSamplingStrategy(ARSamplingStrategy):
    """Sampling strategy for autoregressive models."""

    def __init__(self, pad_token_id: int, dllm_block_length: int) -> None:
        super().__init__(pad_token_id)
        self.dllm_block_length = dllm_block_length

    @record_function('make_sampling_inputs')
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
            'repetition_ngram_size',
            'repetition_ngram_threshold',
        ]
        for name in update_attr_names:
            attr = getattr(out, name, None)
            if attr is None:
                continue
            if attr.dim() == 1:
                repeats = (dllm_block_length, 1)
                attr = attr[None].repeat(*repeats).flatten(0, 1)
            elif attr.dim() == 2:
                repeats = (1, dllm_block_length, 1)
                attr = attr[:, None].repeat(*repeats).flatten(0, 1)
            else:
                repeats = (dllm_block_length, ) + (1, ) * (attr.dim())
                attr = attr[None].repeat(*repeats).flatten(0, 1)
            setattr(out, name, attr)

        # update generated_ids_cpu
        if out.generated_ids_cpu is not None:
            generated_ids_cpu = out.generated_ids_cpu
            if generated_ids_cpu.shape[1] == 0:
                out.generated_ids_cpu = np.repeat(generated_ids_cpu, dllm_block_length, axis=0)
            else:
                generated_ids_cpu = np.repeat(generated_ids_cpu[:, None], dllm_block_length, axis=1)
                generated_ids_cpu = np.reshape(generated_ids_cpu, (-1, generated_ids_cpu.shape[-1]))
                out.generated_ids_cpu = generated_ids_cpu

        if len(out.response_formats) > 0:
            new_resp_formats = []
            for resp in out.response_formats:
                new_resp_formats += [resp] * dllm_block_length
            out.response_formats = tuple(new_resp_formats)

        out.batch_size *= dllm_block_length

        return out
