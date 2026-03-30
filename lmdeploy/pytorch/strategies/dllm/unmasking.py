# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.profiler import record_function

from lmdeploy.pytorch import consts
from lmdeploy.pytorch.config import DLLMConfig, UnmaskingStrategy

DLLM_MASKED = consts.DLLM_MASKED
DLLM_UNMASKED = consts.DLLM_UNMASKED
DLLM_CACHED = consts.DLLM_CACHED


class UnmaskingProcessor:

    def __init__(self, dllm_config: DLLMConfig):
        self.dllm_config = dllm_config

    def _get_scores(self, logits: torch.Tensor, token_ids: torch.Tensor):
        """Get scores."""
        scores = logits.softmax(dim=-1)
        scores = scores.gather(-1, token_ids.unsqueeze(-1)).flatten()
        return scores

    def _get_denoise_num(self):
        """Get denoise num."""
        block_size = self.dllm_config.block_length
        denoising_steps = self.dllm_config.denoising_steps
        if denoising_steps is None:
            denoising_steps = block_size
        num = block_size // self.dllm_config.denoising_steps
        num = max(1, min(num, block_size))
        return num

    def low_confidence_static(self, logits: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """static."""
        block_size = self.dllm_config.block_length
        topk = self._get_denoise_num()
        scores = self._get_scores(logits, token_ids)
        is_masked = dllm_mask == DLLM_MASKED
        scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_size)
        dllm_mask = dllm_mask.view(-1, block_size)
        _, indices = scores.topk(topk, dim=-1)
        dllm_unmasked = dllm_mask.scatter(-1, indices, DLLM_UNMASKED)

        is_masked = is_masked.view_as(dllm_mask)
        dllm_mask = torch.where(is_masked, dllm_unmasked, dllm_mask)
        return dllm_mask.flatten()

    def low_confidence_dynamic(self, logits: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """dynamic."""
        block_size = self.dllm_config.block_length
        threshold = self.dllm_config.confidence_threshold
        scores = self._get_scores(logits, token_ids)
        is_masked = dllm_mask == DLLM_MASKED
        scores = torch.where(is_masked, scores, scores.new_zeros((1, )))

        scores = scores.view(-1, block_size)
        dllm_mask = dllm_mask.view(-1, block_size)
        _, indices = scores.topk(1, dim=-1)
        scores = scores.scatter(-1, indices, threshold)

        is_masked = is_masked.view_as(dllm_mask)
        is_masked &= scores >= threshold
        dllm_mask[is_masked] = DLLM_UNMASKED
        return dllm_mask.flatten()

    def sequential(self, dllm_mask: torch.Tensor):
        """sequential."""
        block_size = self.dllm_config.block_length
        denoise_num = self._get_denoise_num()
        dllm_mask = dllm_mask.view(-1, block_size)
        is_masked = dllm_mask == DLLM_MASKED

        # get indices
        indices = is_masked.int().argmax(dim=1)
        ranges = torch.arange(0, denoise_num, device=indices.device, dtype=indices.dtype)
        indices = indices[:, None] + ranges[None, :]
        indices = indices % block_size

        dllm_unmasked = dllm_mask.clone()
        dllm_unmasked = dllm_unmasked.scatter(-1, indices, DLLM_UNMASKED)
        dllm_mask = torch.where(is_masked, dllm_unmasked, dllm_mask)

        return dllm_mask.flatten()

    @record_function('unmasking')
    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor, token_ids: torch.Tensor, dllm_mask: torch.Tensor):
        """call."""
        strategy = self.dllm_config.unmasking_strategy
        if strategy is None:
            return dllm_mask

        # reshape to [num_blocks, block_size]
        block_size = self.dllm_config.block_length
        dllm_mask = dllm_mask.unflatten(0, (-1, block_size))

        is_same = (dllm_mask == dllm_mask[:, :1]).all(dim=1)
        first_mask = dllm_mask[:, 0]

        # unmasked to cache
        is_block_unmasked = is_same & (first_mask == DLLM_UNMASKED)
        dllm_mask[is_block_unmasked] = DLLM_CACHED

        dllm_mask = dllm_mask.flatten()
        token_ids = torch.where(dllm_mask != DLLM_MASKED, input_ids, token_ids)
        if strategy == UnmaskingStrategy.LOW_CONFIDENCE_STATIC:
            dllm_mask = self.low_confidence_static(logits, token_ids, dllm_mask)
        elif strategy == UnmaskingStrategy.LOW_CONFIDENCE_DYNAMIC:
            dllm_mask = self.low_confidence_dynamic(logits, token_ids, dllm_mask)
        elif strategy == UnmaskingStrategy.SEQUENTIAL:
            dllm_mask = self.sequential(dllm_mask)
        else:
            raise RuntimeError(f'strategy {strategy} not supported.')

        return dllm_mask, token_ids
