# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn


class RejectionSampler(nn.Module):
    """apply rejection sampling according to "Accelerating Large Language Model
    Decoding with Speculative Sampling".

    https://arxiv.org/pdf/2302.01318
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def create_uniform_random(batch_size: int, num_speculative_tokens: int,
                              device: torch.device, dtype: torch.dtype):
        """Generate a batch of random uniform samples.

        Args:
            batch_size (int): The batch size.
            num_speculative_tokens (int): The number of speculative tokens.
            device (torch.device): The device of the output tensor.
            dtype (torch.dtype): The dtype of the output tensor.

        Returns:
            random_uniform (Tensor): The uniform tensor, shape
                (batch_size, num_speculative_tokens).
        """
        return torch.rand(batch_size,
                          num_speculative_tokens,
                          device=device,
                          dtype=dtype)

    @staticmethod
    def adjusted_distribution(target_probs_without_bonus: torch.Tensor,
                              draft_probs: torch.Tensor):
        """Adjust the distribution from the draft_probs if needed.

        Args:
            target_probs_without_bonus (Tensor): The probability distribution
                over token ids from the target model, shape
                 (batch_size, num_speculative_tokens, vocab_size).

            draft_probs (Tensor): The probability distribution over token ids
                from the draft model, shape (batch_size,
                num_speculative_tokens, vocab_size).
        Returns:
            adjusted_probs (Tensor): The adjusted probability distribution,
                shape (batch_size, num_speculative_tokens, vocab_size).
        """
        adjusted_probs = (target_probs_without_bonus -
                          draft_probs).clamp_min_(0)
        adjusted_probs = adjusted_probs / adjusted_probs.sum(
            -1, keepdim=True).clamp_min_(1e-5)  # clamp to avoid div zero
        return adjusted_probs

    # TODO fuse into a triton kernel
    # TODO add seed
    def forward(self, target_probs: torch.Tensor, draft_probs: torch.Tensor,
                draft_token_ids: torch.Tensor) -> torch.Tensor:
        """Reject sampling probs and return token_ids.

        Args:
            target_probs (Tensor): The probability distribution
                over token ids from the target model, shape
                 (batch_size, num_speculative_tokens + 1, vocab_size).

            draft_probs (Tensor): The probability distribution over token ids
                from the draft model, shape (batch_size,
                num_speculative_tokens, vocab_size).

            draft_token_ids (Tensor): The proposal token id from the draft
                model, shape (batch_size, num_speculative_tokens).

        Returns:
            output_token_ids (Tensor): Token ids sampled through rejection
                sampling. shape (batch)size, num_speculative_tokens + 1).
        """
        target_probs_without_bonus = target_probs[:, :-1]
        batch_size, num_speculative_tokens, _ = draft_probs.shape
        device = draft_probs.device
        batch_indices = torch.arange(batch_size, device=device)
        probs_indicies = torch.arange(num_speculative_tokens, device=device)
        draft_token_probs = draft_probs[batch_indices[:, None], probs_indicies,
                                        draft_token_ids]
        target_token_probs = target_probs_without_bonus[batch_indices[:, None],
                                                        probs_indicies,
                                                        draft_token_ids]
        # target model scores draft token ids
        scores = target_token_probs / draft_token_probs
        random_uniform = self.create_uniform_random(batch_size,
                                                    num_speculative_tokens,
                                                    device=device,
                                                    dtype=scores.dtype)
        rejected = scores < random_uniform
        rejected_mask = rejected.cumsum(-1) > 0
        accepted_mask = ~rejected_mask
        rejected_mask = torch.cat(
            [rejected_mask,
             rejected_mask.new_ones(batch_size, 1)], -1)
        reject_idx = rejected_mask.float().argmax(-1, False)
        # compute adjusted token ids
        adjusted_probs = self.adjusted_distribution(target_probs_without_bonus,
                                                    draft_probs)
        adjusted_probs = torch.cat([adjusted_probs, target_probs[:, -1:]], 1)
        adjusted_probs = adjusted_probs[batch_indices, reject_idx]
        adjusted_token_ids = torch.multinomial(adjusted_probs,
                                               num_samples=1,
                                               replacement=True).squeeze(-1)
        output_token_ids = draft_token_ids.new_full(
            (batch_size, num_speculative_tokens + 1), -1)
        output_token_ids[~rejected_mask] = draft_token_ids[accepted_mask]
        output_token_ids[batch_indices, reject_idx] = adjusted_token_ids
        return output_token_ids


def test_rejection_sampler():
    batch_size = 4
    num_speculative_tokens = 5
    vocab_size = 1024
    dtype = torch.float32
    device = torch.device('cuda')
    target_logits_with_bonus = torch.rand(
        (batch_size, num_speculative_tokens + 1, vocab_size),
        dtype=dtype,
        device=device)
    draft_logits = torch.rand((batch_size, num_speculative_tokens, vocab_size),
                              dtype=dtype,
                              device=device)
    draft_token_ids = torch.randint(0,
                                    vocab_size,
                                    (batch_size, num_speculative_tokens),
                                    device=device)
    rejection_sampler = RejectionSampler()
    rejection_sampler.forward(target_logits_with_bonus, draft_logits,
                              draft_token_ids)
