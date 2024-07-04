# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import LongTensor, Tensor


def multinomial_sampling(scores: Tensor,
                         seeds: LongTensor,
                         offsets: LongTensor,
                         indices: Tensor = None):
    sampled_index = torch.multinomial(scores, num_samples=1, replacement=True)
    outputs = torch.gather(indices, dim=1, index=sampled_index)
    return outputs.view(-1)
