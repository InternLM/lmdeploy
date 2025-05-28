# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig


class InternVisionEmbeddings(nn.Module):
    """Mono vision."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.empty(1, 1, self.embed_dim, dtype=dtype, device=device), )

        self.patch_embedding = nn.Conv2d(in_channels=3,
                                         out_channels=self.embed_dim,
                                         kernel_size=self.patch_size,
                                         stride=self.patch_size,
                                         dtype=dtype,
                                         device=device)

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_positions, self.embed_dim, dtype=dtype, device=device))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size,
                                              -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat(
            [self.position_embedding[:, :1, :],
             self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)],
            dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternVisionPatchModel(nn.Module):
    """Mono vision."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.embeddings = InternVisionEmbeddings(config, dtype=dtype, device=device)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        if len(pixel_values.shape) != 4:
            raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')

        hidden_states = self.embeddings(pixel_values)[:, 1:]
        return hidden_states
