# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.utils import get_logger

from .base import SPEC_PROPOSERS
from .deepseek_mtp import DeepseekMTP

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='hy3_mtp')
class Hy3MTP(DeepseekMTP):
    """Hy3 MTP proposer using the autoregressive speculative loop."""

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None, build_model_ctx=None):
        """Build the MTP layer and share the target embedding table."""
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)
        if target_model is None:
            raise ValueError('Hy3 MTP requires the target model to share token embeddings')
        embed_tokens = target_model.get_input_embeddings()
        if embed_tokens is None:
            raise ValueError('Hy3 target model has no input embeddings')
        self.model.set_input_embeddings(embed_tokens)
        logger.info('Using embed_tokens from the Hy3 target model.')
