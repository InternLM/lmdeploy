# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.utils import get_logger

from .base import SPEC_PROPOSERS
from .deepseek_mtp import DeepseekMTP

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='qwen3_5_mtp')
class Qwen3_5MTP(DeepseekMTP):

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None, build_model_ctx=None):
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)
        logger.info('Using embed_tokens from target model.')
        self.model.set_input_embeddings(target_model.get_input_embeddings())
        assert self.model.get_input_embeddings() is not None, 'Input embeddings should not be None.'
