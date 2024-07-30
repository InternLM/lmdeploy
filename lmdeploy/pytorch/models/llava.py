# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List

import torch
from torch import nn

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager

from .patch import get_rewrite_cls


class PatchedLlavaForConditionalGeneration(nn.Module):

    support_cuda_graph = True

    def __init__(self, origin: nn.Module, ctx_mgr: StepContextManager):
        super().__init__()
        self.ctx_mgr = ctx_mgr
        language_model_cls = get_rewrite_cls(origin.language_model)
        self.language_model = language_model_cls(origin.language_model,
                                                 ctx_mgr)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        return self.language_model.forward(input_ids=input_ids,
                                           inputs_embeds=inputs_embeds,
                                           past_key_values=past_key_values,
                                           position_ids=position_ids,
                                           attn_metadata=attn_metadata)

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.language_model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
