# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterable, List, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager

from .patch import build_model_from_hf_config
from .utils.cudagraph import CudaGraphMixin


class LlavaForConditionalGeneration(nn.Module, CudaGraphMixin):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        text_config = config.text_config
        self.language_model = build_model_from_hf_config(text_config,
                                                         dtype=dtype,
                                                         device=device)

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

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.language_model.get_logits(hidden_states)

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""

        prefix_length = len('language_model.')
        new_weights = dict()
        for key, val in weights:
            if not key.startswith('language_model.'):
                continue
            new_key = key[prefix_length:]
            new_weights[new_key] = val

        self.language_model.load_weights(new_weights.items())
