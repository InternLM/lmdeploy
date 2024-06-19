# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class PatchedLlavaLlamaForCausalLM(nn.Module):

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        context = self.context.context
        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)
        outputs = self.model.forward(input_ids=None,
                                     inputs_embeds=inputs_embeds,
                                     past_key_values=past_key_values,
                                     position_ids=position_ids,
                                     return_dict=return_dict,
                                     attention_mask=attention_mask)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return CausalLMOutputWithPast(logits=logits)


class PatchedLlavaForConditionalGeneration(nn.Module):

    def forward(self,
                input_ids: torch.LongTensor = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        """llava hf."""
        context = self.context.context
        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)
        return self.language_model.forward(input_ids=None,
                                           inputs_embeds=inputs_embeds,
                                           past_key_values=past_key_values,
                                           position_ids=position_ids)
