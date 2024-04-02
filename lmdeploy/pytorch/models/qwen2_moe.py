# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from torch import nn


class PatchedQwen2MoeModel(nn.Module):

    def _continuous_batching_forward(
            self,
            input_ids: torch.LongTensor = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None):
        """Rewrite implementation of Qwen2MoeModel.forward."""
        from transformers.modeling_outputs import MoeModelOutputWithPast

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):

            past_key_value = past_key_values[idx]
            layer_outputs = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_value,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(last_hidden_state=hidden_states,
                                      past_key_values=past_key_values,
                                      hidden_states=None,
                                      attentions=None,
                                      router_logits=None)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """Rewrite of Qwen2MoeModel.forward."""
        return self._continuous_batching_forward(
            input_ids,
            position_ids,
            past_key_values,
            inputs_embeds,
        )
