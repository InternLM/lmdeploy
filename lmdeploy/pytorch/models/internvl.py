# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class PatchedInternVLChatModel(nn.Module):

    def forward(self,
                pixel_values: torch.FloatTensor = None,
                input_ids: torch.LongTensor = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs = self.language_model(input_ids=input_ids,
                                      past_key_values=past_key_values,
                                      position_ids=position_ids)
        return CausalLMOutputWithPast(logits=outputs.logits)
