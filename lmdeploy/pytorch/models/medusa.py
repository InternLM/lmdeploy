# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.models.llama import LlamaConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn.linear import build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


class ResBlock(nn.Module):
    """A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual
    connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self,
                 hidden_size,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.linear = build_rowwise_linear(hidden_size,
                                           hidden_size,
                                           bias=True,
                                           dtype=dtype,
                                           device=device)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaModel(nn.Module, CudaGraphMixin):
    """The medusa model architecture."""

    packed_modules_mapping = {
        'qkv_proj': [
            'q_proj',
            'k_proj',
            'v_proj',
        ],
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    def __init__(self,
                 config: LlamaConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build medusa
        self.medusa_head = nn.ModuleList([
            nn.Sequential(
                *([
                    ResBlock(
                        self.config.hidden_size, device=device, dtype=dtype)
                ] * self.config.medusa_num_layers),
                build_rowwise_linear(self.config.hidden_size,
                                     self.config.vocab_size,
                                     bias=False,
                                     dtype=dtype,
                                     device=device),
            ) for _ in range(self.config.medusa_num_heads)
        ])

    def forward(self, last_hidden_states: torch.Tensor,
                **kwargs) -> List[torch.Tensor]:
        outputs = [head[0](last_hidden_states) for head in self.medusa_head]
        outputs = torch.cat(outputs, 0)
        return outputs

    def get_logits(self, hidden_states: List[torch.Tensor]):
        """compute logits of the model output."""
        outputs = []
        for medusa_head, hidden_state in zip(self.medusa_head, hidden_states):
            outputs.append(medusa_head[-1](hidden_state))
        outputs = torch.stack(outputs, 1)
        return outputs

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
        **kwargs,
    ):
        """prepare input."""
        return dict(last_hidden_states=context.last_hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm
        stacked_params_mapping = []

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            name = 'medusa_head.' + name
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)
