# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.models.llama import LlamaConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn.linear import build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin

vicuna_7b_stage2 = [(0, ), (0, 0), (1, ), (0, 1), (0, 0, 0), (1, 0), (2, ),
                    (0, 2), (0, 0, 1), (0, 3), (3, ), (0, 1, 0), (2, 0), (4, ),
                    (0, 0, 2), (0, 4), (1, 1), (1, 0, 0), (0, 0, 0, 0), (5, ),
                    (0, 0, 3), (0, 5), (0, 2, 0), (3, 0), (0, 1, 1), (0, 6),
                    (6, ), (0, 7), (0, 0, 4), (4, 0), (1, 2), (0, 8), (7, ),
                    (0, 3, 0), (0, 0, 0, 1), (0, 0, 5), (2, 1), (0, 0, 6),
                    (1, 0, 1), (0, 0, 1, 0), (2, 0, 0), (5, 0), (0, 9),
                    (0, 1, 2), (8, ), (0, 4, 0), (0, 2, 1), (1, 3), (0, 0, 7),
                    (0, 0, 0, 2), (0, 0, 8), (1, 1, 0), (0, 1, 0, 0), (6, 0),
                    (9, ), (0, 1, 3), (0, 0, 0, 3), (1, 0, 2), (0, 5, 0),
                    (3, 1), (0, 0, 2, 0), (7, 0), (1, 4)]

vicuna_13b_stage2 = [(0, ), (0, 0), (1, ), (0, 0, 0), (0, 1), (1, 0), (2, ),
                     (0, 2), (0, 0, 1), (0, 1, 0), (3, ), (0, 3), (2, 0),
                     (0, 0, 2), (0, 0, 0, 0), (0, 4), (1, 0, 0), (1, 1), (4, ),
                     (0, 0, 3), (0, 5), (0, 2, 0), (5, ), (3, 0), (0, 1, 1),
                     (0, 6), (0, 0, 4), (0, 0, 0, 1),
                     (0, 7), (0, 0, 5), (1, 2), (0, 0, 1, 0), (0, 3, 0),
                     (1, 0, 1), (4, 0), (0, 0, 6), (0, 8), (2, 0, 0), (0, 9),
                     (6, ), (7, ), (2, 1), (5, 0), (0, 1, 2), (0, 0, 0, 2),
                     (8, ), (0, 4, 0), (0, 1, 0, 0), (0, 2, 1), (0, 0, 7),
                     (1, 1, 0), (1, 3), (0, 0, 2, 0), (9, ), (0, 0, 8),
                     (0, 5, 0), (0, 0, 0, 3), (0, 0, 9), (0, 1, 3), (1, 0, 2),
                     (0, 0, 1, 1), (3, 0, 0), (1, 0, 0, 0)]

vicuna_33b_stage2 = [(0, ), (0, 0), (1, ), (0, 1), (0, 0, 0), (1, 0), (2, ),
                     (0, 2), (0, 0, 1), (0, 3), (3, ),
                     (0, 1, 0), (2, 0), (0, 4), (4, ), (0, 0, 2), (1, 1),
                     (1, 0, 0), (0, 5), (5, ), (0, 0, 0, 0), (0, 0, 3), (3, 0),
                     (0, 2, 0), (0, 6), (0, 1, 1), (6, ), (0, 0, 4), (0, 7),
                     (7, ), (1, 2), (4, 0), (8, ), (0, 3, 0), (0, 0, 5),
                     (0, 0, 0, 1), (0, 8), (2, 1), (0, 9), (1, 0, 1),
                     (2, 0, 0), (0, 0, 6), (5, 0), (0, 0, 1, 0), (1, 3),
                     (0, 1, 2), (0, 4, 0), (0, 0, 7), (0, 2, 1), (9, ),
                     (1, 1, 0), (0, 0, 0, 2), (6, 0), (0, 0, 8), (0, 1, 0, 0),
                     (7, 0), (0, 1, 3), (0, 5, 0), (1, 4), (0, 0, 9), (3, 1),
                     (1, 0, 2), (2, 2)]

zephyr_stage2 = [(0, ), (0, 0), (1, ), (0, 1), (2, ),
                 (0, 0, 0), (1, 0), (0, 2), (3, ), (0, 3), (4, ), (2, 0),
                 (0, 0, 1), (0, 4), (5, ), (0, 5), (0, 1, 0), (1, 1), (6, ),
                 (0, 0, 2), (3, 0), (0, 6), (7, ), (0, 7), (0, 8), (0, 0, 3),
                 (1, 0, 0), (0, 9), (0, 2, 0), (1, 2), (4, 0), (8, ), (9, ),
                 (2, 1), (0, 1, 1), (0, 0, 4), (0, 0, 0, 0), (5, 0), (0, 3, 0),
                 (1, 3), (0, 0, 5), (0, 0, 6), (6, 0), (2, 0, 0), (1, 0, 1),
                 (0, 1, 2), (0, 4, 0), (1, 4), (3, 1), (2, 2), (0, 0, 7),
                 (7, 0), (0, 2, 1), (0, 0, 8), (0, 1, 3), (0, 5, 0), (1, 5),
                 (0, 0, 9), (1, 1, 0), (0, 0, 0, 1), (0, 0, 1, 0), (4, 1),
                 (2, 3)]
mc_sim_7b_63 = [[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3],
                [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0,
                                                                      6], [6],
                [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0],
                [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3],
                [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1],
                [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8],
                [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2],
                [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5],
                [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6],
                [0, 7, 0]]

TOPK = 10


def pad_path(path, length, pad_value=-2):
    """Pad the given path list with a specific value up to a specified length.

    Args:
        path (list): The original list that needs padding.
        length (int): The desired length of the padded list.
        pad_value (optional, default=-2): The value to use for padding.

    Returns:
        list: A new list based on the original path but padded to the desired
             length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


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


class MedusaModel(nn.Module, CudaGraphMixin, DeployModelMixin):
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
        self.medusa_choices = None
        if 'vicuna-7b' in config.base_model_name_or_path:
            self.medusa_choices = vicuna_7b_stage2
        elif 'vicuna-13b' in config.base_model_name_or_path:
            self.medusa_choices = vicuna_13b_stage2
        elif 'vicuna-33b' in config.base_model_name_or_path:
            self.medusa_choices = vicuna_33b_stage2
        elif 'zephyr' in config.base_model_name_or_path:
            self.medusa_choices = zephyr_stage2
        else:
            self.medusa_choices = mc_sim_7b_63
        self.generate_medusa_buffers(device=device)

    def generate_medusa_buffers(self, device: torch.dtype = None):
        """Generate buffers for the Medusa structure based on the provided
        choices.

        Args:
            medusa_choices (list): A nested list representing tree in the
                Medusa structure.
            device (str): Device to which the tensors should be moved.
                Default is "cuda".

        Returns:
            dict: A dictionary containing buffers related to the
                Medusa structure.
        """
        if self.medusa_choices is None:
            self.medusa_attn_mask = None
            self.tree_indices = None
            self.medusa_position_ids = None
            self.retrieve_indices = None
            return

        # Sort the medusa_choices based on their lengths and then their values
        sorted_medusa_choices = sorted(self.medusa_choices,
                                       key=lambda x: (len(x), x))
        medusa_len = len(sorted_medusa_choices) + 1

        # Initialize depth_counts to keep track of how many choices have a
        # particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_medusa_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        # Create the attention mask for Medusa
        medusa_attn_mask = torch.eye(medusa_len, medusa_len)
        medusa_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                # retrieve ancestor position
                if len(cur_medusa_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_medusa_choice) - 1):
                    ancestor_idx.append(
                        sorted_medusa_choices.index(cur_medusa_choice[:c +
                                                                      1]) + 1)
                medusa_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        # Generate tree indices for the Medusa structure
        medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
        medusa_tree_indices[0] = 0
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_medusa_choice = sorted_medusa_choices[start + j]
                medusa_tree_indices[start + j +
                                    1] = cur_medusa_choice[-1] + TOPK * i + 1
            start += depth_counts[i]

        # Generate position IDs for the Medusa structure
        medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            medusa_position_ids[start + 1:start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        # Generate retrieval indices for Medusa structure verification
        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_medusa_choices)):
            cur_medusa_choice = sorted_medusa_choices[-i - 1]
            retrieve_indice = []
            if cur_medusa_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_medusa_choice)):
                    retrieve_indice.append(
                        sorted_medusa_choices.index(cur_medusa_choice[:c + 1]))
                    retrieve_paths.append(cur_medusa_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [
            pad_path(path, max_length) for path in retrieve_indices_nest
        ]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([
            torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long),
            retrieve_indices
        ],
                                     dim=1)
        self.medusa_attn_mask = medusa_attn_mask.unsqueeze(0).unsqueeze(0).to(
            device)
        self.tree_indices = medusa_tree_indices.to(device)
        self.medusa_position_ids = medusa_position_ids.to(device)
        self.retrieve_indices = retrieve_indices.to(device)

    def generate_candidates(self, medusa_logits: torch.Tensor,
                            base_token_id: torch.Tensor):
        """Generate candidates based on provided logits and indices.

        Args:
            medusa_logits (torch.Tensor): Logits from a specialized Medusa
                structure, aiding in candidate selection. Shape
                [bs, speculative_num, vocab_size]
            base_token_id (torch.Tensor): Standard logits from a language
                model. Shape [bs]

        Returns:
            tuple (torch.Tensor, torch.Tensor): A tuple containing two sets of candidates:
                1. Cartesian candidates derived from the combined original and Medusa logits.
                2. Tree candidates mapped from the Cartesian candidates using tree indices.
        """  # noqa
        # Greedy decoding: Select the most probable candidate from the original
        # logits. here we only implement greedy decoding
        bs = medusa_logits.shape[0]
        candidates_logit = base_token_id.unsqueeze(-1)
        # Extract the TOPK candidates from the medusa logits.
        candidates_medusa_logits = torch.topk(medusa_logits, TOPK,
                                              dim=-1).indices

        # Combine the selected candidate from the original logits with the
        # topk medusa logits.
        candidates = torch.cat(
            [candidates_logit,
             candidates_medusa_logits.view(bs, -1)], dim=-1)

        # Map the combined candidates to the tree indices to get tree
        # candidates.
        tree_candidates = candidates[:, self.tree_indices]

        # Extend the tree candidates by appending a zero.
        tree_candidates_ext = torch.cat([
            tree_candidates,
            torch.zeros(
                (bs, 1), dtype=torch.long, device=tree_candidates.device)
        ],
                                        dim=-1)

        # Retrieve the cartesian candidates using the retrieve indices.
        cart_candidates = tree_candidates_ext[:, self.retrieve_indices]

        # Unsqueeze the tree candidates for dimension consistency.
        tree_candidates = tree_candidates.unsqueeze(
            1)  # bs, 1, len(self.medusa_choices)
        return (cart_candidates, tree_candidates, self.medusa_attn_mask,
                self.medusa_position_ids, self.retrieve_indices)

    def support_cuda_graph(
        self,
        *args,
        **kwargs,
    ):
        """support cudagraph."""
        return True

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
