# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.distributed import get_dist_group, get_dist_manager, get_tp_world_rank
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader

DEFAULT_VOCAB_PADDING_SIZE = 64


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


class ParallelEmbedding(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 padding_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 is_tp: bool = False,
                 padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
                 layer_type: str = 'mlp'):
        self.dist_ctx = get_dist_manager().current_context()
        super().__init__()

        self.is_tp = is_tp
        self.vocab_size = vocab_size
        self.padding_size = padding_size
        if padding_idx < 0:
            padding_idx = vocab_size - padding_idx
        assert padding_idx > 0
        self.padding_idx = padding_idx

        dist_cfg = get_dist_manager().current_config()
        _, self.rank = get_tp_world_rank(layer_type)
        self.tp, tp_mode = dist_cfg.get_tp_by_layer(layer_type)

        dist_group = get_dist_group(layer_type=layer_type)
        self.tp_group = dist_group.gpu_group

        if is_tp and self.tp > 1:
            self.vocab_size_padded = pad_vocab_size(self.vocab_size, self.padding_size)
            assert self.vocab_size_padded % self.tp == 0
            self.vocab_size_padded = self.vocab_size_padded // self.tp
        else:
            self.vocab_size_padded = self.vocab_size

        self.start_index = self.rank * self.vocab_size_padded
        self.end_index = (self.rank + 1) * self.vocab_size_padded
        self.register_parameter('weight', self.create_weight(self.vocab_size_padded, hidden_size, dtype, device))
        self.weight.weight_loader = self.weight_loader

        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.Embedding)
        self.impl = builder.build(self.start_index, self.end_index)

        self.all_reduce = self.tp > 1

    @staticmethod
    def create_weight(vocab_size: int, hidden_size: int, dtype: torch.dtype = None, device: torch.device = None):
        """Create weight."""
        if dtype is None:
            dtype = torch.float16
        if device is None:
            device = 'cuda'
        weight = torch.nn.Parameter(torch.zeros((vocab_size, hidden_size), dtype=dtype, device=device),
                                    requires_grad=False)
        return weight

    def _weight_loader_tp_rowwise(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader for rowwise embedding."""
        loaded_weight = loaded_weight.to(param.device)

        shard_size = self.vocab_size_padded
        if self.end_index > loaded_weight.shape[0]:
            shard_size = loaded_weight.shape[0] - self.start_index

        loaded_weight = loaded_weight.narrow(0, self.start_index, shard_size)
        param[:loaded_weight.shape[0]].data.copy_(loaded_weight)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader."""
        if not self.is_tp:
            default_weight_loader(param, loaded_weight)
        else:
            self._weight_loader_tp_rowwise(param, loaded_weight)
        if self.padding_idx is not None and self.padding_idx >= self.start_index and self.padding_idx < self.end_index:
            self.weight[self.padding_idx - self.start_index] = 0

    def forward(self, x: torch.Tensor):
        return self.impl.forward(x, self.weight, all_reduce=self.all_reduce, group=self.tp_group)
