# Copyright (c) OpenMMLab. All rights reserved.
import logging

import torch
import torch.distributed as dist
from torch import nn

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.distributed import get_dist_group, get_dist_manager, get_tp_world_rank
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader

DEFAULT_VOCAB_PADDING_SIZE = 64
logger = logging.getLogger(__name__)


def _tp_agree(local_ready: bool, device: torch.device, group: dist.ProcessGroup) -> bool:
    """Resolve an optional LM-head provider decision on every TP rank."""
    ready = torch.tensor(int(local_ready), dtype=torch.int32, device=device)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group)
    return bool(ready.item())


def _tp_same_config(values: tuple[int, ...], device: torch.device, group: dist.ProcessGroup) -> bool:
    """Return whether every TP rank supplied the same integer config."""
    lower = torch.tensor(values, dtype=torch.int64, device=device)
    upper = lower.clone()
    dist.all_reduce(lower, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(upper, op=dist.ReduceOp.MAX, group=group)
    return bool(torch.equal(lower, upper))


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


class ParallelEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: int,
        dtype: torch.dtype = None,
        device: torch.device = None,
        is_tp: bool = False,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        layer_type: str = 'attn',
        force_dtype: torch.dtype = None,
    ):
        self.dist_ctx = get_dist_manager().current_context()
        super().__init__()

        self.is_tp = is_tp
        self.vocab_size = vocab_size
        self.padding_size = padding_size
        if padding_idx is not None:
            if padding_idx < 0:
                padding_idx = vocab_size + padding_idx
            assert padding_idx >= 0 and padding_idx < vocab_size
        self.padding_idx = padding_idx

        dist_cfg = get_dist_manager().current_config()
        _, self.rank = get_tp_world_rank(layer_type)
        self.tp, _ = dist_cfg.get_tp_by_layer(layer_type)

        dist_group = get_dist_group(layer_type=layer_type)
        self.tp_group = dist_group.gpu_group
        self.tp_rank = dist_group.rank

        if is_tp and self.tp > 1:
            self.vocab_size_padded = pad_vocab_size(self.vocab_size, self.padding_size)
            assert self.vocab_size_padded % self.tp == 0, \
                f'vocab_size_padded({self.vocab_size_padded}) must be divisible by tp({self.tp})'
            self.vocab_size_padded = self.vocab_size_padded // self.tp
        else:
            self.vocab_size_padded = self.vocab_size

        self.out_dtype = dtype
        self.start_index = self.rank * self.vocab_size_padded
        self.end_index = (self.rank + 1) * self.vocab_size_padded
        weight_dtype = force_dtype or dtype
        self.register_parameter('weight', self.create_weight(self.vocab_size_padded, hidden_size, weight_dtype, device))
        self.weight.weight_loader = self.weight_loader

        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.Embedding)
        self.impl = builder.build(self.start_index, self.end_index)

        self.all_reduce = self.is_tp and self.tp > 1

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
        shard_size = self.vocab_size_padded
        if self.end_index > loaded_weight.shape[0]:
            shard_size = loaded_weight.shape[0] - self.start_index

        loaded_weight = loaded_weight.narrow(0, self.start_index, shard_size)
        loaded_weight = loaded_weight.to(param.device)
        param[:loaded_weight.shape[0]].data.copy_(loaded_weight)
        param[loaded_weight.shape[0]:].data.fill_(0)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader."""
        if not self.all_reduce:
            default_weight_loader(param, loaded_weight)
            if self.padding_idx is not None:
                self.weight[self.padding_idx] = 0
        else:
            self._weight_loader_tp_rowwise(param, loaded_weight)
            if (self.padding_idx is not None and self.padding_idx >= self.start_index
                    and self.padding_idx < self.end_index):
                self.weight[self.padding_idx - self.start_index] = 0

    def forward(self, x: torch.Tensor):
        embeddings = self.impl.forward(x, self.weight, all_reduce=self.all_reduce, group=self.tp_group)
        if self.out_dtype is not None and embeddings.dtype != self.out_dtype:
            embeddings = embeddings.to(dtype=self.out_dtype)
        return embeddings


class ParallelLMHead(ParallelEmbedding):
    """LM head sharded along the vocabulary dimension."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        bias: bool = False,
        dtype: torch.dtype = None,
        device: torch.device = None,
        is_tp: bool = True,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        layer_type: str = 'attn',
    ):
        super().__init__(vocab_size=vocab_size,
                         hidden_size=hidden_size,
                         padding_idx=None,
                         dtype=dtype,
                         device=device,
                         is_tp=is_tp,
                         padding_size=padding_size,
                         layer_type=layer_type)

        if bias:
            bias_param = self.weight.new_zeros(self.vocab_size_padded)
            self.register_parameter('bias', nn.Parameter(bias_param, requires_grad=False))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter('bias', None)

        builder = get_backend().get_layer_impl_builder(OpType.Linear)
        self.impl = builder.build(hidden_size, self.vocab_size_padded, bias, dtype=dtype)

        self._symm_mem_gatherer = None
        self._symm_mem_device = self.weight.device
        self._symm_mem_dtype = self.weight.dtype
        if self.all_reduce and self.weight.device.type == 'cuda':
            device = self.weight.device
            if device.index is None:
                device = torch.device('cuda', torch.cuda.current_device())
            requested = _envs.enable_symm_mem_lmhead and self.weight.dtype == torch.bfloat16
            if not _tp_agree(requested, device, self.tp_group):
                return

            gathered_width = self.tp * self.vocab_size_padded
            capacity = _envs.symm_mem_lmhead_max_mb * 1024 * 1024
            max_tokens = capacity // (gathered_width * torch.bfloat16.itemsize)
            same_config = _tp_same_config((capacity, gathered_width, max_tokens), device, self.tp_group)
            if max_tokens <= 0 or not same_config:
                if self.tp_rank == 0:
                    logger.warning('symmetric-memory LM-head disabled because TP ranks have inconsistent arena config')
                return

            gatherer_cls = None
            try:
                from lmdeploy.pytorch.backends.cuda.comm.symm_mem_allgather import MultimemAllGatherer
                gatherer_cls = MultimemAllGatherer
            except ImportError as exc:
                if self.tp_rank == 0:
                    logger.warning('symmetric-memory LM-head unavailable: %s', exc)
            if not _tp_agree(gatherer_cls is not None, device, self.tp_group):
                return

            gatherer = gatherer_cls(group=self.tp_group,
                                    rank=self.tp_rank,
                                    gathered_width=gathered_width,
                                    max_tokens=max_tokens)
            if gatherer.prepare(device):
                self._symm_mem_gatherer = gatherer

    def tie_weights(self, embedding: ParallelEmbedding):
        """Tie the local LM-head shard to a parallel embedding shard."""
        self.weight = embedding.weight

    def _apply(self, fn, recurse=True):
        """Keep the symmetric arena aligned with model device moves."""
        previous_device = self._symm_mem_device
        previous_dtype = self._symm_mem_dtype
        result = super()._apply(fn, recurse=recurse)
        current_device = self.weight.device
        current_dtype = self.weight.dtype
        gatherer = self._symm_mem_gatherer
        if gatherer is not None and (current_device != previous_device or current_dtype != previous_dtype):
            if previous_device.type == 'cuda':
                gatherer.release()
            self._symm_mem_device = current_device
            self._symm_mem_dtype = current_dtype
            if current_dtype != torch.bfloat16:
                self._symm_mem_gatherer = None
            elif current_device.type == 'cuda' and not gatherer.prepare(current_device):
                self._symm_mem_gatherer = None
        return result

    def get_local_logits(self, hidden_states: torch.Tensor):
        """Compute logits for the vocabulary shard owned by this rank."""
        if hidden_states.dtype != self.weight.dtype:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.impl.forward(hidden_states, self.weight, self.bias)

    def all_gather_logits(self, local_logits: torch.Tensor) -> torch.Tensor:
        """All-gather full logits on every TP rank."""
        if not self.all_reduce:
            return local_logits[..., :self.vocab_size]

        if self._symm_mem_gatherer is not None:
            local_logits_2d = local_logits.reshape(-1, local_logits.shape[-1])
            gathered = self._symm_mem_gatherer(local_logits_2d)
            if gathered is not None:
                output_shape = local_logits.shape[:-1] + (self.tp * local_logits.shape[-1], )
                return gathered.reshape(output_shape)[..., :self.vocab_size]

        input_size = local_logits.size()
        output_size = (input_size[0] * self.tp, ) + input_size[1:]
        logits = local_logits.new_empty(output_size)
        dist.all_gather_into_tensor(logits, local_logits, group=self.tp_group)
        # The collective concatenates dim 0. Move its rank dimension beside
        # the vocabulary shard before reconstructing the full last dimension.
        logits = logits.reshape((self.tp, ) + input_size)
        logits = logits.movedim(0, local_logits.dim() - 1)
        logits = logits.reshape(input_size[:-1] + (self.tp * input_size[-1], ))
        return logits[..., :self.vocab_size]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute TP-local logits and all-gather them on every rank."""
        local_logits = self.get_local_logits(hidden_states)
        return self.all_gather_logits(local_logits)
