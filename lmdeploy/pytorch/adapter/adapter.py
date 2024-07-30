# Copyright (c) OpenMMLab. All rights reserved.

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
from torch import Tensor

from ..block import LogicalTokenBlocks


def _div_up(a, b):
    """div up."""
    return (a + b - 1) // b


def _cache_weight(cache: Tensor, weight: Tensor, rank_offset: Tensor):
    """cache weight."""
    assert weight.dim() == 2
    assert rank_offset.dim() == 1

    cache = cache.view(-1)
    rank, feat_size = weight.size()
    assert cache.size(-1) >= feat_size, ('cache.size(-1) >= feat_size failed.')
    assert rank <= rank_offset.size(0), ('rank <= rank_offset.size(0) failed.')
    for r in range(rank):
        r_off = rank_offset[r]
        cache[r_off:r_off + feat_size] = weight[r]


def _get_named_loralinears(model: torch.nn.Module):
    """get all named loralinear."""
    from peft.tuners.lora import Linear as LoRALinear
    from peft.tuners.lora.awq import AwqLoraLinear
    named_loralinear: Dict[str, torch.nn.Module] = dict()
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, AwqLoraLinear)):
            named_loralinear[name] = module
    return named_loralinear


def _get_layer_index(key: str, config: Any):
    """get layer index of the lora linear."""
    layers_pattern = getattr(config, 'layers_pattern', None)
    if isinstance(layers_pattern, str):
        layers_pattern = [layers_pattern]
    if layers_pattern is None or len(layers_pattern) == 0:
        layer_index = re.match(r'.*\.[^.]*\.(\d+)\.', key)
        return int(layer_index[1])
    else:
        for pattern in layers_pattern:
            layer_index = re.match(f'.*.{pattern}\\.(\\d+)\\.*', key)

            if layer_index is not None:
                return int(layer_index[1])


def get_indexed_lora_linears(model: torch.nn.Module):
    """get indexed lora linear."""
    named_linears = _get_named_loralinears(model)

    config = None
    peft_config = getattr(model, 'peft_config', dict())
    if len(peft_config) > 0:
        config = next(iter(peft_config.values()))

    indexed_linears = dict()
    for name, layer in named_linears.items():
        index = _get_layer_index(name, config)
        target = name.split('.')[-1]
        indexed_linears.setdefault(index, dict())
        indexed_linears[index][target] = layer
    return indexed_linears


def update_lora_linears(lora_linears: Dict,
                        weight_maps: List['AdapterWeightMap'],
                        device: str = 'cuda'):
    """update lora linears."""

    def __update_linear(linear, idx, target_name, adapter_names):
        """update linear."""
        linear.layer_idx = idx
        linear.target_name = target_name
        for name in adapter_names:
            if name in linear.lora_A:
                linear.lora_A.pop(name)
                linear.lora_B.pop(name)

    adapter_names = [weight_map.adapter_name for weight_map in weight_maps]

    for idx, lora_linear in lora_linears.items():
        for target, linear in lora_linear.items():
            __update_linear(linear,
                            idx,
                            target_name=target,
                            adapter_names=adapter_names)


@dataclass
class LoRALinearInfo:
    """lora linear info."""
    ranks: Dict[str, int]
    scalings: Dict[str, int]
    target_names: List[str]
    in_features: int
    out_features: int
    rank_stride: int = field(default=0, init=False)

    def __post_init__(self):
        """post init."""
        self.rank_stride = max(self.in_features, self.out_features)

    @classmethod
    def from_loralinear(cls, linear: torch.nn.Module):
        """create from lora linear."""
        from peft.tuners.lora import Linear as LoRALinear
        from peft.tuners.lora.awq import AwqLoraLinear
        assert isinstance(linear, (LoRALinear, AwqLoraLinear))

        ranks = linear.r
        scalings = linear.scaling
        in_features = linear.base_layer.in_features
        out_features = linear.base_layer.out_features
        return cls(
            ranks=ranks,
            scalings=scalings,
            target_names=list(ranks.keys()),
            in_features=in_features,
            out_features=out_features,
        )

    def max_ranks_per_block(self, block_numel: int):
        assert block_numel >= self.rank_stride, (
            'LoRA Adapter raquires larger block_size.')
        return block_numel // self.rank_stride

    def ranks_per_block(self, block_numel: int, adapter_name: str):
        """ranks per blocks."""
        max_ranks_per_block = self.max_ranks_per_block(block_numel)
        rank = self.ranks.get(adapter_name, 0)
        return min(rank, max_ranks_per_block)

    def num_required_blocks(self, block_numel: int, adapter_name: str):
        """get num required blocks."""
        ranks_per_block = self.ranks_per_block(block_numel, adapter_name)
        rank = self.ranks.get(adapter_name, 0)
        if rank == 0:
            return 0
        return _div_up(rank, ranks_per_block)

    def inblock_offset(self, block_numel: int, adapter_name: str):
        """in block offset."""
        rank = self.ranks.get(adapter_name, 0)
        ranks_per_block = self.ranks_per_block(block_numel, adapter_name)
        num_required_blocks = self.num_required_blocks(block_numel,
                                                       adapter_name)
        ret = np.arange(ranks_per_block) * self.rank_stride
        ret = ret.repeat(num_required_blocks)[:rank]
        return ret

    def block_idx_per_rank(self, block_numel: int, adapter_name: str):
        """out block idx."""
        rank = self.ranks.get(adapter_name, 0)
        ranks_per_block = self.ranks_per_block(block_numel, adapter_name)
        num_required_blocks = self.num_required_blocks(block_numel,
                                                       adapter_name)
        ret = np.arange(num_required_blocks)
        ret = ret[:, None].repeat(ranks_per_block, 1)
        ret = ret.flatten()[:rank]
        return ret


def get_loralinear_info(model: torch.nn.Module):
    """get loralinear info."""
    indexed_lora_linears = get_indexed_lora_linears(model)
    if len(indexed_lora_linears) == 0:
        return dict()
    lora_linears = indexed_lora_linears[0]
    infos = dict()
    for target_name, linear in lora_linears.items():
        infos[target_name] = LoRALinearInfo.from_loralinear(linear)
    return infos


@dataclass
class AdapterWeightMap:
    adapter_name: str
    rank: List[int]
    rank_offset: np.ndarray
    max_rank: int
    target_modules: List[str]

    @classmethod
    def cache_lora_a(cls, cache: Tensor, weight: Tensor, rank_offset: Tensor):
        """cache lora a weight."""
        return _cache_weight(cache, weight, rank_offset)

    @classmethod
    def cache_lora_b(cls, cache: Tensor, weight: Tensor, rank_offset: Tensor):
        """cache lora b weight."""
        return _cache_weight(cache, weight.t(), rank_offset)

    def cache_lora_linear(self, lora_linear: Dict[str, torch.nn.Module],
                          cache_a: Tensor, cache_b: Tensor):
        """cache lora linear."""
        name = self.adapter_name
        target_modules = self.target_modules
        rank_offset = self.rank_offset.reshape(-1, self.max_rank)
        for tidx, target in enumerate(target_modules):
            linear = lora_linear[target]
            if not (name in linear.lora_A and name in linear.lora_B):
                continue
            linear_a = linear.lora_A[name]
            linear_b = linear.lora_B[name]
            weight_a = linear_a.weight
            weight_b = linear_b.weight
            assert weight_a is not None
            assert weight_b is not None
            r_offset = rank_offset[tidx]
            self.cache_lora_a(cache_a, weight_a, r_offset)
            self.cache_lora_b(cache_b, weight_b, r_offset)

    def cache_adapter(self, lora_linears: Dict, caches: List[List[Tensor]]):
        """cache all linear."""
        assert len(lora_linears) == len(caches), (
            'len(lora_linears) == len(caches)')

        for idx, lora_linear in lora_linears.items():
            cache_a, cache_b = caches[idx]
            self.cache_lora_linear(lora_linear, cache_a, cache_b)


@dataclass
class SchedulerAdapter:
    """lora adapter."""

    adapter_name: str
    rank: List[int]
    scaling: List[int]
    target_modules: List[str]
    logical_blocks: LogicalTokenBlocks
    inblock_offset: np.ndarray
    block_idx_per_rank: np.ndarray
    block_stride: int = 0
    max_rank: int = 0
    num_required_blocks: int = 0
    rank_offset: np.ndarray = field(default=None, init=False)
    _active: bool = field(default=False, init=False)

    @classmethod
    def new(cls, adapter_name: str, linear_infos: Dict[str, LoRALinearInfo],
            block_numel: int, max_rank: int):
        """new."""

        target_modules = list(linear_infos.keys())

        rank = []
        scaling = []
        for linear in linear_infos.values():
            ranks = linear.ranks
            rank.append(ranks.get(adapter_name, 0))
            scaling.append(linear.scalings.get(adapter_name, 1.0))

        inblock_offset = [np.empty((0, ), dtype=np.int64)]
        block_idx_per_rank = [np.empty((0, ), dtype=np.int64)]
        num_required_blocks = 0
        for target_name in target_modules:
            linear = linear_infos[target_name]
            ib_offset = linear.inblock_offset(block_numel, adapter_name)
            pad_ib_offset = np.zeros((max_rank, ), dtype=np.int64)
            pad_ib_offset[:ib_offset.shape[0]] = ib_offset
            inblock_offset.append(pad_ib_offset)
            bidx_p_rank = linear.block_idx_per_rank(
                block_numel, adapter_name) + num_required_blocks
            pad_bidx_p_rank = np.zeros((max_rank, ), dtype=np.int64)
            pad_bidx_p_rank[:bidx_p_rank.shape[0]] = bidx_p_rank
            block_idx_per_rank.append(pad_bidx_p_rank)
            num_required_blocks += linear.num_required_blocks(
                block_numel, adapter_name)
        inblock_offset = np.concatenate(inblock_offset)
        block_idx_per_rank = np.concatenate(block_idx_per_rank)

        ret = cls(
            adapter_name=adapter_name,
            rank=rank,
            scaling=scaling,
            target_modules=target_modules,
            logical_blocks=LogicalTokenBlocks(),
            inblock_offset=inblock_offset,
            block_idx_per_rank=block_idx_per_rank,
            block_stride=block_numel,
            max_rank=max_rank,
            num_required_blocks=num_required_blocks,
        )

        return ret

    def update_rank_offset(self, phy_blocks: np.ndarray):
        """update rank offset."""
        if len(phy_blocks) > 0:
            rank_offset = phy_blocks[
                self.block_idx_per_rank] * self.block_stride
            rank_offset += self.inblock_offset
        else:
            rank_offset = np.zeros_like(self.inblock_offset)
        self.rank_offset = rank_offset
        return rank_offset

    def is_actived(self):
        """check if adapter is active."""
        return self._active

    def active(self, flag: bool = True):
        """active adapter."""
        self._active = flag

    @property
    def name(self):
        return self.adapter_name

    def build_weight_map(self):
        """build weight map."""
        assert self.rank_offset is not None
        return AdapterWeightMap(
            adapter_name=self.name,
            rank=self.rank,
            rank_offset=self.rank_offset,
            max_rank=self.max_rank,
            target_modules=self.target_modules,
        )


class AdapterManager:
    """adapter manager."""

    def __init__(self, linear_infos: Dict[str, LoRALinearInfo],
                 block_numel: int):
        self.linear_infos = linear_infos
        self.block_numel = block_numel
        self._adapters: Dict[str, SchedulerAdapter] = dict()

        self.max_rank = self._get_max_rank()
        self._add_non_adapter()

    def _get_max_rank(self):
        """get max rank."""
        max_rank = 0
        for linear in self.linear_infos.values():
            ranks = linear.ranks
            if len(ranks) > 0:
                max_rank = max(max_rank, max(ranks.values()))
        return max_rank

    def _add_non_adapter(self):
        """add non adapter."""
        self.add_adapter(None)

    def _register_adapter(self, adapter: SchedulerAdapter):
        """register adapter."""
        assert adapter.adapter_name not in self._adapters
        self._adapters[adapter.adapter_name] = adapter
        return adapter

    def get_adapter(self, name: str, default=None):
        """get adapter."""
        return self._adapters.get(name, default)

    def num_adapters(self):
        """get num adapters."""
        return len(self._adapters)

    def add_adapter(self, adapter_name: str):
        """add adapter."""
        adapter = SchedulerAdapter.new(
            adapter_name,
            self.linear_infos,
            self.block_numel,
            max_rank=self.max_rank,
        )
        self._register_adapter(adapter)
        return adapter
