# Copyright (c) OpenMMLab. All rights reserved.

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch import Tensor

from ..block import LogicalTokenBlocks


def _cache_weight(cache: Tensor, weight: Tensor, block_table: Tensor):
    """cache weight."""
    assert cache.dim() == 2
    assert weight.dim() == 2
    assert block_table.dim() == 1
    assert block_table.size(0) == weight.size(0)

    feat_size = weight.size(-1)
    assert cache.size(-1) >= feat_size

    cache[block_table, :feat_size] = weight.to(device=cache.device,
                                               dtype=cache.dtype)


def _get_named_loralinears(model: torch.nn.Module):
    """get all named loralinear."""
    from peft.tuners.lora import Linear as LoRALinear
    named_loralinear: Dict[str, torch.nn.Module] = dict()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            named_loralinear[name] = module
    return named_loralinear


def _get_layer_index(key: str, config: Any):
    """get layer index of the lora linear."""
    from peft.utils.other import COMMON_LAYERS_PATTERN
    layer_indexing_pattern = getattr(config, 'layers_pattern', None)
    layers_pattern = layer_indexing_pattern or COMMON_LAYERS_PATTERN
    if isinstance(layers_pattern, str):
        layers_pattern = [layers_pattern]
    for pattern in layers_pattern:
        layer_index = re.match(f'.*.{pattern}\\.(\\d+)\\.*', key)

        if layer_index is not None:
            return int(layer_index[1])


def get_indexed_lora_linears(model: torch.nn.Module):
    """get indexed lora linear."""
    named_linears = _get_named_loralinears(model)

    config = None
    peft_config = getattr(model, 'peft_config', dict)
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

    def __get_rank_and_start(target_names):
        """get rank and start."""
        rank_map = dict()
        start_map = dict()
        for target in target_names:
            ranks = [0] + [
                weight_map.target_modules[target].rank
                for weight_map in weight_maps
            ]
            block_starts = [0] + [
                weight_map.target_modules[target].block_start
                for weight_map in weight_maps
            ]
            rank_map[target] = torch.tensor(ranks)
            start_map[target] = torch.tensor(block_starts)
        return rank_map, start_map

    def __update_linear(linear, idx, rank_map, start_map, adapter_names):
        """update linear."""
        linear.layer_idx = idx
        linear.ranks = rank_map[target].to(device)
        linear.block_starts = start_map[target].to(device)
        for name in adapter_names:
            if name in linear.lora_A:
                linear.lora_A.pop(name)
                linear.lora_B.pop(name)

    adapter_names = [weight_map.adapter_name for weight_map in weight_maps]

    all_targets = set()
    for weight_map in weight_maps:
        targets = weight_map.target_modules.keys()
        all_targets.update(targets)

    for weight_map in weight_maps:
        weight_map.expand_targets(all_targets)

    rank_map, start_map = __get_rank_and_start(all_targets)

    for idx, lora_linear in lora_linears.items():
        for target, linear in lora_linear.items():
            __update_linear(linear,
                            idx,
                            rank_map=rank_map,
                            start_map=start_map,
                            adapter_names=adapter_names)


@dataclass
class TargetMeta:
    rank: int
    block_start: int


@dataclass
class AdapterWeightMap:
    adapter_name: str
    block_table: Tensor
    target_modules: Dict[str, TargetMeta]

    @classmethod
    def new(cls, adapter_name: str, rank: int, target_names: List[str],
            block_table: Tensor):
        """create new weightmap."""
        block_start = 0
        target_modules: Dict[str, TargetMeta] = dict()
        for name in target_names:
            target_modules[name] = TargetMeta(rank, block_start)
            block_start += rank

        return AdapterWeightMap(adapter_name,
                                block_table=block_table,
                                target_modules=target_modules)

    def expand_targets(self,
                       target_names: List[str],
                       ignore_exists: bool = True):
        for name in target_names:
            if name in self.target_modules:
                if ignore_exists:
                    continue
                else:
                    raise RuntimeError(f'target {name} exists.')
            self.target_modules[name] = TargetMeta(0, 0)

    @classmethod
    def cache_lora_a(cls, cache: Tensor, weight: Tensor, block_table: Tensor):
        """cache lora a weight."""
        return _cache_weight(cache, weight, block_table)

    @classmethod
    def cache_lora_b(cls, cache: Tensor, weight: Tensor, block_table: Tensor):
        """cache lora b weight."""
        return _cache_weight(cache, weight.t(), block_table)

    def cache_lora_linear(self, lora_linear: torch.nn.Module, cache_a: Tensor,
                          cache_b: Tensor):
        """cache lora linear."""
        name = self.adapter_name
        target_modules = self.target_modules
        block_table = self.block_table
        block_start = 0
        for target in target_modules:
            linear = lora_linear[target]
            if not (name in linear.lora_A and name in linear.lora_B):
                continue
            linear_a = linear.lora_A[name]
            linear_b = linear.lora_B[name]
            assert linear_a.weight is not None
            assert linear_b.weight is not None
            rank = linear_a.weight.size(0)
            block_offset = block_table[block_start:block_start + rank]
            block_start += rank
            self.cache_lora_a(cache_a, linear_a.weight, block_offset)
            self.cache_lora_b(cache_b, linear_b.weight, block_offset)

    def cache_adapter(self, lora_linears: Dict, caches: List[List[Tensor]]):
        """cache all linear."""
        assert len(lora_linears) == len(caches)

        for idx, lora_linear in lora_linears.items():
            assert idx < len(caches)
            cache_a, cache_b = caches[idx]
            self.cache_lora_linear(lora_linear, cache_a, cache_b)


class SchedulerAdapter:
    """lora adapter."""

    def __init__(self, idx: int, adapter_path: str, adapter_name: str):
        from peft import PeftConfig
        self._idx = idx
        self._adapter_path = adapter_path
        self._name = adapter_name
        self._config = PeftConfig.from_pretrained(self._adapter_path)
        self._target_modules = list(self._config.target_modules)
        self._logical_blocks = LogicalTokenBlocks(1)
        self._active = False
        self._manager: AdapterManager = None

    def set_manager(self, manager: 'AdapterManager'):
        """set manager."""
        self._manager = manager

    @property
    def idx(self):
        return self._idx

    @property
    def name(self):
        """get adapter name."""
        return self._name

    @property
    def rank(self):
        """get rank."""
        return self._config.r

    @property
    def target_modules(self):
        """get target modules."""
        return self._target_modules

    @property
    def logical_blocks(self):
        """get logical blocks."""
        return self._logical_blocks

    def is_actived(self):
        """check if adapter is active."""
        return self._active

    def active(self, flag: bool = True):
        """active adapter."""
        if self._active != flag:
            if flag:
                self._manager._active_count += 1
            else:
                self._manager._active_count -= 1
            self._active = flag

    def num_blocks(self):
        """get num blocks."""
        # ranks * (lora_a + lora_b) * num_targets
        return self.rank * len(self.target_modules)

    def num_required_blocks(self):
        """get num required blocks."""
        if self.is_actived():
            return 0
        else:
            return self.num_blocks()

    def set_logical_blocks(self, logical_blocks: LogicalTokenBlocks):
        self._logical_blocks = logical_blocks

    def build_weight_map(self, block_table: Tensor):
        return AdapterWeightMap.new(self.name,
                                    rank=self.rank,
                                    target_names=self.target_modules,
                                    block_table=block_table)


class NoneAdapter(SchedulerAdapter):
    """for sequence without adapter."""

    def __init__(self):
        self._idx = 0
        self._adapter_path = None
        self._name = None
        self._config = None
        self._target_modules = []
        self._logical_blocks = LogicalTokenBlocks(1)
        self._active = True
        self._manager: AdapterManager = None

    @property
    def rank(self):
        """get rank."""
        return 0


class AdapterManager:
    """Adapter manager."""

    def __init__(self) -> None:
        self._adapters: Dict[str, SchedulerAdapter] = dict()
        self._adapters[None] = NoneAdapter()
        self._adapter_count = 1
        self._active_count = 1

    def add_adapter(self, adapter_path: str, adapter_name: str):
        """add adapter."""
        assert adapter_name not in self._adapters
        adapter = SchedulerAdapter(self._adapter_count,
                                   adapter_path,
                                   adapter_name=adapter_name)
        adapter.set_manager(self)
        self._adapters[adapter_name] = adapter
        self._adapter_count += 1
        return adapter

    def get_adapter(self, name: str, default=None):
        """get adapter."""
        return self._adapters.get(name, default)

    def num_adapters(self):
        """get num adapters."""
        return len(self._adapters)


ADAPTER_MANAGER = AdapterManager()
