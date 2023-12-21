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


def _get_named_loralinear(model: torch.nn.Module):
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


@dataclass
class AdapterWeightMap:
    adapter_name: str
    adapter_path: str
    block_table: Tensor
    target_modules: List[str]

    def load_adapter(self, model: torch.nn.Module):
        """load adapter."""
        model.load_adapter(self.adapter_path, self.adapter_name)

    def get_lora_linears(self, model: torch.nn.Module):
        """get all lora linears."""
        name = self.adapter_name
        config = model.peft_config[name]
        target_modules = list(config.target_modules)
        named_linears = _get_named_loralinear(model)
        num_layers = len(named_linears) // len(target_modules)
        indexed_linears = [dict() for _ in range(num_layers)]
        for name, linear in named_linears.items():
            index = _get_layer_index(name, config)
            target = name.split('.')[-1]
            assert target in target_modules
            indexed_linears[index][target] = linear
        return indexed_linears

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
            assert name in linear.lora_A and name in linear.lora_B
            linear_a = linear.lora_A[name]
            linear_b = linear.lora_B[name]
            assert linear_a.weight is not None
            assert linear_b.weight is not None
            rank = linear_a.weight.size(0)
            block_offset = block_table[block_start:block_start + rank]
            block_start += rank
            self.cache_lora_a(cache_a, linear_a.weight, block_offset)
            self.cache_lora_b(cache_b, linear_b.weight, block_offset)

    def cache_adapter(self, lora_linears: List[Dict],
                      caches: List[List[Tensor]]):
        """cache all linear."""
        assert len(lora_linears) == len(caches)

        for cache, lora_linear in zip(caches, lora_linears):
            cache_a, cache_b = cache
            self.cache_lora_linear(lora_linear, cache_a, cache_b)

    def update_linears(self, lora_linears: List[Dict]):
        """remove linear weights, add index."""
        target_modules = self.target_modules
        for idx, lora_linear in enumerate(lora_linears):
            for target in target_modules:
                linear = lora_linear[target]
                linear.layer_idx = idx
                linear.lora_A.clear()
                linear.lora_B.clear()


class SchedulerAdapter:
    """lora adapter."""

    def __init__(self, adapter_path: str, adapter_name: str):
        from peft import PeftConfig
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
        return AdapterWeightMap(adapter_name=self._name,
                                adapter_path=self._adapter_path,
                                block_table=block_table,
                                target_modules=self.target_modules)


class NoneAdapter(SchedulerAdapter):
    """for sequence without adapter."""

    def __init__(self):
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
        self._active_count = 1

    def add_adapter(self, adapter_path: str, adapter_name: str):
        """add adapter."""
        assert adapter_name not in self._adapters
        adapter = SchedulerAdapter(adapter_path, adapter_name=adapter_name)
        adapter.set_manager(self)
        self._adapters[adapter_name] = adapter
        return adapter

    def get_adapter(self, name: str, default=None):
        """get adapter."""
        return self._adapters.get(name, default)

    def num_adapters(self):
        """get num adapters."""
        return len(self._adapters)


ADAPTER_MANAGER = AdapterManager()
