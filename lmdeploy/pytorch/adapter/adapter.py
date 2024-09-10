# Copyright (c) OpenMMLab. All rights reserved.

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from ..block import LogicalTokenBlocks


def _div_up(a, b):
    """div up."""
    return (a + b - 1) // b


def get_ranks_and_scalings(target_name: str,
                           cfgs: Iterable,
                           device: torch.device = None):
    """get ranks and scalings."""
    ranks = []
    scalings = []
    for cfg in cfgs:
        if target_name not in cfg.target_modules:
            ranks.append(0)
            scalings.append(1)
            continue
        ranks.append(cfg.r)
        scalings.append(float(cfg.lora_alpha / cfg.r))
    ranks = torch.tensor(ranks, device=device)
    scalings = torch.tensor(scalings, device=device)
    return ranks, scalings


def find_all_target(model: torch.nn.Module, target_name: str):
    """find all targets."""
    # find packed name
    packed_name = target_name
    pack_idx = None
    packed_modules_mapping = getattr(model, 'packed_modules_mapping', dict())
    for name, sub_names in packed_modules_mapping.items():
        if target_name in sub_names:
            pack_idx = sub_names.index(target_name)
            packed_name = name
            break

    found_mods = []
    name_postfix = f'.{packed_name}'
    for name, mod in model.named_modules():
        if not name.endswith(name_postfix):
            continue
        found_mods.append((name, mod))

    return found_mods, pack_idx


def get_max_ranks_per_block(block_numel: int, rank_stride: int):
    assert block_numel >= rank_stride, (
        'LoRA Adapter requires larger block_size.')
    return block_numel // rank_stride


def get_ranks_per_block(block_numel: int, rank_stride: int, rank: int):
    """ranks per blocks."""
    max_ranks_per_block = get_max_ranks_per_block(block_numel, rank_stride)
    return min(rank, max_ranks_per_block)


def get_num_required_blocks(block_numel: int, rank_stride: int, rank: int):
    """get num required blocks."""
    ranks_per_block = get_ranks_per_block(block_numel, rank_stride, rank)
    if rank == 0:
        return 0
    return _div_up(rank, ranks_per_block)


def get_inblock_offset(block_numel: int, rank_stride: int, rank: int):
    """in block offset."""
    ranks_per_block = get_ranks_per_block(block_numel, rank_stride, rank)
    num_required_blocks = get_num_required_blocks(block_numel, rank_stride,
                                                  rank)
    ret = np.arange(ranks_per_block) * rank_stride
    ret = ret.repeat(num_required_blocks)[:rank]
    return ret


def get_block_idx_per_rank(block_numel: int, rank_stride: int, rank: int):
    """out block idx."""
    ranks_per_block = get_ranks_per_block(block_numel, rank_stride, rank)
    num_required_blocks = get_num_required_blocks(block_numel, rank_stride,
                                                  rank)
    ret = np.arange(num_required_blocks)
    ret = ret[:, None].repeat(ranks_per_block, 1)
    ret = ret.flatten()[:rank]
    return ret


def get_layer_index(key: str, layers_pattern: str = None):
    """get layer index of the lora linear."""
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


@dataclass
class LoRATargetInfo:
    """lora linear info."""
    in_features: int
    out_features: int
    colwise: bool
    rank_stride: int = field(default=0, init=False)

    def __post_init__(self):
        """post init."""
        self.rank_stride = max(self.in_features, self.out_features)


def _get_rank_and_world():
    """get rank and world size."""
    rank = 0
    world_size = 1
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return rank, world_size


@dataclass
class AdapterWeightMap:
    adapter_name: str
    path: str
    rank: List[int]
    rank_offset: np.ndarray
    max_rank: int
    target_modules: List[str]
    colwise: List[bool]

    @staticmethod
    def _get_weight(weight: torch.Tensor, is_lora_a: bool, is_col: bool,
                    rank: int, world_size: int):
        """get sliced weight."""
        if world_size == 1:
            return weight

        if not is_col and is_lora_a:
            # rowwise
            weight = weight.chunk(world_size, dim=1)[rank]
        else:
            # colwise
            weight = weight.chunk(world_size, dim=0)[rank]
        return weight

    @staticmethod
    def _fill_a_cache(weight: torch.Tensor, cache: torch.Tensor,
                      rank_off: torch.Tensor):
        """fill a cache."""
        num_ranks, feat_size = weight.shape

        for rank in range(num_ranks):
            off = rank_off[rank]
            cache[off:off + feat_size].copy_(weight[rank])

    @staticmethod
    def _fill_b_cache(weight: torch.Tensor, cache: torch.Tensor,
                      rank_off: torch.Tensor):
        """fill a cache."""
        feat_size, num_ranks = weight.shape

        for rank in range(num_ranks):
            off = rank_off[rank]
            cache[off:off + feat_size].copy_(weight[:, rank])

    def cache_adapter(self, caches: List[List[Tensor]]):
        """cache all linear."""
        if self.path is None:
            return
        checkpoint_path = f'{self.path}/adapter_model.bin'
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        dist_rank, world_size = _get_rank_and_world()

        target_modules = self.target_modules
        target_map = dict(
            (name, idx) for idx, name in enumerate(target_modules))
        num_targets = len(target_modules)
        rank_offset = self.rank_offset.view(num_targets, -1)
        for key, weight in state_dict.items():
            layer_idx = get_layer_index(key, None)
            a_cache, b_cache = caches[layer_idx]
            a_cache = a_cache.view(-1)
            b_cache = b_cache.view(-1)

            split_key = key.split('.')
            assert split_key[-1] == 'weight'
            target_name = split_key[-3]
            if split_key[-2] == 'lora_A':
                is_lora_a = True
            elif split_key[-2] == 'lora_B':
                is_lora_a = False
            else:
                raise RuntimeError(f'Unexpected key: {key}')

            target_id = target_map[target_name]
            rank_off = rank_offset[target_id]
            is_col = self.colwise[target_id]
            weight = self._get_weight(weight,
                                      is_lora_a,
                                      is_col,
                                      rank=dist_rank,
                                      world_size=world_size)
            if is_lora_a:
                self._fill_a_cache(weight, a_cache, rank_off)
            else:
                self._fill_b_cache(weight, b_cache, rank_off)


@dataclass
class SchedulerAdapter:
    """lora adapter."""

    adapter_id: int
    adapter_name: str
    rank: List[int]
    scaling: List[int]
    target_modules: List[str]
    target_infos: List[LoRATargetInfo]
    logical_blocks: LogicalTokenBlocks
    inblock_offset: np.ndarray
    block_idx_per_rank: np.ndarray
    adapter_path: str = None
    block_stride: int = 0
    max_rank: int = 0
    num_required_blocks: int = 0
    rank_offset: np.ndarray = field(default=None, init=False)
    _active: bool = field(default=False, init=False)

    @classmethod
    def new(cls, adapter_id: int, adapter_name: str, adapter_path: str,
            adapter_cfg: Any, target_infos: Dict[str, LoRATargetInfo],
            block_numel: int, max_rank: int):
        """new."""

        target_modules = list(target_infos.keys())

        rank = []
        scaling = []
        inblock_offset = [np.empty((0, ), dtype=np.int64)]
        block_idx_per_rank = [np.empty((0, ), dtype=np.int64)]
        num_required_blocks = 0
        for target_name in target_modules:

            # get rank and scaling
            r = 0
            s = 1.0
            if target_name in adapter_cfg.target_modules:
                r = adapter_cfg.r
                if r != 0:
                    s = adapter_cfg.lora_alpha / r
            rank.append(r)
            scaling.append(s)

            info = target_infos[target_name]
            rank_stride = info.rank_stride
            ib_offset = get_inblock_offset(block_numel, rank_stride, r)
            pad_ib_offset = np.zeros((max_rank, ), dtype=np.int64)
            pad_ib_offset[:ib_offset.shape[0]] = ib_offset
            inblock_offset.append(pad_ib_offset)
            bidx_p_rank = get_block_idx_per_rank(block_numel, rank_stride,
                                                 r) + num_required_blocks
            pad_bidx_p_rank = np.zeros((max_rank, ), dtype=np.int64)
            pad_bidx_p_rank[:bidx_p_rank.shape[0]] = bidx_p_rank
            block_idx_per_rank.append(pad_bidx_p_rank)
            num_required_blocks += get_num_required_blocks(
                block_numel, rank_stride, r)
        inblock_offset = np.concatenate(inblock_offset)
        block_idx_per_rank = np.concatenate(block_idx_per_rank)

        ret = cls(
            adapter_id=adapter_id,
            adapter_name=adapter_name,
            rank=rank,
            scaling=scaling,
            target_modules=target_modules,
            target_infos=target_infos,
            logical_blocks=LogicalTokenBlocks(),
            inblock_offset=inblock_offset,
            block_idx_per_rank=block_idx_per_rank,
            adapter_path=adapter_path,
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
        colwise = [
            self.target_infos[name].colwise for name in self.target_modules
        ]
        return AdapterWeightMap(
            adapter_name=self.name,
            path=self.adapter_path,
            rank=self.rank,
            rank_offset=self.rank_offset,
            max_rank=self.max_rank,
            target_modules=self.target_modules,
            colwise=colwise,
        )


class NoneLoraConfig:

    def __init__(self):
        self.r = 0
        self.lora_alpha = 8
        self.target_modules = []


class AdapterManager:
    """adapter manager."""

    def __init__(self, adapters: Dict[str, str],
                 target_infos: Dict[str, LoRATargetInfo], block_numel: int):
        self.target_infos = target_infos
        self.block_numel = block_numel
        if adapters is None:
            adapters = dict()

        self.adapter_paths = dict(
            (name, path) for name, path in adapters.items())
        self.adapter_paths[None] = None

        self.adapter_cfgs = self._get_adapter_cfgs(adapters)

        adapter_names = list(adapters.keys())
        self.adapter_id_map = dict(
            (name, idx + 1) for idx, name in enumerate(adapter_names))
        self.adapter_id_map[None] = 0

        self._adapters: Dict[str, SchedulerAdapter] = dict()
        self.max_rank = self._get_max_rank()
        self._add_non_adapter()

    @staticmethod
    def _get_adapter_cfgs(adapters: Dict[str, str]):
        """get adapter cfgs."""
        if len(adapters) == 0:
            return {None: NoneLoraConfig()}
        from peft import PeftConfig
        adapter_cfgs = dict((name, PeftConfig.from_pretrained(path))
                            for name, path in adapters.items())
        adapter_cfgs[None] = NoneLoraConfig()
        return adapter_cfgs

    def _get_max_rank(self):
        """get max rank."""
        max_rank = 0
        for cfg in self.adapter_cfgs.values():
            max_rank = max(max_rank, cfg.r)
        return max_rank

    def _add_non_adapter(self):
        """add non adapter."""
        adapter = self.add_adapter(None)
        rank_offset = adapter.inblock_offset.copy()
        adapter.update_rank_offset(rank_offset)

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
        adapter_id = self.adapter_id_map[adapter_name]
        adapter_cfg = self.adapter_cfgs[adapter_name]
        adapter_path = self.adapter_paths[adapter_name]
        adapter = SchedulerAdapter.new(
            adapter_id,
            adapter_name,
            adapter_path,
            adapter_cfg,
            self.target_infos,
            self.block_numel,
            max_rank=self.max_rank,
        )
        self._register_adapter(adapter)
        return adapter
