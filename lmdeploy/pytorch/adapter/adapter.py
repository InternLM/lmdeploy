# Copyright (c) OpenMMLab. All rights reserved.

import re
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn


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


def _get_reverse_pack_map(model: nn.Module):
    """get reverse pack map."""
    packed_modules_mapping = getattr(model, 'packed_modules_mapping', dict())
    reverse_map = dict()
    for pack_name, names in packed_modules_mapping.items():
        for name in names:
            reverse_map[name] = pack_name
    return reverse_map


def _get_key_map(reverse_map: Dict[str, str]):
    """get key map."""
    key_map = dict()
    for name, pack_name in reverse_map.items():
        key = f'.{name}'
        val = f'.{pack_name}.lora_adapters.{name}'
        key_map[key] = val

    return key_map


def load_lora_weights(model: nn.Module, weights: Iterable[Tuple[str,
                                                                torch.Tensor]],
                      adapter_id: int):
    """load lora weights."""
    from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
    prefix_len = len('base_model.model.')
    w_len = len('.weight')
    reverse_map = _get_reverse_pack_map(model)
    key_map = _get_key_map(reverse_map)

    params_dict = dict(model.named_parameters())
    for name, loaded_weight in weights:
        name = name[prefix_len:]
        splited_name = name.split('.')
        assert splited_name[-1] == 'weight'
        assert splited_name[-2] in ['lora_A', 'lora_B']
        mod_name = splited_name[-3]
        dot_mod_name = f'.{mod_name}'
        if dot_mod_name in key_map:
            replace_name = key_map[dot_mod_name]
        else:
            replace_name = f'.{mod_name}.lora_adapters.{mod_name}'
        name = name[:-w_len]
        param_name = name.replace(dot_mod_name, replace_name)

        param = params_dict[param_name]
        load_weight(param, loaded_weight, adapter_id=adapter_id)


class AdapterManager:
    """adapter manager."""

    def __init__(self, adapters: Dict[str, str]):
        if adapters is None:
            adapters = dict()

        adapter_names = list(adapters.keys())
        adapter_names = sorted(adapter_names)
        adapter_names = [None] + adapter_names

        adapter_id_map = dict(zip(adapter_names, range(len(adapter_names))))
        self.adapter_id_map = adapter_id_map

    def get_adapter_ids(self, names: List[str]):
        return [self.adapter_id_map[name] for name in names]

    def num_adapters(self):
        return len(self.adapter_id_map)
