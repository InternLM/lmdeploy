# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

from torch import nn


def collect_target_modules(model: nn.Module,
                           target: Union[str, type],
                           skip_names: List[str] = [],
                           prefix: str = '') -> Dict[str, nn.Module]:
    """Collects the specific target modules from the model.

    Args:
        model : The PyTorch module from which to collect the target modules.
        target : The specific target to be collected. It can be a class of a
            module or the name of a module.
        skip_names : List of names of modules to be skipped during collection.
        prefix : A string to be added as a prefix to the module names.

    Returns:
        A dictionary mapping from module names to module instances.
    """

    if not isinstance(target, (type, str)):
        raise TypeError('Target must be a string (name of the module) '
                        'or a type (class of the module)')

    def _is_target(n, m):
        if isinstance(target, str):
            return target == type(m).__name__ and n not in skip_names
        return isinstance(m, target) and n not in skip_names

    name2mod = {}
    for name, mod in model.named_modules():
        m_name = f'{prefix}.{name}' if prefix else name
        if _is_target(name, mod):
            name2mod[m_name] = mod
    return name2mod


def collect_target_weights(model: nn.Module, target: Union[str, type],
                           skip_names: List[str]) -> Dict[str, nn.Module]:
    """Collects weights of the specific target modules from the model.

    Args:
        model : The PyTorch module from which to collect the weights of
            target modules.
        target : The specific target whose weights to be collected. It can be
            a class of a module or the name of a module.
        skip_names : Names of modules to be skipped during weight collection.

    Returns:
        A dictionary mapping from module instances to their
            corresponding weights.
    """

    named_modules = collect_target_modules(model, target, skip_names)
    mod2weight = {}
    for _, mod in named_modules.items():
        assert hasattr(
            mod, 'weight'), "The module does not have a 'weight' attribute"
        mod2weight[mod] = mod.weight
    return mod2weight


def bimap_name_mod(
    name2mod_mappings: List[Dict[str, nn.Module]]
) -> Tuple[Dict[str, nn.Module], Dict[nn.Module, str]]:
    """Generates bidirectional maps from module names to module instances and
    vice versa.

    Args:
        name2mod_mappings : List of dictionaries each mapping from module
            names to module instances.

    Returns:
        Two dictionaries providing bidirectional mappings between module
            names and module instances.
    """

    name2mod = {}
    mod2name = {}
    for mapping in name2mod_mappings:
        mod2name.update({v: k for k, v in mapping.items()})
        name2mod.update(mapping)
    return name2mod, mod2name
