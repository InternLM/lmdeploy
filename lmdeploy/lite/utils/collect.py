# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn


def collect_target_weights(model: nn.Module, target_module_types: type,
                           skip_modules: list) -> dict:
    """Collects target weight tensors in the model and returns them in a
    dictionary.

    Args:
        model (nn.Module): Model containing the target modules.
        target (type): Target module type, e.g., nn.Linear.
        skip_modules (list): List of modules that should not be included in
            the result.

    Returns:
        dict: A dictionary containing the target weight tensors in the model.
    """
    target_weights = {}
    for name, module in model.named_modules():
        if isinstance(module,
                      target_module_types) and name not in skip_modules:
            assert hasattr(module, 'weight')
            target_weights[name] = module.weight

    return target_weights


def collect_target_modules(model: nn.Module,
                           target_module_types: type,
                           skip_modules: list = []) -> dict:
    """Collects target weight tensors in the model and returns them in a
    dictionary.

    Args:
        model (nn.Module): Model containing the target modules.
        target (type): Target module type, e.g., nn.Linear.
        skip_modules (list): List of modules that should not be included in
            the result.

    Returns:
        dict: A dictionary containing the target weight tensors in the model.
    """
    target_modules = {}
    for name, module in model.named_modules():
        if isinstance(module,
                      target_module_types) and name not in skip_modules:
            target_modules[name] = module

    return target_modules
