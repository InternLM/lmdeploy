# Copyright (c) OpenMMLab. All rights reserved.
import torch


def enable_micro_batch(func):
    """Decorator to enable micro-batch computation."""
    def wrapper(self, hidden_states, *args, **kwargs):
        if isinstance(hidden_states, list):
            # Apply forward computation to each micro-batch
            return [func(self, hs, *args, **kwargs) for hs in hidden_states]
        else:
            # If not a list, directly apply the forward computation
            return func(self, hidden_states, *args, **kwargs)
    return wrapper


def split_batch(func, param_name, num_splits=2):
    """Decorator to split along the 0th dimension into a specified number of chunks."""
    def wrapper(*args, **kwargs):
        inputs = kwargs.get(param_name, None)
        if inputs is not None:
            split_inputs = list(torch.chunk(inputs, num_splits, dim=0))
            kwargs[param_name] = split_inputs
            results = func(*args, **kwargs)
        return torch.cat(results, dim=0)
    return wrapper