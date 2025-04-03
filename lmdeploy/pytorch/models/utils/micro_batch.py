# Copyright (c) OpenMMLab. All rights reserved.
import functools

import torch


def enable_micro_batch(param_name):
    """Decorator factory to enable micro-batch computation."""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            inputs = kwargs.get(param_name, None)
            if isinstance(inputs, list):
                # Apply forward computation to each micro-batch
                results = []
                for input in inputs:
                    kwargs[param_name] = input
                    result = func(self, *args, **kwargs)
                    results.append(result)
                return results
            else:
                # If not a list, directly apply the forward computation
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def split_batch(func, param_name, num_splits=2):
    """Decorator to split along the 0th dimension into a specified number of
    chunks."""

    def wrapper(*args, **kwargs):
        inputs = kwargs.get(param_name, None)
        if inputs is not None:
            split_inputs = list(torch.chunk(inputs, num_splits, dim=0))
            kwargs[param_name] = split_inputs
            results = func(*args, **kwargs)
            return torch.cat(results, dim=0)
        else:
            return func(*args, **kwargs)

    return wrapper
