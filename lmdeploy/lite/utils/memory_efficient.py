# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def memory_efficient_inference(model: nn.Module,
                               target=(nn.Linear, ),
                               device='cuda'):
    """Context manager for memory-efficient inference on specified modules of a
    PyTorch model.

    Args:
      model (nn.Module): The model to be used for inference.
      target (tuple): A tuple containing the target module classes to move to
            GPU during forward pass.
      device (str): The device ('cpu' or 'cuda') where the model will be
            moved during inference.

    Yields:
      None

    Example:
      with memory_efficient_inference(model, target=nn.Linear, device='cuda'):
          output = model(input)
    """

    def _before_forward_hook(m, input):
        m.to(device)

    def _after_forward_hook(m, input, output):
        m.to('cpu')
        torch.cuda.empty_cache()

    def _to_device(m, spec_modules, dev):
        if len(spec_modules) == 0:
            m.to(dev)
            return

        for child in m.children():
            if isinstance(child, spec_modules):
                child.to('cpu')
            else:
                _to_device(child, spec_modules, dev)
        m.to(dev)

    _to_device(model, target, device)
    # enter
    hook_handles = []
    for module in model.modules():
        if isinstance(module, target):
            before_h = module.register_forward_pre_hook(_before_forward_hook)
            after_h = module.register_forward_hook(_after_forward_hook)
            hook_handles.append(before_h)
            hook_handles.append(after_h)

    with torch.inference_mode():
        yield

    # exit
    for h in hook_handles:
        h.remove()

    model.to('cpu')
    torch.cuda.empty_cache()
