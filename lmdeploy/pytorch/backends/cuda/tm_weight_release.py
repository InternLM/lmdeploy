# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Spill TurboMind-packed torch weights to exact CPU copies.

After ``update_weights`` packs a layer into the TurboMind channel, forward
runs on the packed handle only. The raw torch parameters are dead weight on
the GPU. Spilling replaces their storage with an empty tensor while keeping
an exact CPU copy, so a hot update can restore the module and re-pack.

Hot updates write through ``param.weight_loader``. Once armed, each wrapped
loader first restores the whole module from the CPU copies, then delegates to
the original loader. Restoration happens before any size assertion in the
default loader, which would otherwise reject the emptied storage.
"""

import torch


class TorchWeightSpill:
    """Keep CPU copies of released parameters and restore them on hot
    update."""

    def __init__(self):
        self._spills: dict[str, tuple[torch.nn.Parameter, torch.Tensor, torch.device]] = {}
        self._armed: set[int] = set()

    def spill(self, named_params: dict[str, torch.nn.Parameter]):
        """Copy each parameter to CPU, then free its GPU storage."""
        for name, param in named_params.items():
            if param is None or param.numel() == 0:
                # Already spilled without an intervening restore: the CPU
                # copy stays authoritative, never overwrite it with the
                # emptied storage.
                continue
            self._spills[name] = (param, param.data.detach().to('cpu', copy=True), param.device)
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            self._arm(param)

    def _arm(self, param: torch.nn.Parameter):
        """Wrap the parameter loader so the first touch restores the module."""
        if id(param) in self._armed:
            return
        self._armed.add(id(param))
        original = getattr(param, 'weight_loader', None)

        def loader(wrapped_param, loaded_weight, **kwargs):
            self.restore()
            if original is None:
                return None
            return original(wrapped_param, loaded_weight, **kwargs)

        param.weight_loader = loader

    def restore(self):
        """Reallocate the released GPU storages from their CPU copies."""
        for param, copy, device in self._spills.values():
            param.data = torch.empty(copy.shape, dtype=copy.dtype, device=device)
            param.data.copy_(copy)
        self._spills.clear()

    @property
    def resident_bytes(self) -> int:
        """Bytes currently held in CPU copies."""
        return sum(copy.numel() * copy.element_size() for _, copy, _ in self._spills.values())
