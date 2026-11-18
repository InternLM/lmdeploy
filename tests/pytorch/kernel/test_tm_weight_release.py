# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Spill tracker tests that must pass on a PyTorch-only installation.

``TorchWeightSpill`` arms one restore-on-touch loader per parameter
generation. ``register_all_parameters`` replaces Parameters on every
repack, so the tracker must key on the live parameter object: a bare
``id()`` survives the object and skips arming a replacement whose id got
recycled, which would send the next hot update into empty spilled
storage. Holding the parameter itself makes id recycling of tracked
entries impossible in the first place.
"""

import torch

from lmdeploy.pytorch.backends.cuda.tm_weight_release import TorchWeightSpill


def _make_param():
    param = torch.nn.Parameter(torch.empty(8), requires_grad=False)
    param.weight_loader = lambda param, loaded_weight: None
    return param


def test_arm_tracks_live_parameter_object():
    """The tracker holds the parameter object itself, not a bare id."""
    spill = TorchWeightSpill()
    param = _make_param()
    spill.spill(dict(qweight=param))
    assert any(armed is param for armed in spill._armed)


def test_tracker_retains_armed_parameters():
    """Armed entries hold live references, so tracked ids cannot be handed out
    again.

    A bare-id tracker would let CPython recycle the id of a replaced parameter; the replacement then looks already-armed
    and its loader is never wrapped, so the next hot update writes into empty spilled storage.
    """
    spill = TorchWeightSpill()
    param = _make_param()
    spill.spill(dict(qweight=param))
    ref = param
    del param  # only the test scope drops it; the tracker must keep it alive
    assert any(armed is ref for armed in spill._armed)


def test_arm_is_idempotent_for_the_same_object():
    """Re-spilling the same live parameter never stacks wrappers."""
    spill = TorchWeightSpill()
    param = _make_param()
    spill.spill(dict(qweight=param))
    original_wrapper = param.weight_loader
    spill.spill(dict(qweight=param))
    assert param.weight_loader is original_wrapper
    assert sum(armed is param for armed in spill._armed) == 1


def test_replaced_generation_is_armed_and_restores():
    """A replacement parameter gets its own wrapped loader despite a same-named
    predecessor."""
    spill = TorchWeightSpill()
    first = _make_param()
    spill.spill(dict(qweight=first))
    second = _make_param()
    spill.spill(dict(qweight=second))

    assert any(armed is second for armed in spill._armed)
    assert second.data.numel() == 0  # spilled by the second round

    # Touching the replacement's loader must run the wrapped restore first,
    # then delegate: a skipped arming would leave the storage emptied.
    second.weight_loader(second, torch.zeros(8))
    assert second.data.numel() == 8


def test_prune_drops_replaced_generations():
    """A new spill retires armed entries whose parameters were replaced."""
    spill = TorchWeightSpill()
    first = _make_param()
    spill.spill(dict(qweight=first))
    second = _make_param()
    spill.spill(dict(qweight=second))
    assert all(armed is second for armed in spill._armed)
    assert [name for name in spill._spills] == ['qweight']
