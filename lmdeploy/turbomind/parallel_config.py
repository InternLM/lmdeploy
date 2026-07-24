# Copyright (c) OpenMMLab. All rights reserved.
"""Derivation of TurboMind's per-module parallel configuration.

Pure stdlib module (no torch / no compiled extension).  Constraint system:

    dev == outer_dp * attn_dp * attn_tp * cp == outer_dp * ep * mlp_tp
    dp  == outer_dp * attn_dp
    attn_tp == tp || mlp_tp == tp
    tp % attn_tp == 0 && tp % mlp_tp == 0

When ``device_num`` is unset, ``dev`` is scanned over the divisors of the
``derive_device_num`` upper bound in descending order (capped by the
available GPU count) and the largest device count admitting a valid layout
wins.  Within a given ``dev``, ``outer_dp``
is scanned over the divisors of ``dp`` in ascending order and the first
valid config wins (smallest ``outer_dp``, i.e. largest valid ``mlp_tp``).
``tp % mlp_tp == 0`` forces ``mlp_tp = 1`` when ``tp == 1``, which
reproduces the legacy replicated-island configs for pure-DP setups without
special-casing.  Configs that cannot satisfy the system are rejected;
notably the legacy spelling ``tp=1, ep=8`` (dp=1) must now be written
``tp=8, ep=8``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd


@dataclass
class ParallelConfig:
    device_num: int
    outer_dp_size: int
    attn_dp_size: int
    attn_tp_size: int
    attn_cp_size: int
    mlp_dp_size: int
    mlp_tp_size: int
    mlp_ep_size: int


def derive_device_num(dp: int, tp: int, ep: int, cp: int) -> int:
    """Upper bound on the device count for the requested parallelism.

    Any valid layout satisfies ``dev == dp * attn_tp * cp`` with
    ``attn_tp | tp`` and ``dev == outer_dp * ep * mlp_tp`` with
    ``outer_dp | dp`` and ``mlp_tp | tp``, so ``dev`` divides both
    ``dp*tp*cp`` and ``dp*tp*ep`` — hence their gcd.
    ``derive_parallel_config`` scans its divisors for the largest
    attainable device count.
    """
    return dp * tp * gcd(cp, ep)


def _try_layout(dp: int, tp: int, ep: int, cp: int, dev: int) -> ParallelConfig | None:
    """Solve the constraint system for a fixed device count; None if
    unsatisfiable."""
    if dev % (dp * cp) != 0:
        return None
    attn_tp = dev // (dp * cp)
    if tp % attn_tp != 0:
        return None
    for outer_dp in (x for x in range(1, dp + 1) if dp % x == 0):
        if dev % (outer_dp * ep) != 0:
            continue
        mlp_tp = dev // (outer_dp * ep)
        if tp % mlp_tp != 0:
            continue
        if attn_tp != tp and mlp_tp != tp:
            continue
        return ParallelConfig(device_num=dev,
                              outer_dp_size=outer_dp,
                              attn_dp_size=dp // outer_dp,
                              attn_tp_size=attn_tp,
                              attn_cp_size=cp,
                              mlp_dp_size=1,
                              mlp_tp_size=mlp_tp,
                              mlp_ep_size=ep)
    return None


def _device_num_candidates(total: int, device_num: int | None, available_devices: int | None) -> list[int]:
    """Device counts to try, in order of preference.

    An explicit ``device_num`` is the only candidate (it must divide the
    ``total`` upper bound); otherwise every divisor of ``total`` within
    ``available_devices``, largest first.
    """
    if device_num is not None:
        assert isinstance(device_num, int) and device_num >= 1, \
            f'device_num must be a positive integer, got {device_num!r}'
        assert total % device_num == 0, (f'config requires {total} devices; '
                                         f'device_num={device_num} is not a divisor')
        return [device_num]
    cap = min(total, available_devices) if available_devices is not None else total
    return [dev for dev in range(cap, 0, -1) if total % dev == 0]


def derive_parallel_config(dp: int,
                           tp: int,
                           ep: int,
                           cp: int,
                           device_num: int | None,
                           available_devices: int | None = None) -> ParallelConfig:
    """Derive per-module parallel sizes from user-facing dp/tp/ep/cp.

    ``device_num`` is the explicit device count; pass None to derive it as
    the largest divisor of ``derive_device_num(...)`` within
    ``available_devices`` that admits a valid layout.  Asserts on invalid
    configs.
    """
    for name, value in (('dp', dp), ('tp', tp), ('ep', ep), ('cp', cp)):
        assert isinstance(value, int) and value >= 1, f'{name} must be a positive integer, got {value!r}'
    assert available_devices is None or (isinstance(available_devices, int) and available_devices >= 1), \
        f'available_devices must be a positive integer or None, got {available_devices!r}'

    total = derive_device_num(dp, tp, ep, cp)
    for dev in _device_num_candidates(total, device_num, available_devices):
        config = _try_layout(dp, tp, ep, cp, dev)
        if config is not None:
            return config

    raise AssertionError(f'no valid parallel config: dp={dp}, tp={tp}, ep={ep}, cp={cp}, '
                         f'device_num={device_num}, available_devices={available_devices}; '
                         f'note the legacy "tp=1, ep=N" spelling must be "tp=N, ep=N"')
