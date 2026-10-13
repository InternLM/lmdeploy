# Copyright (c) OpenMMLab. All rights reserved.
"""Bounded SM90 launch configurations for GDN ReplaySSM kernels.

These are deliberately offline-selected defaults: production dispatch never
runs a Triton autotuner or allocates a tuning workspace.
"""

STANDARD_SM90_CONFIG_TABLE = (
    (1, dict(BV=32, NK=1, num_warps=1, num_stages=1, dot_precision='ieee')),
    (8, dict(BV=32, NK=2, num_warps=1, num_stages=4, dot_precision='ieee')),
    (32, dict(BV=128, NK=4, num_warps=1, num_stages=4, dot_precision='ieee')),
    (256, dict(BV=128, NK=4, num_warps=1, num_stages=3, dot_precision='ieee')),
    (None, dict(BV=128, NK=4, num_warps=1, num_stages=4, dot_precision='ieee')),
)
# Backward-compatible production-shape alias. New dispatch should call the
# batch-aware helper below.
STANDARD_SM90_CONFIG = STANDARD_SM90_CONFIG_TABLE[3][1]
MTP_RECONSTRUCT_SM90_CONFIG = dict(BV=32, num_warps=1, num_stages=1, dot_precision='ieee')
MTP_EXACT_FOLD_SM90_CONFIG_TABLE = (
    (1, dict(BV=32, num_warps=2, num_stages=3)),
    (None, dict(BV=16, num_warps=1, num_stages=1)),
)
MTP_EXACT_FOLD_SM90_CONFIG = MTP_EXACT_FOLD_SM90_CONFIG_TABLE[1][1]
STANDARD_CURSOR_SM90_CONFIG = dict(BLOCK=128)
MTP_COMMIT_CURSOR_SM90_CONFIG = dict(BLOCK=256)
RESET_CURSOR_SM90_CONFIG = dict(BLOCK=128)


def get_standard_replayssm_config(batch: int) -> dict:
    """Return the offline-selected standard configuration for a batch."""
    for max_batch, config in STANDARD_SM90_CONFIG_TABLE:
        if max_batch is None or batch <= max_batch:
            return config
    raise AssertionError('standard ReplaySSM configuration table is empty')


def get_mtp_exact_fold_config(batch: int) -> dict:
    """Return the offline-selected exact-fold configuration for a batch."""
    for max_batch, config in MTP_EXACT_FOLD_SM90_CONFIG_TABLE:
        if max_batch is None or batch <= max_batch:
            return config
    raise AssertionError('MTP exact-fold configuration table is empty')


def get_replayssm_config(operation: str, major: int | None = None, batch: int | None = None) -> dict:
    """Return a copy of the qualified fixed configuration."""
    configs = {
        'standard': get_standard_replayssm_config(batch) if batch is not None else STANDARD_SM90_CONFIG,
        'mtp_reconstruct': MTP_RECONSTRUCT_SM90_CONFIG,
        'mtp_exact_fold': get_mtp_exact_fold_config(batch) if batch is not None else MTP_EXACT_FOLD_SM90_CONFIG,
        'standard_cursor': STANDARD_CURSOR_SM90_CONFIG,
        'mtp_commit_cursor': MTP_COMMIT_CURSOR_SM90_CONFIG,
        'reset_cursor': RESET_CURSOR_SM90_CONFIG,
    }
    if operation not in configs:
        raise KeyError(f'unknown ReplaySSM operation: {operation}')
    if major is not None and major != 9:
        raise ValueError(f'ReplaySSM launch tables are qualified only for SM90, got SM{major}x.')
    return dict(configs[operation])
