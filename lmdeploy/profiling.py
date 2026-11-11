# Copyright (c) OpenMMLab. All rights reserved.
"""Opt-in phase records for correlating serving, paging and worker costs.

Records contain identifiers and sizes, never prompt text or tensor payloads. Callers guard expensive metadata
construction with PROFILE_PHASES.
"""
import json
import os
import time

from lmdeploy.utils import get_logger

PROFILE_PHASES = os.getenv('LMDEPLOY_PROFILE_PHASES', '0') == '1'
logger = get_logger('lmdeploy')


def trace_phase(stage: str, **fields) -> None:
    """Emit a wall-clock correlation point; durations use caller-owned
    clocks."""
    if PROFILE_PHASES:
        logger.info('PHASE_TRACE %s', json.dumps(dict(stage=stage, timestamp=time.time(), pid=os.getpid(), **fields)))
