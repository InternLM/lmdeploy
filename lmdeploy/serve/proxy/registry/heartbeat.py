# Copyright (c) OpenMMLab. All rights reserved.

import os
import threading
import time

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

CONTROLLER_HEART_BEAT_EXPIRATION = int(os.getenv('LMDEPLOY_CONTROLLER_HEART_BEAT_EXPIRATION', 90))


def start_heartbeat(pool) -> threading.Thread:
    """Start daemon thread that evicts unhealthy replicas."""

    def _loop():
        while True:
            time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
            logger.info('Start heart beat check')
            pool.remove_stale_replicas()

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread
