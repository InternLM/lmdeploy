# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import asyncio

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


async def cleanup_result_generators(result_generators, sessions, session_mgr):
    """Close engine generators and remove API sessions idempotently."""
    for generator in result_generators:
        try:
            await generator.aclose()
        except (asyncio.CancelledError, GeneratorExit):
            pass
        except Exception:
            logger.exception('Close result generator failed.')
    for session in sessions:
        session_mgr.remove(session)


async def with_request_cleanup(generator, result_generators, sessions, session_mgr):
    """Yield from an API generator and cleanup when the HTTP task exits."""
    try:
        async for item in generator:
            yield item
    finally:
        cleanup_task = asyncio.create_task(
            cleanup_result_generators(result_generators, sessions, session_mgr),
            name='api_request_cleanup')
        try:
            await asyncio.shield(cleanup_task)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception:
            logger.exception('API request cleanup failed.')
