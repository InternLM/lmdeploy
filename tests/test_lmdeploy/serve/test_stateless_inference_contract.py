import asyncio
import inspect

import pytest

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.core.exceptions import SafeRunException
from lmdeploy.serve.managers import Session, SessionManager


class _CountingHandle:

    def __init__(self):
        self.cancelled = 0
        self.ended = 0

    async def async_cancel(self, session_id: int):
        self.cancelled += 1

    async def async_end(self, session_id: int):
        self.ended += 1

    async def async_stream_infer(self, session_id: int, **kwargs):
        yield EngineOutput(ResponseType.SUCCESS, [1])
        raise RuntimeError('boom')


def test_async_engine_generate_signature_is_stateless():
    params = inspect.signature(AsyncEngine.generate).parameters

    assert 'sequence_start' not in params
    assert 'sequence_end' not in params
    assert 'step' not in params


def test_session_has_no_step_state():
    session = Session(0, SessionManager())

    assert not hasattr(session, 'step')
    assert 'step=' not in repr(session)
    assert 'step=' not in str(session)


async def _run_safe_run_error_cleanup():
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.backend = 'pytorch'
    handle = _CountingHandle()
    session = Session(123, SessionManager())

    with pytest.raises(SafeRunException):
        async with engine.safe_run(handle, session=session) as gen:
            async for _ in gen:
                pass

    assert handle.cancelled == 1
    assert handle.ended == 0


def test_safe_run_cancels_without_backend_specific_end():
    asyncio.run(_run_safe_run_error_cleanup())
