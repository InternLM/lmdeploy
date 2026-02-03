# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import asyncio
import itertools
import weakref
from contextlib import asynccontextmanager
from typing import Any, List, Tuple

from lmdeploy.messages import GenerationConfig, Response
from lmdeploy.serve.core.exceptions import SafeRunException
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class Session:
    """Session for the engine."""

    def __init__(self, session_id: int, session_mgr: SessionManager, **kwargs):
        self.session_id = session_id
        self.prompt: Any = None
        self.response: Response | None = None
        self.history: List[Tuple[Any, str]] = []
        self.gen_config: GenerationConfig | None = None
        self.step: int = 0
        # event to wait for the session to be active
        self._active: asyncio.Event | None = None
        self._handle = None  # inference instance
        self._session_mgr: SessionManager = weakref.ref(session_mgr)
        self.update(**kwargs)

    def update(self, **kwargs):
        """Update the session."""
        self.prompt = kwargs.get('prompt', self.prompt)
        self.gen_config = kwargs.get('gen_config', self.gen_config)
        self.step = kwargs.get('step', self.step)

    def __repr__(self) -> str:
        """Return a string representation of the Session object."""
        return (f'Session(session_id={self.session_id}, '
                f'step={self.step}, history_len={len(self.history)}, '
                f'has_response={self.response is not None}, '
                f'has_gen_config={self.gen_config is not None})')

    def __str__(self) -> str:
        """Return a human-readable string representation of the Session."""
        res = f'Session(id={self.session_id}, step={self.step})'
        if self.history:
            res += '\nHistory:\n'
            for user, assistant in self.history:
                if isinstance(user, list):
                    user = str(user)
                res += f'USER: \n{user}\nASSISTANT: \n{assistant}\n'
        return res

    def reset(self):
        """Reset the session to initial state.

        This method resets all session data (prompt, response, history, etc.) but keeps the session_id.
        """
        self.prompt = None
        self.response = None
        self.history = []
        self.gen_config = None
        self.step = 0
        self._active = None
        self._handle = None
        self._session_mgr = None
        logger.debug(f'Session {self.session_id} has been reset.')

    @asynccontextmanager
    async def request_handle(self):
        if self._handle is not None:
            raise RuntimeError(f'Session {self.session_id} already has an inference instance.')
        logger.debug(f'[request_handle] session {self.session_id} acquiring an instance')

        hnd_pool = self._session_mgr().request_handle_pool
        self._handle = await hnd_pool.get()
        self._active = asyncio.Event()
        logger.debug(f'[request_handle] session {self.session_id} acquired an instance')
        try:
            yield self._handle
        except SafeRunException:
            pass
        except (asyncio.CancelledError, GeneratorExit) as e:
            logger.error(f'[request_handle] session {self.session_id} exception caught: {e}')
            await self._handle.async_cancel(self.session_id)
        except Exception as e:
            logger.error(f'Session {self.session_id} failed to acquire an inference instance: {e}')
            raise e
        finally:
            logger.debug(f'[request_handle] session {self.session_id} releasing the instance')
            # Return inference instance if it was acquired
            if self._handle is not None:
                hnd_pool.put(self._handle)
                self._handle = None
            # MUST set the signal after releasing the instance to avoid race condition
            # refer to async_end method
            self._active.set()

    async def async_abort(self):
        """Abort the session."""
        logger.info(f'[session] Aborting session {self.session_id}')
        if self._handle is not None:
            await self._handle.async_cancel(self.session_id)
        # DO NOT reset the session here because it might be used by other components.
        # Leave the cleanup to the caller.

    async def async_close(self):
        """End the session."""
        logger.info(f'[session] Ending session {self.session_id}')
        if self._handle is not None:
            await self._active.wait()
        async with self.request_handle() as handle:
            try:
                await handle.async_end(self.session_id)
            except (Exception, asyncio.CancelledError, GeneratorExit) as e:
                logger.error(f'[async_end] exception caught: {e}')
        self.reset()

    def abort(self):
        """Abort the session in sync mode."""
        self._run(self.async_abort())

    def close(self):
        """End the session in sync mode."""
        self._run(self.async_close())

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._session_mgr().loop)


class RequestHandlePool:
    """Manages a pool of request handles for concurrent request processing.

    This class maintains a fixed-size pool of request handles that can be reused
    across multiple inference requests. It implements a lazy-initialized queue-based
    pool pattern to efficiently manage handle lifecycle and enable concurrent
    request handling.

    Each session or request should acquire a handle from the pool before inference and
    return it after completion. The manager supports:
    - Pool-based handle allocation and deallocation
    - Lazy initialization of the async queue (required for asyncio.Queue)
    - Handle rebuilding after engine wakeup (e.g., turbomind backend)
    - Complete pool cleanup

    Args:
        engine (AsyncEngine): The async inference engine that creates handles.
        size (int): The size of the handle pool, typically set to max_batch_size.

    Note:
        The pool queue is lazily initialized on first access via `get()` method,
        as `asyncio.Queue` must be created within an async context.
    """

    def __init__(self, engine, size: int):
        self.size = size
        self.handles = [engine.create_instance() for _ in range(size)]
        # `asyncio.Queue` must be created in an async context, refer to `get` method
        self.pool: asyncio.Queue = None

    async def get(self):
        """Get a handle from pool."""
        # Lazy initialization: create pool on first use
        if self.pool is None:
            self.pool = asyncio.Queue()
            for inst in self.handles:
                self.pool.put_nowait(inst)

        return await self.pool.get()

    def put(self, handle):
        """Put a handle back to the pool."""
        if handle is not None and self.pool is not None:
            self.pool.put_nowait(handle)

    def clear(self):
        """Clear all handles."""
        self.handles = []
        self.pool = None


class SessionManager:
    """Session manager."""

    def __init__(self):
        """Initialize the session manager."""

        self.sessions = {}
        self.session_id_generator = itertools.count(1)
        self.request_handle_pool = None
        self.loop = None

    def get(self, session_id: int | None = None, **kwargs) -> Session:
        """Create a new session."""
        session_id = session_id or next(self.session_id_generator)
        if session_id in self.sessions:
            logger.debug(f'[SessionManager] session {session_id} already exists. Updating...')
            session = self.sessions[session_id]
            session.update(**kwargs)
            return session
        else:
            logger.info(f'[SessionManager] session {session_id} not found. Creating...')
            session = Session(session_id, self, **kwargs)
            self.sessions[session_id] = session
            return session

    async def async_abort_all(self):
        """Abort all sessions."""
        tasks = []
        for session in list(self.sessions.values()):
            tasks.append(session.async_abort())
        await asyncio.gather(*tasks, return_exceptions=True)
        # "abort all" is designed for async RL. The aborted sessions will be no longer used,
        # so we reset and clear the sessions here.
        for session in list(self.sessions.values()):
            session.reset()
        self.sessions.clear()

    def has(self, session_id):
        return session_id in self.sessions

    def remove(self, session: Session):
        self.sessions.pop(session.session_id)

    def clear(self):
        self.sessions.clear()
        # reset the session id generator
        self.session_id_generator = itertools.count(1)

    def attach_event_loop(self, loop):
        self.loop = loop

    def build_request_handle_pool(self, engine, size):
        """Build the request handle's pool."""
        self.request_handle_pool = RequestHandlePool(engine, size)
