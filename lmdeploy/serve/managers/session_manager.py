# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import itertools
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from lmdeploy.messages import GenerationConfig, Response
from lmdeploy.serve.core.exceptions import SafeRunException
from lmdeploy.serve.utils import singleton
from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from .request_handle_manager import RequestHandleManager

logger = get_logger('lmdeploy')


class Session:
    """Session for the engine."""

    def __init__(self, session_id: int, **kwargs):
        self.session_id = session_id
        self.prompt: Any = None
        self.response: Optional[Response] = None
        self.history: List[Tuple[Any, str]] = []
        self.gen_config: Optional[GenerationConfig] = None
        self.step: int = 0
        # event to wait for the session to be active
        self._active: Optional[asyncio.Event] = None
        self._hnd = None  # inference instance
        self._hnd_mgr: Optional['RequestHandleManager'] = None
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
        self._hnd = None
        self._hnd_mgr = None
        logger.debug(f'Session {self.session_id} has been reset.')

    @asynccontextmanager
    async def acquire_request_handle(self, req_hnd_mgr: 'RequestHandleManager'):
        if self._hnd is not None:
            raise RuntimeError(f'Session {self.session_id} already has an inference instance.')
        logger.debug(f'[acquire_request_handle] session {self.session_id} acquiring an instance')
        self._hnd_mgr = req_hnd_mgr
        self._hnd = await self._hnd_mgr.get()
        self._active = asyncio.Event()
        logger.debug(f'[acquire_request_handle] session {self.session_id} acquired an instance')
        try:
            yield self._hnd
        except SafeRunException:
            # SafeRunException is raised by AsyncEngine.safe_run. We don't need to handle it here.
            pass
        except Exception as e:
            logger.error(f'Session {self.session_id} failed to acquire an inference instance: {e}')
            raise e
        finally:
            logger.debug(f'[acquire_request_handle] session {self.session_id} releasing the instance')
            # Return inference instance if it was acquired
            if self._hnd is not None:
                req_hnd_mgr.ret(self._hnd)
                self._hnd = None
            # MUST set the signal after releasing the instance to avoid race condition
            # refer to async_end method
            self._active.set()

    async def async_abort(self):
        """Abort the session."""
        logger.info(f'Aborting session {self.session_id}')
        if self._hnd is not None:
            await self._hnd.async_cancel(self.session_id)
        # DO NOT reset the session here because it might be used by other components.
        # Leave the cleanup to the caller.

    async def async_end(self):
        """End the session."""
        logger.debug(f'Ending session {self.session_id}')
        if self._hnd_mgr is None:
            logger.warning(f'Session {self.session_id} has no handle.')
            return

        if self._hnd is not None:
            await self._active.wait()
            if self._hnd is not None:
                raise RuntimeError(f'Session {self.session_id} is not finished yet.')
        handle = await self._hnd_mgr.get()
        try:
            await handle.async_end(self.session_id)
        except (Exception, asyncio.CancelledError, GeneratorExit) as e:  # noqa
            logger.error(f'[async_end] exception caught: {e}')
        finally:
            self._hnd_mgr.ret(handle)
            self.reset()


@singleton
class SessionManager:
    """Session manager."""

    def __init__(self):
        """Initialize the session manager.

        DO KEEP the initializing list empty because it is a singleton class and might be initialized by api_server or
        pipeline or other components.
        """

        self.sessions = {}
        self.session_id_generator = itertools.count()

    def reserve(self) -> int:
        """Reserve a new session id."""
        return next(self.session_id_generator)

    def get(self, session_id: Optional[int] = None, **kwargs) -> Session:
        """Create a new session."""
        if session_id is None:
            session_id = self.reserve()
        if session_id in self.sessions:
            logger.debug(f'[SessionManager] session {session_id} already exists. Updating...')
            session = self.sessions[session_id]
            session.update(**kwargs)
            return session
        else:
            logger.info(f'[SessionManager] session {session_id} not found. Creating...')
            session = Session(session_id, **kwargs)
            self.sessions[session_id] = session
            return session

    async def async_abort(self, session: Session):
        """Abort a session."""
        logger.info(f'Aborting session {session.session_id}')
        await session.async_abort()
        # DO not remove the session here because it might be used by other components.

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

    async def async_end(self, session: Session):
        """End a session."""
        logger.info(f'Ending session {session.session_id}')
        await session.async_end()
        self.sessions.pop(session.session_id)

    def has(self, session_id: int) -> bool:
        """Check if a session exists."""
        return session_id in self.sessions

    def clear(self):
        """Clear all sessions."""
        self.sessions.clear()
        # reset the session id generator
        self.session_id_generator = itertools.count()
