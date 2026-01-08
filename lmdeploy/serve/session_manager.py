# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import enum
import itertools
from contextlib import asynccontextmanager
from typing import Any, List, Optional, Tuple

from lmdeploy.messages import GenerationConfig, Response
from lmdeploy.utils import get_logger

from .exceptions import SafeRunException
from .inst_manager import InferInstManager
from .utils import singleton

logger = get_logger('lmdeploy')


class SessionState(enum.Enum):
    """Session state enumeration."""
    IDLE = 'idle'  # Initial state, no active inference
    ACQUIRING = 'acquiring'  # Acquiring an inference instance
    ACTIVE = 'active'  # Has an inference instance and is active


class Session:
    """Session for the engine."""

    def __init__(self, session_id: int, **kwargs):
        self.session_id = session_id
        self.prompt: Any = None
        self.response: Optional[Response] = None
        self.history: List[Tuple[Any, str]] = []
        self.gen_config: Optional[GenerationConfig] = None
        self.step: int = 0
        self._state = SessionState.IDLE
        self._abort_event: Optional[asyncio.Event] = None  # cancellation signal for abort
        self._inst = None  # inference instance
        self.update(**kwargs)

    @property
    def state(self) -> SessionState:
        """Get the current session state."""
        return self._state

    def update(self, **kwargs):
        """Update the session."""
        self.gen_config = kwargs.get('gen_config', self.gen_config)
        self.step = kwargs.get('step', self.step)

    def __repr__(self) -> str:
        res = ''
        for user, assistant in self.history:
            if isinstance(user, list):
                user = str(user)
            res += f'USER: \n{user}\nASSISTANT: \n{assistant}\n'
        return res

    @asynccontextmanager
    async def acquire_inst(self, inst_mgr: InferInstManager):
        """Acquire an inference instance for an session."""
        if self._inst is not None:
            raise RuntimeError(f'Session {self.session_id} already has an inference instance.')

        if self._state != SessionState.IDLE:
            raise RuntimeError(f'Session {self.session_id} is not in IDLE state.')

        self._state = SessionState.ACQUIRING
        get_free_inst = inst_mgr.get()
        get_task = asyncio.create_task(get_free_inst)

        # Create or reuse abort event for cancellation
        self._abort_event = self._abort_event or asyncio.Event()
        self._abort_event.clear()  # Reset event to ensure it's not set

        pending = set()

        try:
            logger.info(f'Session {self.session_id} acquiring an inference instance...')
            abort_task = asyncio.create_task(self._abort_event.wait())
            done, pending = await asyncio.wait([get_task, abort_task], return_when=asyncio.FIRST_COMPLETED)

            inst = None
            if get_task in done:
                try:
                    inst = get_task.result()
                    self._inst = inst
                    self._state = SessionState.ACTIVE
                    logger.info(f'Session {self.session_id} acquired an inference instance.')
                except Exception as e:
                    logger.error(f'Session {self.session_id} failed to get an inference instance: {e}')
                    self._state = SessionState.IDLE
            else:
                # Abort was triggered (abort_task completed)
                logger.info(f'Session {self.session_id} aborted before acquiring an inference instance.')
                # Cancel get_task if it's still pending
                if get_task in pending:
                    get_task.cancel()
                    # Try to get the inference instance if it was already retrieved before cancellation
                    try:
                        await get_task
                    except asyncio.CancelledError:
                        pass
                    # Edge case: Task might have retrieved an inference instance before being cancelled
                    if not get_task.cancelled() and get_task.exception() is None:
                        inst = get_task.result()
                        # Return it immediately if we got it
                        if inst is not None:
                            self.inst_mgr.ret(inst)
                            inst = None
                    # Remove get_task from pending since it's already been handled
                    pending.discard(get_task)
                # abort_task is in done, so it will be cleaned up in finally block

            yield self._inst
        except SafeRunException:
            # SafeRunException is raised by AsyncEngine.safe_run. We don't need to handle it here.
            pass
        except Exception as e:
            logger.error(f'Session {self.session_id} failed to acquire an inference instance: {e}')
            raise e
        finally:
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            # Return inference instance if it was acquired
            if self._inst is not None:
                inst_mgr.ret(self._inst)
                self._inst = None
            # Reset state to IDLE
            self._state = SessionState.IDLE

    async def async_abort(self):
        """Abort the session."""
        if self._state == SessionState.IDLE:
            return

        logger.info(f'Aborting session {self.session_id}')
        if self._state == SessionState.ACTIVE:
            await self._inst.async_cancel(self.session_id)
        elif self._state == SessionState.ACQUIRING:
            # Signal abort by setting the event
            self._abort_event.set()
        else:
            raise RuntimeError(f'Session {self.session_id} is not in ACTIVE or ACQUIRING state.')
        self._state = SessionState.IDLE

    async def async_end(self):
        """End the session."""
        if self._state != SessionState.ACTIVE:
            raise RuntimeError(f'Session {self.session_id} is not in ACTIVE state.')
        logger.debug(f'Ending session {self.session_id}')
        await self._inst.async_end(self.session_id)


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

    def create(self, session_id: Optional[int] = None, **kwargs) -> Session:
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

    async def async_abort_all(self):
        """Abort all sessions."""
        tasks = []
        for session in list(self.sessions.values()):
            tasks.append(session.async_abort())
        await asyncio.gather(*tasks, return_exceptions=True)
        self.sessions.clear()

    async def async_end(self, session: Session):
        """End a session."""
        logger.info(f'Ending session {session.session_id}')
        await session.async_end()

    def has(self, session_id: int) -> bool:
        """Check if a session exists."""
        return session_id in self.sessions
