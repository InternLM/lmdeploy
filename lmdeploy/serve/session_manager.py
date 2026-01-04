# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import itertools
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

from lmdeploy.messages import GenerationConfig, Response
from lmdeploy.utils import get_logger

from .inst_manager import InferInstManager
from .utils import singleton

if TYPE_CHECKING:
    from lmdeploy.serve.async_engine import AsyncEngine

logger = get_logger('lmdeploy')


class Session:
    """Session for the engine."""

    def __init__(self, session_id: int, engine: 'AsyncEngine', inst_mgr: InferInstManager, **kwargs):
        self.session_id = session_id
        self.engine = engine
        self.prompt: Any = None
        self.inst_mgr = inst_mgr
        self._response: Response = None
        self.gen_config: GenerationConfig = None
        self.step: int = 0
        self.abort_event = asyncio.Event()  # event to signal the session is aborted
        self.active_event = asyncio.Event(
        )  # event to signal the session is active, from getting an inference instance to finishing the inference
        self.inst = None  # inference instance
        self.generator = None  # generator for streaming response
        self.history: List[Tuple[Any, str]] = []
        self.update(**kwargs)

    def update(self, **kwargs):
        """Update the session."""
        self.gen_config = kwargs.get('gen_config', self.gen_config)
        self.step = kwargs.get('step', self.step)
        if 'response' in kwargs:
            self._response = kwargs['response']

    @property
    def response(self) -> Response:
        """Return response."""
        return self._response

    def close_chat(self):
        """Close the chat session."""
        if self.engine and self.prompt and self.inst:
            self.engine._run(coro=self.inst.end_session(self.session_id)).result()
            self.engine = None

    def stop(self):
        if self.engine and self.prompt:
            self.engine._run(coro=self.engine.stop_session(self.session_id)).result()

    def __repr__(self) -> str:
        res = ''
        for user, assistant in self.history:
            if isinstance(user, list):
                user = str(user)
            res += f'USER: \n{user}\nASSISTANT: \n{assistant}\n'
        return res

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_chat()

    def __call__(self,
                 prompt: str,
                 gen_config: Optional[GenerationConfig] = None,
                 stream_response: bool = True,
                 do_preprocess: bool = True,
                 adapter_name: str = None,
                 **kwargs) -> Union[Response, Iterator[Response]]:
        self.engine.chat(prompt,
                         gen_config=gen_config or self.gen_config,
                         stream_response=stream_response,
                         do_preprocess=do_preprocess,
                         session=self,
                         adapter_name=adapter_name,
                         **kwargs)
        if stream_response:
            # self.generator is assigned in self.engine.chat
            return self.generator
        else:
            return self.response

    @asynccontextmanager
    async def acquire_inst(self):
        """Acquire an inference instance for the session."""
        if self.inst is not None:
            raise RuntimeError(f'Session {self.session_id} already has an inference instance.')

        get_free_inst = self.inst_mgr.get()
        get_task = asyncio.create_task(get_free_inst)
        wait_task = asyncio.create_task(self.abort_event.wait())

        try:
            logger.info(f'Session {self.session_id} acquiring an inference instance...')
            done, pending = await asyncio.wait([get_task, wait_task], return_when=asyncio.FIRST_COMPLETED)

            inst = None
            if get_task in done:
                try:
                    inst = get_task.result()
                    self.inst = inst
                    self.active_event = asyncio.Event()
                    logger.info(f'Session {self.session_id} acquired an inference instance.')
                except Exception as e:
                    logger.error(f'Session {self.session_id} failed to get an inference instance: {e}')
            else:
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

            yield self.inst
        finally:
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            # Return inference instance if it was acquired
            if self.inst is not None:
                self.inst_mgr.ret(self.inst)
                self.inst = None
            if self.active_event is not None:
                self.active_event.clear()

    async def abort(self):
        """Abort the session."""
        logger.info(f'Aborting session {self.session_id}')
        self.abort_event.set()
        if self.generator is not None:
            try:
                await self.generator.async_cancel(self.session_id)
            except Exception as e:
                logger.error(f'Error cancelling generator for session {self.session_id}: {e}')
            finally:
                # Return generator to pool
                self.inst_mgr.ret(self.generator)
                self.generator = None


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
        self.inst_mgr = None

    def attach(self, inst_mgr: InferInstManager):
        """Attach a generator manager to the session manager."""
        self.inst_mgr = inst_mgr

    def reserve(self) -> int:
        """Reserve a new session id."""
        return next(self.session_id_generator)

    def create(self, engine: 'AsyncEngine', session_id: Optional[int] = None, **kwargs) -> Session:
        """Create a new session."""
        if self.inst_mgr is None:
            raise ValueError('InferInstance manager is not attached to the session manager.')
        session_id = session_id or next(self.session_id_generator)
        if session_id in self.sessions:
            logger.info(f'[SessionManager] session {session_id} already exists. Updating...')
            session = self.sessions[session_id]
            session.update(**kwargs)
            return session
        else:
            logger.info(f'[SessionManager] session {session_id} not found. Creating...')
            session = Session(session_id, engine, self.inst_mgr, **kwargs)
            self.sessions[session_id] = session
            return session

    async def abort(self, session: Session):
        """Abort a session."""
        logger.info(f'Aborting session {session.session_id}')
        await session.abort()

    async def abort_all(self):
        """Abort all sessions."""
        tasks = []
        for session in list(self.sessions.values()):
            tasks.append(session.abort())
        await asyncio.gather(*tasks, return_exceptions=True)
        self.sessions.clear()

    async def end(self, session: Session):
        """End a session."""
        logger.info(f'Ending session {session.session_id}')
        await session.end()

    def has(self, session_id: int) -> bool:
        """Check if a session exists."""
        return session_id in self.sessions
