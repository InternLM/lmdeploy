# # Copyright (c) OpenMMLab. All rights reserved.
# import asyncio
# from contextlib import asynccontextmanager
# from typing import TYPE_CHECKING, Any

# from lmdeploy.utils import get_logger

# from .session_manager import Session
# from .utils import singleton

# logger = get_logger('lmdeploy')

# @singleton
# class GeneratorManager:
#     """Manages Generator objects, which are also referred to as inference
#     instances."""

#     # TODO(lvhan): define a base class for engine to avoid `Any` type
#     def __init__(self, engine: Any, size: int):
#         self.engine = engine
#         self.size = size
#         self.generators = [self.engine.create_instance() for _ in range(size)]
#         # `asyncio.Queue` must be created in an async context, refer to `get` method
#         self.pool: asyncio.Queue = None
#         # # mapping from session's id to its assigned generator
#         # self.id2generator = dict()
#         self.session_mgr = None

#     @asynccontextmanager
#     async def apply(self, session: Session):
#         """Get a free instance from the pool."""
#         # if session_id in self.id2generator:
#         #     raise RuntimeError(f"Session {session_id} is already assigned to a generator.")
#         # session = self.session_mgr.get(session_id)
#         # if session is None:
#         #     raise RuntimeError(f"Session {session_id} does not exist.")
#         if session.generator is not None:
#             raise RuntimeError(f'Session {session.session_id} already has a generator assigned.')

#         # lazy create pool
#         if self.pool is None:
#             self.pool = asyncio.Queue()
#             for gen in self.generators:
#                 self.pool.put_nowait(gen)

#         generator = None
#         get_generator_task = asyncio.create_task(self.pool.get())
#         abort_event = session.abort_event
#         wait_abort_task = asyncio.create_task(abort_event.wait())

#         try:
#             # Wait for either an instance or an abort signal
#             done, pending = await asyncio.wait([get_generator_task, wait_abort_task],
#                                                return_when=asyncio.FIRST_COMPLETED)

#             # Check if we actually got an instance even if aborted
#             if get_generator_task in done:
#                 generator = get_generator_task.result()

#             if abort_event.is_set():
#                 logger.debug(f'[GeneratorManager] session {session_id} aborted while waiting')
#                 if generator is not None:
#                     logger.debug('[GeneratorManager] generator retrieved during abort, returning to queue.')
#                     self.pool.put_nowait(generator)
#                 else:
#                     # If we didn't get it yet, cancel the pending get_generator_task
#                     get_generator_task.cancel()
#                     try:
#                         await get_generator_task
#                     except asyncio.CancelledError:
#                         pass
#                     # Edge case: Task might finish during cancellation
#                     if not get_generator_task.cancelled() and get_generator_task.exception() is None:
#                         self.pool.put_nowait(get_generator_task.result())
#                 # yield None to indicate the session is aborted and failed to acquire an instance
#                 yield None
#                 return

#             # Clean up the unused abort waiter
#             wait_abort_task.cancel()
#             try:
#                 await wait_abort_task
#             except asyncio.CancelledError:
#                 pass

#             generator._active = asyncio.Event()
#             try:
#                 yield generator
#             # TODO(lvhan): handle exception raised by safe_run
#             except (Exception, asyncio.CancelledError, GeneratorExit) as e:
#                 logger.error(f'[GeneratorManager] session {session_id} exception caught: {e}')
#             finally:
#                 self.id2generator.pop(session_id, None)
#                 generator._active.set()
#                 self.pool.put_nowait(generator)
#         finally:
#             # clean up abort event
#             logger.debug(f'[GeneratorManager] session {session_id} releasing abort event')
#             self.session_mgr.release_abort_event(session_id)

#     def get_session_generator(self, session_id: int):
#         """Get the generator assigned to the given session."""
#         return self.id2generator.get(session_id, None)

#     async def finish(self, generator):
#         await generator._active.wait()

#     @property
#     def generators(self):
#         return self.id2generator
