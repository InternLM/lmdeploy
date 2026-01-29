# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Coroutine, Dict, List

from lmdeploy.messages import RequestMetrics, ResponseType
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class RequestType(enum.Enum):
    """Request type."""

    ADD_SESSION = enum.auto()
    ADD_MESSAGE = enum.auto()
    STOP_SESSION = enum.auto()
    END_SESSION = enum.auto()
    STOP_ENGINE = enum.auto()
    RESUME_ENGINE = enum.auto()


@dataclass
class Response:
    """Response."""

    type: ResponseType
    sender_id: int
    event: asyncio.Event
    data: Any = None
    err_msg: str = ''
    is_done: bool = False
    req_metrics: RequestMetrics = None


@dataclass
class Request:
    """Request."""

    type: RequestType
    sender_id: int
    data: Any = None
    resp: Response = None


ReqList = List[Request]


def _run_until_complete(future: Awaitable):
    """Run untile complete."""
    try:
        event_loop = asyncio.get_event_loop()
    except Exception:
        logger.warning('Can not found event loop in current thread.'
                       ' Create a new event loop.')
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
    return event_loop.run_until_complete(future)


@dataclass
class RequestSender:
    """Request sender.

    Args:
        sender_id (int): The id of the sender
    """
    sender_id: int
    manager: 'RequestManager'
    resp_dict: Dict[int, List[Response]] = field(default_factory=dict)

    @classmethod
    def new(cls, sender_id: int, manager: 'RequestManager'):
        """new."""
        obj = cls(sender_id=sender_id, manager=manager)
        return obj

    @property
    def req_que(self):
        """Request queue."""
        return self.manager.requests

    @property
    def event_loop(self):
        """Get event loop."""
        return self.manager.event_loop

    def is_loop_alive(self):
        """Is loop alive."""
        return self.manager.is_loop_alive()

    def run_until_complete(self, future: Awaitable):
        """Run untile complete."""
        return self.manager.run_until_complete(future)

    def _req_put(self, reqs: Any):
        """Async rq_que put."""
        self.req_que.put_nowait(reqs)

    def _gather_request(self, req_types: List[RequestType], data: List[Any]):
        """Gather requests."""
        if self.manager._loop_task is None:
            self.manager.create_loop_task()
        assert len(req_types) == len(data)

        reqs = []
        resps = []
        for rtype, rdata in zip(req_types, data):
            event = asyncio.Event()
            resp = Response(type=ResponseType.INTERNAL_ENGINE_ERROR,
                            sender_id=self.sender_id,
                            event=event,
                            data=None,
                            err_msg=None)
            req = Request(type=rtype, sender_id=self.sender_id, data=rdata, resp=resp)
            resps.append(resp)
            reqs.append(req)
        return resps, reqs

    def batched_send_async(self, req_types: List[RequestType], data: List[Any]):
        """Batched send request asynchronize."""
        resps, reqs = self._gather_request(req_types, data)
        self._req_put(reqs)
        return resps

    def send_async(self, req_type: RequestType, data: Any):
        """Send request asynchronize."""
        return self.batched_send_async(req_types=[req_type], data=[data])[0]

    async def async_recv(self, resp: Response, wait_main: bool = False) -> Response:
        """Receive response of given request id async."""
        if wait_main:
            await self.manager.prepare_send()
        event = resp.event
        while not event.is_set():
            try:
                await asyncio.wait_for(event.wait(), 1)
            except asyncio.TimeoutError:
                if self.is_loop_alive():
                    continue
                logger.debug('Engine main loop failed.')
                resp.type = ResponseType.ENGINE_STOP_ERROR
                break
        event.clear()
        return resp

    def recv(self, resp: Response) -> Response:
        """Receive response of given request id."""
        coro = self.async_recv(resp)
        return self.run_until_complete(coro)

    async def async_send(self, req_type: RequestType, data: Any):
        """Send and receive synchronize."""
        resp = self.send_async(req_type, data)
        return await self.async_recv(resp)

    def send(self, req_type: RequestType, data: Any) -> Response:
        """Send and receive synchronize."""
        resp = self.send_async(req_type, data)
        return self.recv(resp)


class RequestManager:
    """Request manager."""

    def __init__(self):
        self.senders: Dict[int, RequestSender] = dict()
        self.callbacks: Dict[RequestType, Callable] = dict()
        self.request_priority: List[RequestType] = [
            RequestType.STOP_ENGINE, RequestType.ADD_SESSION, RequestType.STOP_SESSION, RequestType.END_SESSION,
            RequestType.ADD_MESSAGE
        ]
        self.requests: asyncio.Queue = None
        self._loop_task: asyncio.Future = None
        self._loop_coro: Callable = None
        self._next_sender_id = 0

        # sender speed limiter
        self._condition: asyncio.Condition = None
        self._sender_wait_task: asyncio.Task = None
        self._send_count = 0
        self._send_event = None

    async def prepare_send(self):
        if self._condition is None:
            return

        self._send_count += 1
        self._send_event.set()
        async with self._condition:
            await self._condition.wait()
        self._send_count -= 1
        if self._send_count == 0:
            self._send_event.clear()

    async def sender_wait_loop(self):
        """Wait for loop to be created."""
        self._condition = asyncio.Condition()
        self._send_count = 0
        self._send_event = asyncio.Event()

        try:
            while True:
                await self._send_event.wait()
                # notify one sender to control send speed
                async with self._condition:
                    self._condition.notify()
                await asyncio.sleep(0.0001)
        finally:
            # notify all senders to exit
            async with self._condition:
                self._condition.notify_all()
            self._condition = None
            self._send_event = None

    def create_loop_task(self):
        """Create coro task."""
        if self._loop_task is not None:
            logger.debug('loop task has been created.')
            return self._loop_task
        logger.debug('creating engine loop task.')
        event_loop = asyncio.get_event_loop()
        assert self._loop_coro is not None, ('Please set loop task with manager.start_loop')
        loop_unshielded = event_loop.create_task(self._loop_coro(), name='EngineMainLoop')
        self._loop_task = loop_unshielded
        self._sender_wait_task = event_loop.create_task(self.sender_wait_loop(), name='SenderWaitLoop')
        self.requests = asyncio.Queue()
        return self._loop_task

    async def wait_tasks(self):
        """Wait for loop task and sender wait task to finish."""
        if self._loop_task is None:
            return

        try:
            await self._loop_task
        except asyncio.CancelledError:
            logger.info('Engine main loop task has been cancelled.')
            raise
        finally:
            if self._sender_wait_task is not None:
                self._sender_wait_task.cancel()
                try:
                    await self._sender_wait_task
                except Exception:
                    logger.debug('Sender wait task has been cancelled.')

    @property
    def event_loop(self):
        """Get event loop."""
        if self._loop_task is None:
            return None
        else:
            return self._loop_task.get_loop()

    def set_main_loop_func(self, loop: Callable[[Coroutine], asyncio.Task]):
        """Start main loop."""
        self._loop_coro = loop

    def stop_loop(self):
        if self.is_loop_alive():
            self._loop_task.cancel()
        self._loop_task = None
        if self._sender_wait_task is not None:
            self._sender_wait_task.cancel()
            self._sender_wait_task = None

    def is_loop_alive(self):
        """Check if main loop is alive."""

        if self._loop_task is None:
            logger.debug('loop task has not been created.')
            return False
        if self._loop_task.get_loop() != asyncio.get_event_loop():
            logger.warning('Current event loop is different from'
                           ' the one bound to loop task!')
            return False
        return not self._loop_task.done()

    def build_sender(self):
        """Create a new sender."""
        sender_id = self._next_sender_id
        self._next_sender_id += 1
        new_sender = RequestSender.new(sender_id, self)
        self.senders[sender_id] = new_sender
        return new_sender

    def has_requests(self):
        """Has unprocessed request."""
        if self.requests is None:
            return False
        return not self.requests.empty()

    async def get_all_requests(self) -> Dict[RequestType, List[Request]]:
        """Get all requests in current queue."""
        num_reqs = self.requests.qsize()
        reqs: ReqList = []

        def __proc_reqs(elem):
            """Proc reqs."""
            nonlocal reqs
            if isinstance(elem, Request):
                elem = [elem]
            reqs += elem

        if num_reqs == 0:
            elem = await self.requests.get()
            __proc_reqs(elem)
            num_reqs = self.requests.qsize()

        for _ in range(num_reqs):
            elem = self.requests.get_nowait()
            __proc_reqs(elem)

        # gather requests
        reqs_by_type: Dict[RequestType, List[Request]] = dict((t, []) for t in RequestType)
        for req in reqs:
            reqs_by_type[req.type].append(req)
        return reqs_by_type

    def bind_func(self, req_type: RequestType, callback: Callable):
        """Bind handler for given request type."""
        self.callbacks[req_type] = callback

    def set_request_priority(self, priority: List[RequestType]):
        """Set the priority of request type."""
        self.request_priority = priority

    def response(self, resp: Response):
        """Send response."""
        resp.event.set()

    def process_request(self, req_type: RequestType, reqs: ReqList, **kwargs):
        """Process reqs with given req type."""
        # get callback
        func = self.callbacks.get(req_type, None)
        if func is not None:
            func(reqs, **kwargs)
        else:
            # TODO: send error message
            for req in reqs:
                resp = req.resp
                resp.type = ResponseType.HANDLER_NOT_EXIST
                resp.err_msg = (f'callback for {req_type}'
                                ' not exists.')
                self.response(resp)

    async def step(self, **kwargs):
        """Handle requests.

        Should only be called in loop task.
        """

        def _log_reqs(reqs: ReqList):
            num_reqs = len(reqs)
            if num_reqs == 0:
                return
            logger_level = logger.level
            if logger_level <= logging.DEBUG:
                sender_id = [req.sender_id for req in reqs]
                logger.debug(f'Receive {req_type.name} Request: senders: {sender_id}')
            elif logger_level <= logging.INFO:
                logger.info(f'Receive {req_type.name} Request: {num_reqs}')

        reqs_by_type = await self.get_all_requests()

        # handle requests
        for req_type in self.request_priority:
            reqs: ReqList = reqs_by_type.get(req_type, [])
            if not reqs:
                continue

            _log_reqs(reqs)
            self.process_request(req_type, reqs, **kwargs)

    def run_until_complete(self, future: Awaitable):
        """Run untile complete."""
        return _run_until_complete(future)
