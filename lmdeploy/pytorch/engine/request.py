# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import enum
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Awaitable, Callable, Dict, List

from lmdeploy.messages import ResponseType
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _raise_exception_on_finish(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as e:
        logger.exception(f'Engine loop failed with error: {e}')


def _ignore_exception_on_finish(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as exc:
        logger.info(f'task: {task.get_name()} ended.')
        logger.debug(f'task: {task.get_name()} exception: {exc}')


class RequestType(enum.Enum):
    """Request type."""

    ADD_SESSION = enum.auto()
    ADD_MESSAGE = enum.auto()
    STOP_SESSION = enum.auto()
    END_SESSION = enum.auto()
    STOP_ENGINE = enum.auto()
    RESUME_ENGINE = enum.auto()


@dataclass
class Request:
    """Request."""

    type: RequestType
    sender_id: int
    req_id: int
    data: Any = None


@dataclass
class Response:
    """Response."""

    type: ResponseType
    sender_id: int
    req_id: int
    data: Any = None
    err_msg: str = ''


ReqList = List[Request]


def _run_until_complete(future: Awaitable):
    """run untile complete."""
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
    _next_req_id: int = 0
    _resp_que: asyncio.Queue = None
    _resp_thread_que: Queue = None

    @classmethod
    def new(cls, sender_id: int, manager: 'RequestManager'):
        """new."""
        return cls(sender_id=sender_id, manager=manager)

    @property
    def resp_que(self):
        """response queue."""
        if self.is_thread_safe():
            return self.manager.responses
        if self.manager._loop_task is None and not self.is_thread_safe():
            self.manager.create_loop_task()
        if self._resp_que is None:
            self._resp_que = asyncio.Queue()
        return self._resp_que

    @property
    def req_que(self):
        """request queue."""
        return self.manager.requests

    @property
    def resp_thread_que(self):
        """response threadsafe queue."""
        if self._resp_thread_que is None:
            self._resp_thread_que = Queue()
        return self._resp_thread_que

    @property
    def req_thread_que(self):
        """request threadsafe queue."""
        return self.manager.thread_requests

    @property
    def event_loop(self):
        """get event loop."""
        return self.manager.event_loop

    def is_thread_safe(self):
        """is thread safe."""
        return self.manager.is_thread_safe()

    def is_loop_alive(self):
        """is loop alive."""
        return self.manager.is_loop_alive()

    def run_until_complete(self, future: Awaitable):
        """run untile complete."""
        return self.manager.run_until_complete(future)

    def _resp_get(self):
        """resp_que.get."""
        timeout = 1

        while True:
            try:
                ret = self.resp_thread_que.get(timeout=timeout)
                return ret
            except Empty:
                if not self.manager.is_loop_alive():
                    logger.debug('Engine loop is not alive.')
                    exit(1)
                continue
            except Exception as e:
                logger.exception(
                    f'sender[{self.sender_id}] get response failed: {e}')
                raise e

    async def _async_resp_get(self):
        """get resp.

        Different behavior in threadsafe mode.
        """
        timeout = 1

        async def __no_threadsafe_get():
            while True:
                try:
                    return await asyncio.wait_for(self.resp_que.get(), timeout)
                except asyncio.TimeoutError:
                    if not self.manager.is_loop_alive():
                        logger.debug('Engine loop is not alive.')
                        exit(1)
                    continue
                except Exception as e:
                    logger.exception(
                        f'sender[{self.sender_id}] get response failed: {e}')
                    raise e

        if self.is_thread_safe():
            ret = self._resp_get()
            await asyncio.sleep(0)
            return ret
        else:
            return await __no_threadsafe_get()

    def _req_put(self, reqs: Any):
        """req put."""
        self.req_thread_que.put(reqs)

    async def _async_req_put(self, reqs: Any):
        """async rq_que put.

        Different behavior in threadsafe mode.
        """
        if self.is_thread_safe():
            self._req_put(reqs)
            await asyncio.sleep(0)
        else:
            await self.req_que.put(reqs)

    def _prefetch_resps(self):
        """prefetch from resp que.

        Different behavior in threadsafe mode.
        """
        if self.is_thread_safe():
            resp_que = self.resp_thread_que
        else:
            resp_que = self.resp_que
        num_resps = resp_que.qsize()
        for _ in range(num_resps):
            resp: Response = resp_que.get_nowait()
            req_id = resp.req_id
            self._push_resp(req_id, resp)

    def _push_resp(self, req_id: int, resp: Response):
        """push response."""
        self.resp_dict.setdefault(req_id, [])
        self.resp_dict[req_id].append(resp)

    def _pop_resp(self, req_id: int, default: Any = None):
        """pop response."""
        if req_id not in self.resp_dict:
            return default
        resps = self.resp_dict[req_id]
        ret = resps.pop(0)
        if len(resps) == 0:
            self.resp_dict.pop(req_id)
        return ret

    def _gather_request(self, req_types: List[RequestType], data: List[Any]):
        """gather requests."""
        if self.manager._loop_task is None and not self.is_thread_safe():
            self.manager.create_loop_task()
        if not self.is_loop_alive():
            logger.error('Engine main loop stopped.')
            exit(1)
        assert len(req_types) == len(data)
        batch_size = len(req_types)

        req_ids = list(range(self._next_req_id,
                             self._next_req_id + batch_size))
        self._next_req_id += batch_size

        reqs = [
            Request(type=rtype,
                    sender_id=self.sender_id,
                    req_id=req_id,
                    data=rdata)
            for req_id, rtype, rdata in zip(req_ids, req_types, data)
        ]
        return req_ids, reqs

    async def async_batched_send_async(self, req_types: List[RequestType],
                                       data: List[Any]):
        """Batched send request asynchronize."""
        req_ids, reqs = self._gather_request(req_types, data)
        await self._async_req_put(reqs)
        return req_ids

    async def async_send_async(self, req_type: RequestType, data: Any):
        """send request asynchronize."""
        return (await self.async_batched_send_async(req_types=[req_type],
                                                    data=[data]))[0]

    def batched_send_async(self, req_types: List[RequestType],
                           data: List[Any]) -> List[int]:
        """Batched send request asynchronize.

        Different behavior in threadsafe mode.
        """
        if not self.is_thread_safe():
            coro = self.async_batched_send_async(req_types, data)
            return self.run_until_complete(coro)

        req_ids, reqs = self._gather_request(req_types, data)
        self._req_put(reqs)
        return req_ids

    def send_async(self, req_type: RequestType, data: Any) -> int:
        """send request asynchronize."""
        return self.batched_send_async(req_types=[req_type], data=[data])[0]

    async def async_recv_any(self, que_timeout: float = None) -> Response:
        """receive any response."""
        self._prefetch_resps()
        for req_id in self.resp_dict:
            ret = self._pop_resp(req_id, default=None)
            if ret is not None:
                return ret
        return await self._async_resp_get()

    def recv_any(self, que_timeout: float = None) -> Response:
        """receive any response."""
        coro = self.async_recv_any(que_timeout)
        return self.run_until_complete(coro)

    def recv_all(self, req_id: int, block: bool = True):
        """revceive all response with req_id."""
        self._prefetch_resps()
        resps = self.resp_dict.pop(req_id, [])
        return resps

    async def async_recv(self,
                         req_id: int,
                         que_timeout: float = None) -> Response:
        """receive response of given request id async."""
        ret = self._pop_resp(req_id, default=None)
        if ret is not None:
            return ret

        # check resp que
        while True:
            resp: Response = await self._async_resp_get()
            if resp.req_id != req_id:
                self._push_resp(req_id, resp)
            else:
                return resp

    def recv(self, req_id: int, que_timeout: float = None) -> Response:
        """receive response of given request id.

        Different behavior in threadsafe mode.
        """
        if not self.is_thread_safe():
            coro = self.async_recv(req_id, que_timeout)
            return self.run_until_complete(coro)

        ret = self._pop_resp(req_id, default=None)
        if ret is not None:
            return ret

        # check resp que
        while True:
            resp: Response = self._resp_get()
            if resp.req_id != req_id:
                self._push_resp(req_id, resp)
            else:
                return resp

    async def async_send(self,
                         req_type: RequestType,
                         data: Any,
                         que_timeout: float = None):
        """send and receive synchronize."""
        req_id = await self.async_send_async(req_type, data)
        return await self.async_recv(req_id, que_timeout=que_timeout)

    def send(self,
             req_type: RequestType,
             data: Any,
             que_timeout: float = None) -> Response:
        """send and receive synchronize."""
        req_id = self.send_async(req_type, data)
        return self.recv(req_id, que_timeout=que_timeout)

    def response_callback(self, resp: Response):
        """response callback."""
        self.resp_que.put_nowait(resp)


class RequestManager:
    """Request manager."""

    def __init__(self, thread_safe: bool = False):
        self.senders: Dict[int, RequestSender] = dict()
        self.callbacks: Dict[RequestType, Callable] = dict()
        self.request_priority: List[RequestType] = [
            RequestType.STOP_ENGINE, RequestType.STOP_SESSION,
            RequestType.END_SESSION, RequestType.ADD_SESSION,
            RequestType.ADD_MESSAGE
        ]
        self.requests: asyncio.Queue = None
        self._loop_task: asyncio.Future = None
        self._loop_coro: Callable = None
        self._thread_safe = thread_safe
        self._next_sender_id = 0
        self._mutex = Lock()
        self._loop_thread: Thread = None

        self.thread_requests: Queue = None
        # every sender has it's own responses, this responses is
        # only used in thread safe mode.
        self.responses: asyncio.Queue = None
        if thread_safe:
            self.thread_requests = Queue()

    def create_loop_task(self):
        """create coro task."""
        logger.debug('creating engine loop task.')
        event_loop = asyncio.get_event_loop()
        assert self._loop_coro is not None, (
            'Please set loop task with manager.start_loop')
        loop_unshielded = event_loop.create_task(self._loop_coro(),
                                                 name='EngineMainLoop')
        loop_unshielded.add_done_callback(_raise_exception_on_finish)
        self._loop_task = asyncio.shield(loop_unshielded)
        self.requests = asyncio.Queue()
        return self._loop_task

    @property
    def event_loop(self):
        """get event loop."""
        if self._loop_task is None:
            return None
        else:
            return self._loop_task.get_loop()

    def is_thread_safe(self):
        """is thread safe."""
        return self._thread_safe

    def start_loop(self, loop: asyncio.Task):
        """start main loop."""
        self._loop_coro = loop

        def __get_thread_reqs():
            """get thread reqs."""
            num_reqs = self.thread_requests.qsize()
            reqs = []
            for _ in range(num_reqs):
                tmp_reqs = self.thread_requests.get_nowait()
                if isinstance(tmp_reqs, Request):
                    tmp_reqs = [tmp_reqs]
                reqs += tmp_reqs
            return reqs

        async def __req_loop():
            """req loop."""
            while True:
                # get reqs
                reqs = __get_thread_reqs()

                if len(reqs) > 0:
                    await self.requests.put(reqs)
                else:
                    await asyncio.sleep(0.02)

        def __put_thread_resps(resps: List[Response]):
            """put thread resps."""
            for resp in resps:
                sender = self.senders.get(resp.sender_id, None)
                if sender is None:
                    continue
                sender.resp_thread_que.put_nowait(resp)

        async def __resp_loop():
            """resp loop."""
            while True:
                num_resps = self.responses.qsize()
                resps = []
                for _ in range(num_resps):
                    resps.append(self.responses.get_nowait())
                if len(resps) > 0:
                    __put_thread_resps(resps)
                else:
                    await asyncio.sleep(0.02)

        def __run_forever(event_loop: asyncio.BaseEventLoop):
            """run forever."""
            logger.debug('start thread run forever.')
            asyncio.set_event_loop(event_loop)
            self.create_loop_task()
            req_loop = event_loop.create_task(__req_loop(),
                                              name='RunForeverReqLoop')
            req_loop.add_done_callback(_ignore_exception_on_finish)
            resp_loop = event_loop.create_task(__resp_loop(),
                                               name='RunForeverRespLoop')
            resp_loop.add_done_callback(_ignore_exception_on_finish)
            self.event_loop.run_forever()

        if self.is_thread_safe():
            event_loop = asyncio.new_event_loop()
            self.responses = asyncio.Queue()
            self._loop_thread = Thread(target=__run_forever,
                                       args=(event_loop, ),
                                       daemon=True)
            self._loop_thread.start()

    def is_loop_alive(self):
        """check if main loop is alive."""

        def __check_threadsafe():
            if self._loop_thread is None:
                return False
            if not self._loop_thread.is_alive():
                return False
            if self._loop_task is None:
                return False
            return not self._loop_task.done()

        if self.is_thread_safe():
            return __check_threadsafe()

        if self._loop_task is None:
            logger.debug('loop task has not been created.')
            return False
        if self._loop_task.get_loop() != asyncio.get_event_loop():
            logger.warning('Current event loop is different from'
                           ' the one bound to loop task!')
            return False
        return not self._loop_task.done()

    def build_sender(self):
        """create a new sender."""
        with self._mutex:
            sender_id = self._next_sender_id
            self._next_sender_id += 1
            new_sender = RequestSender.new(sender_id, self)
            self.senders[sender_id] = new_sender
            return new_sender

    def has_requests(self):
        """has unprocessed request."""
        if self.requests is None:
            return False
        return not self.requests.empty()

    def get_all_requests(self) -> Dict[RequestType, Request]:
        """get all requests in current queue."""
        num_reqs = self.requests.qsize()
        reqs: ReqList = []
        for _ in range(num_reqs):
            elem = self.requests.get_nowait()
            if isinstance(elem, Request):
                elem = [elem]
            reqs += elem

        # gather requests
        reqs_by_type: Dict[RequestType, Request] = dict(
            (t, []) for t in RequestType)
        for req in reqs:
            reqs_by_type[req.type].append(req)
        return reqs_by_type

    def bind_func(self, req_type: RequestType, callback: Callable):
        """bind handler for given request type."""
        self.callbacks[req_type] = callback

    def set_request_priority(self, priority: List[RequestType]):
        """set the priority of request type."""
        self.request_priority = priority

    def response(self, resp: Response):
        """send response."""
        if resp.sender_id not in self.senders:
            logger.warning(f'sender {resp.sender_id} not exist. '
                           f'Send {resp} failed.')
            return
        self.senders[resp.sender_id].response_callback(resp)

    def process_request(self, req_type: RequestType, reqs: ReqList, **kwargs):
        """process reqs with given req type."""
        # get callback
        func = self.callbacks.get(req_type, None)
        if func is not None:
            func(reqs, **kwargs)
        else:
            # TODO: send error message
            for req in reqs:
                resp = Response(ResponseType.HANDLER_NOT_EXIST,
                                sender_id=req.sender_id,
                                req_id=req.req_id,
                                err_msg=(f'callback for {req_type}'
                                         ' not exists.'))
                self.response(resp)

    def step(self, **kwargs):
        """handle requests.

        Should only be called in loop task.
        """
        reqs_by_type = self.get_all_requests()

        # handle requests
        for req_type in self.request_priority:
            # request exists
            if req_type not in reqs_by_type or len(reqs_by_type) == 0:
                continue

            reqs: ReqList = reqs_by_type[req_type]
            self.process_request(req_type, reqs, **kwargs)

    def run_until_complete(self, future: Awaitable):
        """run untile complete."""
        return _run_until_complete(future)
