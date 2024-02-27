# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from lmdeploy.messages import ResponseType
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _raise_exception_on_finish(task: asyncio.Task) -> None:
    msg = ('Engine loop failed!')
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception as exc:
        raise RuntimeError(msg) from exc


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

    @classmethod
    def new(cls, sender_id: int, manager: 'RequestManager'):
        """new."""
        return cls(sender_id=sender_id, manager=manager)

    @property
    def resp_que(self):
        if self.manager._loop_task is None:
            self.manager.create_loop_task()
        if self._resp_que is None:
            self._resp_que = asyncio.Queue()
        return self._resp_que

    @property
    def req_que(self):
        return self.manager.requests

    async def _async_resp_get(self):
        """get resp."""
        timeout = 1
        while True:
            if not self.manager.is_loop_alive():
                logger.debug('Engine loop is not alive.')
                exit(1)
            try:
                return await asyncio.wait_for(self.resp_que.get(), timeout)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                raise e

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

    def _prefetch_resps(self):
        """prefetch from resp que."""
        num_resps = self.resp_que.qsize()
        for _ in range(num_resps):
            resp: Response = self.resp_que.get_nowait()
            req_id = resp.req_id
            self._push_resp(req_id, resp)

    def is_thread_alive(self):
        """is thread alive."""
        return self._thread and self._thread.is_alive()

    def _gather_request(self, req_types: List[RequestType], data: List[Any]):
        """gather requests."""
        if self.manager._loop_task is None:
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
        await self.req_que.put(reqs)
        return req_ids

    async def async_send_async(self, req_type: RequestType, data: Any):
        """send request asynchronize."""
        return (await self.async_batched_send_async(req_types=[req_type],
                                                    data=[data]))[0]

    def batched_send_async(self, req_types: List[RequestType],
                           data: List[Any]) -> List[int]:
        """Batched send request asynchronize."""
        coro = self.async_batched_send_async(req_types, data)
        return asyncio.get_event_loop().run_until_complete(coro)

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
        return asyncio.get_event_loop().run_until_complete(
            self.async_recv_any(que_timeout))

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
        """receive response of given request id."""
        coro = self.async_recv(req_id, que_timeout)
        return asyncio.get_event_loop().run_until_complete(coro)

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

    def response_callback(self, resp: Response, timeout: float = None):
        """response callback."""
        self.resp_que.put_nowait(resp)

    def is_loop_alive(self):
        """is loop alive."""
        return self.manager.is_loop_alive()


class RequestManager:
    """Request manager."""

    def __init__(self):
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
        self._next_sender_id = 0

    def create_loop_task(self):
        """create coro task."""
        logger.debug('creating engine loop task.')
        loop_unshielded = asyncio.get_event_loop().create_task(
            self._loop_coro())
        loop_unshielded.add_done_callback(_raise_exception_on_finish)
        self._loop_task = asyncio.shield(loop_unshielded)
        self.requests = asyncio.Queue()
        return self._loop_task

    def start_loop(self, loop: asyncio.Task):
        """start main loop."""
        self._loop_coro = loop

    def is_loop_alive(self):
        """check if main loop is alive."""
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

    def response(self, resp: Response, timeout: float = None):
        """send response."""
        if resp.sender_id not in self.senders:
            logger.warning(f'sender {resp.sender_id} not exist. '
                           f'Send {resp} failed.')
            return
        self.senders[resp.sender_id].response_callback(resp, timeout=timeout)

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
        """handle requests."""
        reqs_by_type = self.get_all_requests()

        # handle requests
        for req_type in self.request_priority:
            # request exists
            if req_type not in reqs_by_type or len(reqs_by_type) == 0:
                continue

            reqs: ReqList = reqs_by_type[req_type]
            self.process_request(req_type, reqs, **kwargs)
