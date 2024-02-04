# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import enum
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Callable, ClassVar, Dict, List

from lmdeploy.messages import ResponseType
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


@dataclass
class RequestSender:
    """Request sender.

    Args:
        sender_id (int): The id of the sender
    """

    sender_id: int
    req_que: Queue
    resp_que: Queue = field(default_factory=Queue)
    resp_dict: Dict[int, List[Response]] = field(default_factory=dict)
    THREAD_ALIVE_INTERVAL: ClassVar[float] = 1.0
    _next_req_id: int = 0
    _thread: Thread = None

    @classmethod
    def new(cls, sender_id: int, req_que: Queue, thread: Thread):
        """new sender."""
        return cls(sender_id=sender_id, req_que=req_que, _thread=thread)

    def _resp_que_get(self, block: bool = True, timeout: float = None):
        """warp of resp_que.get."""
        if not block:
            return self.resp_que(block=block, timeout=timeout)
        timeout_counter = timeout or float(1 << 30)
        while timeout_counter > self.THREAD_ALIVE_INTERVAL:
            try:
                return self.resp_que.get(timeout=self.THREAD_ALIVE_INTERVAL)
            except Empty:
                timeout_counter -= self.THREAD_ALIVE_INTERVAL
            if self._thread and not self._thread.is_alive():
                logger.error('Engine main loop stopped.')
                exit(1)

        return self.resp_que.get(timeout=timeout_counter)

    async def _async_resp_que_get(self,
                                  block: bool = True,
                                  timeout: float = None):
        """warp of resp_que.get."""
        if not block:
            return self.resp_que(block=block, timeout=timeout)
        timeout_counter = timeout or float(1 << 30)
        while timeout_counter > self.THREAD_ALIVE_INTERVAL:
            if self.resp_que.qsize() == 0:
                await asyncio.sleep(self.THREAD_ALIVE_INTERVAL)
                timeout_counter -= self.THREAD_ALIVE_INTERVAL
            else:
                return self.resp_que.get(block=False)
            if self._thread and not self._thread.is_alive():
                logger.error('Engine main loop stopped.')
                exit(1)

        await asyncio.sleep(self.THREAD_ALIVE_INTERVAL)
        return self.resp_que.get(block=False)

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
            resp: Response = self._resp_que_get()
            req_id = resp.req_id
            self._push_resp(req_id, resp)

    def is_thread_alive(self):
        """is thread alive."""
        return self._thread and self._thread.is_alive()

    def batched_send_async(self, req_types: List[RequestType],
                           data: List[Any]) -> List[int]:
        """Batched send request asynchronize."""
        if self._thread and not self._thread.is_alive():
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
        self.req_que.put(reqs)

        return req_ids

    def send_async(self, req_type: RequestType, data: Any) -> int:
        """send request asynchronize."""
        return self.batched_send_async(req_types=[req_type], data=[data])[0]

    def recv_any(self, que_timeout: float = None) -> Response:
        """receive any response."""
        # check resp dict
        self._prefetch_resps()
        for req_id in self.resp_dict:
            ret = self._pop_resp(req_id, default=None)
            if ret is not None:
                return ret

        # check resp que
        return self._resp_que_get(timeout=que_timeout)

    def recv_all(self, req_id: int):
        """revceive all response with req_id."""
        self._prefetch_resps()
        resps = self.resp_dict.pop(req_id, [])
        return resps

    def recv(self, req_id: int, que_timeout: float = None) -> Response:
        """receive response of given request id."""
        # check resp dict
        ret = self._pop_resp(req_id, default=None)
        if ret is not None:
            return ret

        # check resp que
        while True:
            resp: Response = self._resp_que_get(timeout=que_timeout)
            if resp.req_id != req_id:
                self._push_resp(req_id, resp)
            else:
                return resp

    async def async_recv(self,
                         req_id: int,
                         que_timeout: float = None) -> Response:
        """receive response of given request id async."""
        ret = self._pop_resp(req_id, default=None)
        if ret is not None:
            return ret

        # check resp que
        while True:
            resp: Response = await self._async_resp_que_get(timeout=que_timeout
                                                            )
            if resp.req_id != req_id:
                self._push_resp(req_id, resp)
            else:
                return resp

    def send(self,
             req_type: RequestType,
             data: Any,
             que_timeout: float = None) -> Response:
        """send and receive synchronize."""
        req_id = self.send_async(req_type, data)

        return self.recv(req_id, que_timeout=que_timeout)


class RequestManager:
    """Request manager."""

    def __init__(self):
        self._next_sender_id = 0
        self.senders: Dict[int, RequestSender] = dict()
        self.callbacks: Dict[RequestType, Callable] = dict()
        self.request_priority: List[RequestType] = [
            RequestType.STOP_ENGINE, RequestType.STOP_SESSION,
            RequestType.END_SESSION, RequestType.ADD_SESSION,
            RequestType.ADD_MESSAGE
        ]
        self.requests = Queue()
        self.mutex = Lock()

    def build_sender(self, thread: Thread = None):
        """create a new sender."""
        with self.mutex:
            sender_id = self._next_sender_id
            self._next_sender_id += 1
            new_sender = RequestSender.new(sender_id, self.requests, thread)
            self.senders[sender_id] = new_sender
            return new_sender

    def bind_func(self, req_type: RequestType, callback: Callable):
        """bind handler for given request type."""
        self.callbacks[req_type] = callback

    def set_request_priority(self, priority: List[RequestType]):
        """set the priority of request type."""
        self.request_priority = priority

    def has_requests(self):
        """has unprocessed request."""
        return not self.requests.empty()

    def response(self, resp: Response, timeout: float = None):
        """send response."""
        if resp.sender_id not in self.senders:
            logger.warning(f'sender {resp.sender_id} not exist. '
                           f'Send {resp} failed.')
            return
        resp_que = self.senders[resp.sender_id].resp_que
        resp_que.put(resp, timeout=timeout)

    def get_all_requests(self) -> Dict[RequestType, Request]:
        """get all requests in current queue."""
        num_reqs = self.requests.qsize()
        reqs: List[Request] = []
        tmp = num_reqs
        while tmp:
            tmp -= 1
            elem = self.requests.get()
            if isinstance(elem, Request):
                elem = [elem]
            reqs += elem

        # gather requests
        reqs_by_type: Dict[RequestType, Request] = dict(
            (t, []) for t in RequestType)
        for req in reqs:
            reqs_by_type[req.type].append(req)
        return reqs_by_type

    def process_request(self, req_type, reqs, **kwargs):
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

            reqs: List[Request] = reqs_by_type[req_type]
            self.process_request(req_type, reqs, **kwargs)
