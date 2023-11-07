# Copyright (c) OpenMMLab. All rights reserved.
import enum
from dataclasses import dataclass
from queue import Queue
from threading import Lock
from typing import Any, Callable, Dict, List

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


class ResponseType(enum.Enum):
    """Response type."""

    SUCCESS = enum.auto()
    FINISH = enum.auto()
    ENGINE_STOP_ERROR = enum.auto()
    SESSION_REPEAT = enum.auto()
    SESSION_NOT_EXIST = enum.auto()
    HANDLER_NOT_EXIST = enum.auto()


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


class RequestSender:
    """Request sender.

    Args:
        sender_id (int): The id of the sender
    """

    def __init__(self, sender_id: int, req_que: Queue):
        self._next_req_id = 0
        self.sender_id = sender_id
        self.req_que = req_que
        self.resp_que = Queue()
        self.resp_dict = dict()

    def batched_send_async(self, req_types: List[RequestType],
                           data: List[Any]) -> List[int]:
        """Batched send request asynchronize."""
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
        for req_id, resps in self.resp_dict.items():
            ret = resps.pop(0)
            if len(resps) == 0:
                self.resp_dict.pop(req_id)
            return ret

        # check resp que
        return self.resp_que.get(timeout=que_timeout)

    def recv(self, req_id: int, que_timeout: float = None) -> Response:
        """receive response of given request id."""
        # check resp dict
        if req_id in self.resp_dict:
            resps = self.resp_dict[req_id]
            ret = resps.pop(0)
            if len(resps) == 0:
                self.resp_dict.pop(req_id)
            return ret

        # check resp que
        while True:
            resp: Response = self.resp_que.get(timeout=que_timeout)
            if resp.req_id != req_id:
                self.resp_dict.setdefault(req_id, [])
                self.resp_dict[req_id].append(resp)
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

    def create_sender(self):
        """create a new sender."""
        with self.mutex:
            sender_id = self._next_sender_id
            self._next_sender_id += 1
            new_sender = RequestSender(sender_id, self.requests)
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

    def step(self, **kwargs):
        """handle requests."""
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

        # handle requests
        for req_type in self.request_priority:
            # request exists
            if req_type not in reqs_by_type or len(reqs_by_type) == 0:
                continue

            cur_reqs: List[Request] = reqs_by_type[req_type]

            # get callback
            func = self.callbacks.get(req_type, None)
            if func is not None:
                func(cur_reqs, **kwargs)
            else:
                # TODO: send error message
                for req in cur_reqs:
                    resp = Response(ResponseType.HANDLER_NOT_EXIST,
                                    sender_id=req.sender_id,
                                    req_id=req.req_id,
                                    err_msg=(f'callback for {req_type}'
                                             ' not exists.'))
                    self.response(resp)
