# Copyright (c) OpenMMLab. All rights reserved.

LATENCY_DEQUE_LEN = 15


class APIServerException(Exception):

    def __init__(self, status_code: int, body: bytes, headers: dict | None = None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        if 'content-type' not in self.headers:
            self.headers['content-type'] = 'application/json'
