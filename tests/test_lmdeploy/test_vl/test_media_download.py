# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise the real requests/urllib3 streaming and redirect machinery."""
import gzip
from io import BytesIO
from unittest.mock import Mock

import pytest
import requests
from requests.adapters import BaseAdapter
from urllib3.exceptions import ReadTimeoutError
from urllib3.response import HTTPResponse

from lmdeploy.vl.media import connection


class CountingBody(BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.bytes_read = 0

    def read(self, size=-1):
        data = super().read(size)
        self.bytes_read += len(data)
        return data

    def read1(self, size=-1):
        return self.read(size)


class MediaAdapter(BaseAdapter):
    def __init__(self):
        self.replies = []
        self.bodies = []
        self.requests = []
        self.closed = False

    def send(self, request, **kwargs):
        self.requests.append(request)
        status, headers, data = self.replies.pop(0)
        response = requests.Response()
        response.status_code = status
        response.headers.update(headers)
        response.url, response.request = request.url, request
        body = CountingBody(data)
        self.bodies.append(body)
        response.raw = HTTPResponse(body=body, headers=headers, preload_content=False)
        return response

    def close(self):
        self.closed = True


@pytest.fixture
def transport(monkeypatch):
    session, adapter = requests.Session(), MediaAdapter()
    session.mount('https://', adapter)
    monkeypatch.setattr(connection.requests, 'Session', lambda: session)
    monkeypatch.setattr(connection, '_is_safe_url', lambda url: (True, 'test transport'))
    return adapter


def load(media_io, **kwargs):
    return connection.load_from_url('https://media.example/image', media_io, **kwargs)


@pytest.mark.parametrize('size', [0, 65536, 65537])
def test_download_accepts_exact_budget_and_closes(transport, size):
    data = b'x' * size
    transport.replies = [(200, {}, data)]
    decoder = Mock()
    assert load(decoder, max_download_bytes=size) is decoder.load_bytes.return_value
    decoder.load_bytes.assert_called_once_with(data)
    assert transport.bodies[0].bytes_read == size
    assert transport.bodies[0].closed and transport.closed
    assert transport.requests[0].headers['Accept-Encoding'] == 'identity'


@pytest.mark.parametrize('headers', [
    {}, {'Content-Length': 'invalid'}, {'Transfer-Encoding': 'chunked'},
    {'Content-Length': '1', 'Transfer-Encoding': 'chunked'},
])
def test_unknown_length_download_stops_without_buffering_whole_body(transport, headers):
    transport.replies = [(200, headers, b'x' * (11 * 1024**2))]
    decoder = Mock()
    with pytest.raises(ValueError, match='byte limit'):
        load(decoder, max_download_bytes=10 * 1024**2)
    assert 10 * 1024**2 < transport.bodies[0].bytes_read <= 10 * 1024**2 + 65536
    assert transport.bodies[0].closed and transport.closed
    decoder.load_bytes.assert_not_called()


def test_declared_oversize_is_rejected_without_reading_body(transport):
    transport.replies = [(200, {'Content-Length': str(11 * 1024**2)}, b'x' * (11 * 1024**2))]
    decoder = Mock()
    with pytest.raises(ValueError, match='byte limit'):
        load(decoder, max_download_bytes=10 * 1024**2)
    assert transport.bodies[0].bytes_read == 0
    assert transport.bodies[0].closed and transport.closed
    decoder.load_bytes.assert_not_called()


def test_one_byte_over_budget_is_rejected_before_decode(transport):
    transport.replies = [(200, {}, b'123456789')]
    decoder = Mock()
    with pytest.raises(ValueError, match='byte limit'):
        load(decoder, max_download_bytes=8)
    decoder.load_bytes.assert_not_called()
    assert transport.bodies[0].closed and transport.closed


def test_bounded_download_rejects_compressed_body_before_decompression(transport):
    data = gzip.compress(b'x' * (11 * 1024**2))
    transport.replies = [(200, {'Content-Encoding': 'gzip', 'Content-Length': str(len(data))}, data)]
    decoder = Mock()
    with pytest.raises(ValueError, match='Content-Encoding'):
        load(decoder, max_download_bytes=10 * 1024**2)
    assert transport.bodies[0].bytes_read == 0
    assert transport.bodies[0].closed and transport.closed
    decoder.load_bytes.assert_not_called()


def test_unbounded_callers_keep_compressed_download_support(transport):
    transport.replies = [(200, {'Content-Encoding': 'gzip'}, gzip.compress(b'image'))]
    decoder = Mock()
    load(decoder)
    decoder.load_bytes.assert_called_once_with(b'image')
    assert transport.bodies[0].closed and transport.closed


def test_redirect_body_is_not_read_by_requests_before_url_validation(transport):
    transport.replies = [
        (302, {'Location': '/final', 'Content-Length': str(11 * 1024**2)}, b'x' * (11 * 1024**2)),
        (200, {'Content-Encoding': 'identity'}, b'image'),
    ]
    decoder = Mock()
    load(decoder, max_download_bytes=10, allowed_media_domains=['media.example'])
    decoder.load_bytes.assert_called_once_with(b'image')
    assert [body.bytes_read for body in transport.bodies] == [0, 5]
    assert all(body.closed for body in transport.bodies) and transport.closed
    assert transport.requests[1].url == 'https://media.example/final'
    assert all(request.headers['Accept-Encoding'] == 'identity' for request in transport.requests)


@pytest.mark.parametrize('location', ['https://blocked.example/image', 'file:///secret'])
def test_bounded_redirect_still_checks_target_policy(transport, location):
    transport.replies = [(302, {'Location': location}, b'redirect body')]
    decoder = Mock()
    with pytest.raises(ValueError, match='allowed domains|Unsupported scheme'):
        load(decoder, max_download_bytes=10, allowed_media_domains=['media.example'])
    assert len(transport.requests) == 1 and transport.bodies[0].bytes_read == 0
    assert transport.bodies[0].closed and transport.closed
    decoder.load_bytes.assert_not_called()


def test_redirect_limit_closes_all_responses(transport):
    transport.replies = [(302, {'Location': '/loop'}, b'redirect body')] * 4
    with pytest.raises(ValueError, match='maximum media URL redirects'):
        load(Mock(), max_download_bytes=10)
    assert len(transport.requests) == 4
    assert all(body.closed and body.bytes_read == 0 for body in transport.bodies) and transport.closed


@pytest.mark.parametrize('failure', ['http', 'decode'])
def test_download_closes_on_errors(transport, failure):
    transport.replies = [(404 if failure == 'http' else 200, {}, b'invalid')]
    decoder = Mock()
    decoder.load_bytes.side_effect = ValueError('decode failed')
    with pytest.raises((requests.HTTPError, ValueError)):
        load(decoder, max_download_bytes=10)
    assert transport.bodies[0].closed and transport.closed


def test_download_closes_on_stream_failure(transport, monkeypatch):
    transport.replies = [(200, {}, b'x' * 131072)]
    read = CountingBody.read

    def fail_midway(body, size=-1):
        if body.bytes_read:
            raise ReadTimeoutError(None, None, 'test timeout')
        return read(body, size)

    monkeypatch.setattr(CountingBody, 'read', fail_midway)
    decoder = Mock()
    with pytest.raises(requests.RequestException, match='test timeout'):
        load(decoder, max_download_bytes=131072)
    assert transport.bodies[0].closed and transport.closed
    decoder.load_bytes.assert_not_called()


@pytest.mark.parametrize('budget', [-1, True, 1.5])
def test_invalid_budget_does_not_start_download(transport, budget):
    with pytest.raises(ValueError, match='max_download_bytes'):
        load(Mock(), max_download_bytes=budget)
    assert not transport.requests
