# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest
import requests
from PIL import UnidentifiedImageError

from lmdeploy.vl.media.connection import _load_http_url
from lmdeploy.vl.media.image import ImageMediaIO


@pytest.mark.parametrize('outcome', ['success', 'http_error', 'decode_error', 'timeout',
                                    'initial_blocked', 'redirect_blocked', 'redirect_loop', 'missing_location'])
def test_media_fetch_closes_its_http_session_on_every_exit(outcome):
    """A request-owned connection pool must not depend on garbage
    collection."""
    session = MagicMock()
    session.__enter__.return_value = session
    session.__exit__.side_effect = lambda *args: session.close() and False
    response = MagicMock(is_redirect=False, content=b'pixels')
    session.get.return_value = response
    media = MagicMock()
    media.load_bytes.return_value = 'decoded'
    safe = [(True, '')] * 5
    expected = None
    if outcome == 'http_error':
        response.raise_for_status.side_effect = requests.HTTPError('404')
        expected = requests.HTTPError
    elif outcome == 'decode_error':
        media.load_bytes.side_effect = UnidentifiedImageError('not an image')
        expected = UnidentifiedImageError
    elif outcome == 'timeout':
        session.get.side_effect = requests.Timeout('test timeout')
        expected = requests.Timeout
    elif outcome == 'initial_blocked':
        safe = [(False, 'private address')]
        expected = ValueError
    elif outcome in ('redirect_blocked', 'redirect_loop', 'missing_location'):
        response.is_redirect = True
        response.headers = {} if outcome == 'missing_location' else {'Location': 'https://example.com/next.png'}
        if outcome == 'redirect_blocked':
            safe = [(True, ''), (False, 'private redirect')]
        expected = ValueError
    with patch('requests.Session', return_value=session), \
            patch('lmdeploy.vl.media.connection._is_safe_url', side_effect=safe):
        if expected:
            with pytest.raises(expected):
                _load_http_url(urlparse('https://example.com/image.png'), media)
        else:
            assert _load_http_url(urlparse('https://example.com/image.png'), media) == 'decoded'
    session.close.assert_called_once()


@pytest.mark.parametrize('scheme', ['http', 'https'])
def test_image_fetch_timeout_propagates_without_decoder(monkeypatch, scheme):
    monkeypatch.setenv('LMDEPLOY_IMAGE_FETCH_TIMEOUT', '2')
    media = ImageMediaIO()
    with patch('lmdeploy.vl.media.connection._is_safe_url', return_value=(True, '')):
        with patch('requests.Session.get', side_effect=requests.Timeout('test timeout')) as fetch:
            with pytest.raises(requests.Timeout):
                _load_http_url(urlparse(f'{scheme}://example.com/image.png'), media)
    assert fetch.call_args.kwargs['timeout'] == 2
    assert fetch.call_args.kwargs['allow_redirects'] is False


def test_http_error_is_not_decoded_as_image():
    media = MagicMock()
    response = MagicMock(is_redirect=False)
    response.raise_for_status.side_effect = requests.HTTPError('404')
    with patch('lmdeploy.vl.media.connection._is_safe_url', return_value=(True, '')):
        with patch('requests.Session.get', return_value=response):
            with pytest.raises(requests.HTTPError):
                _load_http_url(urlparse('https://example.com/missing.png'), media)
    media.load_bytes.assert_not_called()


def test_successful_non_image_body_is_rejected():
    response = MagicMock(is_redirect=False, content=b'not an image')
    with patch('lmdeploy.vl.media.connection._is_safe_url', return_value=(True, '')):
        with patch('requests.Session.get', return_value=response):
            with pytest.raises(UnidentifiedImageError):
                _load_http_url(urlparse('https://example.com/readme.txt'), ImageMediaIO())
