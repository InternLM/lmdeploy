import socket
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest

from lmdeploy.vl.media.connection import _is_safe_url, _load_http_url


@pytest.mark.parametrize(
    'url,expected_safe,mock_ips',
    [
        ('https://github.com', True, ['140.82.112.3']),  # Public domain
        ('http://8.8.8.8', True, ['8.8.8.8']),  # Public IPv4
        ('ftp://example.com', False, []),  # Forbidden scheme
        ('http://127.0.0.1', False, ['127.0.0.1']),  # IPv4 loopback
        ('http://localhost', False, ['127.0.0.1']),  # Resolves to loopback
        ('http://169.254.169.254', False, ['169.254.169.254']),  # Cloud metadata service
        ('http://[::1]', False, ['::1']),  # IPv6 loopback
        ('http://[fc00::1]', False, ['fc00::1']),  # IPv6 unique local address
        ('http://mixed-dns.com', False, ['1.1.1.1', '10.0.0.1']),  # DNS Rebinding
        ('http://', False, []),  # Empty host
        ('http://invalid-host-name', False, None),  # Invalid hostname (simulate DNS failure)
    ])
def test_is_safe_url(url, expected_safe, mock_ips):
    with patch('socket.getaddrinfo') as mock_gai:
        if mock_ips is None:
            # simulate DNS resolution failure
            mock_gai.side_effect = socket.gaierror('Hostname resolution failed')
        else:
            mock_gai.return_value = [(socket.AF_INET if '.' in ip else socket.AF_INET6, None, None, None, (ip, 80))
                                     for ip in mock_ips]
        is_safe, _ = _is_safe_url(url)
        assert is_safe == expected_safe


@patch('requests.Session.get')
@patch('lmdeploy.vl.media.connection._is_safe_url', return_value=(True, ''))
def test_load_http_url_logic(mock_safe, mock_get):
    media_io = MagicMock()
    url_spec = urlparse('https://example.com/img.jpg')

    # test success with manual redirect handling disabled for the request
    mock_get.return_value = MagicMock(content=b'data', status_code=200, is_redirect=False)
    media_io.load_bytes.return_value = 'loaded'
    assert _load_http_url(url_spec, media_io) == 'loaded'
    assert mock_get.call_args.args == ('https://example.com/img.jpg', )
    assert mock_get.call_args.kwargs['allow_redirects'] is False


@patch('requests.Session.get')
@patch('lmdeploy.vl.media.connection._is_safe_url', return_value=(True, ''))
def test_load_http_url_allowed_media_domains_exact_match(mock_safe, mock_get):
    media_io = MagicMock()
    media_io.load_bytes.return_value = 'loaded'
    mock_get.return_value = MagicMock(content=b'data', status_code=200, is_redirect=False)

    assert _load_http_url(urlparse('https://example.com/img.jpg'), media_io,
                          allowed_media_domains=['example.com']) == 'loaded'


@patch('requests.Session.get')
@patch('lmdeploy.vl.media.connection._is_safe_url', return_value=(True, ''))
def test_load_http_url_allowed_media_domains_rejects_subdomain(mock_safe, mock_get):
    media_io = MagicMock()

    with pytest.raises(ValueError, match='allowed domains'):
        _load_http_url(urlparse('https://cdn.example.com/img.jpg'), media_io, allowed_media_domains=['example.com'])

    mock_get.assert_not_called()


@patch('requests.Session.get')
@patch('lmdeploy.vl.media.connection._is_safe_url', return_value=(True, ''))
def test_load_http_url_rejects_backslash_host_confusion(mock_safe, mock_get):
    media_io = MagicMock()

    with pytest.raises(ValueError, match='allowed domains'):
        _load_http_url(urlparse(r'https://example.com\@safe.example.org/img.jpg'),
                       media_io,
                       allowed_media_domains=['safe.example.org'])

    mock_get.assert_not_called()


@patch('requests.Session.get')
def test_load_http_url_allowed_domain_still_blocks_non_global_ip(mock_get):
    media_io = MagicMock()

    with patch('socket.getaddrinfo', return_value=[(socket.AF_INET, None, None, None, ('127.0.0.1', 80))]):
        with pytest.raises(ValueError, match='Blocked non-global IP detected'):
            _load_http_url(urlparse('http://localhost/img.jpg'), media_io, allowed_media_domains=['localhost'])

    mock_get.assert_not_called()


def _mock_response(content=b'data', *, redirect_location=None, status_code=None):
    status_code = status_code if status_code is not None else (302 if redirect_location is not None else 200)
    response = MagicMock(content=content, status_code=status_code)
    response.is_redirect = redirect_location is not None
    response.headers = {}
    if redirect_location is not None:
        response.headers['Location'] = redirect_location
    return response


@patch('requests.Session.get')
@patch('lmdeploy.vl.media.connection._is_safe_url', return_value=(True, ''))
def test_load_http_url_validates_allowed_domain_after_redirect(mock_safe, mock_get):
    media_io = MagicMock()
    media_io.load_bytes.return_value = 'loaded'
    mock_get.side_effect = [
        _mock_response(redirect_location='https://example.com/final.jpg'),
        _mock_response(content=b'final'),
    ]

    assert _load_http_url(urlparse('https://example.com/img.jpg'), media_io,
                          allowed_media_domains=['example.com']) == 'loaded'
    assert mock_get.call_count == 2
    assert mock_get.call_args_list[1].args == ('https://example.com/final.jpg', )


@patch('requests.Session.get')
@patch('lmdeploy.vl.media.connection._is_safe_url', return_value=(True, ''))
def test_load_http_url_blocks_disallowed_domain_after_redirect(mock_safe, mock_get):
    media_io = MagicMock()
    mock_get.return_value = _mock_response(redirect_location='https://evil.example/final.jpg')

    with pytest.raises(ValueError, match='allowed domains'):
        _load_http_url(urlparse('https://example.com/img.jpg'), media_io, allowed_media_domains=['example.com'])

    assert mock_get.call_count == 1


@patch('requests.Session.get')
@patch('lmdeploy.vl.media.connection._is_safe_url', return_value=(True, ''))
def test_load_http_url_does_not_treat_all_3xx_as_redirect(mock_safe, mock_get):
    media_io = MagicMock()
    media_io.load_bytes.return_value = 'loaded'
    mock_get.return_value = _mock_response(content=b'cached', status_code=304)

    assert _load_http_url(urlparse('https://example.com/img.jpg'), media_io) == 'loaded'
    media_io.load_bytes.assert_called_once_with(b'cached')


@patch('requests.Session.get')
@patch('lmdeploy.vl.media.connection._is_safe_url', return_value=(True, ''))
def test_load_http_url_rejects_redirect_without_location(mock_safe, mock_get):
    media_io = MagicMock()
    mock_get.return_value = MagicMock(content=b'', status_code=302, is_redirect=True, headers={})

    with pytest.raises(ValueError, match='Redirect response missing Location header'):
        _load_http_url(urlparse('https://example.com/img.jpg'), media_io)


@patch('requests.Session.get')
@patch('lmdeploy.vl.media.connection._is_safe_url',
       side_effect=[(True, ''), (False, 'Blocked non-global IP detected: 127.0.0.1')])
def test_load_http_url_blocks_unsafe_url_after_redirect(mock_safe, mock_get):
    media_io = MagicMock()
    mock_get.return_value = _mock_response(redirect_location='http://127.0.0.1/final.jpg')

    with pytest.raises(ValueError, match='Blocked non-global IP detected'):
        _load_http_url(urlparse('https://example.com/img.jpg'),
                       media_io,
                       allowed_media_domains=['example.com', '127.0.0.1'])

    assert mock_get.call_count == 1
