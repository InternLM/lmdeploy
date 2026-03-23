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

    # test redirect blocked
    mock_get.return_value = MagicMock(is_redirect=True)
    with pytest.raises(ValueError, match='Redirects are not allowed'):
        _load_http_url(url_spec, media_io)

    # test success
    mock_get.return_value = MagicMock(is_redirect=False, content=b'data', status_code=200)
    media_io.load_bytes.return_value = 'loaded'
    assert _load_http_url(url_spec, media_io) == 'loaded'
    assert mock_get.call_args.kwargs['allow_redirects'] is False
