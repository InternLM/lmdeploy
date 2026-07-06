# Copyright (c) OpenMMLab. All rights reserved.
import ipaddress
import os
import socket
from pathlib import Path
from typing import TypeVar
from urllib.parse import ParseResult, urljoin, urlparse
from urllib.request import url2pathname

import requests
from urllib3.util import parse_url

from .base import MediaIO
from .image import ImageMediaIO
from .video import VideoMediaIO

_M = TypeVar('_M')

headers = {
    'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Check if the URL is safe to fetch (not internal/private)."""
    try:
        parsed = parse_url(url)
        if parsed.scheme not in ('http', 'https'):
            return False, f'Unsupported scheme: {parsed.scheme}'

        hostname = parsed.hostname
        if not hostname:
            return False, 'Could not parse hostname from URL'
        if hostname.startswith('[') and hostname.endswith(']'):
            hostname = hostname[1:-1]

        # check all IPs (IPv4 + IPv6) using getaddrinfo
        try:
            infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            return False, 'Hostname resolution failed'

        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            # block any IP that is not globally routable (covers private, loopback,
            # link-local, multicast, reserved, unspecified, etc.)
            if not ip.is_global:
                return False, f'Blocked non-global IP detected: {ip}'

        return True, 'URL is safe'
    except Exception as e:
        return False, f'URL validation failed: {str(e)}'


def _load_http_url(url_spec: ParseResult, media_io: MediaIO[_M],
                   allowed_media_domains: list[str] | None = None) -> _M:
    url = url_spec.geturl()
    fetch_timeout = 10
    if isinstance(media_io, ImageMediaIO):
        fetch_timeout = int(os.environ.get('LMDEPLOY_IMAGE_FETCH_TIMEOUT', 10))
    elif isinstance(media_io, VideoMediaIO):
        fetch_timeout = int(os.environ.get('LMDEPLOY_VIDEO_FETCH_TIMEOUT', 30))

    client = requests.Session()
    max_redirects = 3
    for _ in range(max_redirects + 1):
        parsed = parse_url(url)
        if parsed.scheme not in ('http', 'https'):
            raise ValueError(f'URL is blocked for security reasons: Unsupported scheme: {parsed.scheme}')
        if not parsed.hostname:
            raise ValueError('URL is blocked for security reasons: Could not parse hostname from URL')
        if allowed_media_domains and parsed.hostname not in allowed_media_domains:
            raise ValueError('The URL must be from one of the allowed domains: '
                             f'{allowed_media_domains}. Input URL domain: {parsed.hostname}')

        is_safe, reason = _is_safe_url(parsed.url)
        if not is_safe:
            raise ValueError(f'URL is blocked for security reasons: {reason}')

        response = client.get(parsed.url, headers=headers, timeout=fetch_timeout, allow_redirects=False)

        if response.is_redirect:
            location = response.headers.get('Location')
            if not location:
                raise ValueError('Redirect response missing Location header')
            url = urljoin(parsed.url, location)
            continue

        response.raise_for_status()
        return media_io.load_bytes(response.content)

    raise ValueError('Exceeded maximum media URL redirects')


def _load_data_url(url_spec: ParseResult, media_io: MediaIO[_M]) -> _M:
    url_spec_path = url_spec.path or ''
    data_spec, data = url_spec_path.split(',', 1)
    media_type, data_type = data_spec.split(';', 1)
    # media_type starts with a leading "/" (e.g., "/video/jpeg")
    media_type = media_type.lstrip('/')

    if data_type != 'base64':
        msg = 'Only base64 data URLs are supported for now.'
        raise NotImplementedError(msg)

    return media_io.load_base64(media_type, data)


def _load_file_url(url_spec: ParseResult, media_io: MediaIO[_M]) -> _M:
    url_spec_path = url_spec.path or ''
    url_spec_netloc = url_spec.netloc or ''
    filepath = Path(url2pathname(url_spec_netloc + url_spec_path))
    return media_io.load_file(filepath)


def load_from_url(url: str, media_io: MediaIO[_M], allowed_media_domains: list[str] | None = None) -> _M:
    """Load media from a HTTP, data or file url."""
    url_spec = urlparse(url)

    if url_spec.scheme and url_spec.scheme.startswith('http'):
        return _load_http_url(url_spec, media_io, allowed_media_domains)

    if url_spec.scheme == 'data':
        return _load_data_url(url_spec, media_io)

    # file url or raw file path (absolute or relative)
    if url_spec.scheme == 'file' or os.path.exists(url) or os.path.exists(url_spec.path):
        return _load_file_url(url_spec, media_io)

    msg = 'The URL must be either a HTTP, data or file URL.'
    raise ValueError(msg)
