# Copyright (c) OpenMMLab. All rights reserved.
"""Shared utilities for the Mooncake Store worker components."""

from __future__ import annotations

import socket
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import torch

StoreFactory = Callable[[], Any]

def _load_mooncake_store_factory() -> StoreFactory:
    """Import Mooncake only in a worker that actually enables the connector."""
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as e:
        raise ImportError(
            'MooncakeStoreConnector requires the mooncake-transfer-engine package. '
            'Install it before enabling the connector.') from e
    return MooncakeDistributedStore


def _load_mooncake_replicate_config() -> Any:
    """Construct the default replication policy only when the first put
    runs."""
    try:
        from mooncake.store import ReplicateConfig
    except ImportError as e:
        raise ImportError(
            'Mooncake KV-cache saving requires ReplicateConfig from the '
            'mooncake-transfer-engine package.') from e
    return ReplicateConfig()


def _get_local_hostname() -> str:
    """Resolve a routable local address, including on offline hosts."""
    candidates = (
        (socket.AF_INET, ('8.8.8.8', 80)),
        (socket.AF_INET6, ('2001:4860:4860::8888', 80)),
    )
    for family, remote_address in candidates:
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as sock:
                sock.connect(remote_address)
                return str(sock.getsockname()[0])
        except OSError:
            continue

    hostname = socket.gethostname()
    try:
        addresses = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except OSError:
        addresses = ()
    for _family, _type, _proto, _canonname, sockaddr in addresses:
        address = str(sockaddr[0])
        if address not in ('127.0.0.1', '::1'):
            return address
    if hostname:
        return hostname
    raise RuntimeError('cannot determine the local hostname for Mooncake Store')


def _is_tensor(value: object) -> bool:
    """Keep the production tensor check strict while allowing test patching."""
    return isinstance(value, torch.Tensor)


def _result_histogram(results: Sequence[int]) -> dict[int, int]:
    return dict(sorted(Counter(results).items()))
