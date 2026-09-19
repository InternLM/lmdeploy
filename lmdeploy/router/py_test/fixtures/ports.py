# Copyright (c) OpenMMLab. All rights reserved.
import socket


def find_free_port() -> int:
    """Return an available TCP port on localhost."""
    return find_free_ports(1)[0]


def find_free_ports(count: int) -> list[int]:
    """Return unique available TCP ports, binding all sockets before release.

    Keeping the sockets open while allocating avoids the common race where independent port requests return two sockets
    with the same ephemeral port.
    """
    sockets = []
    ports: set[int] = set()
    try:
        while len(ports) < count:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(sock)
            sock.bind(('127.0.0.1', 0))
            ports.add(sock.getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()
    return list(ports)
