"""Ascend multi-node helpers (rank table path, HCCL / Ray env for lmdeploy).

Rank table JSON is resolved in order:

1. ``ASCEND_RANK_TABLE_FILE_PATH`` (already set in the environment)
2. ``run_config['extra_params']['rank-table-file']`` (relative to ``resource_path``)
3. ``{resource_path}/ascend_rank_table.json`` or ``rank_table.json``
4. ``{resource_path}/rank_table/rank_table_{NODE_COUNT}x{devices_per_node}.json``
5. First ``*.json`` under ``{resource_path}/rank_table/``

See: https://github.com/DeepLink-org/dlinfer/blob/main/docs/ascend_multinodes.md
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

DEFAULT_RANK_TABLE_BASENAMES = (
    'ascend_rank_table.json',
    'rank_table.json',
)

RANK_TABLE_EXTRA_KEYS = (
    'rank-table-file',
    'ascend-rank-table-file',
)


def _devices_per_node() -> int:
    explicit = os.getenv('ASCEND_DEVICES_PER_NODE') or os.getenv('PROC_PER_NODE')
    if explicit:
        return max(1, int(explicit))
    visible = os.getenv('ASCEND_RT_VISIBLE_DEVICES', '')
    if visible:
        return max(1, len([x for x in visible.split(',') if x.strip() != '']))
    return 8


def _parallel_world_size(parallel_config: dict[str, int] | None) -> int:
    parallel_config = parallel_config or {}
    ep = int(parallel_config.get('ep', 1) or 1)
    if ep > 1:
        return ep
    tp = int(parallel_config.get('tp', 1) or 1)
    return max(1, tp)


def ascend_needs_rank_table(config: dict[str, Any], parallel_config: dict[str, int] | None = None) -> bool:
    """Return True when Ascend multi-node rank table is required."""
    if config.get('device') != 'ascend':
        return False
    node_count = int(os.getenv('NODE_COUNT', '1'))
    if node_count > 1:
        return True
    world_size = _parallel_world_size(parallel_config)
    return world_size > _devices_per_node()


def _resolve_under_resource_path(config: dict[str, Any], value: str) -> str:
    if os.path.isabs(value):
        return value
    resource_path = config.get('resource_path') or ''
    return os.path.join(resource_path, value)


def resolve_ascend_rank_table_path(
    config: dict[str, Any],
    run_config: dict[str, Any] | None = None,
    *,
    parallel_config: dict[str, int] | None = None,
) -> str | None:
    """Resolve Ascend rank table JSON path; None if not found."""
    existing = os.getenv('ASCEND_RANK_TABLE_FILE_PATH')
    if existing:
        return existing

    if run_config:
        extra = run_config.get('extra_params') or {}
        for key in RANK_TABLE_EXTRA_KEYS:
            candidate = extra.get(key)
            if candidate:
                path = _resolve_under_resource_path(config, str(candidate))
                if os.path.isfile(path):
                    return path
                return path

    resource_path = config.get('resource_path') or ''
    if not resource_path:
        return None

    for name in DEFAULT_RANK_TABLE_BASENAMES:
        path = os.path.join(resource_path, name)
        if os.path.isfile(path):
            return path

    rank_dir = os.path.join(resource_path, 'rank_table')
    if os.path.isdir(rank_dir):
        node_count = int(os.getenv('NODE_COUNT', '1'))
        devices = _devices_per_node()
        for name in (
            f'rank_table_{node_count}x{devices}.json',
            f'ascend_rank_table_{node_count}x{devices}.json',
        ):
            path = os.path.join(rank_dir, name)
            if os.path.isfile(path):
                return path
        json_files = sorted(Path(rank_dir).glob('*.json'))
        if json_files:
            return str(json_files[0])

    return None


def _default_socket_ifname() -> str:
    return os.getenv('HCCL_SOCKET_IFNAME') or os.getenv('GLOO_SOCKET_IFNAME') or 'eth0'


def _hostname_ip() -> str | None:
    """First IPv4 from ``hostname -I`` when available."""
    try:
        import subprocess
        out = subprocess.check_output(['hostname', '-I'], text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            return out.split()[0]
    except Exception:
        pass
    return None


def _rank_table_master_addr(rank_table_path: str | None) -> str | None:
    if not rank_table_path or not os.path.isfile(rank_table_path):
        return None
    try:
        with open(rank_table_path) as f:
            rank_table = json.load(f)
        servers = rank_table.get('server_list') or []
        if servers and servers[0].get('server_id'):
            return str(servers[0]['server_id'])
    except Exception:
        return None
    return None


def _rank_table_local_addr(rank_table_path: str | None) -> str | None:
    """Local ``server_id`` from rank table using ``NODE_RANK``."""
    if not rank_table_path or not os.path.isfile(rank_table_path):
        return None
    node_rank = int(os.getenv('NODE_RANK', '0'))
    try:
        with open(rank_table_path) as f:
            rank_table = json.load(f)
        servers = rank_table.get('server_list') or []
        if 0 <= node_rank < len(servers) and servers[node_rank].get('server_id'):
            return str(servers[node_rank]['server_id'])
    except Exception:
        return None
    return None


def _master_addr(rank_table_path: str | None = None) -> str:
    """Address lmdeploy compares to
    ``rank_table['server_list'][0]['server_id']``."""
    for key in ('LMDEPLOY_DIST_MASTER_ADDR', 'LMDEPLOY_DP_MASTER_ADDR', 'MASTER_ADDR'):
        val = os.getenv(key)
        if val:
            return val
    from_table = _rank_table_master_addr(rank_table_path)
    if from_table:
        return from_table
    return _hostname_ip() or socket.gethostbyname(socket.gethostname())


def _local_addr(rank_table_path: str | None = None) -> str:
    """This node's HCCL / Ray bind IP (worker != master)."""
    for key in ('HCCL_IF_IP', 'RAY_NODE_IP_ADDRESS'):
        val = os.getenv(key)
        if val:
            return val
    from_table = _rank_table_local_addr(rank_table_path)
    if from_table:
        return from_table
    return _hostname_ip() or socket.gethostbyname(socket.gethostname())


def build_ascend_multinode_env(
    config: dict[str, Any],
    run_config: dict[str, Any] | None = None,
    *,
    parallel_config: dict[str, int] | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Merge Ascend multi-node env vars into *base_env* (or ``os.environ``)."""
    env = dict(base_env if base_env is not None else os.environ)
    if config.get('device') != 'ascend':
        return env

    parallel_config = parallel_config or ((run_config or {}).get('parallel_config') or {})
    dp = int(parallel_config.get('dp', 1) or 1)
    ep = int(parallel_config.get('ep', 1) or 1)

    if dp > 1:
        env.setdefault('LMDEPLOY_EXECUTOR_BACKEND', 'ray')
        env.setdefault('RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES', '1')
    if dp > 1 or ep > 1:
        env.setdefault('DEVICE', 'ascend')

    if not ascend_needs_rank_table(config, parallel_config):
        return env

    rank_table = resolve_ascend_rank_table_path(config, run_config, parallel_config=parallel_config)
    if rank_table:
        env['ASCEND_RANK_TABLE_FILE_PATH'] = rank_table

    master = _master_addr(rank_table)
    local = _local_addr(rank_table)
    env.setdefault('MASTER_ADDR', master)
    env.setdefault('LMDEPLOY_DP_MASTER_ADDR', master)
    env.setdefault('LMDEPLOY_DIST_MASTER_ADDR', master)
    env.setdefault('RAY_NODE_IP_ADDRESS', local)
    ifname = _default_socket_ifname()
    env.setdefault('HCCL_SOCKET_IFNAME', ifname)
    env.setdefault('GLOO_SOCKET_IFNAME', ifname)
    env.setdefault('TP_SOCKET_IFNAME', ifname)
    env.setdefault('HCCL_IF_IP', local)
    return env


def ensure_ascend_rank_table(config: dict[str, Any],
                             run_config: dict[str, Any] | None = None,
                             *,
                             parallel_config: dict[str, int] | None = None) -> str | None:
    """Set ``ASCEND_RANK_TABLE_FILE_PATH`` in ``os.environ`` when required."""
    parallel_config = parallel_config or ((run_config or {}).get('parallel_config') or {})
    if not ascend_needs_rank_table(config, parallel_config):
        return os.getenv('ASCEND_RANK_TABLE_FILE_PATH')

    rank_table = resolve_ascend_rank_table_path(config, run_config, parallel_config=parallel_config)
    if not rank_table:
        node_count = int(os.getenv('NODE_COUNT', '1'))
        devices = _devices_per_node()
        resource_path = config.get('resource_path', '')
        raise FileNotFoundError(
            'Ascend multi-node job requires a rank table JSON. '
            f'Place it under {resource_path}/rank_table/rank_table_{node_count}x{devices}.json '
            'or set ASCEND_RANK_TABLE_FILE_PATH.',
        )
    if not os.path.isfile(rank_table):
        raise FileNotFoundError(f'Ascend rank table not found: {rank_table}')
    merged = build_ascend_multinode_env(config, run_config, parallel_config=parallel_config)
    merged['ASCEND_RANK_TABLE_FILE_PATH'] = rank_table
    os.environ.update(merged)
    return rank_table
