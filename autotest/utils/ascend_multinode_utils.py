"""Ascend multi-node helpers (rank table path, HCCL / Ray env for lmdeploy).

Rank table path is taken only from ``ASCEND_RANK_TABLE_FILE_PATH``. When rank
table is required, set it from the model yaml ``engine_config.extra.rank-table-file``
(relative to ``resource_path``) before multi-node Ray/proxy startup, or export
the env var explicitly in the job.

See: https://github.com/DeepLink-org/dlinfer/blob/main/docs/ascend_multinodes.md
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

RANK_TABLE_EXTRA_KEY = 'rank-table-file'


def _is_ascend(config: dict[str, Any]) -> bool:
    return config.get('device') == 'ascend'


def _devices_per_node() -> int:
    explicit = os.getenv('ASCEND_DEVICES_PER_NODE')
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
    """Return True when Ascend rank table is required."""
    if not _is_ascend(config):
        return False
    node_count = int(os.getenv('NODE_COUNT', '1'))
    if node_count > 1:
        return True
    world_size = _parallel_world_size(parallel_config)
    return world_size > _devices_per_node()


def _resolve_under_resource_path(config: dict[str, Any], value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.join(config['resource_path'], value)


def _rank_table_file_from_run_config(config: dict[str, Any], run_config: dict[str, Any] | None) -> str | None:
    if not run_config:
        return None
    extra = run_config.get('extra_params') or {}
    candidate = extra.get(RANK_TABLE_EXTRA_KEY)
    if candidate:
        return _resolve_under_resource_path(config, str(candidate))
    return None


def _ensure_ascend_rank_table_file_path(
    config: dict[str, Any],
    run_config: dict[str, Any] | None = None,
) -> str:
    """Return rank table path; set ``ASCEND_RANK_TABLE_FILE_PATH`` when
    missing."""
    path = os.getenv('ASCEND_RANK_TABLE_FILE_PATH')
    if not path:
        path = _rank_table_file_from_run_config(config, run_config)
        if not path:
            raise FileNotFoundError(
                'Ascend rank table is required but ASCEND_RANK_TABLE_FILE_PATH is unset. '
                f'Set engine_config.extra.rank-table-file in model yaml (under {config["resource_path"]}) '
                'or export ASCEND_RANK_TABLE_FILE_PATH.',
            )
        os.environ['ASCEND_RANK_TABLE_FILE_PATH'] = path

    if not os.path.isfile(path):
        raise FileNotFoundError(f'Ascend rank table not found: {path}')
    return path


def resolve_ascend_rank_table_path(
    config: dict[str, Any],
    run_config: dict[str, Any] | None = None,
    *,
    parallel_config: dict[str, int] | None = None,
) -> str | None:
    """Return ``ASCEND_RANK_TABLE_FILE_PATH`` when rank table is configured."""
    _ = parallel_config
    path = os.getenv('ASCEND_RANK_TABLE_FILE_PATH')
    if path:
        return path
    return _rank_table_file_from_run_config(config, run_config)


def _default_socket_ifname() -> str:
    return os.getenv('HCCL_SOCKET_IFNAME', 'eth0')


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
    from_table = _rank_table_master_addr(rank_table_path)
    if from_table:
        return from_table
    val = os.getenv('MASTER_ADDR')
    if val:
        return val
    return _hostname_ip() or socket.gethostbyname(socket.gethostname())


def _local_addr(rank_table_path: str | None = None) -> str:
    """This node's HCCL / Ray bind IP (worker != master)."""
    from_table = _rank_table_local_addr(rank_table_path)
    if from_table:
        return from_table
    val = os.getenv('HCCL_IF_IP')
    if val:
        return val
    return _hostname_ip() or socket.gethostbyname(socket.gethostname())


def bootstrap_ascend_session_env(config: dict[str, Any]) -> str | None:
    """Resolve rank table + Pod IPs before Ray/proxy start (session scope).

    K8s often sets ``MASTER_ADDR`` to a hostname; rank table ``server_id`` must
    win so worker Ray join and lmdeploy HCCL use routable Pod IPs.
    """
    if int(os.getenv('NODE_COUNT', '1')) <= 1:
        return None
    if not _is_ascend(config):
        return None

    rank_table = _ensure_ascend_rank_table_file_path(config)

    master = _rank_table_master_addr(rank_table)
    local = _rank_table_local_addr(rank_table)
    if not master or not local:
        raise ValueError(f'rank table missing server_id entries: {rank_table}')

    os.environ['MASTER_ADDR'] = master
    os.environ['LMDEPLOY_DP_MASTER_ADDR'] = master
    os.environ['LMDEPLOY_DIST_MASTER_ADDR'] = master
    os.environ['HCCL_IF_IP'] = local
    os.environ['RAY_NODE_IP_ADDRESS'] = local
    ifname = _default_socket_ifname()
    os.environ.setdefault('HCCL_SOCKET_IFNAME', ifname)
    os.environ.setdefault('GLOO_SOCKET_IFNAME', ifname)
    os.environ.setdefault('TP_SOCKET_IFNAME', ifname)
    os.environ.setdefault('RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES', '1')
    os.environ.setdefault('LMDEPLOY_DP_MASTER_PORT', '29555')
    return rank_table


def build_ascend_multinode_env(
    config: dict[str, Any],
    run_config: dict[str, Any] | None = None,
    *,
    parallel_config: dict[str, int] | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Merge Ascend multi-node env vars into *base_env* (or ``os.environ``)."""
    env = dict(base_env if base_env is not None else os.environ)
    if not _is_ascend(config):
        return env

    parallel_config = parallel_config or ((run_config or {}).get('parallel_config') or {})
    dp = int(parallel_config.get('dp', 1) or 1)

    if dp > 1:
        env.setdefault('LMDEPLOY_EXECUTOR_BACKEND', 'ray')
        env.setdefault('RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES', '1')

    if not ascend_needs_rank_table(config, parallel_config):
        return env

    rank_table = _ensure_ascend_rank_table_file_path(config, run_config)
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

    rank_table = _ensure_ascend_rank_table_file_path(config, run_config)
    merged = build_ascend_multinode_env(config, run_config, parallel_config=parallel_config)
    merged['ASCEND_RANK_TABLE_FILE_PATH'] = rank_table
    os.environ.update(merged)
    return rank_table
