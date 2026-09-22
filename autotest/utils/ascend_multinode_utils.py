"""Ascend multi-node helpers (HCCL / Ray env for lmdeploy).

``MASTER_ADDR`` is passed by the job. ``HCCL_IF_IP`` defaults to the first IP from
``hostname -i`` when unset. Other socket/Ray flags are defaulted below.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any


def _is_ascend(config: dict[str, Any]) -> bool:
    return config.get('device') == 'ascend'


def _hostname_ip() -> str | None:
    """First non-loopback IPv4 from ``hostname -i`` when available."""
    try:
        out = subprocess.check_output(['hostname', '-i'], text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            for ip in out.split():
                if not ip.startswith('127.'):
                    return ip
            return out.split()[0]
    except Exception:
        pass
    return None


def resolve_hccl_if_ip() -> str | None:
    return os.getenv('HCCL_IF_IP') or _hostname_ip()


def _apply_ascend_env(env: dict[str, str]) -> None:
    env['OMP_PROC_BIND'] = 'false'
    env['OMP_NUM_THREADS'] = '1'
    env['TASK_QUEUE_ENABLE'] = '1'
    env['HCCL_CONNECT_TIMEOUT'] = '7200'
    env['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'
    env['HCCL_OP_EXPANSION_MODE'] = 'AI_CPU'
    env['HCCL_BUFFSIZE'] = '512'
    env['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'
    env.setdefault('GLOO_SOCKET_IFNAME', 'eth0')
    env.setdefault('TP_SOCKET_IFNAME', 'eth0')
    env.setdefault('RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES', '1')
    hccl_if_ip = env.get('HCCL_IF_IP') or _hostname_ip()
    if hccl_if_ip:
        env.setdefault('HCCL_IF_IP', hccl_if_ip)

    jemalloc = '/usr/lib64/libjemalloc.so.2'
    existing = env.get('LD_PRELOAD', '')
    if existing:
        if jemalloc not in existing.split(':'):
            env['LD_PRELOAD'] = f'{jemalloc}:{existing}'
    else:
        env['LD_PRELOAD'] = jemalloc


def bootstrap_ascend_session_env(config: dict[str, Any]) -> None:
    """Apply Ascend env before Ray/proxy start (session scope)."""
    if int(os.getenv('NODE_COUNT', '1')) <= 1:
        return
    if not _is_ascend(config):
        return

    env = dict(os.environ)
    _apply_ascend_env(env)
    os.environ.update(env)


def build_ascend_multinode_env(
    config: dict[str, Any],
    run_config: dict[str, Any] | None = None,
    *,
    parallel_config: dict[str, int] | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Merge Ascend env vars into *base_env* (or ``os.environ``) for
    api_server."""
    env = dict(base_env if base_env is not None else os.environ)
    if not _is_ascend(config):
        return env

    _apply_ascend_env(env)

    parallel_config = parallel_config or ((run_config or {}).get('parallel_config') or {})
    dp = int(parallel_config.get('dp', 1) or 1)
    if dp > 1:
        env.setdefault('LMDEPLOY_EXECUTOR_BACKEND', 'ray')

    return env


def ensure_ascend_multinode_env(
    config: dict[str, Any],
    run_config: dict[str, Any] | None = None,
    *,
    parallel_config: dict[str, int] | None = None,
) -> None:
    """Apply Ascend env vars to ``os.environ`` when starting api_server."""
    parallel_config = parallel_config or ((run_config or {}).get('parallel_config') or {})
    merged = build_ascend_multinode_env(config, run_config, parallel_config=parallel_config)
    os.environ.update(merged)
