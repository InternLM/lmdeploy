"""Run interface REST suites with a per-test api_server (GPU-concurrent)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from utils.config_utils import get_workerid
from utils.constant import DEFAULT_PORT
from utils.run_restful_chat import start_openai_service, terminate_restful_api

_AUTOTEST_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _AUTOTEST_ROOT.parent

# Nested protocol suite concurrency (HTTP clients against one api_server).
# Matches historical daily/pr restful ``-n 20``; override via env.
INTERFACE_SUITE_WORKERS_ENV = 'INTERFACE_SUITE_WORKERS'
_DEFAULT_SUITE_WORKERS = 20


def _suite_workers() -> int:
    raw = os.environ.get(INTERFACE_SUITE_WORKERS_ENV, '').strip()
    if not raw:
        return _DEFAULT_SUITE_WORKERS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f'{INTERFACE_SUITE_WORKERS_ENV} must be an int, got {raw!r}',
        ) from exc
    if value < 0:
        raise ValueError(f'{INTERFACE_SUITE_WORKERS_ENV} must be >= 0, got {value}')
    return value


def _scrub_outer_xdist_env(env: dict[str, str]) -> dict[str, str]:
    """Drop outer pytest-xdist identity so nested ``-n`` starts clean."""
    cleaned = dict(env)
    for key in list(cleaned):
        if key.startswith('PYTEST_XDIST') or key == 'PYTEST_CURRENT_TEST':
            cleaned.pop(key, None)
    return cleaned


def _pytest_cmd(
    test_path: str,
    *,
    k_expr: str,
    m_expr: str | None,
    env: dict[str, str],
    n_workers: int,
) -> int:
    """Run a nested pytest against one interface suite file."""
    cmd = [
        sys.executable,
        '-m',
        'pytest',
        test_path,
        '-k',
        k_expr,
        '-q',
        '--tb=line',
        '-p',
        'no:cacheprovider',
    ]
    if m_expr:
        cmd.extend(['-m', m_expr])
    # Concurrent HTTP load against the worker-local api_server (fills GPU).
    if n_workers > 1:
        cmd.extend(['-n', str(n_workers), '--dist=load'])
    print('interface suite cmd:', ' '.join(cmd), flush=True)
    completed = subprocess.run(cmd, env=env, cwd=str(_REPO_ROOT))
    return int(completed.returncode)


def run_interface_restful_test(config, run_config, worker_id) -> None:
    """Start api_server for ``run_config``, then run configured interface
    suites.

    Mirrors ``run_llm_test`` GPU/port isolation via ``worker_id``, but executes
    the existing ``autotest/interface/restful`` protocol suites against the
    worker-local port (``LMDEPLOY_PORT``).
    """
    pid, content = start_openai_service(config, run_config, worker_id)
    try:
        assert pid > 0, f'Failed to start RESTful API server: {content}'
        port = DEFAULT_PORT + get_workerid(worker_id)
        model = run_config['model']
        backend = run_config['backend']
        case_info = run_config.get('case_info') or []
        generate_marker = run_config.get('generate_marker') or f'not not_{backend}'
        n_workers = _suite_workers()

        env = _scrub_outer_xdist_env(os.environ.copy())
        env['LMDEPLOY_PORT'] = str(port)
        env['TEST_ENV'] = env.get('TEST_ENV') or os.environ.get('TEST_ENV', 'a100')
        env['CONFIG_COMPARE_SKIP_MKDIRS'] = '1'
        # Nested collection should see autotest utils.
        py_path = env.get('PYTHONPATH', '')
        pieces = [str(_AUTOTEST_ROOT)]
        if py_path:
            pieces.append(py_path)
        env['PYTHONPATH'] = os.pathsep.join(pieces)

        k_expr = f'{model} and {backend}'
        failures: list[str] = []

        suite_map = [
            (
                'chat_completions_v1',
                'autotest/interface/restful/test_restful_chat_completions_v1.py',
                f'not not_{backend}',
            ),
            (
                'completions_v1',
                'autotest/interface/restful/test_restful_completions_v1.py',
                None,
            ),
            (
                'generate',
                'autotest/interface/restful/test_restful_generate.py',
                generate_marker,
            ),
            (
                'toolcall',
                'autotest/interface/restful/tool_parser/',
                f'tool_call and not not_{backend}',
            ),
            (
                'reasoning',
                'autotest/interface/restful/reasoning_parser/',
                f'reasoning and not not_{backend}',
            ),
        ]
        for case_name, rel_path, marker in suite_map:
            if case_name not in case_info:
                continue
            rc = _pytest_cmd(
                rel_path,
                k_expr=k_expr,
                m_expr=marker,
                env=env,
                n_workers=n_workers,
            )
            if rc != 0:
                failures.append(f'{case_name} (exit={rc})')

        assert not failures, (
            f'interface restful failures for {backend} {model} port={port}: '
            + ', '.join(failures)
        )
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)
