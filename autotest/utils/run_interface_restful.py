"""Run interface REST suites with a per-test api_server (GPU-concurrent)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from utils.config_utils import get_workerid
from utils.constant import (
    DEFAULT_PORT,
    PROXY_PORT,
    RESTFUL_BASE_MODEL_LIST,
    RESTFUL_MODEL_LIST,
    TOOL_REASONING_MODEL_LIST,
)
from utils.proxy_distributed_utils import ApiServerPerTest, proxy_worker_node_wait
from utils.ray_distributed_utils import ray_worker_node_wait
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


def _protocol_model_candidates() -> list[str]:
    """Model ids used as pytest params in interface protocol suites."""
    seen: set[str] = set()
    out: list[str] = []
    for name in (*RESTFUL_MODEL_LIST, *RESTFUL_BASE_MODEL_LIST, *TOOL_REASONING_MODEL_LIST):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _pytest_k_expr(model: str, backend: str) -> str:
    """Build ``-k`` expression that does not also match longer sibling ids.

    Pytest ``-k`` is substring match, so ``Qwen/...-A3B`` would also select
    ``Qwen/...-A3B-FP8``. Exclude every known param id that contains ``model``.
    """
    parts = [model, backend]
    for other in _protocol_model_candidates():
        if other != model and model in other:
            parts.append(f'not {other}')
    return ' and '.join(parts)


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


def _run_interface_suites(
    config,
    run_config,
    port: int,
    *,
    via_proxy: bool = False,
) -> None:
    """Execute configured interface protocol suites against ``port``.

    ``via_proxy=True`` skips suites/cases that return large token-id /
    routed-experts payloads (proxy response size limits): the whole
    ``generate`` suite (``/generate`` always emits ``output_ids``), and
    toolcall tests marked ``experts`` (return_token_ids / routed_experts /
    encode+input_ids paths).
    """
    del config  # reserved for future path overrides
    model = run_config['model']
    backend = run_config['backend']
    case_info = list(run_config.get('case_info') or [])
    generate_marker = run_config.get('generate_marker') or f'not not_{backend}'
    n_workers = _suite_workers()

    if via_proxy and 'generate' in case_info:
        case_info = [c for c in case_info if c != 'generate']
        print(
            'proxy: skipping generate suite '
            '(/generate output_ids / routed_experts exceed proxy limits)',
            flush=True,
        )

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

    k_expr = _pytest_k_expr(model, backend)
    failures: list[str] = []

    toolcall_marker = f'tool_call and not not_{backend}'
    if via_proxy:
        # Exclude return_token_ids / routed_experts / encode(input_ids) cases.
        toolcall_marker += ' and not experts'

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
            toolcall_marker,
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
        _run_interface_suites(config, run_config, port)
    finally:
        if pid > 0:
            terminate_restful_api(worker_id)


def run_interface_restful_ray_distributed_test(config, run_config, manager) -> None:
    """Run interface suites against a Ray multi-node api_server (tp16)."""
    assert manager is not None, 'Manager instance must be provided'
    if manager.is_master:
        manager.start_lmdeploy_api_server(config=config, run_config=run_config)
        try:
            _run_interface_suites(config, run_config, PROXY_PORT)
        finally:
            manager.cleanup(force=False)
    else:
        time.sleep(10)
        ray_worker_node_wait(manager, timeout_minutes=4880)


def run_interface_restful_proxy_distributed_test(config, run_config, manager) -> None:
    """Run interface suites against LMDeploy proxy (dp/ep multi-node).

    Skips ``generate`` and toolcall ``experts``-marked cases: proxy cannot
    safely carry large ``/generate`` / encode / return_token_ids /
    routed_experts payloads.
    """
    assert manager is not None, 'Manager instance must be provided'
    api_server = ApiServerPerTest(proxy_manager=manager, config=config, run_config=run_config)
    api_server.start()
    try:
        if manager.is_master:
            api_server.wait_until_ready()
            _run_interface_suites(config, run_config, PROXY_PORT, via_proxy=True)
        else:
            print(f'⏸️ Worker node {manager.node_rank} waiting for master to complete test...')
            proxy_worker_node_wait(manager, timeout_minutes=4880)
    finally:
        api_server.cleanup()
        if manager.is_master:
            time.sleep(1)
