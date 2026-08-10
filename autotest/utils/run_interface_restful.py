"""Run interface REST suites with a per-test api_server (GPU-concurrent)."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import allure
from utils.config_utils import get_case_str_by_config, get_workerid
from utils.constant import (
    DEFAULT_PORT,
    PROXY_PORT,
    RESTFUL_BASE_MODEL_LIST,
    RESTFUL_MODEL_LIST,
    TOOL_REASONING_MODEL_LIST,
)
from utils.proxy_distributed_utils import ApiServerPerTest
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


def _phase_run_configs(run_config: dict) -> list[dict]:
    """Expand ``run_config`` into one cfg per launch profile / api_server
    phase.

    Prefers ``interface_phases`` from :func:`get_interface_run_config_list`
    (each item: ``suites``, ``case_info``, ``extra_params``). Falls back to a
    single phase using top-level ``case_info`` / ``extra_params``.
    """
    phases = list(run_config.get('interface_phases') or [])
    if not phases:
        case_info = list(run_config.get('case_info') or [])
        if not case_info:
            return []
        return [copy.deepcopy(run_config)]

    out: list[dict] = []
    for phase in phases:
        case_info = list(phase.get('case_info') or [])
        if not case_info:
            continue
        cfg = copy.deepcopy(run_config)
        cfg['case_info'] = case_info
        cfg['extra_params'] = copy.deepcopy(phase.get('extra_params') or {})
        if phase.get('suites') is not None:
            cfg['interface_suites'] = list(phase['suites'])
        out.append(cfg)
    return out


def _read_log_tail(log_path: str, max_lines: int = 80) -> str:
    """Return the last ``max_lines`` of a nested suite log for CI output."""
    try:
        with open(log_path, encoding='utf-8', errors='replace') as log_file:
            lines = log_file.readlines()
    except OSError as exc:
        return f'(failed to read log {log_path}: {exc})'
    if not lines:
        return f'(empty log: {log_path})'
    return ''.join(lines[-max_lines:]).rstrip()


def _pytest_cmd(
    test_path: str,
    *,
    k_expr: str,
    m_expr: str | None,
    env: dict[str, str],
    n_workers: int,
    log_path: str,
    reruns: int = 5,
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
        '--tb=short',
        f'--reruns={reruns}',
        '--reruns-delay=1',
        '-p',
        'no:cacheprovider',
    ]
    if m_expr:
        cmd.extend(['-m', m_expr])
    # Concurrent HTTP load against the worker-local api_server (fills GPU).
    if n_workers > 1:
        cmd.extend(['-n', str(n_workers), '--dist=load'])
    with open(log_path, 'w') as log_file:
        log_file.write(f"interface suite cmd: {' '.join(cmd)}\n")
        log_file.flush()
        completed = subprocess.run(cmd, env=env, cwd=str(_REPO_ROOT), stdout=log_file, stderr=subprocess.STDOUT)
    allure.attach.file(log_path, name=os.path.basename(log_path), attachment_type=allure.attachment_type.TEXT)
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
    model = run_config['model']
    backend = run_config['backend']
    case_info = list(run_config.get('case_info') or [])
    generate_marker = run_config.get('generate_marker') or f'not not_{backend}'
    n_workers = _suite_workers()
    log_dir = config.get('log_path') or str(_AUTOTEST_ROOT)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    case_str = get_case_str_by_config(run_config)

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

    anthropic_marker = f'anthropic and not not_{backend}'

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
            'anthropic_v1',
            'autotest/interface/restful/test_restful_anthropic_v1.py',
            anthropic_marker,
        ),
        (
            'anthropic_sdk',
            'autotest/interface/restful/test_restful_anthropic_sdk_messages.py',
            anthropic_marker,
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
        log_path = os.path.join(log_dir, f'log_interface_{case_name}_{case_str}_{port}_{timestamp}.log')
        rc = _pytest_cmd(
            rel_path,
            k_expr=k_expr,
            m_expr=marker,
            env=env,
            n_workers=n_workers,
            log_path=log_path,
            reruns=5,
        )
        if rc != 0:
            tail = _read_log_tail(log_path)
            print(
                f'--- nested suite failed: {case_name} exit={rc} log={log_path} ---\n{tail}\n',
                flush=True,
            )
            failures.append(
                f'{case_name} (exit={rc}, log={log_path})\n'
                f'----- {case_name} log tail -----\n{tail}'
            )

    assert not failures, (
        f'interface restful failures for {backend} {model} port={port}:\n'
        + '\n\n'.join(failures)
    )


def run_interface_restful_test(config, run_config, worker_id) -> None:
    """Start api_server per interface launch profile, then run that phase's
    suites.

    Each yaml ``{suites, extra}`` profile is one phase (own ``extra_params``).
    """
    port = DEFAULT_PORT + get_workerid(worker_id)
    for phase_idx, phase_cfg in enumerate(_phase_run_configs(run_config)):
        print(
            f'interface phase {phase_idx}: suites={phase_cfg.get("interface_suites")} '
            f'cases={phase_cfg.get("case_info")} '
            f'extra_keys={sorted((phase_cfg.get("extra_params") or {}).keys())}',
            flush=True,
        )
        pid, content = start_openai_service(config, phase_cfg, worker_id)
        try:
            assert pid > 0, f'Failed to start RESTful API server (phase {phase_idx}): {content}'
            _run_interface_suites(config, phase_cfg, port)
        finally:
            if pid > 0:
                terminate_restful_api(worker_id)


def run_interface_restful_ray_distributed_test(config, run_config, manager) -> None:
    """Run interface suites against a Ray multi-node api_server (tp16).

    One api_server restart per launch profile. Workers wait on Ray GCS;
    intermediate ``cleanup(force=False)`` does not tear down the cluster.
    """
    assert manager is not None, 'Manager instance must be provided'
    phases = _phase_run_configs(run_config)

    if manager.is_master:
        try:
            for phase_idx, phase_cfg in enumerate(phases):
                print(
                    f'interface phase {phase_idx}: suites={phase_cfg.get("interface_suites")} '
                    f'cases={phase_cfg.get("case_info")} '
                    f'extra_keys={sorted((phase_cfg.get("extra_params") or {}).keys())}',
                    flush=True,
                )
                manager.start_lmdeploy_api_server(config=config, run_config=phase_cfg)
                try:
                    _run_interface_suites(config, phase_cfg, PROXY_PORT)
                finally:
                    manager.cleanup(force=False)
        finally:
            manager.cleanup(force=False)
    else:
        time.sleep(10)
        ray_worker_node_wait(manager, timeout_minutes=4880)


def _proxy_phase_flag_path(config, run_config, phase_idx: int) -> str:
    log_dir = config.get('log_path') or config.get('server_log_path') or '/tmp'
    case_str = get_case_str_by_config(run_config)
    return os.path.join(log_dir, f'.interface_phase_done_{case_str}_{phase_idx}')


def _proxy_worker_wait_phase_done(flag_path: str, timeout_minutes: int = 4880) -> None:
    """Worker waits until master writes the phase-done flag (shared log fs)."""
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        if os.path.exists(flag_path):
            return
        time.sleep(5)
    raise TimeoutError(f'proxy worker timed out waiting for phase flag {flag_path}')


def run_interface_restful_proxy_distributed_test(config, run_config, manager) -> None:
    """Run interface suites against LMDeploy proxy (dp/ep multi-node).

    Skips ``generate`` and toolcall ``experts``-marked cases: proxy cannot
    safely carry large ``/generate`` / encode / return_token_ids /
    routed_experts payloads.

    One ``ApiServerPerTest`` restart per launch profile. All ranks join each
    phase; workers sync via a shared done-flag.
    """
    assert manager is not None, 'Manager instance must be provided'
    phases = _phase_run_configs(run_config)

    for phase_idx, phase_cfg in enumerate(phases):
        if manager.is_master:
            print(
                f'interface phase {phase_idx}: suites={phase_cfg.get("interface_suites")} '
                f'cases={phase_cfg.get("case_info")} '
                f'extra_keys={sorted((phase_cfg.get("extra_params") or {}).keys())}',
                flush=True,
            )

        flag_path = _proxy_phase_flag_path(config, phase_cfg, phase_idx)
        if manager.is_master and os.path.exists(flag_path):
            os.remove(flag_path)

        api_server = ApiServerPerTest(proxy_manager=manager, config=config, run_config=phase_cfg)
        api_server.start()
        try:
            if manager.is_master:
                api_server.wait_until_ready()
                _run_interface_suites(config, phase_cfg, PROXY_PORT, via_proxy=True)
                Path(flag_path).touch()
            else:
                print(
                    f'⏸️ Worker node {manager.node_rank} waiting for master phase '
                    f'{phase_idx} ({phase_cfg.get("case_info")})...',
                )
                _proxy_worker_wait_phase_done(flag_path)
        finally:
            api_server.cleanup()
            if manager.is_master:
                time.sleep(1)
