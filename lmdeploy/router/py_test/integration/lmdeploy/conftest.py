# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests

from py_test.fixtures.ports import find_free_port

DEFAULT_TOKEN_IDS = [
    151644,
    8948,
    198,
    2610,
    525,
    264,
    109050,
    151645,
    198,
    151644,
    77091,
    198,
]


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'lmdeploy: requires externally managed LMDeploy API servers'
    )


def _split_urls(value: str) -> tuple[str, ...]:
    return tuple(url.strip().rstrip('/') for url in value.split(',') if url.strip())


def _discover_models(payload: dict) -> list[str]:
    if isinstance(payload.get('data'), list):
        return [
            item['id']
            for item in payload['data']
            if isinstance(item, dict) and isinstance(item.get('id'), str)
        ]
    if isinstance(payload.get('id'), str):
        return [payload['id']]
    return []


def _wait_for_http(url: str, timeout: float, process=None) -> None:
    deadline = time.monotonic() + timeout
    last_error = 'no response'
    with requests.Session() as session:
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(f'router exited with status {process.returncode}')
            try:
                response = session.get(url, timeout=2)
                if response.status_code == 200:
                    return
                last_error = f'HTTP {response.status_code}: {response.text[:200]}'
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(0.2)
    raise TimeoutError(f'timed out waiting for {url}: {last_error}')


@dataclass(frozen=True)
class LMDeployTestConfig:
    backend_urls: tuple[str, ...]
    model: str
    token_ids: tuple[int, ...]
    request_timeout: float
    pd_prefill_url: str | None
    pd_decode_url: str | None


@dataclass
class RouterHandle:
    process: subprocess.Popen
    url: str
    log_path: Path
    _log_file: object


class LMDeployRouterFactory:
    def __init__(
        self,
        binary: Path,
        config: LMDeployTestConfig,
        artifact_dir: Path,
    ) -> None:
        self.binary = binary
        self.config = config
        self.artifact_dir = artifact_dir
        self._handles: list[RouterHandle] = []

    def start(
        self,
        *,
        policy: str = 'random',
        worker_urls: Sequence[str] | None = None,
        lmdeploy_pd: bool = False,
        prefill_urls: Sequence[str] = (),
        decode_urls: Sequence[str] = (),
    ) -> RouterHandle:
        port = find_free_port()
        prometheus_port = find_free_port()
        if lmdeploy_pd:
            urls = ()
        elif worker_urls is None:
            urls = self.config.backend_urls
        else:
            urls = tuple(worker_urls)
        mode = 'pd' if lmdeploy_pd else 'regular'
        log_path = self.artifact_dir / f'router-{mode}-{policy}-{port}.log'
        log_file = log_path.open('w', encoding='utf-8')

        command = [
            str(self.binary),
            '--host',
            '127.0.0.1',
            '--port',
            str(port),
            '--policy',
            policy,
            '--prometheus-port',
            str(prometheus_port),
            '--prometheus-host',
            '127.0.0.1',
        ]
        if urls:
            command.extend(['--worker-urls', *urls])
        if lmdeploy_pd:
            command.extend(
                [
                    '--lmdeploy-pd-disaggregation',
                    '--lmdeploy-migration-protocol',
                    'rdma',
                    '--lmdeploy-rdma-link-type',
                    'roce',
                    '--prefill-policy',
                    policy,
                    '--decode-policy',
                    policy,
                ]
            )
            for url in prefill_urls:
                command.extend(['--prefill', url])
            for url in decode_urls:
                command.extend(['--decode', url])

        log_file.write('COMMAND: ' + ' '.join(command) + '\n\n')
        log_file.flush()
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle = RouterHandle(
            process=process,
            url=f'http://127.0.0.1:{port}',
            log_path=log_path,
            _log_file=log_file,
        )
        self._handles.append(handle)
        try:
            _wait_for_http(f'{handle.url}/health', timeout=30, process=process)
        except Exception as exc:
            self.stop(handle)
            tail = log_path.read_text(encoding='utf-8', errors='replace')[-4000:]
            raise RuntimeError(f'failed to start router; log tail:\n{tail}') from exc
        return handle

    def stop(self, handle: RouterHandle) -> None:
        if handle.process.poll() is None:
            handle.process.terminate()
            try:
                handle.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handle.process.kill()
                handle.process.wait(timeout=5)
        if not handle._log_file.closed:
            handle._log_file.close()
        if handle in self._handles:
            self._handles.remove(handle)

    def stop_all(self) -> None:
        for handle in list(self._handles):
            self.stop(handle)


@pytest.fixture(scope='session')
def lmdeploy_config() -> LMDeployTestConfig:
    raw_urls = os.getenv('LMDEPLOY_BACKEND_URLS', '')
    if not raw_urls:
        pytest.skip(
            'set LMDEPLOY_BACKEND_URLS=url1,url2 to run real LMDeploy integration tests'
        )
    backend_urls = _split_urls(raw_urls)
    if len(backend_urls) < 2:
        pytest.fail('LMDEPLOY_BACKEND_URLS must contain at least two backend URLs')

    timeout = float(os.getenv('LMDEPLOY_REQUEST_TIMEOUT', '120'))
    discovered_models: list[str] = []
    backend_model_sets: list[set[str]] = []
    with requests.Session() as session:
        for backend_url in backend_urls:
            _wait_for_http(f'{backend_url}/health', timeout=10)
            response = session.get(f'{backend_url}/v1/models', timeout=10)
            response.raise_for_status()
            models = _discover_models(response.json())
            if not models:
                pytest.fail(f'{backend_url}/v1/models returned no model IDs')
            discovered_models.extend(models)
            backend_model_sets.append(set(models))

    model = os.getenv('LMDEPLOY_MODEL') or discovered_models[0]
    if any(model not in models for models in backend_model_sets):
        pytest.fail(
            f'model {model!r} was not reported by the configured LMDeploy backends'
        )

    raw_token_ids = os.getenv('LMDEPLOY_TOKEN_IDS')
    token_ids = DEFAULT_TOKEN_IDS
    if raw_token_ids:
        parsed = json.loads(raw_token_ids)
        if (
            not isinstance(parsed, list)
            or not parsed
            or not all(isinstance(item, int) for item in parsed)
        ):
            pytest.fail('LMDEPLOY_TOKEN_IDS must be a non-empty JSON list of integers')
        token_ids = parsed

    return LMDeployTestConfig(
        backend_urls=backend_urls,
        model=model,
        token_ids=tuple(token_ids),
        request_timeout=timeout,
        pd_prefill_url=os.getenv('LMDEPLOY_PD_PREFILL_URL'),
        pd_decode_url=os.getenv('LMDEPLOY_PD_DECODE_URL'),
    )


@pytest.fixture(scope='session')
def lmdeploy_router_binary(lmdeploy_config) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    configured = os.getenv('LMDEPLOY_ROUTER_BIN')
    if configured:
        candidates = [Path(configured)]
    else:
        target_dir = Path(os.getenv('CARGO_TARGET_DIR', repo_root / 'target'))
        candidates = [
            target_dir / 'release' / 'lmdeploy-router',
            target_dir / 'debug' / 'lmdeploy-router',
        ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    pytest.fail(
        'lmdeploy-router binary not found; run '
        '`cargo build --release --bin lmdeploy-router` or set LMDEPLOY_ROUTER_BIN'
    )


@pytest.fixture(scope='session')
def lmdeploy_router_factory(
    lmdeploy_router_binary: Path,
    lmdeploy_config: LMDeployTestConfig,
) -> Iterable[LMDeployRouterFactory]:
    repo_root = Path(__file__).resolve().parents[3]
    artifact_dir = Path(
        os.getenv(
            'LMDEPLOY_TEST_ARTIFACT_DIR',
            repo_root / '.test-artifacts' / 'lmdeploy',
        )
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    factory = LMDeployRouterFactory(
        binary=lmdeploy_router_binary,
        config=lmdeploy_config,
        artifact_dir=artifact_dir,
    )
    try:
        yield factory
    finally:
        factory.stop_all()


@pytest.fixture(scope='module')
def lmdeploy_router(lmdeploy_router_factory) -> Iterable[RouterHandle]:
    handle = lmdeploy_router_factory.start(policy='random')
    try:
        yield handle
    finally:
        lmdeploy_router_factory.stop(handle)
