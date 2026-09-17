# Copyright (c) OpenMMLab. All rights reserved.
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

DEFAULT_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
DEFAULT_STARTUP_TIMEOUT = 300.0


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _wait_for_health(url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = 'no response'
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'process exited with status {process.returncode}')
        try:
            with urllib.request.urlopen(f'{url}/health', timeout=2) as response:
                if response.status == 200:
                    return
                last_error = f'HTTP {response.status}'
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f'timed out waiting for {url}: {last_error}')


def _start(command: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        log_file.write('COMMAND: ' + ' '.join(command) + '\n\n')
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _api_server_command(
    model: str,
    port: int,
    *,
    model_name: str | None = None,
    role: str = 'Hybrid',
    proxy_url: str | None = None,
    tp: int = 1,
) -> list[str]:
    command = [
        'lmdeploy',
        'serve',
        'api_server',
        model,
        '--server-name',
        '127.0.0.1',
        '--server-port',
        str(port),
        '--role',
        role,
        '--tp',
        str(tp),
    ]
    if model_name:
        command.extend(['--model-name', model_name])
    if proxy_url:
        command.extend(['--proxy-url', proxy_url])
    return command


def _discover_model(url: str) -> str:
    response = requests.get(f'{url}/v1/models', timeout=10)
    response.raise_for_status()
    payload = response.json()
    for item in payload.get('data', []):
        if isinstance(item, dict) and isinstance(item.get('id'), str):
            return item['id']
    if isinstance(payload.get('id'), str):
        return payload['id']
    raise RuntimeError(f'{url}/v1/models returned no model ID')


def pytest_configure(config):
    config.addinivalue_line('markers', 'e2e: requires a real LMDeploy server')


@pytest.fixture(scope='session')
def e2e_model() -> str:
    return os.getenv('LMDEPLOY_E2E_MODEL', DEFAULT_MODEL)


@pytest.fixture(scope='session')
def e2e_model_name() -> str:
    return os.getenv('LMDEPLOY_E2E_MODEL_NAME', Path(e2e_model.__wrapped__()).name)


@pytest.fixture(scope='session')
def e2e_embedding_model() -> str | None:
    return os.getenv('LMDEPLOY_E2E_EMBEDDING_MODEL') or None


@pytest.fixture(scope='session')
def e2e_startup_timeout() -> float:
    return float(os.getenv('LMDEPLOY_E2E_STARTUP_TIMEOUT', str(DEFAULT_STARTUP_TIMEOUT)))


@pytest.fixture(scope='session')
def e2e_artifact_dir() -> Path:
    directory = Path(os.getenv('LMDEPLOY_E2E_ARTIFACT_DIR', '/tmp/lmdeploy-router-e2e'))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture
def e2e_router_only_rr(e2e_artifact_dir):
    port = _find_available_port()
    url = f'http://127.0.0.1:{port}'
    process = _start(
        ['lmdeploy-router', '--host', '127.0.0.1', '--port', str(port), '--policy', 'round_robin'],
        e2e_artifact_dir / f'router-{port}.log',
    )
    try:
        _wait_for_health(url, process, 60)
        yield SimpleNamespace(process=process, url=url)
    finally:
        _stop(process)


@pytest.fixture(scope='session')
def e2e_primary_worker(e2e_model, e2e_model_name, e2e_startup_timeout, e2e_artifact_dir):
    if not os.getenv('LMDEPLOY_E2E_RUN_REGULAR'):
        pytest.skip('set LMDEPLOY_E2E_RUN_REGULAR=1 to run the managed LMDeploy worker e2e')
    port = _find_available_port()
    url = f'http://127.0.0.1:{port}'
    command = _api_server_command(e2e_model, port, model_name=e2e_model_name)
    process = _start(command, e2e_artifact_dir / 'regular-worker.log')
    try:
        _wait_for_health(url, process, e2e_startup_timeout)
        yield SimpleNamespace(process=process, url=url)
    finally:
        _stop(process)


@pytest.fixture(scope='session')
def e2e_embedding_worker(e2e_embedding_model, e2e_startup_timeout, e2e_artifact_dir):
    if not e2e_embedding_model:
        pytest.skip('set LMDEPLOY_E2E_EMBEDDING_MODEL to run embeddings e2e')
    port = _find_available_port()
    url = f'http://127.0.0.1:{port}'
    command = _api_server_command(e2e_embedding_model, port)
    process = _start(command, e2e_artifact_dir / 'embedding-worker.log')
    try:
        _wait_for_health(url, process, e2e_startup_timeout)
        yield SimpleNamespace(process=process, url=url)
    finally:
        _stop(process)


def _split_urls(value: str) -> tuple[str, ...]:
    return tuple(url.strip().rstrip('/') for url in value.split(',') if url.strip())


@pytest.fixture(scope='module')
def e2e_pd_cluster(e2e_model, e2e_artifact_dir):
    prefill_urls = _split_urls(os.getenv('LMDEPLOY_E2E_PREFILL_URLS', ''))
    decode_urls = _split_urls(os.getenv('LMDEPLOY_E2E_DECODE_URLS', ''))
    if not prefill_urls or not decode_urls:
        pytest.skip(
            'set LMDEPLOY_E2E_PREFILL_URLS and LMDEPLOY_E2E_DECODE_URLS to run LMDeploy PD e2e'
        )

    port = _find_available_port()
    url = f'http://127.0.0.1:{port}'
    command = [
        'lmdeploy-router', '--host', '127.0.0.1', '--port', str(port), '--policy', 'round_robin',
        '--lmdeploy-pd-disaggregation',
    ]
    for prefill_url in prefill_urls:
        command.extend(['--prefill', prefill_url])
    for decode_url in decode_urls:
        command.extend(['--decode', decode_url])
    process = _start(command, e2e_artifact_dir / f'pd-router-{port}.log')
    try:
        _wait_for_health(url, process, 60)
        try:
            model_id = _discover_model(prefill_urls[0])
        except RuntimeError:
            model_id = e2e_model
        yield SimpleNamespace(
            process=process, url=url, model=model_id, prefill_urls=prefill_urls, decode_urls=decode_urls
        )
    finally:
        _stop(process)
